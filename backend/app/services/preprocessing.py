import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Dict, Any, List, Tuple
import joblib
import os

from app.models.schemas import CompetitionConfig
from app.core.config import settings

class PreprocessingService:
    def __init__(self):
        self.artifacts_dir = settings.artifacts_dir
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}

    async def process(self, train_path: str, test_path: str, config: CompetitionConfig,
                     eda_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess training and test data
        """
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_col = config.target_column

            # Separate features and target
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            X_test = test_df.copy()

            # Store original column info
            original_columns = {
                "train": train_df.columns.tolist(),
                "test": test_df.columns.tolist(),
                "features": X_train.columns.tolist()
            }

            # Identify column types
            column_info = self._identify_column_types(X_train, eda_results)

            # Handle missing values
            X_train_processed, X_test_processed = self._handle_missing_values(
                X_train, X_test, column_info
            )

            # Encode categorical variables
            X_train_processed, X_test_processed = self._encode_categorical(
                X_train_processed, X_test_processed, column_info
            )

            # Feature scaling
            X_train_processed, X_test_processed = self._scale_features(
                X_train_processed, X_test_processed, column_info
            )

            # Feature selection (optional)
            if config.feature_engineering:
                X_train_processed, X_test_processed = self._feature_selection(
                    X_train_processed, y_train, X_test_processed, config
                )

            # Create preprocessing pipeline info
            preprocessing_info = {
                "original_shape": X_train.shape,
                "processed_shape": X_train_processed.shape,
                "feature_names": X_train_processed.columns.tolist(),
                "target_name": target_col,
                "column_types": column_info,
                "preprocessing_steps": [
                    "missing_value_imputation",
                    "categorical_encoding",
                    "feature_scaling"
                ]
            }

            # Save preprocessing artifacts
            self._save_preprocessing_artifacts(config.task_type)

            return {
                "X_train": X_train_processed,
                "X_test": X_test_processed,
                "y_train": y_train,
                "preprocessing_info": preprocessing_info,
                "original_columns": original_columns
            }

        except Exception as e:
            raise Exception(f"Preprocessing failed: {str(e)}")

    def _identify_column_types(self, df: pd.DataFrame, eda_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Identify different types of columns"""

        # Use EDA results if available
        if "feature_analysis" in eda_results:
            feature_analysis = eda_results["feature_analysis"]

            numerical_cols = [
                col for col, info in feature_analysis.items()
                if info["type"] == "numerical" and col in df.columns
            ]

            categorical_cols = [
                col for col, info in feature_analysis.items()
                if info["type"] == "categorical" and col in df.columns
            ]

        else:
            # Fallback to dtype-based identification
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        return {
            "numerical": numerical_cols,
            "categorical": categorical_cols
        }

    def _handle_missing_values(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                              column_info: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Handle missing values in training and test data"""

        X_train_clean = X_train.copy()
        X_test_clean = X_test.copy()

        # Handle numerical columns
        numerical_cols = column_info["numerical"]
        if numerical_cols:
            # Use median for numerical columns
            num_imputer = SimpleImputer(strategy='median')
            X_train_clean[numerical_cols] = num_imputer.fit_transform(X_train_clean[numerical_cols])
            X_test_clean[numerical_cols] = num_imputer.transform(X_test_clean[numerical_cols])

            self.imputers['numerical'] = num_imputer

        # Handle categorical columns
        categorical_cols = column_info["categorical"]
        if categorical_cols:
            # Use most frequent for categorical columns
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_train_clean[categorical_cols] = cat_imputer.fit_transform(X_train_clean[categorical_cols])
            X_test_clean[categorical_cols] = cat_imputer.transform(X_test_clean[categorical_cols])

            self.imputers['categorical'] = cat_imputer

        return X_train_clean, X_test_clean

    def _encode_categorical(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                          column_info: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical variables"""

        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()

        categorical_cols = column_info["categorical"]

        for col in categorical_cols:
            if col in X_train_encoded.columns:
                # Determine encoding strategy based on cardinality
                unique_count = X_train_encoded[col].nunique()

                if unique_count <= 10:  # Low cardinality - use one-hot encoding
                    # One-hot encode
                    dummies_train = pd.get_dummies(X_train_encoded[col], prefix=col, drop_first=True)
                    dummies_test = pd.get_dummies(X_test_encoded[col], prefix=col, drop_first=True)

                    # Align columns (test set might not have all categories)
                    dummies_test = dummies_test.reindex(columns=dummies_train.columns, fill_value=0)

                    # Drop original column and add encoded columns
                    X_train_encoded = pd.concat([X_train_encoded.drop(col, axis=1), dummies_train], axis=1)
                    X_test_encoded = pd.concat([X_test_encoded.drop(col, axis=1), dummies_test], axis=1)

                else:  # High cardinality - use label encoding
                    le = LabelEncoder()
                    X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))

                    # Handle unseen categories in test set
                    X_test_encoded[col] = X_test_encoded[col].astype(str)
                    mask = X_test_encoded[col].isin(le.classes_)
                    X_test_encoded.loc[mask, col] = le.transform(X_test_encoded.loc[mask, col])
                    X_test_encoded.loc[~mask, col] = -1  # Unknown categories

                    self.encoders[col] = le

        return X_train_encoded, X_test_encoded

    def _scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                       column_info: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale numerical features"""

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        # Get numerical columns (after encoding)
        numerical_cols = X_train_scaled.select_dtypes(include=[np.number]).columns.tolist()

        if numerical_cols:
            scaler = StandardScaler()
            X_train_scaled[numerical_cols] = scaler.fit_transform(X_train_scaled[numerical_cols])
            X_test_scaled[numerical_cols] = scaler.transform(X_test_scaled[numerical_cols])

            self.scalers['standard'] = scaler

        return X_train_scaled, X_test_scaled

    def _feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, config: CompetitionConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform feature selection"""

        if len(X_train.columns) <= 20:  # Skip if already few features
            return X_train, X_test

        # Choose appropriate scoring function
        if config.task_type in ["binary_classification", "multiclass_classification"]:
            score_func = f_classif
        else:  # regression
            score_func = f_regression

        # Select top features
        k = min(50, len(X_train.columns))  # Select top 50 or all if less
        selector = SelectKBest(score_func=score_func, k=k)

        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        # Get selected feature names
        selected_features = X_train.columns[selector.get_support()].tolist()

        # Convert back to DataFrame
        X_train_df = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)

        return X_train_df, X_test_df

    def _save_preprocessing_artifacts(self, task_type: str):
        """Save preprocessing objects for later use"""
        preprocessing_dir = os.path.join(self.artifacts_dir, "preprocessing")
        os.makedirs(preprocessing_dir, exist_ok=True)

        # Save encoders
        if self.encoders:
            joblib.dump(self.encoders, os.path.join(preprocessing_dir, "encoders.joblib"))

        # Save scalers
        if self.scalers:
            joblib.dump(self.scalers, os.path.join(preprocessing_dir, "scalers.joblib"))

        # Save imputers
        if self.imputers:
            joblib.dump(self.imputers, os.path.join(preprocessing_dir, "imputers.joblib"))