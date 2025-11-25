import pandas as pd
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold, KFold, TimeSeriesSplit, GroupKFold,
    cross_val_score, cross_validate
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from app.models.schemas import CompetitionConfig, CVStrategy

class CrossValidationService:
    def __init__(self):
        self.cv_strategies = {
            CVStrategy.STRATIFIED_KFOLD: StratifiedKFold,
            CVStrategy.KFOLD: KFold,
            CVStrategy.TIME_SERIES_SPLIT: TimeSeriesSplit,
            CVStrategy.GROUP_KFOLD: GroupKFold
        }

    async def setup_cv(self, processed_data: Dict[str, Any], config: CompetitionConfig) -> Dict[str, Any]:
        """
        Set up cross-validation strategy based on data characteristics
        """
        try:
            X_train = processed_data["X_train"]
            y_train = processed_data["y_train"]

            # Choose CV strategy
            cv_strategy = self._choose_cv_strategy(X_train, y_train, config)

            # Create CV splitter
            cv_splitter = self._create_cv_splitter(cv_strategy, config.cv_folds, X_train, y_train)

            # Detect potential leakage
            leakage_info = self._detect_leakage(X_train, y_train, cv_splitter)

            # Validate strategy
            validation_results = self._validate_cv_strategy(X_train, y_train, cv_splitter, config)

            return {
                "cv_strategy": cv_strategy,
                "cv_splitter": cv_splitter,
                "leakage_info": leakage_info,
                "validation_results": validation_results,
                "fold_count": config.cv_folds
            }

        except Exception as e:
            raise Exception(f"Cross-validation setup failed: {str(e)}")

    def _choose_cv_strategy(self, X_train: pd.DataFrame, y_train: pd.Series,
                           config: CompetitionConfig) -> str:
        """
        Automatically choose the best CV strategy based on data characteristics
        """

        # If user specified a strategy, use it
        if config.cv_strategy:
            return config.cv_strategy

        # Automatic strategy selection
        if config.task_type == "regression":
            # For regression, check if data is temporal
            if self._is_temporal_data(X_train):
                return CVStrategy.TIME_SERIES_SPLIT
            else:
                return CVStrategy.KFOLD

        else:  # classification
            # Check if classes are balanced
            class_counts = y_train.value_counts()
            min_class_ratio = class_counts.min() / class_counts.max()

            if min_class_ratio < 0.1:  # Highly imbalanced
                return CVStrategy.STRATIFIED_KFOLD
            elif self._has_group_structure(X_train):
                return CVStrategy.GROUP_KFOLD
            else:
                return CVStrategy.STRATIFIED_KFOLD

    def _create_cv_splitter(self, strategy: str, n_folds: int,
                           X_train: pd.DataFrame, y_train: pd.Series):
        """Create the cross-validation splitter"""

        if strategy == CVStrategy.STRATIFIED_KFOLD:
            return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        elif strategy == CVStrategy.KFOLD:
            return KFold(n_splits=n_folds, shuffle=True, random_state=42)

        elif strategy == CVStrategy.TIME_SERIES_SPLIT:
            return TimeSeriesSplit(n_splits=n_folds)

        elif strategy == CVStrategy.GROUP_KFOLD:
            # For group K-fold, we need to identify groups
            # This is a simplified approach - in practice, you'd want domain knowledge
            groups = self._identify_groups(X_train)
            return GroupKFold(n_splits=n_folds)

        else:
            # Default to stratified K-fold
            return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    def _detect_leakage(self, X_train: pd.DataFrame, y_train: pd.Series, cv_splitter) -> Dict[str, Any]:
        """
        Detect potential data leakage issues
        """
        leakage_issues = []

        # Check for near-perfect correlation features
        correlation_matrix = X_train.corr().abs()
        high_corr_pairs = []

        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))

        if high_corr_pairs:
            leakage_issues.append(f"Highly correlated features found: {high_corr_pairs}")

        # Check for target leakage (features that are too predictive)
        leakage_score = self._check_target_leakage(X_train, y_train)
        if leakage_score > 0.98:
            leakage_issues.append(f"Potential target leakage detected (score: {leakage_score:.3f})")

        # Check for duplicate features
        duplicate_features = []
        for i, col1 in enumerate(X_train.columns):
            for col2 in X_train.columns[i+1:]:
                if X_train[col1].equals(X_train[col2]):
                    duplicate_features.append((col1, col2))

        if duplicate_features:
            leakage_issues.append(f"Duplicate features found: {duplicate_features}")

        return {
            "issues": leakage_issues,
            "high_correlation_pairs": high_corr_pairs,
            "duplicate_features": duplicate_features,
            "target_leakage_score": leakage_score
        }

    def _check_target_leakage(self, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """
        Check for potential target leakage by training a simple model
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        # Use a simple Random Forest to check predictability
        if len(X_train.columns) == 0:
            return 0.0

        try:
            rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)

            # Use 3-fold CV to check scores
            scores = cross_val_score(rf, X_train, y_train, cv=3, scoring='accuracy')
            return scores.mean()
        except:
            return 0.0

    def _validate_cv_strategy(self, X_train: pd.DataFrame, y_train: pd.Series,
                            cv_splitter, config: CompetitionConfig) -> Dict[str, Any]:
        """
        Validate the chosen CV strategy
        """
        validation_results = {}

        # Check fold distribution
        fold_distributions = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train, y_train)):
            if hasattr(y_train, 'iloc'):
                val_y = y_train.iloc[val_idx]
            else:
                val_y = y_train[val_idx]

            if config.task_type in ["binary_classification", "multiclass_classification"]:
                fold_dist = val_y.value_counts().to_dict()
                fold_distributions.append({
                    "fold": fold_idx,
                    "size": len(val_idx),
                    "class_distribution": fold_dist
                })
            else:  # regression
                fold_dist = {
                    "mean": float(val_y.mean()),
                    "std": float(val_y.std()),
                    "min": float(val_y.min()),
                    "max": float(val_y.max())
                }
                fold_distributions.append({
                    "fold": fold_idx,
                    "size": len(val_idx),
                    "stats": fold_dist
                })

        validation_results["fold_distributions"] = fold_distributions

        # Calculate CV stability metric
        stability_scores = self._calculate_cv_stability(X_train, y_train, cv_splitter, config)
        validation_results["stability_scores"] = stability_scores

        return validation_results

    def _calculate_cv_stability(self, X_train: pd.DataFrame, y_train: pd.Series,
                              cv_splitter, config: CompetitionConfig) -> Dict[str, float]:
        """
        Calculate CV stability using a simple model
        """
        try:
            from sklearn.linear_model import LogisticRegression, LinearRegression
            from sklearn.metrics import get_scorer

            # Choose appropriate simple model
            if config.task_type in ["binary_classification", "multiclass_classification"]:
                model = LogisticRegression(max_iter=100, random_state=42)
                scoring = 'accuracy'
            else:  # regression
                model = LinearRegression()
                scoring = 'neg_mean_squared_error'

            # Get scoring function
            scorer = get_scorer(scoring)

            # Perform CV
            scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring=scoring)

            # Calculate stability metrics
            mean_score = scores.mean()
            std_score = scores.std()
            cv_score = std_score / mean_score if mean_score != 0 else float('inf')

            return {
                "mean_score": float(mean_score),
                "std_score": float(std_score),
                "cv_coefficient": float(cv_score),
                "stability": "high" if cv_score < 0.1 else "medium" if cv_score < 0.2 else "low"
            }

        except Exception as e:
            return {
                "error": str(e),
                "stability": "unknown"
            }

    def _is_temporal_data(self, X_train: pd.DataFrame) -> bool:
        """
        Check if data appears to be temporal
        """
        # Look for datetime columns or temporal patterns in column names
        datetime_cols = X_train.select_dtypes(include=['datetime64']).columns

        if len(datetime_cols) > 0:
            return True

        # Check column names for temporal patterns
        temporal_keywords = ['date', 'time', 'year', 'month', 'day', 'timestamp']
        temporal_cols = [
            col for col in X_train.columns
            if any(keyword in col.lower() for keyword in temporal_keywords)
        ]

        return len(temporal_cols) > 0

    def _has_group_structure(self, X_train: pd.DataFrame) -> bool:
        """
        Check if data has obvious group structure
        """
        # This is a simplified check - in practice, you'd need domain knowledge
        # Look for categorical columns with reasonable group sizes
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            unique_count = X_train[col].nunique()
            total_count = len(X_train)

            # If we have a reasonable number of groups (not too many, not too few)
            if 5 <= unique_count <= total_count // 10:
                return True

        return False

    def _identify_groups(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Identify groups for GroupKFold
        """
        # This is a simplified approach - in practice, you'd want domain knowledge
        # Use the first suitable categorical column as group identifier
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            unique_count = X_train[col].nunique()
            total_count = len(X_train)

            if 5 <= unique_count <= total_count // 10:
                return X_train[col].values

        # If no suitable column found, use a simple grouping
        return np.arange(len(X_train)) // (len(X_train) // 5)