import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from typing import Dict, Any, List
import json

from app.models.schemas import CompetitionConfig
from app.core.config import settings
from app.utils.plotting import create_correlation_heatmap, create_distribution_plots

class EDAService:
    def __init__(self):
        self.artifacts_dir = settings.artifacts_dir

    async def analyze(self, train_path: str, test_path: str, config: CompetitionConfig) -> Dict[str, Any]:
        """
        Perform comprehensive exploratory data analysis
        """
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Basic dataset information
            dataset_info = {
                "train_shape": train_df.shape,
                "test_shape": test_df.shape,
                "train_columns": train_df.columns.tolist(),
                "test_columns": test_df.columns.tolist(),
                "target_column": config.target_column
            }

            # Data types analysis
            dtypes_info = self._analyze_dtypes(train_df)

            # Missing values analysis
            missing_values = self._analyze_missing_values(train_df, test_df)

            # Target variable analysis
            target_analysis = self._analyze_target_variable(train_df, config)

            # Feature analysis
            feature_analysis = self._analyze_features(train_df, config)

            # Correlation analysis
            correlation_analysis = self._analyze_correlations(train_df, config)

            # Data quality assessment
            quality_assessment = self._assess_data_quality(train_df, test_df)

            # Generate visualizations
            visualizations = await self._generate_visualizations(
                train_df, test_df, config, correlation_analysis
            )

            return {
                "dataset_info": dataset_info,
                "dtypes_info": dtypes_info,
                "missing_values": missing_values,
                "target_analysis": target_analysis,
                "feature_analysis": feature_analysis,
                "correlation_analysis": correlation_analysis,
                "quality_assessment": quality_assessment,
                "visualizations": visualizations
            }

        except Exception as e:
            raise Exception(f"EDA failed: {str(e)}")

    def _analyze_dtypes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types and their distributions"""
        dtype_counts = df.dtypes.value_counts().to_dict()

        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()

        return {
            "dtype_counts": {str(k): v for k, v in dtype_counts.items()},
            "categorical_columns": categorical_columns,
            "numerical_columns": numerical_columns,
            "datetime_columns": datetime_columns,
            "column_details": {
                col: {
                    "dtype": str(df[col].dtype),
                    "unique_values": int(df[col].nunique()),
                    "sample_values": df[col].dropna().head(3).tolist()
                }
                for col in df.columns
            }
        }

    def _analyze_missing_values(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values patterns"""
        train_missing = train_df.isnull().sum()
        test_missing = test_df.isnull().sum()

        train_missing_pct = (train_missing / len(train_df)) * 100
        test_missing_pct = (test_missing / len(test_df)) * 100

        return {
            "train_missing": {
                col: {
                    "count": int(count),
                    "percentage": float(pct)
                }
                for col, (count, pct) in zip(train_missing.index, zip(train_missing, train_missing_pct))
                if count > 0
            },
            "test_missing": {
                col: {
                    "count": int(count),
                    "percentage": float(pct)
                }
                for col, (count, pct) in zip(test_missing.index, zip(test_missing, test_missing_pct))
                if count > 0
            },
            "complete_features": train_df.columns[train_df.isnull().sum() == 0].tolist()
        }

    def _analyze_target_variable(self, df: pd.DataFrame, config: CompetitionConfig) -> Dict[str, Any]:
        """Analyze target variable distribution"""
        target_col = config.target_column

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        target_series = df[target_col]

        if config.task_type in ["binary_classification", "multiclass_classification"]:
            value_counts = target_series.value_counts()
            class_distribution = value_counts.to_dict()

            return {
                "type": "classification",
                "unique_classes": len(value_counts),
                "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
                "class_balance": {
                    "majority_class": str(value_counts.index[0]),
                    "majority_ratio": float(value_counts.iloc[0] / len(target_series)),
                    "minority_class": str(value_counts.index[-1]),
                    "minority_ratio": float(value_counts.iloc[-1] / len(target_series))
                }
            }
        else:  # regression
            return {
                "type": "regression",
                "statistics": {
                    "mean": float(target_series.mean()),
                    "median": float(target_series.median()),
                    "std": float(target_series.std()),
                    "min": float(target_series.min()),
                    "max": float(target_series.max()),
                    "q25": float(target_series.quantile(0.25)),
                    "q75": float(target_series.quantile(0.75))
                }
            }

    def _analyze_features(self, df: pd.DataFrame, config: CompetitionConfig) -> Dict[str, Any]:
        """Analyze individual features"""
        target_col = config.target_column
        feature_cols = [col for col in df.columns if col != target_col]

        feature_info = {}
        for col in feature_cols:
            col_data = df[col]

            info = {
                "dtype": str(col_data.dtype),
                "unique_count": int(col_data.nunique()),
                "missing_count": int(col_data.isnull().sum()),
                "missing_percentage": float(col_data.isnull().sum() / len(col_data) * 100)
            }

            if col_data.dtype in ['object', 'category']:
                # Categorical feature
                info["type"] = "categorical"
                value_counts = col_data.value_counts()
                info["top_categories"] = {
                    str(k): int(v) for k, v in value_counts.head().items()
                }
            elif np.issubdtype(col_data.dtype, np.number):
                # Numerical feature
                info["type"] = "numerical"
                info["statistics"] = {
                    "mean": float(col_data.mean()) if not col_data.isnull().all() else None,
                    "median": float(col_data.median()) if not col_data.isnull().all() else None,
                    "std": float(col_data.std()) if not col_data.isnull().all() else None,
                    "min": float(col_data.min()) if not col_data.isnull().all() else None,
                    "max": float(col_data.max()) if not col_data.isnull().all() else None
                }

            feature_info[col] = info

        return feature_info

    def _analyze_correlations(self, df: pd.DataFrame, config: CompetitionConfig) -> Dict[str, Any]:
        """Analyze correlations between features and target"""
        target_col = config.target_column

        # Get numerical columns only
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if target_col in numerical_cols:
            numerical_cols.remove(target_col)

        correlation_matrix = None
        target_correlations = None

        if len(numerical_cols) > 0:
            # Calculate correlation matrix
            numerical_df = df[numerical_cols + [target_col]] if target_col in df.columns else df[numerical_cols]
            correlation_matrix = numerical_df.corr().fillna(0).round(3)

            # Get correlations with target
            if target_col in correlation_matrix.columns:
                target_correlations = correlation_matrix[target_col].drop(target_col).sort_values(ascending=False)
                target_correlations = target_correlations.to_dict()

        return {
            "correlation_matrix": correlation_matrix.to_dict() if correlation_matrix is not None else None,
            "target_correlations": {k: float(v) for k, v in target_correlations.items()} if target_correlations else None,
            "high_correlation_pairs": self._find_high_correlation_pairs(correlation_matrix) if correlation_matrix is not None else []
        }

    def _find_high_correlation_pairs(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find pairs of highly correlated features"""
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr_pairs.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": float(corr_val)
                    })

        return high_corr_pairs

    def _assess_data_quality(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality"""
        issues = []

        # Check for duplicate rows
        train_duplicates = train_df.duplicated().sum()
        if train_duplicates > 0:
            issues.append(f"Training data has {train_duplicates} duplicate rows")

        # Check for constant features
        constant_features = []
        for col in train_df.columns:
            if train_df[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            issues.append(f"Constant features found: {constant_features}")

        # Check cardinality of categorical features
        high_cardinality_features = []
        for col in train_df.select_dtypes(include=['object', 'category']).columns:
            if train_df[col].nunique() > len(train_df) * 0.5:  # More than 50% unique values
                high_cardinality_features.append(col)

        if high_cardinality_features:
            issues.append(f"High cardinality features: {high_cardinality_features}")

        return {
            "issues": issues,
            "duplicate_rows": int(train_duplicates),
            "constant_features": constant_features,
            "high_cardinality_features": high_cardinality_features,
            "quality_score": max(0, 100 - len(issues) * 10)  # Simple quality score
        }

    async def _generate_visualizations(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                      config: CompetitionConfig, correlation_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualization plots and save as HTML"""
        visualizations = {}

        try:
            # Target distribution plot
            target_plot = await create_distribution_plots(train_df, config.target_column, config.task_type)
            visualizations["target_distribution"] = target_plot

            # Correlation heatmap
            if correlation_analysis["correlation_matrix"]:
                heatmap_plot = await create_correlation_heatmap(correlation_analysis["correlation_matrix"])
                visualizations["correlation_heatmap"] = heatmap_plot

            # Missing values pattern
            missing_plot = self._create_missing_values_plot(train_df, test_df)
            visualizations["missing_values"] = missing_plot

        except Exception as e:
            print(f"Visualization generation failed: {e}")
            visualizations["error"] = str(e)

        return visualizations

    def _create_missing_values_plot(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
        """Create missing values pattern plot"""
        import plotly.express as px

        train_missing = train_df.isnull().sum()
        test_missing = test_df.isnull().sum()

        # Create DataFrame for plotting
        missing_data = pd.DataFrame({
            'Feature': train_missing.index,
            'Train_Missing': train_missing.values,
            'Test_Missing': test_missing.reindex(train_missing.index, fill_value=0).values
        })

        # Filter to features with missing values
        missing_data = missing_data[
            (missing_data['Train_Missing'] > 0) | (missing_data['Test_Missing'] > 0)
        ]

        if len(missing_data) == 0:
            return "No missing values found in the dataset"

        # Create bar plot
        fig = px.bar(
            missing_data.melt(id_vars=['Feature'], var_name='Dataset', value_name='Missing_Count'),
            x='Feature',
            y='Missing_Count',
            color='Dataset',
            barmode='group',
            title='Missing Values by Feature'
        )

        return fig.to_html(include_plotlyjs='cdn')