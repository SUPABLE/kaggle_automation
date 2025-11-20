import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
import optuna
from typing import Dict, Any, List
import joblib
import os
import time

from app.models.schemas import CompetitionConfig, TaskType
from app.core.config import settings

class ModelingService:
    def __init__(self):
        self.artifacts_dir = settings.artifacts_dir
        self.trained_models = {}
        self.model_configs = {}

    async def train_models(self, processed_data: Dict[str, Any], cv_strategy: Dict[str, Any],
                          config: CompetitionConfig) -> Dict[str, Any]:
        """
        Train multiple models with cross-validation
        """
        try:
            X_train = processed_data["X_train"]
            y_train = processed_data["y_train"]
            cv_splitter = cv_strategy["cv_splitter"]
            evaluation_metric = config.evaluation_metric

            # Get appropriate models for the task
            models = self._get_models_for_task(config.task_type)

            model_results = []
            best_model = None
            best_score = float('-inf') if 'accuracy' in evaluation_metric or 'r2' in evaluation_metric else float('inf')

            # Train each model
            for model_name, model in models.items():
                print(f"Training {model_name}...")

                start_time = time.time()

                # Perform cross-validation
                cv_results = self._perform_cv(model, X_train, y_train, cv_splitter, config)

                training_time = time.time() - start_time

                # Create model result object
                model_result = {
                    "model_name": model_name,
                    "model": model,
                    "cv_score": cv_results["mean_score"],
                    "cv_std": cv_results["std_score"],
                    "training_time": training_time,
                    "cv_scores": cv_results["all_scores"],
                    "additional_metrics": cv_results["additional_metrics"]
                }

                model_results.append(model_result)

                # Update best model
                is_better = self._is_better_score(model_result["cv_score"], best_score, evaluation_metric)
                if is_better:
                    best_score = model_result["cv_score"]
                    best_model = model_result

            # Train final models on full dataset
            final_models = self._train_final_models(model_results, X_train, y_train, config)

            # Get feature importances
            feature_importance = self._get_feature_importance(final_models, X_train.columns.tolist())

            # Save models
            self._save_models(final_models, config.task_type)

            return {
                "model_results": model_results,
                "best_model": best_model,
                "final_models": final_models,
                "feature_importance": feature_importance,
                "evaluation_metric": evaluation_metric
            }

        except Exception as e:
            raise Exception(f"Model training failed: {str(e)}")

    def _get_models_for_task(self, task_type: str) -> Dict[str, Any]:
        """
        Get appropriate models for the task type
        """
        if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            return {
                "logistic_regression": LogisticRegression(max_iter=1000, random_state=settings.random_state),
                "random_forest": RandomForestClassifier(n_estimators=100, random_state=settings.random_state),
                "lightgbm": lgb.LGBMClassifier(random_state=settings.random_state, verbose=-1),
                "xgboost": xgb.XGBClassifier(random_state=settings.random_state, eval_metric='logloss'),
                "catboost": CatBoostClassifier(random_state=settings.random_state, verbose=False)
            }
        else:  # regression
            return {
                "linear_regression": LinearRegression(),
                "ridge_regression": Ridge(random_state=settings.random_state),
                "random_forest": RandomForestRegressor(n_estimators=100, random_state=settings.random_state),
                "lightgbm": lgb.LGBMRegressor(random_state=settings.random_state, verbose=-1),
                "xgboost": xgb.XGBRegressor(random_state=settings.random_state, eval_metric='rmse'),
                "catboost": CatBoostRegressor(random_state=settings.random_state, verbose=False)
            }

    def _perform_cv(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                   cv_splitter, config: CompetitionConfig) -> Dict[str, Any]:
        """
        Perform cross-validation for a model
        """
        # Choose scoring metric
        scoring = self._get_scoring_metric(config.evaluation_metric)

        # Perform cross-validation with multiple metrics
        cv_results = cross_validate(
            model, X_train, y_train,
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        # Get the primary score
        test_scores = cv_results[f"test_{scoring}"]
        mean_score = test_scores.mean()
        std_score = test_scores.std()

        # Calculate additional metrics
        additional_metrics = self._calculate_additional_metrics(
            model, X_train, y_train, cv_splitter, config
        )

        return {
            "mean_score": mean_score,
            "std_score": std_score,
            "all_scores": test_scores.tolist(),
            "additional_metrics": additional_metrics
        }

    def _get_scoring_metric(self, evaluation_metric: str) -> str:
        """
        Convert evaluation metric to sklearn scoring metric
        """
        metric_mapping = {
            "accuracy": "accuracy",
            "f1": "f1_macro",
            "auc": "roc_auc",
            "log_loss": "neg_log_loss",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2"
        }

        return metric_mapping.get(evaluation_metric, "accuracy")

    def _calculate_additional_metrics(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                                    cv_splitter, config: CompetitionConfig) -> Dict[str, float]:
        """
        Calculate additional evaluation metrics
        """
        additional_metrics = {}

        try:
            # Get predictions using cross-validation
            from sklearn.model_selection import cross_val_predict

            if config.task_type in ["binary_classification", "multiclass_classification"]:
                y_pred = cross_val_predict(model, X_train, y_train, cv=cv_splitter)
                y_proba = None

                if hasattr(model, "predict_proba"):
                    try:
                        y_proba = cross_val_predict(model, X_train, y_train, cv=cv_splitter, method='predict_proba')
                        if len(y_proba[0]) <= 2:  # Binary classification
                            y_proba = y_proba[:, 1]
                    except:
                        pass

                # Calculate metrics
                additional_metrics["accuracy"] = accuracy_score(y_train, y_pred)
                additional_metrics["f1_macro"] = f1_score(y_train, y_pred, average='macro')

                if y_proba is not None and len(np.unique(y_train)) == 2:  # Binary classification
                    additional_metrics["auc"] = roc_auc_score(y_train, y_proba)

            else:  # regression
                y_pred = cross_val_predict(model, X_train, y_train, cv=cv_splitter)

                additional_metrics["mse"] = mean_squared_error(y_train, y_pred)
                additional_metrics["mae"] = mean_absolute_error(y_train, y_pred)
                additional_metrics["r2"] = r2_score(y_train, y_pred)
                additional_metrics["rmse"] = np.sqrt(mean_squared_error(y_train, y_pred))

        except Exception as e:
            print(f"Failed to calculate additional metrics: {e}")

        return additional_metrics

    def _is_better_score(self, current_score: float, best_score: float, metric: str) -> bool:
        """
        Determine if current score is better than best score
        """
        # Higher is better for these metrics
        higher_is_better = ["accuracy", "f1", "auc", "r2"]

        if metric in higher_is_better:
            return current_score > best_score
        else:  # Lower is better for error metrics
            return current_score < best_score

    def _train_final_models(self, model_results: List[Dict[str, Any]], X_train: pd.DataFrame,
                           y_train: pd.Series, config: CompetitionConfig) -> Dict[str, Any]:
        """
        Train final models on the full dataset
        """
        final_models = {}

        for model_result in model_results:
            model_name = model_result["model_name"]
            model = model_result["model"]

            # Clone and train on full dataset
            try:
                from sklearn.base import clone
                final_model = clone(model)
                final_model.fit(X_train, y_train)

                final_models[model_name] = {
                    "model": final_model,
                    "cv_score": model_result["cv_score"],
                    "training_time": model_result["training_time"]
                }

            except Exception as e:
                print(f"Failed to train final model {model_name}: {e}")

        return final_models

    def _get_feature_importance(self, final_models: Dict[str, Any], feature_names: List[str]) -> Dict[str, float]:
        """
        Get aggregated feature importance from all models
        """
        feature_importance_dict = {}

        for model_name, model_info in final_models.items():
            model = model_info["model"]

            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_)
                    if importances.ndim > 1:
                        importances = importances.mean(axis=0)
                else:
                    continue

                # Normalize importances
                importances = importances / importances.sum()

                # Add to dictionary
                for i, feature in enumerate(feature_names):
                    if feature not in feature_importance_dict:
                        feature_importance_dict[feature] = []
                    feature_importance_dict[feature].append(importances[i])

            except Exception as e:
                print(f"Failed to get feature importance from {model_name}: {e}")

        # Average importances across models
        avg_importance = {}
        for feature, scores in feature_importance_dict.items():
            avg_importance[feature] = np.mean(scores)

        return avg_importance

    def _save_models(self, final_models: Dict[str, Any], task_type: str):
        """
        Save trained models to disk
        """
        models_dir = os.path.join(self.artifacts_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        for model_name, model_info in final_models.items():
            model = model_info["model"]
            model_path = os.path.join(models_dir, f"{model_name}_{task_type}.joblib")
            joblib.dump(model, model_path)

    async def hyperparameter_optimization(self, model_name: str, X_train: pd.DataFrame,
                                         y_train: pd.Series, cv_splitter, config: CompetitionConfig) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using Optuna
        """
        if not config.hyperparameter_optimization:
            return None

        def objective(trial):
            # Define hyperparameter search space based on model
            if model_name == "lightgbm":
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                }
                model = lgb.LGBMClassifier(**params, random_state=settings.random_state, verbose=-1)

            elif model_name == "xgboost":
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                }
                model = xgb.XGBClassifier(**params, random_state=settings.random_state, eval_metric='logloss')

            elif model_name == "random_forest":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                }
                model = RandomForestClassifier(**params, random_state=settings.random_state)

            else:
                return 0.0  # Skip optimization for unsupported models

            # Perform cross-validation
            scoring = self._get_scoring_metric(config.evaluation_metric)
            scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring=scoring)

            return scores.mean()

        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=settings.max_trials)

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "n_trials": len(study.trials)
        }