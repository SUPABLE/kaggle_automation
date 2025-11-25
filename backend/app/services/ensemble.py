import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from scipy.optimize import minimize
import joblib
import os
from typing import Dict, Any, List, Tuple

from app.models.schemas import CompetitionConfig, TaskType
from app.core.config import settings

class EnsembleService:
    def __init__(self):
        self.artifacts_dir = settings.artifacts_dir

    async def create_ensemble(self, model_results: Dict[str, Any], processed_data: Dict[str, Any],
                            config: CompetitionConfig) -> Dict[str, Any]:
        """
        Create ensemble models from trained individual models
        """
        try:
            X_train = processed_data["X_train"]
            y_train = processed_data["y_train"]
            final_models = model_results["final_models"]

            # Create different ensemble strategies
            ensembles = {}

            # 1. Weighted Average Ensemble
            weighted_ensemble = self._create_weighted_ensemble(final_models, X_train, y_train, config)
            ensembles["weighted"] = weighted_ensemble

            # 2. Stacking Ensemble
            stacking_ensemble = self._create_stacking_ensemble(final_models, X_train, y_train, config)
            ensembles["stacking"] = stacking_ensemble

            # 3. Simple Average Ensemble
            simple_ensemble = self._create_simple_ensemble(final_models, X_train, y_train, config)
            ensembles["simple"] = simple_ensemble

            # Evaluate ensembles
            ensemble_scores = self._evaluate_ensembles(ensembles, X_train, y_train, config)

            # Select best ensemble
            best_ensemble_type = max(ensemble_scores.keys(), key=lambda k: ensemble_scores[k]["score"])
            best_ensemble = {
                "type": best_ensemble_type,
                "ensemble": ensembles[best_ensemble_type],
                "score": ensemble_scores[best_ensemble_type]["score"]
            }

            return {
                "ensembles": ensembles,
                "ensemble_scores": ensemble_scores,
                "best_ensemble": best_ensemble,
                "ensemble_performance": {
                    "individual_best": model_results["best_model"]["cv_score"],
                    "ensemble_best": best_ensemble["score"],
                    "improvement": best_ensemble["score"] - model_results["best_model"]["cv_score"]
                }
            }

        except Exception as e:
            raise Exception(f"Ensemble creation failed: {str(e)}")

    def _create_weighted_ensemble(self, final_models: Dict[str, Any], X_train: pd.DataFrame,
                                 y_train: pd.Series, config: CompetitionConfig) -> Dict[str, Any]:
        """
        Create weighted average ensemble with optimized weights
        """
        # Get model names and models
        model_names = list(final_models.keys())
        models = [final_models[name]["model"] for name in model_names]
        model_scores = [final_models[name]["cv_score"] for name in model_names]

        # Get out-of-fold predictions
        oof_predictions = self._get_oof_predictions(models, X_train, y_train, config)

        # Optimize weights
        optimal_weights = self._optimize_weights(oof_predictions, y_train, config)

        weighted_ensemble = {
            "models": models,
            "model_names": model_names,
            "weights": optimal_weights,
            "type": "weighted"
        }

        return weighted_ensemble

    def _create_stacking_ensemble(self, final_models: Dict[str, Any], X_train: pd.DataFrame,
                                y_train: pd.Series, config: CompetitionConfig) -> Dict[str, Any]:
        """
        Create stacking ensemble
        """
        # Select top performing models for stacking (avoid overfitting with too many)
        sorted_models = sorted(final_models.items(), key=lambda x: x[1]["cv_score"], reverse=True)
        top_models = dict(sorted_models[:4])  # Use top 4 models

        # Prepare base estimators
        base_estimators = [(name, model_info["model"]) for name, model_info in top_models.items()]

        # Choose meta-learner based on task type
        if config.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            meta_learner = LogisticRegression(max_iter=1000, random_state=settings.random_state)
            stacking_class = StackingClassifier
        else:  # regression
            meta_learner = LinearRegression()
            stacking_class = StackingRegressor

        # Create stacking ensemble
        stacking_model = stacking_class(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )

        # Train stacking model
        stacking_model.fit(X_train, y_train)

        stacking_ensemble = {
            "model": stacking_model,
            "base_models": top_models,
            "meta_learner": meta_learner,
            "type": "stacking"
        }

        return stacking_ensemble

    def _create_simple_ensemble(self, final_models: Dict[str, Any], X_train: pd.DataFrame,
                               y_train: pd.Series, config: CompetitionConfig) -> Dict[str, Any]:
        """
        Create simple average ensemble
        """
        model_names = list(final_models.keys())
        models = [final_models[name]["model"] for name in model_names]

        simple_ensemble = {
            "models": models,
            "model_names": model_names,
            "weights": [1.0 / len(models)] * len(models),  # Equal weights
            "type": "simple"
        }

        return simple_ensemble

    def _get_oof_predictions(self, models: List, X_train: pd.DataFrame, y_train: pd.Series,
                           config: CompetitionConfig) -> np.ndarray:
        """
        Get out-of-fold predictions from models
        """
        oof_predictions = []

        for model in models:
            try:
                # Get cross-validated predictions
                if config.task_type in ["binary_classification", "multiclass_classification"]:
                    if hasattr(model, "predict_proba"):
                        pred = cross_val_predict(model, X_train, y_train, cv=5, method='predict_proba')
                        if len(pred[0]) <= 2:  # Binary classification
                            pred = pred[:, 1]  # Take positive class probability
                    else:
                        pred = cross_val_predict(model, X_train, y_train, cv=5)
                else:  # regression
                    pred = cross_val_predict(model, X_train, y_train, cv=5)

                oof_predictions.append(pred)

            except Exception as e:
                print(f"Failed to get OOF predictions: {e}")
                # Fallback to regular predictions
                try:
                    model.fit(X_train, y_train)
                    if hasattr(model, "predict_proba"):
                        pred = model.predict_proba(X_train)
                        if len(pred[0]) <= 2:
                            pred = pred[:, 1]
                    else:
                        pred = model.predict(X_train)
                    oof_predictions.append(pred)
                except:
                    # If all else fails, use zeros
                    oof_predictions.append(np.zeros(len(y_train)))

        return np.column_stack(oof_predictions)

    def _optimize_weights(self, oof_predictions: np.ndarray, y_train: pd.Series,
                         config: CompetitionConfig) -> np.ndarray:
        """
        Optimize ensemble weights using validation performance
        """
        def objective(weights):
            # Ensure weights sum to 1
            weights = weights / np.sum(weights)

            # Calculate weighted predictions
            weighted_pred = np.dot(oof_predictions, weights)

            # Calculate score based on task type
            if config.task_type in ["binary_classification", "multiclass_classification"]:
                if len(np.unique(y_train)) == 2:  # Binary classification
                    # Convert probabilities to classes
                    pred_classes = (weighted_pred > 0.5).astype(int)
                    score = accuracy_score(y_train, pred_classes)
                else:  # Multiclass
                    pred_classes = np.argmax(weighted_pred.reshape(-1, len(np.unique(y_train))), axis=1)
                    score = accuracy_score(y_train, pred_classes)
            else:  # regression
                score = -mean_squared_error(y_train, weighted_pred)  # Negative because we minimize

            return -score  # Negative because we minimize

        # Initial weights (equal)
        initial_weights = np.ones(oof_predictions.shape[1]) / oof_predictions.shape[1]

        # Constraints: weights must be non-negative
        bounds = [(0, 1)] * oof_predictions.shape[1]

        # Optimize weights
        result = minimize(
            objective,
            initial_weights,
            method='L-BFGS-B',
            bounds=bounds
        )

        if result.success:
            optimal_weights = result.x / np.sum(result.x)  # Normalize to sum to 1
        else:
            # Fallback to equal weights
            optimal_weights = initial_weights

        return optimal_weights

    def _evaluate_ensembles(self, ensembles: Dict[str, Any], X_train: pd.DataFrame,
                          y_train: pd.Series, config: CompetitionConfig) -> Dict[str, Dict[str, float]]:
        """
        Evaluate different ensemble strategies
        """
        ensemble_scores = {}

        for ensemble_type, ensemble in ensembles.items():
            try:
                if ensemble_type == "stacking":
                    # For stacking, use the trained model
                    predictions = ensemble["model"].predict(X_train)
                    if config.task_type in ["binary_classification", "multiclass_classification"]:
                        score = accuracy_score(y_train, predictions)
                    else:
                        score = -mean_squared_error(y_train, predictions)  # Negative MSE for minimization

                else:
                    # For weighted/simple ensembles, calculate weighted predictions
                    models = ensemble["models"]
                    weights = ensemble["weights"]

                    # Get predictions from all models
                    all_predictions = []
                    for model in models:
                        if hasattr(model, "predict_proba"):
                            pred = model.predict_proba(X_train)
                            if len(pred[0]) <= 2:  # Binary classification
                                pred = pred[:, 1]
                        else:
                            pred = model.predict(X_train)
                        all_predictions.append(pred)

                    # Calculate weighted predictions
                    all_predictions = np.column_stack(all_predictions)
                    weighted_pred = np.dot(all_predictions, weights)

                    # Calculate score
                    if config.task_type in ["binary_classification", "multiclass_classification"]:
                        if len(np.unique(y_train)) == 2:  # Binary classification
                            pred_classes = (weighted_pred > 0.5).astype(int)
                            score = accuracy_score(y_train, pred_classes)
                        else:  # Multiclass - this is simplified
                            score = 0.5  # Placeholder
                    else:  # regression
                        score = -mean_squared_error(y_train, weighted_pred)

                ensemble_scores[ensemble_type] = {
                    "score": score
                }

            except Exception as e:
                print(f"Failed to evaluate ensemble {ensemble_type}: {e}")
                ensemble_scores[ensemble_type] = {
                    "score": float('-inf')
                }

        return ensemble_scores

    async def generate_submission(self, ensemble_results: Dict[str, Any], processed_data: Dict[str, Any],
                                config: CompetitionConfig) -> List[Dict[str, Any]]:
        """
        Generate submission predictions using the best ensemble
        """
        try:
            X_test = processed_data["X_test"]
            best_ensemble = ensemble_results["best_ensemble"]
            ensemble = best_ensemble["ensemble"]

            # Generate predictions on test set
            if ensemble["type"] == "stacking":
                # Use stacking model
                if hasattr(ensemble["model"], "predict_proba"):
                    test_predictions = ensemble["model"].predict_proba(X_test)
                    if len(test_predictions[0]) <= 2:  # Binary classification
                        test_predictions = test_predictions[:, 1]
                else:
                    test_predictions = ensemble["model"].predict(X_test)

            else:
                # Use weighted/simple ensemble
                models = ensemble["models"]
                weights = ensemble["weights"]

                # Get predictions from all models
                all_predictions = []
                for model in models:
                    if hasattr(model, "predict_proba"):
                        pred = model.predict_proba(X_test)
                        if len(pred[0]) <= 2:  # Binary classification
                            pred = pred[:, 1]
                    else:
                        pred = model.predict(X_test)
                    all_predictions.append(pred)

                # Calculate weighted predictions
                all_predictions = np.column_stack(all_predictions)
                test_predictions = np.dot(all_predictions, weights)

            # Format submission according to config
            submission_data = self._format_submission(test_predictions, processed_data, config)

            return submission_data

        except Exception as e:
            raise Exception(f"Submission generation failed: {str(e)}")

    def _format_submission(self, predictions: np.ndarray, processed_data: Dict[str, Any],
                          config: CompetitionConfig) -> List[Dict[str, Any]]:
        """
        Format predictions according to competition requirements
        """
        # Get submission format from config
        submission_format = config.submission_format
        id_column = submission_format.get("id_column", "id")
        prediction_column = submission_format.get("prediction_column", "target")

        # Get test IDs (assuming there's an ID column in test data)
        # This is a simplified approach - in practice, you'd need to track this through preprocessing
        X_test = processed_data["X_test"]

        if id_column in X_test.columns:
            test_ids = X_test[id_column].values
        else:
            # If no ID column found, create sequential IDs
            test_ids = np.arange(len(predictions))

        # Create submission data
        submission_data = []
        for i, test_id in enumerate(test_ids):
            submission_data.append({
                id_column: test_id,
                prediction_column: predictions[i]
            })

        return submission_data

    def save_ensemble(self, ensemble: Dict[str, Any], task_id: str):
        """
        Save the best ensemble model
        """
        ensemble_dir = os.path.join(self.artifacts_dir, "ensembles")
        os.makedirs(ensemble_dir, exist_ok=True)

        ensemble_path = os.path.join(ensemble_dir, f"best_ensemble_{task_id}.joblib")
        joblib.dump(ensemble, ensemble_path)