from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum

class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"

class EvaluationMetric(str, Enum):
    ACCURACY = "accuracy"
    F1 = "f1"
    AUC = "auc"
    LOG_LOSS = "log_loss"
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"

class CVStrategy(str, Enum):
    STRATIFIED_KFOLD = "stratified_kfold"
    KFOLD = "kfold"
    TIME_SERIES_SPLIT = "time_series_split"
    GROUP_KFOLD = "group_kfold"

class FileUploadResponse(BaseModel):
    success: bool
    message: str
    file_info: Optional[Dict[str, Any]] = None
    train_columns: Optional[List[str]] = None
    test_columns: Optional[List[str]] = None

class CompetitionConfig(BaseModel):
    target_column: str
    task_type: TaskType
    evaluation_metric: EvaluationMetric
    cv_strategy: CVStrategy = CVStrategy.STRATIFIED_KFOLD
    cv_folds: int = Field(default=5, ge=3, le=10)
    submission_format: Dict[str, str]
    competition_rules: Optional[Dict[str, Any]] = None
    feature_engineering: bool = True
    hyperparameter_optimization: bool = True

class AnalysisRequest(BaseModel):
    config: CompetitionConfig
    file_session_id: str

class AnalysisStartResponse(BaseModel):
    success: bool
    task_id: str
    message: str

class AnalysisStatus(BaseModel):
    task_id: str
    status: Literal["pending", "running", "completed", "failed"]
    current_stage: Optional[str] = None
    progress_percentage: float = Field(ge=0, le=100)
    message: Optional[str] = None
    error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

class ModelPerformance(BaseModel):
    model_name: str
    cv_score: float
    cv_std: float
    training_time: float

class EDAResults(BaseModel):
    dataset_shape: tuple
    missing_values: Dict[str, int]
    data_types: Dict[str, str]
    target_distribution: Dict[str, Any]
    correlation_matrix: Optional[List[List[float]]] = None
    feature_importance: Optional[Dict[str, float]] = None

class AnalysisResults(BaseModel):
    task_id: str
    status: str
    eda_results: EDAResults
    model_performances: List[ModelPerformance]
    best_model: ModelPerformance
    ensemble_score: float
    feature_importance: Dict[str, float]
    submission_predictions_count: int

class ReportRequest(BaseModel):
    task_id: str
    format: Literal["html", "markdown"] = "html"

class ReportResponse(BaseModel):
    task_id: str
    report_content: str
    report_format: str
    generated_at: str

class SubmissionDownloadRequest(BaseModel):
    task_id: str

class TaskProgress(BaseModel):
    stage: str
    progress: float
    message: str
    timestamp: str