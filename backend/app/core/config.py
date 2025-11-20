import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App settings
    app_name: str = "Kaggle Automation API"
    debug: bool = True

    # Data directories
    data_dir: str = "../data"
    uploads_dir: str = "../data/uploads"
    artifacts_dir: str = "../data/artifacts"
    reports_dir: str = "../data/reports"
    shared_dir: str = "../shared"

    # File upload settings
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: list = [".csv"]

    # ML settings
    random_state: int = 42
    cv_folds: int = 5
    max_trials: int = 100

    # WebSocket settings
    ws_ping_interval: int = 20
    ws_ping_timeout: int = 10

    class Config:
        env_file = ".env"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.uploads_dir, exist_ok=True)
os.makedirs(settings.artifacts_dir, exist_ok=True)
os.makedirs(settings.reports_dir, exist_ok=True)
os.makedirs(settings.shared_dir, exist_ok=True)