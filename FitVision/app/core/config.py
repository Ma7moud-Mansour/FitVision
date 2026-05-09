from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ──
    APP_NAME: str = "FitVision API"
    DEBUG: bool = False

    # ── Database ──
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/fitvision"

    # ── Media Paths ──
    UPLOAD_DIR: str = "media/uploads"
    PROCESSED_DIR: str = "media/processed"

    # ── Video Processing ──
    TARGET_FPS: int = 10
    MAX_FILE_SIZE_MB: int = 100  # الحد الأقصى لحجم الفيديو

    # ── Auth ──
    SECRET_KEY: str = "change-me-in-production-ya-basha"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
