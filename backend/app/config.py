from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    MODELS_DIR: str = "./saved_models"
    PRECOMPUTED_DIR: str = "./precomputed"
    CATALOG_DB_PATH: str = "./catalog.duckdb"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    LRU_CACHE_SIZE: int = 2048
    LOG_LEVEL: str = "INFO"
    MAX_BULK_UPLOAD_ITEMS: int = 500

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
