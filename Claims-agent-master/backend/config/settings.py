from functools import lru_cache

from pydantic import BaseSettings


class Settings(BaseSettings):
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    openai_api_key: str
    vector_store_path: str = "./data/vector_store"
    rag_data_path: str = "./data/raw"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
