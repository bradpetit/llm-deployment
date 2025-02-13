# app/config.py
from pydantic_settings import BaseSettings
from typing import List
import json

class Settings(BaseSettings):
    # Model Settings
    MODEL_NAME: str = "mistral"
    MODEL_DEVICE: str = "cpu"  # Not needed for Ollama but keep for compatibility
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Database Settings
    CHROMA_DB_PATH: str = "./data/chroma_db"
    CHROMA_DB_CACHE_PATH: str ="./data/chroma_cache"
    
    # Generation Settings
    MAX_LENGTH: int = 2048
    DEFAULT_TEMPERATURE: float = 0.7

    class Config:
        env_file = ".env"

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str):
            if field_name == "ALLOWED_ORIGINS":
                try:
                    return json.loads(raw_val)
                except json.JSONDecodeError:
                    return raw_val.split(",")
            return raw_val

settings = Settings()