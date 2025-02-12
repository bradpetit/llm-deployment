# app/config.py
from pydantic_settings import BaseSettings
from typing import List
import json

class Settings(BaseSettings):
    # Model Settings
    MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_DEVICE: str = "cuda"  # or "cpu"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Database Settings
    CHROMA_DB_PATH: str = "./data/chroma_db"
    
    # Generation Settings
    MAX_LENGTH: int = 2048
    DEFAULT_TEMPERATURE: float = 0.7
    
    # HuggingFace Settings
    HUGGINGFACE_TOKEN: str

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