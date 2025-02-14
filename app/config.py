# app/config.py
from pydantic_settings import BaseSettings
from typing import List
import json

class Settings(BaseSettings):
    # Model Settings
    MODEL_NAME: str = "mistral"
    MODEL_DEVICE: str = "cpu"  # Not needed for Ollama but keep for compatibility
    
    # RAG-specific settings
    CHUNK_MIN_SIZE: int = 100
    CHUNK_MAX_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MIN_SIMILARITY_SCORE: float = 0.6
    MAX_CONTEXT_CHUNKS: int = 3
    RERANKING_ENABLED: bool = True
    
    # Add these to your existing settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Database Settings
    CHROMA_DB_PATH: str = "/app/data/chroma_db"
    
    # Generation Settings
    MAX_LENGTH: int = 2048
    DEFAULT_TEMPERATURE: float = 0.65

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