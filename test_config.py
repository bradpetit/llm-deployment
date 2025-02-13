# test_config.py
from app.config import settings
import os

def test_settings():
    print("Current Settings:")
    print(f"ALLOWED_ORIGINS: {settings.ALLOWED_ORIGINS}")
    print(f"MODEL_NAME: {settings.MODEL_NAME}")
    print(f"API_HOST: {settings.API_HOST}")
    print(f"API_PORT: {settings.API_PORT}")
    
    # Test HuggingFace token
    # token = settings.HUGGINGFACE_TOKEN
    # if token:
    #     # Print first and last 4 characters of token for verification
    #     print(f"HuggingFace token found: {token[:4]}...{token[-4:]}")
    # else:
    #     print("Warning: No HuggingFace token found!")

if __name__ == "__main__":
    test_settings()