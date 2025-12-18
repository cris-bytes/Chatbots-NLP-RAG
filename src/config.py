"""
Configuration module for the RAG chatbot system
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # OpenAI Configuration
    openai_api_key: Optional[str] = None

    # Model Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-3.5-turbo"

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    vector_store_path: Path = data_dir / "vectorstore"

    # Document Processing
    chunk_size: int = 500
    chunk_overlap: int = 50

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # RAG Configuration
    top_k_chunks: int = 3
    temperature: float = 0.7
    max_tokens: int = 500

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def ensure_directories():
    """Ensure all required directories exist"""
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)


# Initialize directories on import
ensure_directories()
