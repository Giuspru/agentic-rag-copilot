"""Centralized configuration.

All settings live here and can be overridden via environment variables (prefixed with
``COPILOT_``) or a local ``.env`` file. Every value has a sensible default, so the app
runs out of the box with no configuration.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: this file is at src/copilot/config.py -> parents[2] == project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="COPILOT_",
        extra="ignore",
    )

    # --- Ollama (local models) ---
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text"

    # --- Paths ---
    manuals_dir: Path = PROJECT_ROOT / "data" / "manuals"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    collection_name: str = "maintenance_manuals"

    # --- RAG parameters ---
    chunk_size: int = 800          # characters per chunk
    chunk_overlap: int = 120       # overlap between consecutive chunks
    retrieval_k: int = 4           # how many chunks to retrieve per question


# Single shared instance imported across the app.
settings = Settings()
