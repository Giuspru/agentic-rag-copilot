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
    chunk_size: int = 800  # characters per chunk
    chunk_overlap: int = 120  # overlap between consecutive chunks
    retrieval_k: int = 4  # how many chunks to retrieve per question

    # --- Agent (self-correction budget) ---
    max_rewrites: int = 2  # query rewrites before giving up
    max_generations: int = 2  # answer attempts before giving up

    # --- Robustness (calls to Ollama) ---
    llm_timeout: float = 120.0  # seconds to wait for Ollama without receiving any data
    llm_max_attempts: int = 3  # attempts per node on transient errors (1 = no retry)

    # --- Observability ---
    log_level: str = "INFO"
    # Arize Phoenix tracing (optional)
    tracing_enabled: bool = False  # off by default: the app runs without a Phoenix server
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project: str = "agentic-rag-copilot"


# Single shared instance imported across the app.
settings = Settings()
