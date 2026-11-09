"""Embedding provider — the single place that builds the embedding model.

Embeddings turn text into numeric vectors that capture meaning. They are used both when
we ingest documents (to index them) and at query time (to find similar chunks). The same
model MUST be used for both, otherwise the vectors would not be comparable — that is why
this lives in one shared function.
"""

from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from copilot.config import settings


@lru_cache
def get_embeddings() -> OllamaEmbeddings:
    """Return the shared embedding model instance (built once, then reused)."""
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"timeout": settings.llm_timeout},  # see get_llm()
    )
