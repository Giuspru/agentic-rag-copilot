"""Vector store — the single place that connects to the Chroma database.

Chroma is a *vector database*: it stores the embedding vectors of our document chunks
(together with their original text and metadata) and, given a query vector, returns the
most semantically similar chunks. This is the "R" (retrieval) in RAG.

Both sides of the app go through this one function:
  - ingestion writes chunks into the store;
  - the retriever reads the most relevant chunks back out at query time.

Keeping it in one place guarantees they use the SAME collection, the SAME on-disk
location, and the SAME embedding function — the three things that must match for
retrieval to work at all.
"""

from functools import lru_cache

from langchain_chroma import Chroma

from copilot.config import settings
from copilot.llm.embeddings import get_embeddings


@lru_cache
def get_vectorstore() -> Chroma:
    """Return the shared Chroma vector store (built once, then reused).

    The store is persisted to ``settings.chroma_dir`` on disk, so the index survives
    restarts: we ingest the documents once and can query them from any later run.
    """
    # Chroma's persistent client creates this directory on first use, but we make sure
    # it exists so the very first run never fails on a missing path.
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=settings.collection_name,
        # The embedding function is used to vectorize incoming queries. It MUST be the
        # same model used at ingestion time, otherwise the vectors are not comparable.
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )
