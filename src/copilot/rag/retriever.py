"""Retriever — the READ side of the RAG pipeline.

A *retriever* is LangChain's standard interface for "given a question, return the most
relevant documents". We build ours from the Chroma vector store.

Why wrap the store in a retriever instead of calling ``similarity_search`` directly?
  - Composability: a retriever is a LangChain ``Runnable``, so it plugs straight into
    chains (and later into LangGraph nodes) without manual glue.
  - Abstraction: it hides *how* documents are fetched. Today that is cosine similarity
    in Chroma; swapping in re-ranking, hybrid search, or MMR later would not change any
    code downstream — exactly the reasoning behind ``get_llm()`` and ``get_embeddings()``.
"""

from functools import lru_cache

from langchain_core.vectorstores import VectorStoreRetriever

from copilot.config import settings
from copilot.rag.vectorstore import get_vectorstore


@lru_cache
def get_retriever() -> VectorStoreRetriever:
    """Return the shared retriever (built once, then reused).

    ``k`` (how many chunks to fetch per question) comes from config and must be passed
    inside ``search_kwargs`` — a common gotcha with ``as_retriever``.
    """
    return get_vectorstore().as_retriever(
        search_kwargs={"k": settings.retrieval_k},
    )
