"""LLM provider — the single place that builds the language model.

Every part of the app that needs the LLM calls ``get_llm()`` instead of importing
``ChatOllama`` directly. This keeps the app *model-agnostic*: to switch model or
provider we change only this file (and config), not the rest of the codebase.
"""

from functools import lru_cache

from langchain_ollama import ChatOllama

from copilot.config import settings


@lru_cache
def get_llm() -> ChatOllama:
    """Return the shared chat LLM instance (built once, then reused).

    Temperature is 0.0 for deterministic, grounded answers — important for a RAG
    assistant where we want faithful responses, not creative ones.
    """
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
    )
