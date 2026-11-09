"""Readiness checks — can the app actually answer a question right now?

*Liveness* (``/health``) only says the process is up. *Readiness* (``/ready``) checks
the dependencies an answer needs, so a load balancer or an orchestrator (Kubernetes,
docker-compose) can stop sending traffic to an instance that would fail anyway:

  - **ollama**: the server answers, and both the chat and the embedding models are pulled;
  - **vectorstore**: the collection exists and is not empty (ingestion has been run).

The checks are cheap (no generation, no embedding): they can run every few seconds.
"""

import httpx

from copilot.config import settings
from copilot.rag.vectorstore import get_vectorstore


def _with_tag(model: str) -> str:
    """Ollama lists models with their tag: ``nomic-embed-text`` is ``...:latest``."""
    return model if ":" in model else f"{model}:latest"


def check_ollama() -> tuple[bool, str]:
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return False, f"unreachable at {settings.ollama_base_url} ({type(exc).__name__})"
    available = {m["name"] for m in response.json().get("models", [])}
    required = {_with_tag(settings.llm_model), _with_tag(settings.embedding_model)}
    missing = sorted(required - available)
    if missing:
        return False, f"missing models: {', '.join(missing)} (run `ollama pull`)"
    return True, f"models available: {', '.join(sorted(required))}"


def check_vectorstore() -> tuple[bool, str]:
    try:
        # Chroma's raw collection: ``count()`` reads the index without embedding anything.
        count = get_vectorstore()._collection.count()
    except Exception as exc:  # noqa: BLE001 — any failure means "not ready"
        return False, f"cannot open the vector store ({type(exc).__name__})"
    if count == 0:
        return False, "the collection is empty (run scripts/ingest.py)"
    return True, f"{count} chunks indexed"


def readiness() -> dict[str, tuple[bool, str]]:
    """Run all checks. The app is ready only if every check passes."""
    return {"ollama": check_ollama(), "vectorstore": check_vectorstore()}
