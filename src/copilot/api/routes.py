"""API routes — the HTTP endpoints of the copilot.

The routes are kept thin on purpose: they validate input (via the schemas), delegate to
the RAG layer, and wrap the result. No retrieval or prompt logic lives here, so swapping
the linear chain for the LangGraph agent later only changes the call inside ``chat()``.
"""

from fastapi import APIRouter

from copilot.api.schemas import ChatRequest, ChatResponse, HealthResponse
from copilot.rag.chain import answer

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check: the API process is up and serving requests."""
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question about the maintenance manuals.

    Declared with plain ``def`` (not ``async def``) because ``answer()`` is blocking: it
    waits on the vector store and on Ollama. FastAPI runs sync endpoints in a thread
    pool, so a slow LLM call doesn't freeze the server for other requests.
    """
    return ChatResponse(answer=answer(request.question))
