"""API routes — the HTTP endpoints of the copilot.

The routes are kept thin on purpose: they validate input (via the schemas), delegate to
the agent, and wrap the result. No retrieval or prompt logic lives here: moving from the
linear chain to the LangGraph agent only changed the call inside ``chat()``.

Errors are not handled here either: the agent raises domain errors (``errors.py``) and a
single exception handler in ``main.py`` turns them into HTTP responses.
"""

from fastapi import APIRouter, Response

from copilot.agent.graph import run_agent
from copilot.api.schemas import (
    ChatRequest,
    ChatResponse,
    CheckResult,
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
)
from copilot.observability.health import readiness

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check: the API process is up and serving requests."""
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse, "description": "A dependency is down"}},
)
def ready(response: Response) -> ReadinessResponse:
    """Readiness check: Ollama is up with the models pulled, and the index is not empty."""
    checks = {name: CheckResult(ok=ok, detail=detail) for name, (ok, detail) in readiness().items()}
    is_ready = all(check.ok for check in checks.values())
    if not is_ready:
        response.status_code = 503
    return ReadinessResponse(status="ready" if is_ready else "not_ready", checks=checks)


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Blocked by the input guardrail"},
        503: {"model": ErrorResponse, "description": "Ollama unreachable or model missing"},
        504: {"model": ErrorResponse, "description": "Ollama did not respond in time"},
    },
)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question about the maintenance manuals.

    Declared with plain ``def`` (not ``async def``) because the agent is blocking: it
    waits on the vector store and on Ollama. FastAPI runs sync endpoints in a thread
    pool, so a slow LLM call doesn't freeze the server for other requests.
    """
    return ChatResponse(**run_agent(request.question))
