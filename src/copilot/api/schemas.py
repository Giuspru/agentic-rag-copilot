"""API schemas — the request/response contract of the HTTP layer.

These Pydantic models describe what a client must send and what it gets back. FastAPI
uses them to validate incoming JSON (rejecting bad input with a 422 before our code
runs), to serialize responses, and to generate the interactive docs at ``/docs``.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """A question for the copilot.

    The length limits are a first, minimal input guardrail: empty questions never reach
    the LLM, and oversized ones can't flood the prompt. Surrounding whitespace is
    stripped first, so a question made only of spaces counts as empty. The content
    checks (prompt injection) come next, in ``guardrails/validators.py``.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question about the maintenance manuals.",
        examples=["What grease should I use for the bearings?"],
    )


class ChatResponse(BaseModel):
    """The copilot's answer, with the documents it rests on and how it got there."""

    answer: str = Field(..., description="Answer grounded in the manuals, with citations.")
    sources: list[str] = Field(
        default_factory=list,
        description="Manuals the answer is grounded in (empty if it didn't use them).",
        examples=[["centrifugal-pump-maintenance-manual.md"]],
    )
    steps: list[str] = Field(
        default_factory=list,
        description="Trace of the agent's path through the graph.",
        examples=[["route -> retrieve", "retrieve -> 4 chunks", "grade -> 2/4 relevant"]],
    )


class HealthResponse(BaseModel):
    """Liveness check: tells a client (or a monitor) that the API is up."""

    status: str = Field(..., examples=["ok"])


class CheckResult(BaseModel):
    """Outcome of one readiness check."""

    ok: bool
    detail: str = Field(..., examples=["models available: llama3.2:3b, nomic-embed-text:latest"])


class ReadinessResponse(BaseModel):
    """Readiness: can the app answer right now? Returned with 200 or 503."""

    status: Literal["ready", "not_ready"]
    checks: dict[str, CheckResult]


class ErrorResponse(BaseModel):
    """Body of every error raised by the app (4xx/5xx), in one predictable shape."""

    error: str = Field(..., description="Machine-readable code.", examples=["llm_unavailable"])
    detail: str = Field(..., description="Human-readable explanation.")
