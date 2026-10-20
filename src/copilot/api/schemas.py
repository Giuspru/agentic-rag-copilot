"""API schemas — the request/response contract of the HTTP layer.

These Pydantic models describe what a client must send and what it gets back. FastAPI
uses them to validate incoming JSON (rejecting bad input with a 422 before our code
runs), to serialize responses, and to generate the interactive docs at ``/docs``.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """A question for the copilot.

    The length limits are a first, minimal input guardrail: empty questions never reach
    the LLM, and oversized ones can't flood the prompt.
    """

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
