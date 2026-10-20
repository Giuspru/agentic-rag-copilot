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
    """The copilot's answer.

    Citations are inside the text for now (the linear chain returns a plain string);
    structured ``sources`` will come with the agent, whose state holds the documents.
    """

    answer: str = Field(..., description="Answer grounded in the manuals, with citations.")


class HealthResponse(BaseModel):
    """Liveness check: tells a client (or a monitor) that the API is up."""

    status: str = Field(..., examples=["ok"])
