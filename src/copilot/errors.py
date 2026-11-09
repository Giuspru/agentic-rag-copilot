"""Domain errors — what can go wrong, in the app's own vocabulary.

The libraries underneath report the same failure in different ways. With Ollama switched
off, for example, the chat model raises ``httpx.ConnectError`` while the embedding model
raises the built-in ``ConnectionError``; a missing model is an ``ollama.ResponseError``
with status 404; a hung server is an ``httpx.ReadTimeout``.

The rest of the app should not know any of that. ``llm_errors()`` translates those
low-level exceptions into a few domain errors, and each domain error carries the HTTP
status it maps to, so the API needs a single exception handler for all of them.

Two questions are kept separate on purpose:
  - *is it worth retrying?* (``is_transient``) — used by the graph's retry policy;
  - *what do we tell the client?* (the domain error) — used once retries are exhausted.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import httpx
from ollama import ResponseError

from copilot.config import settings


class CopilotError(Exception):
    """Base class of the app's errors. Subclasses set the HTTP mapping."""

    status_code: int = 500
    code: str = "internal_error"


class InputRejectedError(CopilotError):
    """The question was refused by the input guardrail (it never reaches the LLM)."""

    status_code = 400
    code = "input_rejected"


class LLMUnavailableError(CopilotError):
    """Ollama is unreachable, or a required model is missing. Retrying later may work."""

    status_code = 503
    code = "llm_unavailable"


class LLMTimeoutError(CopilotError):
    """Ollama stopped sending data for longer than ``settings.llm_timeout``."""

    status_code = 504
    code = "llm_timeout"


def is_transient(exc: BaseException) -> bool:
    """Should the failed call be retried?

    Yes for connection hiccups and server-side errors (Ollama restarting, model being
    reloaded). No for timeouts — a call that already waited ``llm_timeout`` seconds would
    only multiply the user's wait — and no for client errors such as a missing model,
    which no retry can fix.
    """
    if isinstance(exc, (ConnectionError, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    return isinstance(exc, ResponseError) and exc.status_code >= 500


@contextmanager
def llm_errors() -> Iterator[None]:
    """Translate low-level Ollama/HTTP exceptions raised inside the block."""
    try:
        yield
    except httpx.TimeoutException as exc:
        raise LLMTimeoutError(
            f"The language model did not respond within {settings.llm_timeout:.0f} seconds."
        ) from exc
    except (ConnectionError, httpx.TransportError) as exc:
        raise LLMUnavailableError(
            f"Cannot reach Ollama at {settings.ollama_base_url}. Is it running?"
        ) from exc
    except ResponseError as exc:
        if exc.status_code == 404:
            message = f"A required model is missing in Ollama ({exc.error}). Pull it first."
        else:
            message = f"Ollama returned an error (status {exc.status_code}): {exc.error}"
        raise LLMUnavailableError(message) from exc
