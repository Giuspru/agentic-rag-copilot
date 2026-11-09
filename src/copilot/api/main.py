"""API entry point — builds the FastAPI app.

Run the server with:

    uvicorn copilot.api.main:app --reload

then open http://localhost:8000/docs for the interactive documentation.

Besides the routes, the app wires the cross-cutting concerns of phase 3:
  - **startup** (lifespan): logging, and tracing when enabled;
  - **middleware**: a request ID and the processing time on every response, plus one
    log line per request;
  - **exception handlers**: domain errors -> their HTTP status with a JSON body;
    anything unexpected -> a generic 500 that leaks no internals, logged with traceback.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from copilot import __version__
from copilot.api.routes import router
from copilot.errors import CopilotError, LLMUnavailableError
from copilot.observability.logs import request_id_var, setup_logging
from copilot.observability.tracing import setup_tracing

logger = logging.getLogger("copilot.api")

# How long a client should wait before retrying when Ollama is down (seconds).
RETRY_AFTER_SECONDS = 30


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    setup_tracing()
    logger.info("Agentic RAG Copilot %s started", __version__)
    yield


app = FastAPI(
    title="Agentic RAG Copilot",
    description="Answers questions about maintenance manuals, grounded in the documents.",
    version=__version__,
    lifespan=lifespan,
)
app.include_router(router)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Tag the request with an ID, time it, and log one line when it completes."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    # Everything that logs stays inside the try: the ID is reset only in ``finally``, after
    # the last log line of the request.
    try:
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{elapsed:.3f}"
        logger.info(
            "%s %s -> %d (%.2fs)", request.method, request.url.path, response.status_code, elapsed
        )
        return response
    except Exception:
        # The 500 response itself is built by ``unexpected_error_handler``, outside this
        # middleware, where the request ID is no longer set: the traceback is logged here.
        elapsed = time.perf_counter() - start
        logger.exception(
            "%s %s -> unhandled error (%.2fs)", request.method, request.url.path, elapsed
        )
        raise
    finally:
        request_id_var.reset(token)


@app.exception_handler(CopilotError)
async def copilot_error_handler(request: Request, exc: CopilotError) -> JSONResponse:
    """Domain error -> its HTTP status, with the ``ErrorResponse`` body."""
    logger.warning("%s: %s", exc.code, exc)
    headers = None
    if isinstance(exc, LLMUnavailableError):
        headers = {"Retry-After": str(RETRY_AFTER_SECONDS)}
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.code, "detail": str(exc)},
        headers=headers,
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: a bug. The client gets no stack trace (the middleware logged it)."""
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "Unexpected server error."},
    )
