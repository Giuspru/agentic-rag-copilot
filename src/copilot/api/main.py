"""API entry point — builds the FastAPI app.

Run the server with:

    uvicorn copilot.api.main:app --reload

then open http://localhost:8000/docs for the interactive documentation.
"""

from fastapi import FastAPI

from copilot import __version__
from copilot.api.routes import router

app = FastAPI(
    title="Agentic RAG Copilot",
    description="Answers questions about maintenance manuals, grounded in the documents.",
    version=__version__,
)
app.include_router(router)
