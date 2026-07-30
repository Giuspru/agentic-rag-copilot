"""Ingestion — the WRITE side of the RAG pipeline.

Turning raw documents into something we can retrieve happens in three steps:

  1. load   — read the manuals from disk into LangChain ``Document`` objects;
  2. split  — cut each document into overlapping chunks small enough to embed and to
              fit into a prompt;
  3. index  — store the chunks in Chroma, which embeds each one and saves the vector.

Each step is its own small function so it can be understood and tested in isolation.
``ingest()`` simply wires them together.
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from copilot.config import settings
from copilot.rag.vectorstore import get_vectorstore


def load_documents() -> list[Document]:
    """Load every ``.md`` file under ``settings.manuals_dir`` as plain text.

    We read Markdown as raw text (via ``TextLoader``) instead of parsing its structure:
    for chunking and retrieval the plain text is enough, and it avoids heavier parsing
    dependencies.
    """
    loader = DirectoryLoader(
        str(settings.manuals_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    # TextLoader records the full file path in metadata["source"]; keep only the file
    # name so later citations read like "centrifugal-pump-maintenance-manual.md".
    for doc in documents:
        doc.metadata["source"] = Path(doc.metadata["source"]).name

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks using sizes from config.

    ``RecursiveCharacterTextSplitter`` tries a list of separators in order (paragraphs,
    then lines, then sentences, then words), so it keeps semantic units together as much
    as the chunk size allows. The overlap carries a bit of context across the cut, so a
    fact that sits on a chunk boundary is not lost.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,  # record each chunk's character offset in the source doc
    )
    return splitter.split_documents(documents)


def ingest(reset: bool = True) -> int:
    """Load, split, and index the manuals in Chroma. Return the number of chunks stored.

    When ``reset`` is True the collection is emptied first, so re-running ingestion does
    not create duplicate chunks — the operation is idempotent.
    """
    documents = load_documents()
    chunks = split_documents(documents)

    vectorstore = get_vectorstore()
    if reset:
        vectorstore.reset_collection()

    # add_documents embeds each chunk (via the store's embedding function) and persists it.
    vectorstore.add_documents(chunks)

    return len(chunks)
