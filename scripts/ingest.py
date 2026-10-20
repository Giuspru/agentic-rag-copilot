"""(Re)build the vector store from the manuals in ``data/manuals/``.

Usage:

    python scripts/ingest.py

Run it once before starting the API, and again whenever the manuals change. Ingestion
is idempotent (the collection is reset first), so re-running never duplicates chunks.
"""

from copilot.config import settings
from copilot.rag.ingestion import ingest


def main() -> None:
    print(f"Ingesting manuals from {settings.manuals_dir} ...")
    n_chunks = ingest()
    print(f"Done: {n_chunks} chunks indexed in collection '{settings.collection_name}'.")


if __name__ == "__main__":
    main()
