"""TEST 05 — load_documents() (rag/ingestion.py)

COSA FA load_documents()
------------------------
E' il primo passo dell'ingestion (il lato SCRITTURA del RAG). Legge ogni
file .md sotto data/manuals usando DirectoryLoader + TextLoader e restituisce
una lista di oggetti Document (Famiglia 2 - oggetti dato).

PUNTO CHIAVE: 1 FILE = 1 Document. Qui NON c'e' ancora nessuna divisione in
chunk: quella arriva dopo, con split_documents(). Un Document rappresenta un
intero manuale.

Inoltre normalizza metadata["source"] tenendo solo il NOME del file (non il
path assoluto), cosi' le citazioni leggono "centrifugal-pump-...md".

COSA VERIFICHIAMO
-----------------
1. che restituisca una list[Document];
2. che ci sia 1 Document per ogni file .md presente (1 file = 1 Document);
3. che ogni Document abbia page_content (testo) e metadata["source"];
4. che "source" sia il solo NOME del file, non il path assoluto;
5. che il testo NON sia vuoto.
"""

from pathlib import Path

from langchain_core.documents import Document

from copilot.config import settings
from copilot.rag.ingestion import load_documents

print("=" * 70)
print("TEST 05 — load_documents()")
print("=" * 70)

# quanti file .md ci sono davvero sul disco?
file_md = list(settings.manuals_dir.glob("**/*.md"))
print("\n[0] File .md trovati in data/manuals:", len(file_md))
for f in file_md:
    print("     -", f.name)

# --- 1-2. Caricamento ---
docs = load_documents()
print("\n[1] Tipo restituito:", type(docs).__name__)
print("     numero di Document:", len(docs))
print("\n[2] Regola '1 file = 1 Document':")
print("     file .md:", len(file_md), " ==  Document:", len(docs), "?")

# --- 3-4. Struttura di ogni Document ---
print("\n[3] Ispezione dei Document:")
for i, d in enumerate(docs, 1):
    print(f"     [{i}] source={d.metadata.get('source')!r}")
    print(f"         caratteri di testo: {len(d.page_content)}")
    print(f"         inizio testo: {d.page_content[:60]!r}")

# --- ASSERT ---
print("\n[ASSERT] verifiche automatiche...")

assert isinstance(docs, list)
assert all(isinstance(d, Document) for d in docs)

# 1 file = 1 Document
assert len(docs) == len(file_md), "numero Document != numero file .md!"

for d in docs:
    # ogni Document ha testo non vuoto
    assert len(d.page_content) > 0, "Document con testo vuoto!"
    # ogni Document ha il metadato 'source'
    assert "source" in d.metadata, "manca metadata['source']!"
    # 'source' deve essere solo il NOME del file, non un path
    source = d.metadata["source"]
    assert source == Path(source).name, f"source e' un path, non un nome: {source}"
    assert source.endswith(".md")

print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: load_documents() legge 1 Document per file, con testo e")
print("           source (solo nome file). NIENTE chunk ancora. Load OK.")
