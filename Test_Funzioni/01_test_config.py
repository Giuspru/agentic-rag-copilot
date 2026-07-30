"""TEST 01 — settings (config.py)

COSA TESTIAMO
-------------
`settings` NON e' una funzione: e' un'ISTANZA della classe Settings
(che eredita da pydantic BaseSettings). E' la FONDAZIONE del progetto:
ogni altra funzione (get_llm, get_vectorstore, ingest, ...) legge i suoi
parametri da qui. Se questa e' rotta, e' rotto tutto il resto.

COSA VERIFICHIAMO
-----------------
1. che l'oggetto settings esista e sia importabile;
2. che contenga i valori attesi (i default definiti in config.py);
3. che i path siano oggetti Path e puntino dentro al progetto;
4. che PROJECT_ROOT sia calcolato correttamente.
"""

from pathlib import Path

from copilot.config import settings, PROJECT_ROOT

print("=" * 70)
print("TEST 01 — settings (config)")
print("=" * 70)

# --- 1. Esiste ed e' l'istanza giusta ---
print("\n[1] Tipo dell'oggetto settings:")
print("    ", type(settings).__name__)

# --- 2. Valori Ollama ---
print("\n[2] Parametri Ollama:")
print("     ollama_base_url :", settings.ollama_base_url)
print("     llm_model       :", settings.llm_model)
print("     embedding_model :", settings.embedding_model)

# --- 3. Path del progetto ---
print("\n[3] Path:")
print("     PROJECT_ROOT    :", PROJECT_ROOT)
print("     manuals_dir     :", settings.manuals_dir)
print("     chroma_dir      :", settings.chroma_dir)
print("     collection_name :", settings.collection_name)

# --- 4. Parametri RAG ---
print("\n[4] Parametri RAG:")
print("     chunk_size      :", settings.chunk_size)
print("     chunk_overlap   :", settings.chunk_overlap)
print("     retrieval_k     :", settings.retrieval_k)

# --- ASSERT: controlli automatici (falliscono con errore se qualcosa non torna) ---
print("\n[ASSERT] verifiche automatiche...")

assert settings.llm_model == "llama3.2:3b"
assert settings.embedding_model == "nomic-embed-text"
assert settings.collection_name == "maintenance_manuals"
assert settings.chunk_size == 800
assert settings.chunk_overlap == 120
assert settings.retrieval_k == 4

# i path devono essere oggetti Path (non semplici stringhe)
assert isinstance(settings.manuals_dir, Path)
assert isinstance(settings.chroma_dir, Path)

# i path devono stare DENTRO la root del progetto
assert PROJECT_ROOT in settings.manuals_dir.parents
assert PROJECT_ROOT in settings.chroma_dir.parents

# la cartella dei manuali deve esistere davvero su disco
assert settings.manuals_dir.exists(), "manuals_dir non trovata su disco!"

print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: settings caricato correttamente. Fondazione OK.")
