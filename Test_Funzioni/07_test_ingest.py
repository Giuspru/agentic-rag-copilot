"""TEST 07 — ingest() (rag/ingestion.py)

COSA FA ingest()
----------------
E' la funzione che ORCHESTRA l'intero lato scrittura del RAG, mettendo
insieme i mattoni gia' testati (05 e 06) e collegandoli a Chroma:

    documents = load_documents()          # 1 Document per file
    chunks    = split_documents(documents) # -> 11 chunk
    vectorstore = get_vectorstore()
    if reset: vectorstore.reset_collection()   # svuota prima (idempotenza)
    vectorstore.add_documents(chunks)      # EMBEDDA ogni chunk e lo salva
    return len(chunks)                     # quanti chunk indicizzati

Restituisce il numero di chunk salvati. add_documents() e' il momento in cui
i chunk (testo) diventano VETTORI su disco (via la embedding_function).

COSA VERIFICHIAMO
-----------------
1. che ingest() restituisca il numero di chunk atteso (== output di split);
2. che dopo l'ingest la collection Chroma contenga davvero quei vettori
   (count() sul database == numero restituito);
3. l'IDEMPOTENZA (reset=True): eseguire ingest DUE volte NON raddoppia i
   chunk (niente duplicati). Con reset=True il conteggio resta stabile.
4. che dopo l'ingest una similarity_search restituisca chunk pertinenti.
"""

from copilot.rag.ingestion import ingest, load_documents, split_documents
from copilot.rag.vectorstore import get_vectorstore

print("=" * 70)
print("TEST 07 — ingest()")
print("=" * 70)

# quanti chunk ci aspettiamo? lo sappiamo dai mattoni gia' testati
attesi = len(split_documents(load_documents()))
print("\n[0] Chunk attesi (da load+split):", attesi)

# --- 1. Prima ingest ---
n1 = ingest(reset=True)
print("\n[1] Prima ingest(reset=True) -> chunk indicizzati:", n1)

# --- 2. Il database contiene davvero quei vettori? ---
vs = get_vectorstore()
count_db = vs._collection.count()
print("\n[2] Conteggio reale nella collection Chroma:", count_db)

# --- 3. IDEMPOTENZA: seconda ingest, il totale NON deve raddoppiare ---
n2 = ingest(reset=True)
count_db_2 = vs._collection.count()
print("\n[3] Seconda ingest(reset=True):")
print("     chunk restituiti:", n2)
print("     conteggio nel DB dopo la 2a ingest:", count_db_2)
print("     (senza reset sarebbe diventato", attesi * 2, "-> duplicati!)")

# --- 4. Il retrieval funziona dopo l'ingest? ---
print("\n[4] similarity_search di prova ('lubrication grease'):")
risultati = vs.similarity_search("increase flow", k=2)
for i, d in enumerate(risultati, 1):
    print(f"     [{i}] {d.page_content[:70]!r}")

# --- ASSERT ---
print("\n[ASSERT] verifiche automatiche...")

# 1. ingest restituisce il numero di chunk atteso
assert n1 == attesi, f"ingest ha indicizzato {n1} chunk invece di {attesi}!"

# 2. il DB contiene esattamente quei vettori
assert count_db == attesi, "il conteggio nel DB non corrisponde ai chunk!"

# 3. idempotenza: dopo la seconda ingest il totale e' identico (no duplicati)
assert n2 == attesi
assert count_db_2 == attesi, "reset non funziona: la collection e' cresciuta!"

# 4. il retrieval restituisce qualcosa di sensato
assert len(risultati) == 2
assert all(len(d.page_content) > 0 for d in risultati)

print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: ingest() carica, spezza, vettorizza e salva in Chroma;")
print("           reset=True garantisce idempotenza (niente duplicati). Ingest OK.")
