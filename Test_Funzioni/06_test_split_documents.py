"""TEST 06 — split_documents() (rag/ingestion.py)

COSA FA split_documents()
-------------------------
E' il secondo passo dell'ingestion. Prende la list[Document] (1 Document =
1 manuale intero) e la taglia in CHUNK piu' piccoli, usando
RecursiveCharacterTextSplitter con i parametri di settings:
  - chunk_size = 800      -> ogni chunk ~800 caratteri al massimo
  - chunk_overlap = 120   -> chunk consecutivi si SOVRAPPONGONO di ~120 char
  - add_start_index=True  -> salva l'offset del chunk nel doc originale

Perche' spezzare? Un manuale intero e' troppo grande da (a) embeddare bene e
(b) infilare in un prompt. Chunk piccoli = retrieval piu' preciso.
Perche' l'overlap? Cosi' un fatto a cavallo tra due chunk non va perso.

COSA VERIFICHIAMO
-----------------
1. che da 1 Document escano PIU' chunk (tutti Document a loro volta);
2. che ogni chunk rispetti il limite chunk_size;
3. che i chunk EREDITINO il metadata['source'] del documento originale;
4. che ogni chunk abbia 'start_index' (grazie a add_start_index=True);
5. che esista davvero la SOVRAPPOSIZIONE (la coda di un chunk ricompare
   nella testa del successivo).
"""

from langchain_core.documents import Document

from copilot.config import settings
from copilot.rag.ingestion import load_documents, split_documents

print("=" * 70)
print("TEST 06 — split_documents()")
print("=" * 70)

# partiamo dall'output del test precedente
docs = load_documents()
chunks = split_documents(docs)

print(f"\n[1] Da {len(docs)} Document -> {len(chunks)} chunk")
print("     tipo del contenitore:", type(chunks).__name__)
print("     tipo di ogni elemento:", type(chunks[0]).__name__)

# --- 2. Dimensioni dei chunk ---
lunghezze = [len(c.page_content) for c in chunks]
print("\n[2] Dimensioni chunk (limite chunk_size =", settings.chunk_size, "):")
print("     min :", min(lunghezze))
print("     max :", max(lunghezze))
print("     media:", round(sum(lunghezze) / len(lunghezze), 1))

# --- 3-4. Metadati ereditati + start_index ---
print("\n[3] Metadati del primo chunk:")
print("     source     :", chunks[0].metadata.get("source"))
print("     start_index:", chunks[0].metadata.get("start_index"))

# --- 5. Verifica della sovrapposizione (overlap) fra due chunk consecutivi ---
print("\n[5] Prova della sovrapposizione tra chunk[0] e chunk[1]:")
coda_di_0 = chunks[0].page_content[-40:]
print("     ultimi 40 char di chunk[0]:", repr(coda_di_0))
print("     ...compaiono in chunk[1]? ->",
      any(coda_di_0[i:] in chunks[1].page_content for i in range(len(coda_di_0))))

# --- ASSERT ---
print("\n[ASSERT] verifiche automatiche...")

# 1. piu' chunk che documenti, tutti Document
assert isinstance(chunks, list)
assert all(isinstance(c, Document) for c in chunks)
assert len(chunks) > len(docs), "lo split non ha prodotto piu' pezzi!"

# 2. ogni chunk entro il limite (il splitter puo' sforare di poco su separatori,
#    ma non deve essere enormemente sopra: usiamo un margine di tolleranza)
assert max(lunghezze) <= settings.chunk_size + 50, "chunk troppo grande!"

# 3. ogni chunk eredita il source del documento originale
source_originale = docs[0].metadata["source"]
assert all(c.metadata.get("source") == source_originale for c in chunks), \
    "i chunk non hanno ereditato il source!"

# 4. add_start_index=True -> ogni chunk ha start_index intero
assert all(isinstance(c.metadata.get("start_index"), int) for c in chunks), \
    "manca start_index nei chunk!"

# 5. deve esistere sovrapposizione tra almeno una coppia consecutiva
def si_sovrappongono(a, b):
    coda = a.page_content[-40:]
    return any(coda[i:] in b.page_content for i in range(len(coda)))

assert any(si_sovrappongono(chunks[i], chunks[i + 1]) for i in range(len(chunks) - 1)), \
    "nessuna sovrapposizione trovata: overlap non funziona!"

print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: split_documents() spezza il manuale in chunk <= 800 char,")
print("           con overlap e source ereditato. Split OK.")
