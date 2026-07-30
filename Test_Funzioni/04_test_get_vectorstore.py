"""TEST 04 — get_vectorstore() (rag/vectorstore.py)

COSA FA get_vectorstore()
-------------------------
E' l'UNICO punto che si connette al database vettoriale Chroma. Restituisce
un oggetto Chroma configurato con:
  - collection_name = "maintenance_manuals"  (la "tabella logica")
  - persist_directory = data/chroma           (il DB fisico su disco)
  - embedding_function = get_embeddings()      (il motore che vettorizza)
E' @lru_cache: la connessione viene aperta una volta e riusata.

Chroma NON e' un Runnable (Famiglia 3): niente .invoke(). Si aziona con i
suoi metodi propri: add_documents(), similarity_search(), count(), ...

COSA VERIFICHIAMO
-----------------
1. che restituisca un oggetto Chroma;
2. @lru_cache: due chiamate -> stessa connessione (stesso oggetto);
3. che la cartella di persistenza esista su disco;
4. che usi la stessa embedding_function di get_embeddings() (invariante
   CRITICA: stesso modello in scrittura e lettura, altrimenti i vettori
   non sono confrontabili);
5. che NON abbia .invoke() (conferma che e' Famiglia 3, non un Runnable).
"""

from langchain_chroma import Chroma
from langchain_core.runnables import Runnable

from copilot.config import settings
from copilot.llm.embeddings import get_embeddings
from copilot.rag.vectorstore import get_vectorstore

print("=" * 70)
print("TEST 04 — get_vectorstore()")
print("=" * 70)

# --- 1. Tipo restituito ---
vs = get_vectorstore()
print("\n[1] Tipo restituito:", type(vs).__name__)
print("\n[1] HAHAHAHAHAHAHHAHA", type(vs))


# --- 2. @lru_cache: stessa connessione riusata ---
vs_a = get_vectorstore()
vs_b = get_vectorstore()
print("\n[2] Effetto @lru_cache:")
print("     stesso oggetto? ->", vs_a is vs_b)

# --- 3. Cartella di persistenza su disco ---
print("\n[3] Persistenza su disco:")
print("     chroma_dir        :", settings.chroma_dir)
print("     esiste su disco?  ->", settings.chroma_dir.exists())

# --- 4. Nome collection ed embedding function ---
print("\n[4] Configurazione interna:")
print("     collection attesa :", settings.collection_name)
# la embedding function usata dal vector store deve essere LA STESSA
# istanza restituita da get_embeddings() (grazie a @lru_cache)
emb_dello_store = vs._embedding_function
emb_diretta     = get_embeddings()
print("     embedding_function e' get_embeddings()? ->", emb_dello_store is emb_diretta)

# --- 5. Chroma NON e' un Runnable ---
print("\n[5] Interfaccia:")
print("     ha .invoke()?          ->", hasattr(vs, "invoke"))
print("     ha .add_documents()?   ->", hasattr(vs, "add_documents"))
print("     ha .similarity_search()? ->", hasattr(vs, "similarity_search"))
print("     e' un Runnable?        ->", isinstance(vs, Runnable))

# --- ASSERT ---
print("\n[ASSERT] verifiche automatiche...")

assert isinstance(vs, Chroma)
assert vs_a is vs_b, "lru_cache non riusa la connessione!"
assert settings.chroma_dir.exists(), "cartella di persistenza mancante!"

# invariante critica: stesso modello di embedding in scrittura e lettura
assert emb_dello_store is emb_diretta, "embedding_function diversa da get_embeddings()!"

# conferma Famiglia 3: motore, non Runnable
assert not hasattr(vs, "invoke")
assert not isinstance(vs, Runnable)
assert hasattr(vs, "add_documents") and hasattr(vs, "similarity_search")

print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: get_vectorstore() apre UNA connessione Chroma persistente,")
print("           con lo stesso embedding di ingestion/query. Vector store OK.")
