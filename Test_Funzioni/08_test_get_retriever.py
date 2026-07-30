"""TEST 08 — get_retriever() (rag/retriever.py)

COSA FA get_retriever()
-----------------------
E' il PONTE tra Famiglia 3 e Famiglia 1. Prende il Chroma (motore, senza
.invoke()) e lo "veste" da VectorStoreRetriever chiamando:

    get_vectorstore().as_retriever(search_kwargs={"k": settings.retrieval_k})

Il retriever e' un Runnable: si aziona con .invoke(domanda) e restituisce
i k chunk piu' rilevanti come list[Document]. E' l'interfaccia standard
"data una domanda, dammi i documenti", pronta da mettere in una pipe | .

Gotcha: k va DENTRO search_kwargs, non passato diretto ad as_retriever().

COSA VERIFICHIAMO
-----------------
1. che restituisca un VectorStoreRetriever;
2. @lru_cache: due chiamate -> stesso oggetto;
3. che ORA sia un Runnable con .invoke() (a differenza del Chroma da cui nasce);
4. che k sia configurato da settings.retrieval_k (=4);
5. che .invoke(domanda) restituisca esattamente k Document pertinenti.
"""

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import Runnable
from langchain_core.documents import Document

from copilot.config import settings
from copilot.rag.retriever import get_retriever
from copilot.rag.vectorstore import get_vectorstore

print("=" * 70)
print("TEST 08 — get_retriever()")
print("=" * 70)

# --- 1. Tipo restituito ---
retriever = get_retriever()
print("\n[1] Tipo restituito:", type(retriever).__name__)

# --- 2. @lru_cache ---
r_a = get_retriever()
r_b = get_retriever()
print("\n[2] Effetto @lru_cache:")
print("     stesso oggetto? ->", r_a is r_b)

# --- 3. IL PONTE: il Chroma NON e' Runnable, il retriever SI ---
vs = get_vectorstore()
print("\n[3] Il ponte Famiglia 3 -> Famiglia 1:")
print("     Chroma    ha .invoke()? ->", hasattr(vs, "invoke"),
      "| e' Runnable? ->", isinstance(vs, Runnable))
print("     Retriever ha .invoke()? ->", hasattr(retriever, "invoke"),
      "| e' Runnable? ->", isinstance(retriever, Runnable))

# --- 4. k configurato da settings ---
print("\n[4] Parametro k (dentro search_kwargs):")
print("     search_kwargs:", retriever.search_kwargs)
print("     settings.retrieval_k:", settings.retrieval_k)

# --- 5. .invoke() restituisce k Document ---
print("\n[5] retriever.invoke('bearing lubrication interval'):")
docs = retriever.invoke("bearing lubrication interval")
print("     tipo output:", type(docs).__name__, "di", len(docs), "elementi")
for i, d in enumerate(docs, 1):
    print(f"     [{i}] {d.page_content[:65]!r}")

# --- 5. .invoke() restituisce k Document ---
print("\n[6] retriever.invoke(Tech Spec):")
docs = retriever.invoke("Technical Specification")
print("     tipo output:", type(docs).__name__, "di", len(docs), "elementi")
for i, d in enumerate(docs, 1):
    print(f"     [{i}] {d.page_content[:65]!r}")


# --- 5. .invoke() restituisce k Document ---
print("\n[7] retriever.invoke(Replacement):")
docs = retriever.invoke("Replacement")
print("     tipo output:", type(docs).__name__, "di", len(docs), "elementi")
for i, d in enumerate(docs, 1):
    print(f"     [{i}] {d.page_content[:65]!r}")

# --- ASSERT ---
print("\n[ASSERT] verifiche automatiche...")

assert isinstance(retriever, VectorStoreRetriever)
assert r_a is r_b, "lru_cache non riusa l'istanza!"

# il ponte: Chroma NON Runnable, retriever SI
assert not isinstance(vs, Runnable)
assert isinstance(retriever, Runnable)
assert hasattr(retriever, "invoke")

# k viene da settings ed e' dentro search_kwargs
assert retriever.search_kwargs.get("k") == settings.retrieval_k

# .invoke() -> esattamente k Document
assert isinstance(docs, list)
assert all(isinstance(d, Document) for d in docs)
assert len(docs) == settings.retrieval_k, f"attesi {settings.retrieval_k} chunk!"

print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: get_retriever() trasforma il Chroma in un Runnable")
print("           azionabile con .invoke(), che rende k chunk. Retriever OK.")
