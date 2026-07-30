"""TEST 09 — format_docs() (rag/chain.py)

COSA FA format_docs()
---------------------
E' la piccola funzione-"traduttore" che collega il retriever al prompt.
Il retriever restituisce una list[Document] (oggetti), ma il placeholder
{context} del prompt vuole UNA stringa. format_docs fa da ponte:

    list[Document]  ->  str unica

Per ogni chunk scrive:
    [Source: <nome_file>]
    <testo del chunk>
e incolla i blocchi separati da una riga vuota (\\n\\n).

Due scelte di design:
  - il prefisso [Source: ...] rende possibile la CITAZIONE (il modello vede
    da dove viene ogni pezzo);
  - il separatore \\n\\n fa leggere i chunk come passaggi distinti.

NOTA: format_docs e' una funzione Python normale (NON un Runnable). Nella
chain viene avvolta automaticamente in un RunnableLambda dall'operatore | .

COSA VERIFICHIAMO
-----------------
1. che da una list[Document] produca una singola str;
2. che il testo di OGNI chunk sia presente nella stringa;
3. che ci sia un [Source: ...] per ogni chunk (n. prefissi == n. chunk);
4. che i blocchi siano separati da riga vuota;
5. che funzioni sia con dati REALI (dal retriever) sia con Document finti.
"""

from langchain_core.documents import Document

from copilot.rag.chain import format_docs
from copilot.rag.retriever import get_retriever

print("=" * 70)
print("TEST 09 — format_docs()")
print("=" * 70)

# ---------- CASO A: Document FINTI (controllo totale sull'input) ----------
print("\n[A] Con Document costruiti a mano (input controllato):")
finti = [
    Document(page_content="Testo del primo chunk.", metadata={"source": "a.md"}),
    Document(page_content="Testo del secondo chunk.", metadata={"source": "b.md"}),
]
out = format_docs(finti)
print(out)
print("     ---")
print("     tipo output:", type(out).__name__)

# ---------- CASO B: Document REALI (dal retriever) ----------
print("\n[B] Con Document reali dal retriever ('lubrication'):")
docs_reali = get_retriever().invoke("lubrication")
out_reale = format_docs(docs_reali)
print("     n. chunk:", len(docs_reali))
print("     n. '[Source:' nella stringa:", out_reale.count("[Source:"))
print("     lunghezza stringa:", len(out_reale), "caratteri")
print("     --- primi 200 char ---")
print(out_reale[:200])

# --- ASSERT ---
print("\n[ASSERT] verifiche automatiche...")

# 1. output e' una stringa
assert isinstance(out, str)

# 2. il testo di ogni chunk finto e' dentro la stringa
assert "Testo del primo chunk." in out
assert "Testo del secondo chunk." in out

# 3. un [Source: ...] per ogni chunk
assert out.count("[Source:") == len(finti)
assert "[Source: a.md]" in out
assert "[Source: b.md]" in out

# 4. blocchi separati da riga vuota (\n\n)
assert "\n\n" in out

# 5. stesse proprieta' con i dati reali
assert isinstance(out_reale, str)
assert out_reale.count("[Source:") == len(docs_reali)
for d in docs_reali:
    # il testo di ogni chunk reale deve comparire nel contesto formattato
    assert d.page_content in out_reale

print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: format_docs() fonde i Document in una stringa unica con")
print("           prefissi [Source:] per la citazione. Format OK.")
