"""TEST 14 — rewrite_query() (agent/nodes.py)

COSA FA rewrite_query()
-----------------------
Quando grade_documents non trova chunk rilevanti, il grafo non si arrende:
chiede all'LLM (stavolta come SCRITTORE) di riformulare la query di
ricerca — breve, specifica, in inglese, coi termini tecnici che un manuale
userebbe. Per una domanda in italiano la riscrittura fa anche da
TRADUZIONE verso la lingua dei manuali.

Aggiorna lo stato: nuova query, rewrites + 1, e azzera generations (nuovi
documenti = nuovo budget di tentativi di risposta).

COSA VERIFICHIAMO
-----------------
1. la domanda italiana viene riscritta in una query inglese diversa;
2. i contatori vengono aggiornati correttamente;
3. L'EFFETTO CHE CONTA: la query originale (italiano) non porta chunk
   rilevanti, quella riscritta si' — la riscrittura migliora il retrieval.
"""

from copilot.agent.nodes import grade_documents, retrieve, rewrite_query

print("=" * 70)
print("TEST 14 — rewrite_query()")
print("=" * 70)

domanda = "quale grasso devo usare per i cuscinetti?"
stato = {"question": domanda, "query": domanda, "rewrites": 0, "generations": 2}


def chunk_rilevanti(query: str) -> int:
    s = {"question": domanda, "query": query}
    return len(grade_documents({**s, **retrieve(s)})["documents"])


# --- 1-2. La riscrittura ---
update = rewrite_query(stato)
print("\n[1] Domanda originale:", repr(domanda))
print("     Query riscritta:  ", repr(update["query"]))
print("\n[2] Contatori: rewrites", stato["rewrites"], "->", update["rewrites"],
      "| generations", stato["generations"], "->", update["generations"])

# --- 3. L'effetto sul retrieval ---
prima = chunk_rilevanti(domanda)
dopo = chunk_rilevanti(update["query"])
print(f"\n[3] Chunk rilevanti: query originale {prima}/4 -> query riscritta {dopo}/4")

# --- ASSERT ---
print("\n[ASSERT] verifiche automatiche...")
assert update["query"] and update["query"] != domanda
assert update["query"].isascii(), "la query riscritta dovrebbe essere in inglese"
assert update["rewrites"] == 1 and update["generations"] == 0
assert dopo > prima, "la riscrittura non ha migliorato il retrieval!"
print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: quando la ricerca fallisce, l'agente riformula e riprova —")
print("           e la nuova query trova cio' che la prima non trovava.")
