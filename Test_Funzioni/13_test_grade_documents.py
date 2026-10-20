"""TEST 13 — grade_documents() (agent/nodes.py)

COSA FA grade_documents()
-------------------------
E' il passo "corrective" di CRAG. Il retriever restituisce SEMPRE k chunk
(i piu' simili), anche quando non c'entrano nulla. grade_documents chiede
all'LLM-giudice, chunk per chunk: "questo estratto aiuta a rispondere?"
(risposta strutturata "yes"/"no") e TIENE SOLO quelli rilevanti.

Perche' serve: i chunk irrilevanti sono rumore nel prompt e un invito ad
allucinare. Il RAG lineare li passa tutti all'LLM; l'agente li filtra.

NB: si valuta rispetto a state["query"] (la query di ricerca in inglese),
non alla domanda originale: il modello 3B giudica molto meglio quando
estratto e query sono nella stessa lingua.

COSA VERIFICHIAMO
-----------------
1. "pump vibrating": dei 4 chunk recuperati sopravvive quello con la
   sezione 9.3 Excessive vibration, il rumore viene scartato;
2. domanda fuori corpus (bicicletta): 0 chunk rilevanti su 4 — e' questo
   segnale che fara' scattare la riscrittura della query nel grafo.
"""

from copilot.agent.nodes import grade_documents, retrieve

print("=" * 70)
print("TEST 13 — grade_documents()")
print("=" * 70)


def recupera_e_valuta(query: str) -> tuple[dict, dict]:
    stato = {"question": query, "query": query}
    recuperati = retrieve(stato)
    valutati = grade_documents({**stato, **recuperati})
    return recuperati, valutati


# --- 1. Dentro il corpus: il filtro toglie il rumore ---
print("\n[1] 'pump vibrating'")
rec, val = recupera_e_valuta("pump vibrating")
for doc in rec["documents"]:
    tenuto = "TENUTO  " if doc in val["documents"] else "scartato"
    print(f"     {tenuto} {doc.page_content[:55]!r}")
print("    ", val["steps"][0])
vibrazione_tenuta = any("9.3 Excessive vibration" in d.page_content for d in val["documents"])

# --- 2. Fuori corpus: nessun chunk sopravvive ---
print("\n[2] 'What is the correct tire pressure for a road bicycle?'")
rec2, val2 = recupera_e_valuta("What is the correct tire pressure for a road bicycle?")
print("    ", val2["steps"][0])

# --- ASSERT ---
print("\n[ASSERT] verifiche automatiche...")
assert len(rec["documents"]) == 4, "il retriever dovrebbe restituire k=4 chunk"
assert vibrazione_tenuta, "il chunk con la sezione 9.3 doveva essere tenuto!"
assert len(val["documents"]) < len(rec["documents"]), "il filtro non ha scartato nulla!"
assert val2["documents"] == [], "fuori corpus nessun chunk deve essere rilevante!"
print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: al generatore arrivano solo i chunk utili;")
print("           se non ce n'e' nessuno, il grafo lo sa e puo' correggersi.")
