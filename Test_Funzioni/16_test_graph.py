"""TEST 16 (FINALE FASE 2) — il grafo completo (agent/graph.py)

COSA FA graph.py
----------------
Collega i nodi in un StateGraph LangGraph:
  - add_node:               registra i nodi (il lavoro);
  - add_edge:               archi fissi (retrieve -> grade, sempre);
  - add_conditional_edges:  archi CONDIZIONALI: una funzione decide_*
                            legge lo stato e restituisce il nome del
                            prossimo nodo. E' qui che nascono i CICLI.
run_agent(q) invoca il grafo compilato e restituisce
{"answer", "sources", "steps"}.

COSA VERIFICHIAMO
-----------------
1. le funzioni decide_* SENZA LLM: sono funzioni pure (stato -> nome
   nodo), quindi testabili in modo deterministico e istantaneo — compresi
   i limiti di budget che garantiscono che il grafo TERMINI SEMPRE;
2. la struttura del grafo compilato (i nodi ci sono tutti);
3. end-to-end, i quattro percorsi tipici:
   a. saluto                -> respond_direct, nessuna fonte;
   b. domanda nel corpus    -> risposta citata al primo colpo;
   c. domanda fuori corpus  -> riscritture, poi fallback "I don't know";
   d. domanda in italiano   -> riscrittura (traduzione) e risposta fondata.
"""

from langgraph.graph import END

from copilot.agent.graph import (
    decide_after_check,
    decide_after_grade,
    get_agent,
    run_agent,
)
from copilot.agent.nodes import FALLBACK_ANSWER
from copilot.config import settings

print("=" * 70)
print("TEST 16 (FINALE FASE 2) — il grafo self-corrective")
print("=" * 70)

# --- 1. Archi condizionali: funzioni pure, nessun LLM ---
print(f"\n[1] decide_* (max_rewrites={settings.max_rewrites}, "
      f"max_generations={settings.max_generations}):")
casi_grade = [
    ({"documents": ["chunk"], "rewrites": 0}, "generate"),
    ({"documents": [], "rewrites": 0}, "rewrite_query"),
    ({"documents": [], "rewrites": settings.max_rewrites}, "fallback"),
]
casi_check = [
    ({"verdict": "useful", "generations": 1, "rewrites": 0}, END),
    ({"verdict": "not_grounded", "generations": 1, "rewrites": 0}, "generate"),
    ({"verdict": "not_grounded", "generations": settings.max_generations, "rewrites": 0},
     "fallback"),
    ({"verdict": "not_useful", "generations": 1, "rewrites": 0}, "rewrite_query"),
    ({"verdict": "not_useful", "generations": 1, "rewrites": settings.max_rewrites},
     "fallback"),
]
for stato, atteso in casi_grade:
    print(f"     after_grade {stato} -> {decide_after_grade(stato)}")
for stato, atteso in casi_check:
    print(f"     after_check {stato} -> {decide_after_check(stato)}")

# --- 2. Struttura ---
nodi = sorted(n for n in get_agent().get_graph().nodes if not n.startswith("__"))
print("\n[2] Nodi del grafo:", nodi)

# --- 3. End-to-end ---
print("\n[3] Percorsi end-to-end:")
domande = {
    "a": "Hi! What can you do?",
    "b": "What is the part number of the mechanical seal?",
    "c": "What is the correct tire pressure for a road bicycle?",
    "d": "quale grasso devo usare per i cuscinetti?",
}
risultati = {}
for chiave, domanda in domande.items():
    r = run_agent(domanda)
    risultati[chiave] = r
    print(f"\n  ({chiave}) Q: {domanda}")
    for step in r["steps"]:
        print("        -", step)
    print("      A:", r["answer"][:150].replace("\n", " "))
    print("      sources:", r["sources"])

# --- ASSERT ---
print("\n[ASSERT] verifiche automatiche...")
for stato, atteso in casi_grade:
    assert decide_after_grade(stato) == atteso
for stato, atteso in casi_check:
    assert decide_after_check(stato) == atteso
assert {"route_question", "retrieve", "grade_documents", "rewrite_query",
        "generate", "check_answer", "respond_direct", "fallback"} <= set(nodi)

a, b, c, d = (risultati[k] for k in "abcd")
assert a["steps"] == ["route -> direct", "respond_direct"] and a["sources"] == []
assert "SP-CP2000-SEAL" in b["answer"] and b["sources"]
assert b["steps"][-1] == "check -> useful"
assert c["answer"] == FALLBACK_ANSWER and c["sources"] == []
assert any(s.startswith("rewrite") for s in c["steps"])
assert any(s.startswith("rewrite") for s in d["steps"]), "nessuna riscrittura per l'italiano"
assert "nlgi" in d["answer"].lower() and d["sources"]
print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: l'agente sceglie il percorso, filtra, riformula, controlla")
print("           e, quando non trova supporto nei manuali, ammette di non sapere.")
print("\n*** FASE 2 (agente self-corrective) verificata end-to-end. ***")
