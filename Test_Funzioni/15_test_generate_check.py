"""TEST 15 — generate() + check_answer() (agent/nodes.py)

COSA FANNO
----------
generate()      scrive la risposta usando SOLO i chunk sopravvissuti al
                grading (al 2° tentativo usa STRICT_PROMPT, piu' severo:
                a temperatura 0 lo stesso prompt darebbe la stessa risposta).

check_answer()  il controllo Self-RAG, con DUE giudici distinti:
                1. GROUNDED: ogni fatto della risposta sta nei chunk?
                   (no -> "not_grounded": ha inventato qualcosa)
                2. ANSWERS:  la risposta parla del problema chiesto?
                   (no -> "not_useful": e' fondata ma fuori tema)
                Entrambi ok -> "useful".

COSA VERIFICHIAMO
-----------------
1. risposta generata dai chunk giusti -> contiene il fatto, verdetto "useful";
2. risposta INVENTATA (numero e codice falsi) -> "not_grounded":
   il giudice becca l'allucinazione;
3. risposta FONDATA ma su un ALTRO problema -> "not_useful".
"""

from copilot.agent.nodes import check_answer, generate, grade_documents, retrieve

print("=" * 70)
print("TEST 15 — generate() + check_answer()")
print("=" * 70)

domanda = "What is the part number of the mechanical seal?"
stato = {"question": domanda, "query": domanda, "generations": 0}
stato.update(retrieve(stato))
stato.update(grade_documents(stato))
print("\nChunk rilevanti:", [d.page_content[:30] for d in stato["documents"]])

# --- 1. Generazione reale + check ---
gen = generate(stato)
verdetto_vero = check_answer({**stato, **gen})["verdict"]
print("\n[1] Risposta generata:", gen["generation"])
print("     tentativo:", gen["generations"], "| verdetto:", verdetto_vero)

# --- 2. Risposta inventata ---
inventata = "The mechanical seal part number is SP-CP2000-MS42; tighten it to 120 Nm."
verdetto_inventata = check_answer({**stato, "generation": inventata})["verdict"]
print("\n[2] Risposta inventata:", inventata)
print("     verdetto:", verdetto_inventata)

# --- 3. Risposta fondata ma fuori tema ---
fuori_tema = "Keep spare bearings in a dry environment between 5 °C and 40 °C."
verdetto_fuori_tema = check_answer({**stato, "generation": fuori_tema})["verdict"]
print("\n[3] Risposta fuori tema:", fuori_tema)
print("     verdetto:", verdetto_fuori_tema)

# --- ASSERT ---
print("\n[ASSERT] verifiche automatiche...")
assert "SP-CP2000-SEAL" in gen["generation"], "manca il part number corretto!"
assert gen["generations"] == 1
assert verdetto_vero == "useful"
assert verdetto_inventata == "not_grounded", "il giudice non ha beccato l'invenzione!"
assert verdetto_fuori_tema in ("not_useful", "not_grounded"), "fuori tema promosso!"
print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: l'agente controlla il proprio lavoro prima di consegnarlo.")
