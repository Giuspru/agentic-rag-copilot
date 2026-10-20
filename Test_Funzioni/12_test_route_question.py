"""TEST 12 — route_question() (agent/nodes.py)

COSA FA route_question()
------------------------
E' il PRIMO nodo del grafo. Usa l'LLM come GIUDICE (non come scrittore):
decide se il messaggio richiede i manuali ("retrieve") oppure e' solo
conversazione ("direct": saluti, ringraziamenti, "cosa sai fare?").

L'LLM non risponde a testo libero: con with_structured_output(RouteDecision)
DEVE produrre un JSON valido secondo lo schema Pydantic, quindi il grafo
riceve un valore affidabile ("retrieve" | "direct") su cui ramificare.

Il nodo inizializza anche lo stato: query = domanda, contatori a zero.

COSA VERIFICHIAMO
-----------------
1. saluti/ringraziamenti (anche in italiano) -> "direct";
2. domande vere (anche fuori tema, anche in italiano) -> "retrieve"
   (meglio cercare e non trovare che rispondere "a memoria");
3. lo stato iniziale e' corretto (query, rewrites, generations, steps).
"""

from copilot.agent.nodes import route_question

print("=" * 70)
print("TEST 12 — route_question()")
print("=" * 70)

casi = {
    "Hi! What can you do?":                                  "direct",
    "Ciao, cosa sai fare?":                                  "direct",
    "Thanks, that helped":                                   "direct",
    "What grease should I use for the bearings?":            "retrieve",
    "la pompa perde dalla tenuta, cosa faccio?":             "retrieve",
    "What is the correct tire pressure for a road bicycle?": "retrieve",
}

print("\n[1/2] Decisioni di routing:")
risultati = {}
for messaggio, atteso in casi.items():
    update = route_question({"question": messaggio})
    risultati[messaggio] = update
    esito = "✓" if update["route"] == atteso else "✗"
    print(f"     {esito} {messaggio!r:56} -> {update['route']:8} (atteso {atteso})")

print("\n[3] Stato iniziale scritto dal nodo (per l'ultima domanda):")
ultimo = risultati["What is the correct tire pressure for a road bicycle?"]
for chiave in ["query", "rewrites", "generations", "steps"]:
    print(f"     {chiave:12} = {ultimo[chiave]!r}")

# --- ASSERT ---
print("\n[ASSERT] verifiche automatiche...")
for messaggio, atteso in casi.items():
    assert risultati[messaggio]["route"] == atteso, f"routing sbagliato: {messaggio!r}"
assert ultimo["query"] == "What is the correct tire pressure for a road bicycle?"
assert ultimo["rewrites"] == 0 and ultimo["generations"] == 0
assert ultimo["steps"] == ["route -> retrieve"]
print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: il router separa conversazione e domande, in inglese e in italiano.")
