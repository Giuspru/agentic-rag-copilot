"""TEST 11 — l'API FastAPI (api/schemas.py + api/routes.py + api/main.py)

COSA FA L'API
-------------
Espone il RAG lineare via HTTP, cosi' qualunque client (browser, curl, un
frontend) puo' fare domande senza importare Python:

    GET  /health  -> {"status": "ok"}
    POST /chat    {"question": "..."} -> {"answer": "..."}

Lo strato web e' SOTTILE: valida l'input (schemas.py), chiama answer()
(il test 10) e impacchetta la risposta. Nessuna logica RAG vive qui.

COME LO TESTIAMO
----------------
Con il TestClient di FastAPI: simula le richieste HTTP chiamando l'app
DIRETTAMENTE in memoria, senza avviare uvicorn ne' aprire porte. Stesso
codice che gira in produzione, zero setup.

COSA VERIFICHIAMO
-----------------
1. /health risponde 200 con {"status": "ok"};
2. VALIDAZIONE: domanda vuota, mancante o troppo lunga -> 422, e l'LLM non
   viene nemmeno chiamato (ci pensa Pydantic prima del nostro codice);
3. /chat dentro il corpus -> 200, risposta corretta e citata (come test 10,
   ma passando da HTTP + JSON);
4. lo schema OpenAPI (quello che alimenta /docs) contiene i due endpoint.
"""

from fastapi.testclient import TestClient

from copilot.api.main import app

print("=" * 70)
print("TEST 11 — API FastAPI (/health, /chat)")
print("=" * 70)

client = TestClient(app)

# --- 1. /health ---
r_health = client.get("/health")
print("\n[1] GET /health ->", r_health.status_code, r_health.json())

# --- 2. Validazione dell'input (nessuna chiamata all'LLM) ---
print("\n[2] Input NON validi (devono essere rifiutati con 422):")
casi_non_validi = {
    "domanda vuota":     {"question": ""},
    "campo mancante":    {},
    "troppo lunga":      {"question": "x" * 2001},
    "tipo sbagliato":    {"question": 42},
}
codici = {}
for nome, body in casi_non_validi.items():
    r = client.post("/chat", json=body)
    codici[nome] = r.status_code
    print(f"     {nome:<16} -> {r.status_code}  ({r.json()['detail'][0]['type']})")

# --- 3. /chat dentro il corpus ---
print("\n[3] POST /chat, domanda DENTRO il corpus:")
q = "What grease should I use for the bearings?"
print("     Q:", q)
r_chat = client.post("/chat", json={"question": q})
body = r_chat.json()
print("     status:", r_chat.status_code)
print("     A:", body.get("answer"))

r_low = body.get("answer", "").lower()
fatto_grasso = ("nlgi" in r_low) or ("lithium" in r_low)
cita_fonte   = ("source" in r_low) or (".md" in r_low)
print("     -> parla del grasso giusto?", fatto_grasso, "| cita fonte?", cita_fonte)

# --- 4. Schema OpenAPI (/docs) ---
schema = client.get("/openapi.json").json()
percorsi = sorted(schema["paths"])
print("\n[4] Endpoint documentati in /docs:", percorsi)

# --- ASSERT ---
print("\n[ASSERT] verifiche automatiche...")

# 1. health
assert r_health.status_code == 200
assert r_health.json() == {"status": "ok"}

# 2. validazione: TUTTI rifiutati con 422
assert all(c == 422 for c in codici.values()), f"validazione fallita: {codici}"

# 3. chat: risposta con la forma giusta, corretta e citata
assert r_chat.status_code == 200
assert set(body) == {"answer"}, "la risposta non rispetta ChatResponse!"
assert fatto_grasso, "la risposta non parla del grasso!"
assert cita_fonte, "la risposta non cita la fonte!"

# 4. documentazione automatica
assert "/health" in percorsi and "/chat" in percorsi

print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: il RAG lineare e' esposto via HTTP.")
print("           Input non validi -> 422 senza toccare l'LLM.")
print("           Domanda valida   -> risposta citata in JSON.")
print("\n*** FASE 1 COMPLETA: dalla domanda HTTP alla risposta citata. ***")
