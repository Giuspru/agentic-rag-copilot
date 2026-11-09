"""TEST 17 — errori di dominio, retry e limite di ricorsione (errors.py + agent/graph.py)

IL PROBLEMA
-----------
Con Ollama spento, la STESSA causa produce eccezioni DIVERSE a seconda di
chi la incontra:
    chat model   -> httpx.ConnectError
    embeddings   -> ConnectionError (built-in di Python)
    modello assente -> ollama.ResponseError (status 404)
    Ollama bloccato -> httpx.ReadTimeout
Il resto dell'app non deve conoscere questi dettagli. errors.py li traduce
in pochi ERRORI DI DOMINIO, ognuno col suo status HTTP:
    LLMUnavailableError -> 503   |   LLMTimeoutError -> 504

DUE DOMANDE SEPARATE
--------------------
- "vale la pena RIPROVARE?"  -> is_transient()  (usato dal RetryPolicy)
- "cosa diciamo al CLIENT?"  -> l'errore di dominio (quando i retry finiscono)
Un timeout NON si riprova: ha gia' fatto aspettare llm_timeout secondi.
Un modello mancante nemmeno: nessun retry lo scarica.

RETRY NEL GRAFO
---------------
Ogni nodo che chiama Ollama ha un RetryPolicy di LangGraph (backoff
esponenziale). E' SICURO perche' i nodi sono IDEMPOTENTI: leggono lo stato e
restituiscono un aggiornamento, rieseguirli non ha effetti collaterali.

LIMITE DI RICORSIONE
--------------------
LangGraph ferma ogni run piu' lungo di recursion_limit con
GraphRecursionError. Il default in LangGraph 1.x e' ALTISSIMO: 10007 (nelle
versioni vecchie era 25 — l'avevo dato per scontato, il test 17 mi ha
smentito: verificare sempre sulla versione installata!).
Il rischio quindi e' l'opposto: se un bug rompesse un budget (es. un rewrite
che non incrementa il contatore), l'agente girerebbe per ~10000 chiamate LLM,
cioe' piu' di un giorno, prima di fermarsi. max_graph_steps() calcola dai
budget il limite PIU' STRETTO che non taglia mai un run legittimo (22 passi):
un bug che cicla fallisce subito e in modo visibile.

COSA VERIFICHIAMO (quasi tutto SENZA LLM: veloce e deterministico)
------------------
1. is_transient su ogni tipo di eccezione;
2. llm_errors() traduce eccezioni VERE (porta morta, timeout, modello finto);
3. il caso peggiore del grafo, con nodi FINTI: esattamente max_graph_steps()
   passi, e il limite e' stretto (uno in meno -> errore);
4. un BUG che cicla: col limite calcolato si ferma subito; col default di
   LangGraph eseguirebbe ~10000 nodi;
5. il retry: un nodo che fallisce 2 volte per errore transitorio viene
   rieseguito e il grafo completa; un errore NON transitorio non si riprova.
"""

import time
from unittest.mock import patch

import httpx
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from ollama import ResponseError

from copilot.agent import graph as graph_module
from copilot.config import settings
from copilot.errors import LLMTimeoutError, LLMUnavailableError, is_transient, llm_errors

print("=" * 70)
print("TEST 17 — errori di dominio, retry, limite di ricorsione")
print("=" * 70)

# --- 1. is_transient ---
print("\n[1] is_transient (si riprova?):")
casi_transient = {
    "ConnectionError (embeddings, Ollama spento)": (ConnectionError("down"), True),
    "httpx.ConnectError (chat, Ollama spento)": (httpx.ConnectError("refused"), True),
    "ResponseError 500 (errore lato server)": (ResponseError("boom", 500), True),
    "ResponseError 404 (modello mancante)": (ResponseError("not found", 404), False),
    "httpx.ReadTimeout (Ollama bloccato)": (httpx.ReadTimeout("slow"), False),
    "ValueError (bug nel nostro codice)": (ValueError("bug"), False),
}
for nome, (exc, atteso) in casi_transient.items():
    print(f"     {nome:<45} -> {is_transient(exc)}")

# --- 2. Traduzione di eccezioni VERE ---
print("\n[2] llm_errors() su eccezioni reali di Ollama:")
scenari = {
    "Ollama spento (porta 9)": ChatOllama(model=settings.llm_model, base_url="http://127.0.0.1:9"),
    "timeout di 1 ms": ChatOllama(model=settings.llm_model, client_kwargs={"timeout": 0.001}),
    "modello inesistente": ChatOllama(model="modello-inesistente:1b"),
}
tradotte = {}
for nome, llm in scenari.items():
    try:
        with llm_errors():
            llm.invoke("hi")
    except Exception as exc:
        tradotte[nome] = exc
        causa = type(exc.__cause__).__name__
        print(f"     {nome:<26} {causa:<14} -> {type(exc).__name__} ({exc.status_code})")
        print(f"     {'':<26} messaggio: {exc}")


# --- 3. Caso peggiore del grafo con nodi FINTI ---
# Sostituiamo i nodi veri con funzioni finte che seguono sempre il percorso piu'
# lungo possibile: documenti sempre trovati, risposta mai buona. Nessun LLM.
def finto_route(state):
    return {"route": "retrieve", "query": state["question"], "rewrites": 0,
            "generations": 0, "steps": ["route"]}


def finto_retrieve(state):
    return {"documents": ["chunk"], "steps": ["retrieve"]}


def finto_grade(state):
    return {"steps": ["grade"]}


def finto_generate(state):
    return {"generation": "bozza", "generations": state["generations"] + 1, "steps": ["generate"]}


def finto_check(state):
    # prima "non fondata" (riprova a generare), all'ultimo tentativo "non utile" (riscrive)
    if state["generations"] < settings.max_generations:
        return {"verdict": "not_grounded", "steps": ["check"]}
    return {"verdict": "not_useful", "steps": ["check"]}


def finto_rewrite(state):
    return {"rewrites": state["rewrites"] + 1, "generations": 0, "steps": ["rewrite"]}


nodi_finti = {
    "route_question": finto_route,
    "retrieve": finto_retrieve,
    "grade_documents": finto_grade,
    "generate": finto_generate,
    "check_answer": finto_check,
    "rewrite_query": finto_rewrite,
}


def costruisci_grafo_finto(**extra_patch):
    """Ricostruisce il grafo VERO (archi, budget, retry) ma con i nodi finti."""
    graph_module.get_agent.cache_clear()
    with patch.multiple(graph_module, **{**nodi_finti, **extra_patch}):
        agente = graph_module.get_agent()
    graph_module.get_agent.cache_clear()  # non lasciare il grafo finto in cache
    return agente


agente_finto = costruisci_grafo_finto()
passi_max = graph_module.max_graph_steps()
finale = agente_finto.invoke({"question": "q"}, config={"recursion_limit": passi_max + 1})
print(f"\n[3] Caso peggiore (max_rewrites={settings.max_rewrites}, "
      f"max_generations={settings.max_generations}):")
print(f"     max_graph_steps() = {passi_max}")
print(f"     passi eseguiti    = {len(finale['steps'])} -> {' > '.join(finale['steps'])}")
try:
    agente_finto.invoke({"question": "q"}, config={"recursion_limit": passi_max})
    limite_stretto = False
except GraphRecursionError:
    limite_stretto = True
print(f"     con recursion_limit = {passi_max} (uno in meno) -> GraphRecursionError? {limite_stretto}")

# --- 4. Un bug che cicla: limite calcolato vs default di LangGraph ---
esecuzioni = {"n": 0}


def rewrite_con_bug(state):
    """BUG voluto: dimentica di incrementare 'rewrites' -> il budget non finisce mai."""
    esecuzioni["n"] += 1
    return {"generations": 0, "steps": ["rewrite"]}


agente_bug_loop = costruisci_grafo_finto(rewrite_query=rewrite_con_bug)
try:
    agente_bug_loop.invoke({"question": "q"}, config={"recursion_limit": passi_max + 1})
except GraphRecursionError:
    pass
rewrite_col_limite = esecuzioni["n"]

esecuzioni["n"] = 0
inizio = time.perf_counter()
try:
    agente_bug_loop.invoke({"question": "q"})  # default di LangGraph
except GraphRecursionError:
    pass
rewrite_col_default = esecuzioni["n"]
durata_default = time.perf_counter() - inizio
print("\n[4] Bug che cicla (rewrite non incrementa il contatore):")
print(f"     limite calcolato ({passi_max + 1}) -> fermo dopo {rewrite_col_limite} rewrite")
print(f"     default LangGraph        -> fermo dopo {rewrite_col_default} rewrite "
      f"({durata_default:.1f}s coi nodi finti; con l'LLM vero, ~6 chiamate da ~10s "
      f"per giro = ~{rewrite_col_default * 60 / 3600:.0f} ore)")

# --- 5. Retry ---
tentativi = {"n": 0}


def route_instabile(state):
    """Fallisce 2 volte come un Ollama che si sta riavviando, poi funziona."""
    tentativi["n"] += 1
    if tentativi["n"] <= 2:
        raise httpx.ConnectError("Ollama si sta riavviando")
    return finto_route(state)


agente_retry = costruisci_grafo_finto(route_question=route_instabile)
risultato = agente_retry.invoke({"question": "q"}, config={"recursion_limit": passi_max + 1})
tentativi_transitorio = tentativi["n"]

tentativi["n"] = 0


def route_con_bug(state):
    tentativi["n"] += 1
    raise ValueError("bug")


agente_bug = costruisci_grafo_finto(route_question=route_con_bug)
try:
    agente_bug.invoke({"question": "q"})
except ValueError:
    pass
tentativi_bug = tentativi["n"]
print(f"\n[5] Retry (llm_max_attempts={settings.llm_max_attempts}):")
print(f"     errore transitorio x2 -> tentativi: {tentativi_transitorio}, "
      f"grafo completato: {risultato['steps'][-1] == 'fallback'}")
print(f"     errore NON transitorio -> tentativi: {tentativi_bug} (nessun retry)")

# --- ASSERT ---
print("\n[ASSERT] verifiche automatiche...")
for nome, (exc, atteso) in casi_transient.items():
    assert is_transient(exc) == atteso, nome
assert isinstance(tradotte["Ollama spento (porta 9)"], LLMUnavailableError)
assert isinstance(tradotte["timeout di 1 ms"], LLMTimeoutError)
assert isinstance(tradotte["modello inesistente"], LLMUnavailableError)
assert "Pull it first" in str(tradotte["modello inesistente"])
assert passi_max == 22 or (settings.max_rewrites, settings.max_generations) != (2, 2)
assert len(finale["steps"]) == passi_max, "il caso peggiore non coincide con la formula!"
assert finale["steps"][-1] == "fallback"
assert limite_stretto, "il limite non e' stretto: la formula sovrastima"
assert rewrite_col_limite <= 5, "il limite calcolato doveva fermare il ciclo subito"
assert rewrite_col_default > 1000, "col default il ciclo doveva andare molto avanti"
assert tentativi_transitorio == 3 and risultato["steps"][-1] == "fallback"
assert tentativi_bug == 1
print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: eccezioni eterogenee -> 2 errori di dominio con status HTTP;")
print("           errori transitori riprovati, bug e timeout no;")
print("           il grafo termina sempre; un bug che cicla si ferma in pochi passi.")
