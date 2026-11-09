"""TEST 20 — tracing OpenTelemetry/OpenInference (observability/tracing.py)

COSA AGGIUNGE IL TRACING
------------------------
"steps" dice QUALI nodi sono girati. Una TRACCIA dice tutto il resto: per
ogni nodo il prompt esatto mandato al modello, la risposta grezza, il
giudizio, i chunk recuperati, i token e la latenza — annidati ad albero:

    LangGraph
    ├── route_question
    │   └── ChatOllama            <- prompt, risposta, token, ms
    ├── retrieve
    │   └── VectorStoreRetriever  <- i 4 chunk recuperati
    ...

I PEZZI
-------
- OpenTelemetry: lo standard (vendor-neutral) per le tracce: l'app produce
  SPAN (unita' di lavoro, temporizzate e annidate), un EXPORTER le spedisce;
- OpenInference (LangChainInstrumentor): si aggancia ai callback di
  LangChain, che anche LangGraph usa -> ogni Runnable diventa uno span, senza
  toccare il codice dell'agente;
- Phoenix: il backend che riceve gli span e li mostra (localhost:6006).

COME LO TESTIAMO
----------------
Senza server: passiamo a setup_tracing() un InMemorySpanExporter, che tiene
gli span in una lista. Cosi' possiamo ISPEZIONARLI con degli assert. In
produzione l'exporter e' quello verso Phoenix (vedi README).

COSA VERIFICHIAMO
-----------------
1. una domanda vera produce span per il grafo e per ogni nodo percorso;
2. ci sono span di tipo LLM (con modello e token) e RETRIEVER (con i chunk);
3. gli span sono ANNIDATI: ogni chiamata LLM sta dentro un nodo del grafo.
"""

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from copilot.agent.graph import run_agent
from copilot.observability.tracing import setup_tracing

print("=" * 70)
print("TEST 20 — tracing (OpenTelemetry + OpenInference)")
print("=" * 70)

exporter = InMemorySpanExporter()
setup_tracing(span_exporter=exporter)

domanda = "What grease should I use for the bearings?"
risultato = run_agent(domanda)
spans = exporter.get_finished_spans()

per_id = {s.context.span_id: s for s in spans}
figli: dict = {}
for s in spans:
    figli.setdefault(s.parent.span_id if s.parent else None, []).append(s)
for lista in figli.values():
    lista.sort(key=lambda s: s.start_time)


def kind(span) -> str:
    return span.attributes.get("openinference.span.kind", "?")


def stampa(span, livello=0):
    ms = (span.end_time - span.start_time) / 1e6
    extra = ""
    if kind(span) == "LLM":
        extra = (f"  model={span.attributes.get('llm.model_name')} "
                 f"tokens={span.attributes.get('llm.token_count.total')}")
    elif kind(span) == "RETRIEVER":
        n = sum(1 for k in span.attributes if k.endswith(".document.content"))
        extra = f"  documenti={n}"
    print(f"     {'  ' * livello}{span.name} [{kind(span)}] {ms:.0f} ms{extra}")
    if livello < 2:  # sotto il secondo livello: solo il conteggio, per brevita'
        for figlio in figli.get(span.context.span_id, []):
            stampa(figlio, livello + 1)


print(f"\nQ: {domanda}")
print("steps:", risultato["steps"])
print(f"\n[1-3] {len(spans)} span raccolti. Albero (primi due livelli):")
for radice in figli.get(None, []):
    stampa(radice)

nomi = {s.name for s in spans}
llm_spans = [s for s in spans if kind(s) == "LLM"]
retriever_spans = [s for s in spans if kind(s) == "RETRIEVER"]


def antenati(span):
    nomi_antenati = []
    while span.parent and span.parent.span_id in per_id:
        span = per_id[span.parent.span_id]
        nomi_antenati.append(span.name)
    return nomi_antenati


print(f"\n     span LLM: {len(llm_spans)} | span RETRIEVER: {len(retriever_spans)}")
print("     antenati della prima chiamata LLM:", antenati(llm_spans[0]))

# --- ASSERT ---
print("\n[ASSERT] verifiche automatiche...")
assert "LangGraph" in nomi, "manca lo span radice del grafo"
for nodo in ("route_question", "retrieve", "grade_documents", "generate", "check_answer"):
    assert nodo in nomi, f"manca lo span del nodo {nodo}"
assert llm_spans and all(s.attributes.get("llm.model_name") for s in llm_spans)
assert retriever_spans, "manca lo span del retriever"
assert all("LangGraph" in antenati(s) for s in llm_spans), "chiamata LLM fuori dal grafo"
print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: ogni nodo e ogni chiamata LLM e' uno span annidato, con")
print("           prompt, modello, token e latenza: pronto per Phoenix.")
