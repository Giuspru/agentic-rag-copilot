# Agentic RAG Copilot

![Agentic RAG Copilot — dal manuale tecnico ai chunk vettoriali, fino alla risposta citata, interamente in locale](img/01.png)

Sistema **RAG agentico self-corrective** su documentazione tecnica di manutenzione,
costruito con **LangChain / LangGraph**, **FastAPI**, **Chroma** e LLM locali serviti da
**Ollama**. Nessun cloud, nessuna API key, nessun costo: tutto gira sulla macchina.

> **Stato:** Fase 1 completata (baseline RAG lineare esposta via API FastAPI, funzionante
> end-to-end e testata). Fase 2 (agente self-corrective con LangGraph) in corso.

## Perché questo progetto

Nasce per dimostrare le competenze centrali di un ruolo *Agentic AI / LLM Developer*:
RAG approfondito, orchestrazione di agenti con LangGraph, robustezza e osservabilità,
CI/CD con valutazione automatica della qualità.

L'approccio è volutamente incrementale: prima una **baseline RAG lineare**, poi lo
**strato agentico** costruito sopra. Così il "prima/dopo" è visibile e ogni scelta
progettuale può essere spiegata e difesa.

### Perché un RAG lineare non basta

La baseline mette già in luce i propri limiti, ed è esattamente questo il punto. La
domanda *"pump vibrating"* sul corpus di esempio **non** recupera la sezione di
troubleshooting ideale tra i primi k risultati: la sola similarità semantica pesca
sezioni vicine ma meno pertinenti. Quel fallimento è la motivazione concreta dello
strato agentico — valutare i documenti recuperati (*grading*) e riformulare la query
quando non sono rilevanti.

## Architettura

L'agente finale è un grafo in stile **CRAG / Self-RAG**: archi condizionali e cicli,
non una linea retta.

```
domanda
   |
   v
[route] ------> serve davvero il retrieval?
   |
   v
[retrieve] ---> recupera i chunk candidati dal vector store
   |
   v
[grade] ------> i chunk sono davvero rilevanti?
   |  sì              | no
   |                  +--> [riscrivi query] --> torna a retrieve
   v
[generate] ---> genera la risposta usando SOLO i chunk
   |
   v
[check] ------> la risposta è fondata sui chunk?
   |  sì              | no
   |                  +--> rigenera, oppure ammette "non lo so"
   v
risposta finale + citazione delle fonti
```

La differenza rispetto a un RAG classico sta tutta nelle frecce che **tornano
indietro**: l'agente non si limita a rispondere, ma valuta il proprio lavoro e si
corregge.

## Stack tecnologico

| Ambito | Strumento |
|---|---|
| Orchestrazione agente | LangGraph + LangChain |
| LLM (locale) | Ollama — `llama3.2:3b` |
| Embeddings (locale) | Ollama — `nomic-embed-text` |
| Vector store | Chroma (persistito su disco) |
| Backend API | FastAPI |
| Configurazione | pydantic-settings |
| Eval / Observability | ragas + Arize Phoenix *(previsti)* |
| CI | GitHub Actions *(prevista)* |

## Struttura del progetto

```
src/copilot/
├── config.py          # configurazione centralizzata (pydantic-settings, prefisso COPILOT_)
├── api/
│   ├── main.py        # l'app FastAPI (entry point per uvicorn)
│   ├── routes.py      # GET /health, POST /chat
│   └── schemas.py     # contratto request/response (Pydantic)
├── llm/
│   ├── provider.py    # get_llm()        — factory dell'LLM, model-agnostic
│   └── embeddings.py  # get_embeddings() — factory del modello di embedding
└── rag/
    ├── ingestion.py   # load_documents(), split_documents(), ingest()
    ├── vectorstore.py # get_vectorstore() — connessione a Chroma
    ├── retriever.py   # get_retriever()   — il vector store come Runnable
    └── chain.py       # get_rag_chain(), answer() — la chain LCEL

data/manuals/          # corpus sorgente (versionato)
data/chroma/           # vector DB generato (non versionato: si rigenera con scripts/ingest.py)
scripts/ingest.py      # comando per (ri)popolare il vector store
Test_Funzioni/         # uno script di test per funzione, in ordine bottom-up
info/                  # documenti di design e note di studio
img/                   # immagini del progetto
```

Il criterio è la **separazione delle responsabilità**: `llm/` conosce solo i modelli,
`rag/` solo il recupero della conoscenza, `api/` solo il protocollo HTTP, `config.py` è
l'unico posto dove vivono i parametri. Ogni modulo è testabile da solo.

## Avvio rapido

**Prerequisiti:** [Ollama](https://ollama.com) attivo in locale, Python 3.11.

```bash
# 1. scarica i modelli locali (~2,3 GB in totale)
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 2. installa il progetto (editable, con le dipendenze di sviluppo)
pip install -e ".[dev]"

# 3. costruisci il vector store a partire da data/manuals/
python scripts/ingest.py

# 4. avvia l'API
uvicorn copilot.api.main:app --reload
```

Poi apri <http://localhost:8000/docs> per provare l'API dal browser, oppure:

```bash
curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What grease should I use for the bearings?"}'
```

```json
{"answer": "According to Section 5 of the manual (source: centrifugal-pump-maintenance-manual.md), you should use a lithium-complex grease with NLGI grade 2 ..."}
```

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/health` | GET | Liveness check → `{"status": "ok"}` |
| `/chat` | POST | `{"question": "..."}` → `{"answer": "..."}`, risposta fondata sui manuali e citata |
| `/docs` | GET | Documentazione interattiva (Swagger), generata automaticamente |

Le domande vuote, mancanti o più lunghe di 2000 caratteri vengono rifiutate con `422`
prima di raggiungere l'LLM.

La configurazione è opzionale: ogni parametro ha un default sensato. Per modificarli,
copia `.env.example` in `.env`.

> Il corpus e i prompt sono in **inglese** per una ragione tecnica: i test sugli
> embedding hanno mostrato una separazione semantica molto più netta in inglese che in
> italiano con `nomic-embed-text`, e un retrieval debole avrebbe compromesso tutto il
> resto della catena.

## Roadmap

| Fase | Focus | Stato |
|---|---|---|
| 1 | Baseline RAG lineare (ingestion, retrieval, chain LCEL, citazioni, API FastAPI) | ✅ completata |
| 2 | Agente self-corrective con LangGraph (route / grade / generate / check) | 🔜 prossima |
| 3 | Guardrail, gestione errori, tracing con Phoenix | prevista |
| 4 | Dataset di valutazione (ragas + trajectory eval) e CI/CD | prevista |
| 5 | Opzionali: tool verso sistemi esterni (CMMS), Docker/K8s, framework alternativi | futura |

## Documentazione

I documenti di progettazione si trovano in [`info/`](info/):

- [`info/PROGETTO.md`](info/PROGETTO.md) — visione, concetti RAG, architettura, struttura
- [`info/FASI.md`](info/FASI.md) — le cinque fasi nel dettaglio
- [`info/TEORIA/`](info/TEORIA/) — note di studio sugli interni di LangChain (Runnable, LCEL, `.invoke()`)
