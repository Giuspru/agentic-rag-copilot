# Agentic RAG Copilot

![Agentic RAG Copilot — dal manuale tecnico ai chunk vettoriali, fino alla risposta citata, interamente in locale](img/01.png)

Sistema **RAG agentico self-corrective** su documentazione tecnica di manutenzione,
costruito con **LangChain / LangGraph**, **FastAPI**, **Chroma** e LLM locali serviti da
**Ollama**. Nessun cloud, nessuna API key, nessun costo: tutto gira sulla macchina.

> **Stato:** Fase 3 completata. L'agente self-corrective (LangGraph) valuta i documenti
> recuperati, riformula la ricerca e controlla le proprie risposte. È protetto da
> guardrail di input e di output, gestisce i guasti di Ollama con retry ed errori HTTP
> chiari, ed è osservabile passo per passo con Arize Phoenix. Prossima: valutazione
> automatica e CI/CD (Fase 4).

## Perché questo progetto

Nasce per dimostrare le competenze centrali di un ruolo *Agentic AI / LLM Developer*:
RAG approfondito, orchestrazione di agenti con LangGraph, robustezza e osservabilità,
CI/CD con valutazione automatica della qualità.

L'approccio è volutamente incrementale: prima una **baseline RAG lineare** (Fase 1), poi
lo **strato agentico** costruito sopra (Fase 2). Così il "prima/dopo" è visibile e ogni
scelta progettuale può essere spiegata e difesa.

### Perché un RAG lineare non basta

Un RAG lineare fa sempre gli stessi passi: recupera i k chunk più simili e li passa
tutti all'LLM, anche quando non c'entrano. Non sa accorgersi di aver cercato male, né di
aver risposto male. Due esempi reali sul corpus di esempio:

- **Rumore nel contesto.** Per *"pump vibrating"* il chunk giusto (§9.3 *Excessive
  vibration*) arriva primo, ma gli altri 3 su 4 sono rumore (panoramica, specifiche,
  allineamento) che finisce comunque nel prompt.
- **Domanda in italiano.** Per *"la pompa perde dalla tenuta, cosa faccio?"* la
  similarità semantica in italiano non trova la §9.5 *Seal leakage*. Il RAG lineare
  risponde sul problema sbagliato (§9.1, adescamento) e cita una fonte che non esiste:

```
[LINEAR]  Non conosco il problema specifico della pompa che non tiene la tenuta.
          Tuttavia, posso suggerirti di controllare se la pompa è stata primata
          correttamente (9.1 Pump does not deliver liquid) ...  Fonte: manual.md

[AGENT]   According to Section 9.5 Seal leakage, possible causes of seal leakage
          include: centering clips left in place after installation, worn seal faces,
          dry running, solids in the pumped liquid ...
          path: route -> retrieve -> grade 0/4 -> rewrite -> retrieve -> grade 1/4
                -> generate -> check: not_useful -> rewrite -> retrieve -> grade 2/4
                -> generate -> check: not_grounded -> generate (stricter) -> check: useful
```

## Architettura

L'agente è un grafo in stile **CRAG / Self-RAG**: archi condizionali e cicli, non una
linea retta. Il diagramma è generato dal grafo compilato
(`get_agent().get_graph().draw_mermaid()`); le frecce tratteggiate sono gli archi
condizionali.

```mermaid
graph TD;
	__start__([start]):::first
	route_question(route_question)
	respond_direct(respond_direct)
	retrieve(retrieve)
	grade_documents(grade_documents)
	rewrite_query(rewrite_query)
	generate(generate)
	check_answer(check_answer)
	fallback(fallback)
	__end__([end]):::last
	__start__ --> route_question;
	route_question -.-> respond_direct;
	route_question -.-> retrieve;
	retrieve --> grade_documents;
	grade_documents -.-> generate;
	grade_documents -.-> rewrite_query;
	grade_documents -.-> fallback;
	rewrite_query --> retrieve;
	generate --> check_answer;
	check_answer -.-> __end__;
	check_answer -.-> generate;
	check_answer -.-> rewrite_query;
	check_answer -.-> fallback;
	fallback --> __end__;
	respond_direct --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

| Nodo | Ruolo dell'LLM | Cosa fa |
|---|---|---|
| `route_question` | giudice | Saluto/chiacchiera → `respond_direct`; qualunque richiesta di informazioni → `retrieve` |
| `retrieve` | — | Recupera i k chunk più simili alla query corrente |
| `grade_documents` | giudice | Valuta ogni chunk (sì/no) e tiene solo quelli rilevanti (passo *corrective* di CRAG) |
| `rewrite_query` | scrittore | Nessun chunk rilevante → riformula la query in inglese tecnico (e traduce le domande in italiano) |
| `generate` | scrittore | Risponde usando **solo** i chunk rilevanti; al secondo tentativo con un prompt più severo |
| `check_answer` | controllo deterministico + 2 giudici | Self-RAG: numeri, codici e file citati sono nei chunk? La risposta è fondata? Risponde alla domanda? |
| `fallback` | — | Tentativi esauriti → "I don't know" fisso, senza improvvisare |

La differenza rispetto al RAG lineare sta tutta nelle frecce che **tornano indietro**:
l'agente valuta il proprio lavoro e si corregge. Ogni ciclo è limitato da un contatore
nello stato (`max_rewrites`, `max_generations`), quindi il grafo termina sempre.

Due scelte di design:

- **LLM come giudice con output strutturato.** I nodi giudice usano
  `with_structured_output`: il modello deve rispondere con un JSON conforme a uno schema
  Pydantic (`"yes"`/`"no"`), così il grafo ramifica su un valore affidabile invece di
  interpretare testo libero.
- **I giudici lavorano in inglese.** I chunk vengono valutati rispetto alla *query di
  ricerca* (in inglese), non alla domanda originale: il modello da 3B sbaglia soprattutto
  quando confronta testi in lingue diverse. Per una domanda in italiano il primo giro non
  trova nulla, la riscrittura fa da traduzione e il secondo giro valuta nella lingua dei
  manuali. I prompt dei giudici sono stati tarati su piccoli banchi di prova etichettati
  a mano (es. fondatezza: da 5/9 a 8/9 casi corretti, con 0 risposte inventate promosse).

## Robustezza e osservabilità (Fase 3)

### Guardrail

Due controlli **deterministici** (niente LLM: istantanei, gratuiti, mai sbagliati nel modo
in cui sbaglia un modello) affiancano i giudici LLM del grafo:

- **Input** (`guardrails/validators.py`), prima dell'agente: normalizza la domanda
  (Unicode NFKC, caratteri di controllo, spazi) e blocca i *prompt injection* evidenti
  ("ignore previous instructions", "rivela il tuo prompt", i token speciali di Llama 3
  come `<|eot_id|>`...), in inglese e in italiano. Una domanda bloccata non costa nessuna
  chiamata all'LLM. I pattern sono tarati per non bloccare domande legittime: in un
  dominio di manutenzione *"show me the instructions for seal replacement"* è una
  domanda normale. È una prima linea, non *la* difesa: la protezione vera è
  l'architettura (nessun tool, risposte solo dai chunk, controllo di fondatezza).
- **Output** (`guardrails/groundedness.py`), dentro `check_answer` e prima dei giudici
  LLM: ogni **numero**, **codice ricambio** e **file citato** nella risposta deve
  comparire nei chunk. Il giudice LLM è bravo con le parafrasi e scarso con
  l'esattezza; un confronto tra stringhe non accetta mai "5000 ore" se il manuale dice
  4000. `4000`, `4,000` e l'italiano `4.000` sono riconosciuti come lo stesso numero.

### La taratura: misurare, non indovinare

Su 10 domande nuove (tutte con risposta nel manuale) l'agente della Fase 2 rispondeva
correttamente solo a **6**: le altre 4 finivano in "non lo so", anche con risposte copiate
parola per parola dal manuale. Il banco di prova della Fase 2 (9 casi) era troppo piccolo
per accorgersene. Con un banco costruito sui contesti reali dell'agente (14 risposte
corrette verificate a mano + 14 inventate), il giudice di fondatezza:

| Variante | Corrette accettate | Inventate bocciate |
|---|---|---|
| output JSON vincolato (Fase 2) | 9/14 | 14/14 |
| JSON + citazioni rimosse | 7/14 | 14/14 |
| JSON + "copia la frase che lo prova" | 9/14 | 14/14 |
| **stesso prompt, risposta in testo libero "yes/no"** | **14/14** | **14/14** |

Sul modello da 3B la decodifica vincolata dallo schema JSON peggiorava il giudizio: il
giudice di fondatezza ora risponde in testo libero, interpretato in modo prudente (senza
un "yes" chiaro vale "no"). Nella stessa analisi è emerso che il modello copiava
l'esempio del prompt `(source: manual.md)` e citava un file inesistente in 5 risposte su
14. Senza esempio, invece, a volte dimenticava di citare. Con un formato segnaposto,
`(source: <file name>)` + "copia il nome esatto dal contesto": 15/15 risposte citate,
tutte con il nome giusto. Risultato sulle 10 domande: **da 6/10 a 10/10**, e
risposte più rapide (meno cicli di autocorrezione inutili).

### Gestione degli errori

La stessa causa produce eccezioni diverse a seconda della libreria (con Ollama spento il
modello di chat solleva `httpx.ConnectError`, quello di embedding `ConnectionError`).
`errors.py` le traduce in pochi **errori di dominio**, ognuno con il suo status HTTP:

| Situazione | Risposta |
|---|---|
| Ollama irraggiungibile o modello non scaricato | `503` `llm_unavailable` + header `Retry-After` |
| Ollama non risponde entro `COPILOT_LLM_TIMEOUT` | `504` `llm_timeout` |
| Domanda bloccata dal guardrail di input | `400` `input_rejected` |
| Domanda vuota, di soli spazi o troppo lunga | `422` (validazione Pydantic) |
| Bug imprevisto | `500` `internal_error`, senza dettagli interni (stack trace solo nel log) |

- **Retry per nodo** con il `RetryPolicy` di LangGraph (backoff esponenziale), solo per
  gli errori transitori. Non si riprovano i timeout (moltiplicherebbero l'attesa) né i
  modelli mancanti (nessun retry li scarica). È sicuro perché i nodi sono
  **idempotenti**: leggono lo stato e restituiscono un aggiornamento.
- **Limite di ricorsione calcolato dai budget** (`max_graph_steps()`: 22 passi nel caso
  peggiore). Il default di LangGraph 1.x è 10007: se un bug rompesse un budget, l'agente
  farebbe migliaia di chiamate LLM (ore) prima di fermarsi. Con il limite calcolato si
  ferma dopo pochi passi, in modo visibile.
- **`/health` vs `/ready`**: il primo dice che il processo è vivo; il secondo controlla
  che Ollama risponda con entrambi i modelli scaricati e che l'indice non sia vuoto, ed
  è quello che un orchestratore usa per decidere se mandare traffico.

### Tracing con Arize Phoenix

I `steps` dicono *quali* nodi sono girati; una traccia mostra tutto il resto: per ogni
nodo il prompt esatto, la risposta grezza del modello, i chunk recuperati, i token e la
latenza, annidati ad albero (grafo → nodo → catena → chiamata LLM). La strumentazione è
**OpenTelemetry + OpenInference**, che si aggancia ai callback di LangChain senza
toccare il codice dell'agente. Phoenix gira in locale, in linea con il resto del
progetto. Il tracing è disattivato di default: l'app non dipende da un server di
monitoraggio acceso.

Già il primo albero di span è istruttivo: su una domanda da ~10 s, `grade_documents` ne
occupa ~4 (quattro chiamate LLM in sequenza, una per chunk): è il primo candidato da
ottimizzare.

Ogni risposta HTTP porta anche `X-Request-ID` (generato o ripreso dal client) e
`X-Process-Time`; il request ID compare in ogni riga di log della richiesta.

### Limiti noti

L'agente è tarato per la **precisione**: preferisce dire "non lo so" piuttosto che
rispondere senza supporto. Il prezzo sono alcuni **falsi negativi** e alcuni percorsi
inutilmente lunghi. Esempio dalle 10 domande di taratura: per *"How often should the
mechanical seal be replaced?"* la risposta finale è giusta (8000 ore), ma ci arriva in
~30 s invece di ~10: il grader scarta il chunk giusto (§4) e la riscrittura aggiunge
"excessive vibration and leakage", parole prese dall'esempio nel prompt di riscrittura.
È lo stesso difetto della citazione `manual.md`: il modello da 3B copia gli esempi dei
prompt. Anche le domande che richiedono di collegare punti diversi del manuale (es.
*"How often should I check the coupling alignment?"*) restano difficili. Inoltre la
comprensione dell'italiano dipende dal modello locale da 3B. Misurare questi compromessi
in modo sistematico è il compito della Fase 4 (eval), che verificherà anche se gli altri
giudici traggono lo stesso vantaggio dal testo libero.

## Stack tecnologico

| Ambito | Strumento |
|---|---|
| Orchestrazione agente | LangGraph + LangChain |
| LLM (locale) | Ollama — `llama3.2:3b` |
| Embeddings (locale) | Ollama — `nomic-embed-text` |
| Vector store | Chroma (persistito su disco) |
| Backend API | FastAPI |
| Configurazione | pydantic-settings |
| Observability | Arize Phoenix (OpenTelemetry + OpenInference), logging con request ID |
| Eval | ragas + trajectory eval *(previsti)* |
| CI | GitHub Actions *(prevista)* |

## Struttura del progetto

```
src/copilot/
├── config.py          # configurazione centralizzata (pydantic-settings, prefisso COPILOT_)
├── errors.py          # errori di dominio (503/504/400) e traduzione delle eccezioni di Ollama
├── agent/
│   ├── state.py       # AgentState — la memoria di lavoro che scorre tra i nodi
│   ├── nodes.py       # i nodi: route, retrieve, grade, rewrite, generate, check, fallback
│   └── graph.py       # il grafo LangGraph (archi condizionali, retry, limite) + run_agent()
├── api/
│   ├── main.py        # l'app FastAPI: startup, middleware request ID, exception handler
│   ├── routes.py      # GET /health, GET /ready, POST /chat
│   └── schemas.py     # contratto request/response/errore (Pydantic)
├── guardrails/
│   ├── validators.py  # input: normalizzazione + filtro prompt injection
│   └── groundedness.py# output: numeri, codici e file citati devono essere nei chunk
├── llm/
│   ├── provider.py    # get_llm()        — factory dell'LLM, model-agnostic
│   └── embeddings.py  # get_embeddings() — factory del modello di embedding
├── observability/
│   ├── tracing.py     # setup_tracing() — OpenTelemetry/OpenInference verso Phoenix
│   ├── logs.py        # setup_logging() — log con request ID (ContextVar)
│   └── health.py      # controlli di readiness (Ollama + modelli, indice non vuoto)
└── rag/
    ├── ingestion.py   # load_documents(), split_documents(), ingest()
    ├── vectorstore.py # get_vectorstore() — connessione a Chroma
    ├── retriever.py   # get_retriever()   — il vector store come Runnable
    └── chain.py       # get_rag_chain(), answer() — la baseline lineare (LCEL)

data/manuals/          # corpus sorgente (versionato)
data/chroma/           # vector DB generato (non versionato: si rigenera con scripts/ingest.py)
scripts/ingest.py      # comando per (ri)popolare il vector store
scripts/compare.py     # prima/dopo: RAG lineare vs agente, fianco a fianco
Test_Funzioni/         # uno script di test per funzione, in ordine bottom-up (01–20)
info/                  # documenti di design e note di studio
img/                   # immagini del progetto
```

Il criterio è la **separazione delle responsabilità**: `llm/` conosce solo i modelli,
`rag/` solo il recupero della conoscenza, `agent/` solo il controllo di flusso, `api/`
solo il protocollo HTTP, `guardrails/` e `observability/` sono trasversali,
`config.py` è l'unico posto dove vivono i parametri. Ogni modulo
è testabile da solo. Nel passaggio dal RAG lineare all'agente, nello strato API è
cambiata una sola riga.

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
{
  "answer": "According to Section 5 of the manual, you should use a lithium-complex grease with NLGI grade 2 ...",
  "sources": ["centrifugal-pump-maintenance-manual.md"],
  "steps": [
    "route -> retrieve",
    "retrieve -> 4 chunks for 'What grease should I use for the bearings?'",
    "grade -> 2/4 relevant",
    "generate (attempt 1)",
    "check -> useful"
  ]
}
```

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/health` | GET | Liveness check → `{"status": "ok"}` |
| `/ready` | GET | Readiness check: `200` se Ollama (con i modelli) e l'indice sono pronti, altrimenti `503` con il dettaglio di ogni controllo |
| `/chat` | POST | `{"question": "..."}` → `answer` fondata sui manuali e citata, `sources` (i manuali usati) e `steps` (il percorso dell'agente nel grafo) |
| `/docs` | GET | Documentazione interattiva (Swagger), generata automaticamente |

Le domande vuote, mancanti o più lunghe di 2000 caratteri vengono rifiutate con `422`,
quelle bloccate dal guardrail con `400`, in entrambi i casi prima di raggiungere l'LLM.
Gli errori hanno sempre la stessa forma: `{"error": "llm_unavailable", "detail": "..."}`.

### Tracing (opzionale)

Il server Phoenix va in un ambiente separato: porta con sé molte dipendenze (pandas,
SQLAlchemy, ...) che l'app non deve avere, ed è così anche in produzione, dove il
collector delle tracce è un servizio a parte. Nell'ambiente dell'app serve solo il lato
client (`.[tracing]`: OpenTelemetry + OpenInference).

```bash
# ambiente separato per il server Phoenix
conda create -n phoenix python=3.11 && conda activate phoenix
pip install arize-phoenix "cryptography<46"   # vedi nota sotto
phoenix serve                       # UI su http://localhost:6006

# nell'ambiente del progetto
pip install -e ".[tracing]"
COPILOT_TRACING_ENABLED=true uvicorn copilot.api.main:app
```

Ogni domanda a `/chat` diventa una traccia nel progetto `agentic-rag-copilot` di Phoenix.

> Nota (conda su macOS): `cryptography` ≥ 46 cerca a runtime simboli di OpenSSL 3.5,
> mentre il Python di conda carica la sua OpenSSL 3.0 (errore `symbol not found in flat
> namespace '_OSSL_get_max_threads'`). Il vincolo `cryptography<46` lo evita.

Per vedere il prima/dopo, RAG lineare e agente sulle stesse domande:

```bash
python scripts/compare.py                        # domande dimostrative
python scripts/compare.py "la pompa vibra molto"  # domande tue
```

La configurazione è opzionale: ogni parametro ha un default sensato (compresi i budget
di autocorrezione `COPILOT_MAX_REWRITES` e `COPILOT_MAX_GENERATIONS`, il timeout
`COPILOT_LLM_TIMEOUT` e i tentativi `COPILOT_LLM_MAX_ATTEMPTS`). Per modificarli,
copia `.env.example` in `.env`.

> Il corpus e i prompt sono in **inglese** per una ragione tecnica: i test sugli
> embedding hanno mostrato una separazione semantica molto più netta in inglese che in
> italiano con `nomic-embed-text`, e un retrieval debole avrebbe compromesso tutto il
> resto della catena. Le domande in italiano sono supportate grazie alla riscrittura
> della query.

## Roadmap

| Fase | Focus | Stato |
|---|---|---|
| 1 | Baseline RAG lineare (ingestion, retrieval, chain LCEL, citazioni, API FastAPI) | ✅ completata |
| 2 | Agente self-corrective con LangGraph (route / grade / rewrite / generate / check) | ✅ completata |
| 3 | Guardrail di input/output, gestione errori (retry, 503/504), readiness, tracing con Phoenix | ✅ completata |
| 4 | Dataset di valutazione (ragas + trajectory eval) e CI/CD | 🔜 prossima |
| 5 | Opzionali: tool verso sistemi esterni (CMMS), Docker/K8s, framework alternativi | futura |

## Documentazione

I documenti di progettazione si trovano in [`info/`](info/):

- [`info/PROGETTO.md`](info/PROGETTO.md) — visione, concetti RAG, architettura, struttura
- [`info/FASI.md`](info/FASI.md) — le cinque fasi nel dettaglio
- [`info/TEORIA/`](info/TEORIA/) — note di studio sugli interni di LangChain (Runnable, LCEL, `.invoke()`)
