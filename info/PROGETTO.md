# Agentic RAG Copilot — Documento di progetto

> Documento di riepilogo e spiegazione. Serve a fissare cosa abbiamo deciso e a
> spiegare **come funzionerà** il progetto, così puoi verificare di aver capito.
> Ultimo aggiornamento: 2026-07-20.

---

## 1. Perché facciamo questo progetto (l'obiettivo)

L'obiettivo **non** è costruire un prodotto da vendere. È costruire un progetto
**da portfolio / CV** che dimostri, in modo credibile, le competenze richieste per
ruoli come *Artificial Intelligence Engineer / Agentic AI Developer / LLM Developer*.

Siamo partiti da un'offerta di lavoro reale (`CapGemini.md`, ruolo *Agentic AI / LLM
Developer*) e abbiamo estratto i **requisiti tecnici core**. Il progetto è progettato
apposta per coprirli in un unico sistema coeso, invece che con tante demo scollegate.

Un solo progetto **verticale e profondo** vale, su un CV, molto più di cinque tutorial.

---

## 2. Cosa abbiamo deciso (le scelte chiave)

| Tema | Decisione | Motivo |
|---|---|---|
| **Tipo di progetto** | **Agentic RAG self-corrective** | Massima densità di "craft AI" con zero sistemi esterni |
| **Dominio / corpus** | Documenti **tecnici / di manutenzione** | Aggancio all'offerta Capgemini, gratis, senza costruire nulla |
| **LLM** | **Solo locale, via Ollama** | Zero costi, dati che non escono dal Mac |
| **Codice** | **Model-agnostic** (`get_llm()`) | Pulizia architetturale + cambio modello locale facile |
| **CMMS / integrazione sistemi esterni** | **Rimandato** (fase futura) | Aggiungeva complessità di dominio, non competenze AI |
| **Docker / Kubernetes / Cloud deploy** | **Rimandato** (fase futura) | Non è il focus iniziale; possibile estensione finale |

### Perché abbiamo scartato il CMMS
All'inizio l'idea era un chatbot per manutenzione **integrato con un CMMS** (il gestionale
della manutenzione). Ci siamo accorti che costruire quel finto sistema (database, endpoint,
ordini di lavoro) aggiungeva **complessità di dominio** senza dimostrare **nuove competenze
AI**. Il rischio era far sembrare il progetto "un gestionale con sopra un chatbot" invece di
"un sistema AI sofisticato". Quindi l'abbiamo rimandato. Resta come *roadmap* — anzi, citarlo
come estensione futura dimostra visione architetturale.

---

## 3. Concetti di base (per capire il resto)

### Cos'è un LLM
Un *Large Language Model* è un modello che, dato del testo, predice il testo successivo.
Sa "ragionare" sul linguaggio ma **non conosce i tuoi documenti** e può **inventare**
risposte plausibili ma false (**allucinazioni**).

### Cos'è il RAG (Retrieval-Augmented Generation)
È la tecnica per far rispondere l'LLM **sui tuoi documenti** invece che sulla sua memoria
generica. Il flusso base:

1. **Ingestion** (fatta una volta): prendi i documenti, li spezzi in pezzi piccoli
   (**chunk**), li converti in vettori numerici (**embeddings**) e li salvi in un
   **vector database**.
2. **Retrieval** (a ogni domanda): converti la domanda in vettore, trovi i chunk più
   *simili* nel database.
3. **Generation**: dai quei chunk all'LLM come contesto e gli chiedi di rispondere
   **basandosi solo su quelli**, citando le fonti.

Così l'LLM risponde con informazioni reali e verificabili, non inventate.

### Termini che useremo
- **Chunk**: un pezzo di documento (es. un paragrafo).
- **Embedding**: rappresentazione numerica (vettore) del significato di un testo.
- **Vector DB**: database che trova testi per *similarità di significato*, non per parole esatte.
- **Retriever**: il componente che, data una domanda, restituisce i chunk rilevanti.
- **Groundedness**: quanto una risposta è *fondata* sui documenti recuperati (vs. inventata).
- **Agente**: un LLM che non risponde e basta, ma **decide** cosa fare a più passi
  (es. "questi documenti non bastano, riprovo con una query diversa").

---

## 4. Il cuore del progetto: RAG "agentico" (self-corrective)

Il RAG base ha un difetto: è **lineare e cieco**. Recupera dei chunk e risponde, anche se
i chunk erano irrilevanti o se la risposta è inventata. Un RAG *ingenuo* non se ne accorge.

Il nostro progetto costruisce un RAG **agentico**: invece di un flusso dritto, l'agente
**valuta e si corregge** a più passi. Questo è il pattern noto come **CRAG / Self-RAG**,
ed è uno degli usi più apprezzati di **LangGraph**.

### Il grafo dell'agente (come "ragiona")

```
        ┌───────────────────────────────────────────────┐
        ▼                                               │ (query riscritta)
  Domanda utente                                         │
        │                                               │
        ▼                                               │
   [1. ROUTE]  ── serve davvero cercare nei documenti? ──┤
        │ sì                                            │
        ▼                                               │
   [2. RETRIEVE]  recupera i chunk dal vector DB         │
        │                                               │
        ▼                                               │
   [3. GRADE]  i chunk sono rilevanti per la domanda?    │
        │            │                                  │
        │ sì         └── no ──> [RISCRIVI QUERY] ────────┘
        ▼
   [4. GENERATE]  genera la risposta usando i chunk
        │
        ▼
   [5. CHECK]  la risposta è fondata sui chunk? (no allucinazioni)
        │            │
        │ sì         └── no ──> torna a GENERATE (o segnala "non lo so")
        ▼
   Risposta finale + citazioni delle fonti
```

### Cosa fa ogni nodo
1. **Route** — decide se la domanda richiede il retrieval o se è banale/fuori tema.
2. **Retrieve** — interroga il vector DB e ottiene i chunk candidati (+ eventuale *re-ranking*
   per ordinarli meglio).
3. **Grade documents** — un controllo (fatto dall'LLM) che verifica se i chunk sono
   *davvero* pertinenti. Se no → riscrive la domanda e riprova. Questo è il "self".
4. **Generate** — produce la risposta basandosi **solo** sui chunk.
5. **Check groundedness** — verifica che la risposta sia supportata dai chunk (anti-allucinazione).
   Se non lo è → rigenera, o ammette "non ho abbastanza informazioni".

Il punto chiave: gli **archi condizionali** e i **cicli** (le frecce che tornano indietro).
Un RAG normale è una linea retta; qui l'agente **decide, valuta e si corregge**. È
esattamente ciò che dimostra "conoscenza approfondita del RAG" e "agenti che orchestrano
task complessi".

---

## 5. Lo stack tecnologico (tutto locale e gratuito)

| Componente | Scelta | A cosa serve |
|---|---|---|
| Linguaggio + gestione pacchetti | Python 3.11 + **uv** | Base del progetto |
| Backend API | **FastAPI** | Espone l'endpoint `/chat` a cui fai le domande |
| Orchestrazione agente | **LangGraph** (+ LangChain) | Costruisce il grafo con nodi, cicli, decisioni |
| LLM | **Ollama** (Qwen 2.5 o Llama 3.2) | Il "cervello" che ragiona, gira in locale |
| Embeddings | **nomic-embed-text** (via Ollama) | Trasforma testi in vettori |
| Vector DB | **Chroma** | Salva/cerca i chunk per similarità |
| Observability | **Arize Phoenix** (locale) | Vede/traccia cosa fa l'agente passo-passo |
| Valutazione | **ragas** + trajectory eval | Misura la qualità di RAG e agente |
| Guardrail | Pydantic + controlli I/O | Robustezza, validazione |
| Test / CI | pytest + **GitHub Actions** | Test automatici e CI/CD |

> Nota: "LLM locale" (Ollama) riguarda il *cervello* dell'agente. È cosa diversa dal
> "deploy cloud" (che abbiamo rimandato). Nessun costo, nessun dato che esce dal Mac.

---

## 6. Struttura delle cartelle (dettagliata)

Struttura completa prevista, con i file principali. La logica è: **codice dentro `src/`,
dati fuori, valutazione e test separati**. Ogni cartella ha una sola responsabilità.

```
progettoAI/
│
├── src/                          # tutto il codice dell'applicazione
│   └── copilot/                  # il package Python principale
│       ├── __init__.py
│       │
│       ├── api/                  # === STRATO WEB (FastAPI) ===
│       │   ├── __init__.py
│       │   ├── main.py           # crea l'app FastAPI, avvia il server
│       │   ├── routes.py         # endpoint: POST /chat, GET /health
│       │   └── schemas.py        # modelli Pydantic di request/response
│       │
│       ├── agent/                # === CERVELLO AGENTICO (LangGraph) ===
│       │   ├── __init__.py
│       │   ├── graph.py          # definizione del grafo: nodi + archi condizionali
│       │   ├── state.py          # lo "stato" che scorre tra i nodi (domanda, chunk, risposta...)
│       │   └── nodes.py          # implementazione dei nodi: route, grade, generate, check
│       │
│       ├── rag/                  # === RECUPERO CONOSCENZA (RAG) ===
│       │   ├── __init__.py
│       │   ├── ingestion.py      # carica documenti, li spezza in chunk, calcola embeddings
│       │   ├── chunking.py       # strategia di suddivisione dei documenti
│       │   ├── vectorstore.py    # interfaccia con Chroma (salva/cerca vettori)
│       │   └── retriever.py      # data una domanda, restituisce i chunk rilevanti (+ re-ranking)
│       │
│       ├── llm/                  # === ASTRAZIONE MODELLO ===
│       │   ├── __init__.py
│       │   ├── provider.py       # get_llm() — restituisce l'LLM (Ollama), config-driven
│       │   └── embeddings.py     # get_embeddings() — modello di embedding (nomic-embed-text)
│       │
│       ├── guardrails/           # === ROBUSTEZZA / SICUREZZA ===
│       │   ├── __init__.py
│       │   ├── groundedness.py   # verifica che la risposta sia fondata sui chunk (anti-allucinazione)
│       │   └── validators.py     # validazioni input/output
│       │
│       ├── observability/        # === MONITORAGGIO ===
│       │   ├── __init__.py
│       │   └── tracing.py        # setup di Arize Phoenix (tracciamento passo-passo)
│       │
│       └── config.py             # configurazione centralizzata (pydantic-settings legge .env)
│
├── data/                         # === DATI (fuori dal codice) ===
│   ├── manuals/                  # il corpus: documenti tecnici / manutenzione (PDF, txt, md)
│   └── chroma/                   # database vettoriale persistito su disco (generato, non versionato)
│
├── eval/                         # === VALUTAZIONE DELLA QUALITÀ ===
│   ├── datasets/                 # domande + risposte attese (il "compito in classe" del sistema)
│   ├── run_eval.py               # esegue la valutazione: ragas (RAG) + trajectory eval (agente)
│   └── metrics.py                # metriche personalizzate
│
├── tests/                        # === TEST AUTOMATICI (pytest) ===
│   ├── test_rag.py               # test su chunking, retrieval
│   ├── test_agent.py             # test sul grafo (con LLM mockato)
│   └── test_api.py               # test sugli endpoint
│
├── scripts/                      # === UTILITÀ DA RIGA DI COMANDO ===
│   └── ingest.py                 # comando per (ri)popolare il vector DB dai documenti
│
├── notebooks/                    # esplorazione/prototipazione (opzionale)
│
├── .github/
│   └── workflows/
│       └── ci.yml                # pipeline CI/CD: lint (ruff) + type-check (mypy) + test + eval
│
├── .env.example                  # esempio di configurazione (senza segreti)
├── .gitignore                    # esclude data/chroma, .env, cache...
├── pyproject.toml                # dipendenze e configurazione progetto (uv)
├── README.md                     # in inglese, curato — la vetrina del progetto
└── PROGETTO.md                   # questo documento
```

### Perché questa organizzazione
- **`src/copilot/` diviso per responsabilità**: `api` (web) / `agent` (orchestrazione) /
  `rag` (recupero) / `llm` (modelli) / `guardrails` (robustezza) / `observability`
  (monitoraggio). Ognuno è indipendente e testabile da solo. È il segnale di "codice
  scalabile e manutenibile" che l'offerta chiede.
- **`data/` separata dal codice**: i documenti e il DB vettoriale non sono codice, quindi
  stanno fuori da `src/`. `data/chroma/` è *generato* e non va versionato.
- **`eval/` e `tests/` separati**: i **test** verificano che il codice *funzioni* (logica);
  l'**eval** misura quanto le *risposte AI* sono *buone* (qualità). Sono due cose diverse
  e vanno tenute distinte — distinzione che in un colloquio fa capire che hai capito il
  problema della qualità nei sistemi LLM.
- **`scripts/`**: comandi operativi (es. "ripopola il database") separati dalla logica.

---

## 7. La roadmap (fasi incrementali)

Ogni fase produce qualcosa di **mostrabile** e si costruisce sulla precedente.

1. **Setup + RAG lineare (baseline)** — struttura repo, ingestion dei documenti, Chroma,
   retriever, endpoint `/chat` che risponde con citazioni. *Serve come "prima" del confronto.*
2. **Agente LangGraph (self-corrective)** — trasformiamo il RAG lineare nel grafo con
   route / grade / check. Qui nasce la parte "agentica".
3. **Robustezza + Observability** — guardrail, gestione errori/retry, tracing con Phoenix.
4. **Eval + CI/CD** — dataset di valutazione, ragas + trajectory eval, GitHub Actions.
5. **(Opzionale, futuro)** — tool verso sistemi esterni (es. **CMMS**), o Docker/Kubernetes,
   o un framework agentico alternativo (CrewAI/AutoGen) per confronto.

Approccio didattico: partiamo dal **baseline lineare** e poi lo evolviamo, così vedi il
"prima/dopo" e capisci ogni pezzo (utile anche per difenderlo a un colloquio).

---

## 8. Come il progetto copre i requisiti dell'offerta

| Requisito Capgemini | Dove lo dimostriamo |
|---|---|
| LangChain / LangGraph | Tutto il grafo dell'agente (fase 2) |
| AI Agents che orchestrano task complessi | Route / grade / check / cicli condizionali |
| RAG (conoscenza approfondita) | Ingestion, chunking, retrieval, re-ranking, grading |
| Python backend production-oriented | FastAPI, codice modulare e tipizzato |
| Robustezza, scalabilità, monitoring | Guardrail, retry, tracing con Phoenix |
| CI/CD per AI/ML, test automatici | GitHub Actions con test + eval automatica |
| Dominio industriale (plus) | Corpus tecnico/manutenzione |
| Framework alternativi (plus) | Fase 5 opzionale (CrewAI/AutoGen) |
| Integrazione sistemi complessi (plus) | Fase 5 opzionale (CMMS) |
| Cloud / Docker / K8s | Fase futura opzionale |

Le competenze non tecniche (inglese, comunicazione) le curi con un **README in inglese**
eccellente e con questa documentazione: saper spiegare un'architettura *è* comunicazione.

---

## 9. Verifica la tua comprensione

Se sai rispondere a queste domande, hai capito il progetto. (Non guardare le sezioni sopra!)

1. Qual è la differenza tra un RAG **lineare** e un RAG **agentico**?
2. A cosa serve il nodo **"grade documents"** e cosa succede se i chunk non sono rilevanti?
3. Cos'è un **embedding** e perché serve un **vector DB** invece di una ricerca per parole?
4. Cos'è la **groundedness** e quale problema degli LLM previene?
5. Perché abbiamo **scartato il CMMS** dallo scope iniziale?
6. Perché teniamo il codice **model-agnostic** anche se usiamo solo Ollama?
7. Cosa vuol dire che il grafo LangGraph ha **archi condizionali e cicli**, e perché è importante?

Se su qualcuna hai dubbi, segnamela: è lì che dobbiamo chiarire prima di scrivere codice.

---

## 10. Cosa manca ancora da decidere

- **Quali documenti** concreti mettere nel corpus (manuali PDF specifici, o li reperiamo insieme).
- **Quale modello Ollama** provare per primo (compromesso qualità/velocità sul tuo M1 8GB).
```
