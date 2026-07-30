# Agentic RAG Copilot

A local-first, **self-corrective Agentic RAG** system over technical and maintenance
documentation, built with **LangChain / LangGraph**, **FastAPI**, **Chroma** and local
LLMs served by **Ollama**. No cloud, no API keys, no costs — everything runs on-device.

> **Status:** Phase 1 complete (linear RAG baseline, end-to-end and tested).
> Phase 2 (the LangGraph self-corrective agent) is in progress.

## Why this project

Built to demonstrate the core skills of an *Agentic AI / LLM Developer*: deep RAG,
agent orchestration with LangGraph, robustness and observability, and CI/CD with
automated evaluation.

The approach is deliberately incremental: first a **linear RAG baseline**, then the
**agentic layer** on top of it — so the "before/after" is visible and every design
decision can be explained.

### Why a linear RAG is not enough

The baseline already exposes its own limits, which is exactly the point. Asking
*"pump vibrating"* against the sample corpus does **not** retrieve the ideal
troubleshooting section in the top-k results — semantic similarity alone picks
neighbouring sections instead. That failure is the concrete motivation for the
agentic layer: grading retrieved documents and rewriting the query when they are
not relevant.

## Architecture

The target agent is a **CRAG / Self-RAG** style graph — conditional edges and loops,
not a straight line:

```
question
   |
   v
[route] -----> does this need retrieval?
   |
   v
[retrieve] --> fetch candidate chunks from the vector store
   |
   v
[grade] -----> are the chunks actually relevant?
   |  yes            | no
   |                 +--> [rewrite query] --> back to retrieve
   v
[generate] --> answer using the chunks only
   |
   v
[check] -----> is the answer grounded in the chunks?
   |  yes            | no
   |                 +--> regenerate, or admit "I don't know"
   v
final answer + source citations
```

## Tech stack

| Area | Tool |
|---|---|
| Agent orchestration | LangGraph + LangChain |
| LLM (local) | Ollama — `llama3.2:3b` |
| Embeddings (local) | Ollama — `nomic-embed-text` |
| Vector store | Chroma (persisted on disk) |
| Backend API | FastAPI |
| Config | pydantic-settings |
| Eval / Observability | ragas + Arize Phoenix *(planned)* |
| CI | GitHub Actions *(planned)* |

## Project layout

```
src/copilot/
├── config.py          # centralized settings (pydantic-settings, COPILOT_ prefix)
├── llm/
│   ├── provider.py    # get_llm()        — model-agnostic LLM factory
│   └── embeddings.py  # get_embeddings() — embedding model factory
└── rag/
    ├── ingestion.py   # load_documents(), split_documents(), ingest()
    ├── vectorstore.py # get_vectorstore() — Chroma connection
    ├── retriever.py   # get_retriever()   — vector store as a Runnable
    └── chain.py       # get_rag_chain(), answer() — the LCEL RAG chain

data/manuals/          # source corpus (versioned)
data/chroma/           # generated vector DB (not versioned — rebuild with ingest())
Test_Funzioni/         # one hands-on test script per function, bottom-up
info/                  # design docs and study notes (Italian)
```

## Quickstart

**Prerequisites:** [Ollama](https://ollama.com) running locally, Python 3.11.

```bash
# 1. pull the local models (~2.3 GB total)
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 2. install the project (editable, with dev extras)
pip install -e ".[dev]"

# 3. build the vector store from data/manuals/
python -c "from copilot.rag.ingestion import ingest; print(ingest())"

# 4. ask a question
python -c "from copilot.rag.chain import answer; print(answer('What grease should I use for the bearings?'))"
```

Configuration is optional — every setting has a default. To override, copy
`.env.example` to `.env`.

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| 1 | Linear RAG baseline (ingestion, retrieval, LCEL chain, citations) | ✅ done |
| 2 | LangGraph self-corrective agent (route / grade / generate / check) | 🔜 next |
| 3 | Guardrails, error handling, tracing with Phoenix | planned |
| 4 | Evaluation datasets (ragas + trajectory eval) and CI/CD | planned |
| 5 | Optional: external tools (CMMS), Docker/K8s, alternative agent frameworks | future |

## Documentation

Design documents live in [`info/`](info/) and are written in Italian:

- [`info/PROGETTO.md`](info/PROGETTO.md) — vision, RAG concepts, architecture, folder layout
- [`info/FASI.md`](info/FASI.md) — the five phases in detail
- [`info/TEORIA/`](info/TEORIA/) — study notes on LangChain internals (Runnables, LCEL, `.invoke()`)
