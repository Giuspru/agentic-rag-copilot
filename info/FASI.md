# Fasi del progetto — Agentic RAG Copilot

> Suddivisione del progetto in fasi. Ogni fase produce un **risultato concreto e
> mostrabile** e si costruisce sulla precedente.
> Ultimo aggiornamento: 2026-07-20.

**Logica di fondo:** partiamo da un RAG lineare (baseline, fase 1) e lo evolviamo in
agentico (fase 2), così si vede il "prima/dopo" e si capisce ogni pezzo — utile anche
per saperlo difendere a un colloquio.

---

## Fase 1 — Setup + RAG lineare (baseline)

La base solida su cui tutto si appoggia.

- Struttura del repo, dipendenze (uv), configurazione
- Ingestion dei documenti → chunking → embeddings → Chroma
- Retriever + endpoint `/chat` (FastAPI)

**Risultato:** puoi fare una domanda e ottenere una risposta con citazioni. È il "prima"
del confronto — un RAG dritto, senza intelligenza agentica.

---

## Fase 2 — Agente LangGraph (self-corrective)

Qui nasce la parte davvero agentica: il cuore del progetto.

- Trasformiamo il flusso lineare nel **grafo** con nodi *route → retrieve → grade →
  generate → check*
- Archi condizionali e cicli (riscrittura query, retry anti-allucinazione)

**Risultato:** l'agente valuta i documenti, si corregge e rifiuta di inventare. Il "dopo"
del confronto.

---

## Fase 3 — Robustezza + Observability

Da prototipo a qualcosa che sembra "production-oriented".

- Guardrail (groundedness, validazioni input/output)
- Gestione errori, retry, timeout
- Tracing con **Arize Phoenix** (vedi passo-passo cosa fa l'agente)

**Risultato:** il sistema è robusto e osservabile — copre il requisito "monitoring".

---

## Fase 4 — Eval + CI/CD

La parte che più impressiona un recruiter tecnico.

- Dataset di valutazione (domande + risposte attese)
- **ragas** (qualità RAG) + trajectory eval (qualità agente)
- **GitHub Actions**: lint + type-check + test + eval automatica

**Risultato:** ogni modifica viene testata e valutata automaticamente — copre "CI/CD per AI/ML".

---

## Fase 5 — Estensioni (opzionale, futuro)

Le cose rimandate, da aggiungere se ci sarà tempo/voglia.

- Tool verso sistemi esterni (es. **CMMS**)
- **Docker / Kubernetes / cloud deploy**
- Framework agentico alternativo (**CrewAI / AutoGen**) per confronto

**Risultato:** estensioni che aumentano copertura dei requisiti e mostrano visione
architetturale.

---

## Riepilogo

| Fase | Focus | Risultato mostrabile |
|---|---|---|
| 1 | RAG lineare (baseline) | Q&A con citazioni |
| 2 | Agente LangGraph | RAG self-corrective |
| 3 | Robustezza + Observability | Sistema robusto e tracciato |
| 4 | Eval + CI/CD | Valutazione e test automatici |
| 5 | Estensioni (opzionale) | CMMS / Docker-K8s / framework alternativi |
