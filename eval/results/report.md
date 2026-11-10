# Evaluation report — 2026-11-10T02:23:01

Version 0.4.0 · LLM `llama3.2:3b` · embeddings `nomic-embed-text` · k=4 · 72 questions

## Holdout split (unseen until the final evaluation) — 15 questions

| Metric | linear | agent |
|---|---|---|
| Accuracy (all expected behaviors) | 100% | 93% |
| Answerable: correct answer | 100% | 91% |
| Answerable: fact recall | 100% | 91% |
| Answerable: wrongly refused (safe error) | 0 | 1 |
| Confidently wrong answers (dangerous error) | 0 | 0 |
| Unanswerable: correctly refused | 100% | 100% |
| Small talk: direct route | — | 100% |
| Answers with invented numbers/codes | 0% | 0% |
| Correct sources | 100% | 91% |
| Agent: correct final state | — | 93% |
| Agent: rewrites per question | — | 0.27 |
| Agent: LLM calls per question | — | 8.53 |
| Latency per question (s) | 3.48 | 9.85 |
| Retrieval: answer chunk in top-k | 100% | 100% |
| Retrieval: MRR | 0.70 | 0.70 |

## Test split (held out from tuning; checked by the CI gate) — 34 questions

| Metric | linear | agent |
|---|---|---|
| Accuracy (all expected behaviors) | 94% | 85% |
| Answerable: correct answer | 92% | 79% |
| Answerable: fact recall | 92% | 79% |
| Answerable: wrongly refused (safe error) | 2 | 5 |
| Confidently wrong answers (dangerous error) | 0 | 0 |
| Unanswerable: correctly refused | 100% | 100% |
| Small talk: direct route | — | 100% |
| Answers with invented numbers/codes | 3% | 0% |
| Correct sources | 92% | 79% |
| Agent: correct final state | — | 85% |
| Agent: rewrites per question | — | 0.62 |
| Agent: LLM calls per question | — | 10.03 |
| Latency per question (s) | 3.65 | 11.45 |
| Retrieval: answer chunk in top-k | 92% | 92% |
| Retrieval: MRR | 0.77 | 0.77 |

## Dev split (used to tune the prompts: optimistic) — 23 questions

| Metric | linear | agent |
|---|---|---|
| Accuracy (all expected behaviors) | 87% | 87% |
| Answerable: correct answer | 86% | 86% |
| Answerable: fact recall | 86% | 86% |
| Answerable: wrongly refused (safe error) | 2 | 3 |
| Confidently wrong answers (dangerous error) | 1 | 0 |
| Unanswerable: correctly refused | 100% | 100% |
| Answers with invented numbers/codes | 0% | 0% |
| Correct sources | 90% | 86% |
| Agent: correct final state | — | 87% |
| Agent: rewrites per question | — | 0.43 |
| Agent: LLM calls per question | — | 9.87 |
| Latency per question (s) | 3.68 | 12.05 |
| Retrieval: answer chunk in top-k | 90% | 90% |
| Retrieval: MRR | 0.78 | 0.78 |

## All questions — 72 questions

| Metric | linear | agent |
|---|---|---|
| Accuracy (all expected behaviors) | 93% | 88% |
| Answerable: correct answer | 91% | 84% |
| Answerable: fact recall | 91% | 84% |
| Answerable: wrongly refused (safe error) | 4 | 9 |
| Confidently wrong answers (dangerous error) | 1 | 0 |
| Unanswerable: correctly refused | 100% | 100% |
| Small talk: direct route | — | 100% |
| Answers with invented numbers/codes | 1% | 0% |
| Correct sources | 93% | 84% |
| Agent: correct final state | — | 88% |
| Agent: rewrites per question | — | 0.49 |
| Agent: LLM calls per question | — | 9.67 |
| Latency per question (s) | 3.62 | 11.31 |
| Retrieval: answer chunk in top-k | 93% | 93% |
| Retrieval: MRR | 0.76 | 0.76 |

## Accuracy per category (all splits)

| Category | n | linear | agent |
|---|---|---|---|
| fact | 41 | 98% | 83% |
| italian | 10 | 60% | 90% |
| multi_section | 5 | 100% | 80% |
| out_of_corpus | 4 | 100% | 100% |
| smalltalk | 4 | — | 100% |
| unanswerable | 8 | 100% | 100% |

## Failures

**linear** — 5 failed
- `test-22` *A quanti giri al minuto gira la pompa?* → 'Non so cosa significa "quantità di giri al minuto" in questo contesto.'
- `test-24` *Quanto pesa la pompa senza motore?* → "Non so. (I don't know.)"
- `dev-15` *What happens if new bearings are heated above 120 °C?* → 'The source document does not provide information on what happens if new bearings are heated above 120 °C. \n\n(source: centrifugal-pump-maintenance-manual.md)'
- `dev-20` *Cosa devo fare se la pompa non eroga liquido?* → 'Se la pompa non eroga liquido, segui il protocollo di interruzione e disconnessione (LOTO) per garantire la sicurezza. Chiudi le valvole isolamento del getto e '
- `dev-21` *Qual è il codice ricambio del cuscinetto lato accoppiamento?* → 'Non so. (source: centrifugal-pump-maintenance-manual.md)'

**agent** — 9 failed
- `dev-02` *How often should the mechanical seal be replaced?* → "I don't know: the maintenance manuals don't contain enough information to answer this question." · path: route -> retrieve → retrieve -> 4 chunks for 'How often should the mechanical seal be replaced?' → grade -> 0/4 relevant → rewrite -> 'How often replace mechanical seal in centrifugal pump?' → retrieve -> 4 chunks for 'How often replace mechanical seal in centrifugal pump?' → grade -> 1/4 relevant → generate (attempt 1) → check -> not_grounded → generate (attempt 2) → check -> not_grounded → fallback
- `test-02` *How much grease goes into the non-drive-end bearing?* → "I don't know: the maintenance manuals don't contain enough information to answer this question." · path: route -> retrieve → retrieve -> 4 chunks for 'How much grease goes into the non-drive-end bearing?' → grade -> 1/4 relevant → generate (attempt 1) → check -> not_grounded → generate (attempt 2) → check -> not_grounded → fallback
- `test-13` *How should coupling misalignment be corrected?* → "I don't know: the maintenance manuals don't contain enough information to answer this question." · path: route -> retrieve → retrieve -> 4 chunks for 'How should coupling misalignment be corrected?' → grade -> 2/4 relevant → generate (attempt 1) → check -> not_useful → rewrite -> 'How to adjust for shaft misalignment in a coupling system?' → retrieve -> 4 chunks for 'How to adjust for shaft misalignment in a coupling system?' → grade -> 0/4 relevant → rewrite -> 'How to correct for shaft misalignment in a coupled assembly?' → retrieve -> 4 chunks for 'How to correct for shaft misalignment in a coupled assembly?' → grade -> 0/4 relevant → fallback
- `test-14` *What happens if the pump runs dry?* → "I don't know: the maintenance manuals don't contain enough information to answer this question." · path: route -> retrieve → retrieve -> 4 chunks for 'What happens if the pump runs dry?' → grade -> 1/4 relevant → generate (attempt 1) → check -> not_grounded → generate (attempt 2) → check -> not_grounded → fallback
- `test-18` *How often should I check the coupling alignment?* → "I don't know: the maintenance manuals don't contain enough information to answer this question." · path: route -> retrieve → retrieve -> 4 chunks for 'How often should I check the coupling alignment?' → grade -> 0/4 relevant → rewrite -> 'How frequently should coupling misalignment be inspected?' → retrieve -> 4 chunks for 'How frequently should coupling misalignment be inspected?' → grade -> 0/4 relevant → rewrite -> 'How often should coupling alignment be checked?' → retrieve -> 4 chunks for 'How often should coupling alignment be checked?' → grade -> 0/4 relevant → fallback
- `test-24` *Quanto pesa la pompa senza motore?* → "I don't know: the maintenance manuals don't contain enough information to answer this question." · path: route -> retrieve → retrieve -> 4 chunks for 'Quanto pesa la pompa senza motore?' → grade -> 0/4 relevant → rewrite -> 'How much does a pump weigh without its motor?' → retrieve -> 4 chunks for 'How much does a pump weigh without its motor?' → grade -> 0/4 relevant → rewrite -> 'How much does a pump weigh without its drive unit?' → retrieve -> 4 chunks for 'How much does a pump weigh without its drive unit?' → grade -> 0/4 relevant → fallback
- `dev-15` *What happens if new bearings are heated above 120 °C?* → "I don't know: the maintenance manuals don't contain enough information to answer this question." · path: route -> retrieve → retrieve -> 4 chunks for 'What happens if new bearings are heated above 120 °C?' → grade -> 0/4 relevant → rewrite -> 'What effects do elevated temperatures have on newly installed bearings?' → retrieve -> 4 chunks for 'What effects do elevated temperatures have on newly installed bearings?' → grade -> 2/4 relevant → generate (attempt 1) → check -> not_grounded → generate (attempt 2) → check -> not_grounded → fallback
- `dev-18` *Can different types of grease be mixed?* → "I don't know: the maintenance manuals don't contain enough information to answer this question." · path: route -> retrieve → retrieve -> 4 chunks for 'Can different types of grease be mixed?' → grade -> 0/4 relevant → rewrite -> 'Can various lubricating oils be combined?' → retrieve -> 4 chunks for 'Can various lubricating oils be combined?' → grade -> 0/4 relevant → rewrite -> 'Can various greases be blended?' → retrieve -> 4 chunks for 'Can various greases be blended?' → grade -> 0/4 relevant → fallback
- `hold-07` *What is the minimum flow the pump should run at for extended periods?* → "I don't know: the maintenance manuals don't contain enough information to answer this question." · path: route -> retrieve → retrieve -> 4 chunks for 'What is the minimum flow the pump should run at for extended periods?' → grade -> 1/4 relevant → generate (attempt 1) → check -> not_grounded → generate (attempt 2) → check -> not_grounded → fallback
