"""Evaluation runner — run a system on the dataset, score it, write a report.

Two systems, evaluated with the same metrics so they can be compared:
  - ``linear``: the phase-1 RAG chain (the "before");
  - ``agent``: the self-corrective LangGraph agent (the "after").

Retrieval is scored once per question, independently of the system: it measures the
vector store, which both systems share (the agent then *improves* on it by grading and
rewriting).
"""

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from copilot import __version__
from copilot.agent.graph import run_agent
from copilot.config import settings
from copilot.evaluation import metrics
from copilot.evaluation.dataset import EvalItem
from copilot.rag.chain import answer as linear_answer
from copilot.rag.ingestion import load_documents
from copilot.rag.retriever import get_retriever

SYSTEMS = ("linear", "agent")


def corpus_text() -> str:
    """The whole corpus, as one string, plus a ``[Source: ...]`` header per document."""
    return "\n\n".join(
        f"[Source: {doc.metadata['source']}]\n{doc.page_content}" for doc in load_documents()
    )


def evaluate_item(item: EvalItem, system: str, corpus: str) -> dict:
    """Run one system on one question and score the result."""
    start = time.perf_counter()
    if system == "agent":
        result = run_agent(item.question, include_contexts=True)
        answer, steps, sources = result["answer"], result["steps"], result["sources"]
        contexts = result["contexts"]
    else:
        answer, steps, contexts = linear_answer(item.question), None, None
        # The linear chain always uses the retrieved chunks: its "sources" are the files
        # it cites in the text.
        sources = sorted({s for s in [item.source] if s and s in answer})
    latency = time.perf_counter() - start

    observed = metrics.behavior(answer, steps)
    correct = (
        None
        if (system == "linear" and item.expected == "direct")
        else (metrics.is_correct(item.expected, item.facts, answer, observed))
    )
    if item.expected == "answer":
        source_ok = sources == [item.source]
    elif system == "agent":
        source_ok = sources == []
    else:
        source_ok = None

    record = {
        "id": item.id,
        "split": item.split,
        "category": item.category,
        "expected": item.expected,
        "question": item.question,
        "answer": answer,
        "behavior": observed,
        "correct": correct,
        "fact_recall": metrics.fact_recall(item.facts, answer)
        if item.expected == "answer"
        else None,
        "hallucinated": metrics.hallucinated_claims(answer, corpus),
        "source_ok": source_ok,
        "latency_s": round(latency, 2),
    }
    if steps is not None:
        stats = metrics.trajectory_stats(steps)
        stats["path_ok"] = metrics.path_ok(item.expected, stats["terminal"])
        record["trajectory"] = stats
        record["steps"] = steps
        record["contexts"] = contexts
    return record


def rescore(report: dict, items: list[EvalItem]) -> dict:
    """Recompute the answer-level metrics from the answers saved in a report.

    Generation is slow, scoring is instant: when a metric changes, the saved answers are
    scored again without calling any model. Trajectory, latency, sources and retrieval
    ranks do not depend on the answer-level metrics and are kept as they are.
    """
    by_id = {item.id: item for item in items}
    corpus = corpus_text()
    for system, data in report["systems"].items():
        for record in data["records"]:
            item = by_id[record["id"]]
            answer, steps = record["answer"], record.get("steps")
            record["behavior"] = metrics.behavior(answer, steps)
            record["correct"] = (
                None
                if (system == "linear" and item.expected == "direct")
                else metrics.is_correct(item.expected, item.facts, answer, record["behavior"])
            )
            record["fact_recall"] = (
                metrics.fact_recall(item.facts, answer) if item.expected == "answer" else None
            )
            record["hallucinated"] = metrics.hallucinated_claims(answer, corpus)
        data["summary"] = summarize(data["records"])
    report["meta"]["rescored"] = datetime.now().isoformat(timespec="seconds")
    return report


def retrieval_ranks(items: list[EvalItem]) -> dict[str, int | None]:
    """Rank of the first chunk containing the answer, for each answerable question."""
    retriever = get_retriever()
    return {
        item.id: metrics.first_relevant_rank(
            item.facts, [doc.page_content for doc in retriever.invoke(item.question)]
        )
        for item in items
        if item.expected == "answer"
    }


def run(items: list[EvalItem], systems: tuple[str, ...] = SYSTEMS, progress=print) -> dict:
    """Evaluate the systems on the items. Returns the full report (records + summaries)."""
    corpus = corpus_text()
    ranks = retrieval_ranks(items)
    report: dict = {
        "meta": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "version": __version__,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "retrieval_k": settings.retrieval_k,
            "n_items": len(items),
        },
        "systems": {},
    }
    for system in systems:
        records = []
        for i, item in enumerate(items, start=1):
            record = evaluate_item(item, system, corpus)
            if item.id in ranks:
                record["retrieval_rank"] = ranks[item.id]
            records.append(record)
            mark = {True: "ok ", False: "XX ", None: " - "}[record["correct"]]
            progress(f"[{system} {i}/{len(items)}] {mark} {item.id} ({record['latency_s']}s)")
        report["systems"][system] = {"summary": summarize(records), "records": records}
    return report


def summarize(records: list[dict]) -> dict:
    """Aggregate overall, per split and per category."""
    by_split, by_category = defaultdict(list), defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)
        by_category[r["category"]].append(r)
    return {
        "all": metrics.aggregate(records),
        "split": {k: metrics.aggregate(v) for k, v in sorted(by_split.items())},
        "category": {k: metrics.aggregate(v) for k, v in sorted(by_category.items())},
    }


# --- Report -----------------------------------------------------------------------------

_ROWS = [
    ("accuracy", "Accuracy (all expected behaviors)", "pct"),
    ("answer_accuracy", "Answerable: correct answer", "pct"),
    ("fact_recall", "Answerable: fact recall", "pct"),
    ("false_refusals", "Answerable: wrongly refused (safe error)", "int"),
    ("wrong_answers", "Confidently wrong answers (dangerous error)", "int"),
    ("refusal_accuracy", "Unanswerable: correctly refused", "pct"),
    ("direct_accuracy", "Small talk: direct route", "pct"),
    ("hallucination_rate", "Answers with invented numbers/codes", "pct"),
    ("source_accuracy", "Correct sources", "pct"),
    ("path_accuracy", "Agent: correct final state", "pct"),
    ("rewrites", "Agent: rewrites per question", "num"),
    ("llm_calls", "Agent: LLM calls per question", "num"),
    ("latency_s", "Latency per question (s)", "num"),
    ("retrieval_hit_rate", "Retrieval: answer chunk in top-k", "pct"),
    ("retrieval_mrr", "Retrieval: MRR", "num"),
]


def _fmt(value, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{value * 100:.0f}%"
    if kind == "int":
        return str(value)
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def to_markdown(report: dict) -> str:
    """A compact Markdown report: systems side by side, then per-category accuracy."""
    systems = list(report["systems"])
    meta = report["meta"]
    lines = [
        f"# Evaluation report — {meta['date']}",
        "",
        (
            f"Version {meta['version']} · LLM `{meta['llm_model']}` · embeddings "
            f"`{meta['embedding_model']}` · k={meta['retrieval_k']} · {meta['n_items']} questions"
        ),
        "",
    ]
    for name in ("holdout", "test", "dev", "all"):
        if name == "all":
            sums = {s: report["systems"][s]["summary"]["all"] for s in systems}
        elif all(name in report["systems"][s]["summary"]["split"] for s in systems):
            sums = {s: report["systems"][s]["summary"]["split"][name] for s in systems}
        else:
            continue
        n = next(iter(sums.values()))["n"]
        title = {
            "holdout": "Holdout split (unseen until the final evaluation)",
            "test": "Test split (held out from tuning; checked by the CI gate)",
            "dev": "Dev split (used to tune the prompts: optimistic)",
            "all": "All questions",
        }[name]
        lines += [
            f"## {title} — {n} questions",
            "",
            "| Metric | " + " | ".join(systems) + " |",
            "|---|" + "---|" * len(systems),
        ]
        for key, label, kind in _ROWS:
            values = [sums[s].get(key) for s in systems]
            if all(v is None for v in values):
                continue
            lines.append(f"| {label} | " + " | ".join(_fmt(v, kind) for v in values) + " |")
        lines.append("")
    lines += [
        "## Accuracy per category (all splits)",
        "",
        "| Category | n | " + " | ".join(systems) + " |",
        "|---|---|" + "---|" * len(systems),
    ]
    categories = report["systems"][systems[0]]["summary"]["category"]
    for cat, summary in categories.items():
        accs = [report["systems"][s]["summary"]["category"][cat]["accuracy"] for s in systems]
        lines.append(
            f"| {cat} | {summary['n']} | " + " | ".join(_fmt(a, "pct") for a in accs) + " |"
        )
    lines += ["", "## Failures", ""]
    for s in systems:
        failed = [r for r in report["systems"][s]["records"] if r["correct"] is False]
        lines.append(f"**{s}** — {len(failed)} failed")
        for r in failed:
            path = f" · path: {' → '.join(r['steps'])}" if r.get("steps") else ""
            lines.append(f"- `{r['id']}` *{r['question']}* → {r['answer'][:160]!r}{path}")
        lines.append("")
    return "\n".join(lines)


def write_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "report.json", out_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return json_path, md_path
