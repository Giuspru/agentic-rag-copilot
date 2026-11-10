"""Evaluation metrics — pure, deterministic functions (no LLM involved).

Why deterministic metrics first? An LLM judge (ragas included) is itself a model that can
be wrong — phase 3 showed a 3B judge rejecting answers copied verbatim from the manual.
For questions with a checkable answer, string checks are exact, instant, free and
reproducible, so they can gate a CI pipeline. LLM-based metrics are a complement
(``scripts/ragas_eval.py``), not the foundation.

Four levels:
  - retrieval: does the top-k contain a chunk with the answer, and at which rank?
  - answer: does it contain the expected facts? does it refuse when it should? does it
    state numbers or part codes that are not in the manual?
  - trajectory (agent only): did it end in the right place, and at what cost?
  - aggregation over items, per split and category.
"""

import re
from collections.abc import Iterable

from copilot.agent.nodes import FALLBACK_ANSWER
from copilot.guardrails.groundedness import unsupported_claims

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_NUMERIC_FACT_RE = re.compile(r"^\d+(?:[.,]\d+)*$")
_NON_SO_RE = re.compile(r"\bnon so\b")

# Phrases that mean "I don't know", in the two languages the copilot speaks.
REFUSAL_SIGNALS = [
    "don't know",
    "do not know",
    "does not contain",
    "doesn't contain",
    "not contain",
    "no information",
    "not mentioned",
    "does not mention",
    "doesn't mention",
    "not specified",
    "does not specify",
    "doesn't specify",
    "not provided",
    "does not provide",
    "not available",
    "cannot answer",
    "can't answer",
    "non lo so",
    "non contiene",
    "non è indicat",
    "non è specificat",
    "non viene menzionat",
    "non ho informazioni",
    "non dispongo",
]


# --- Answer-level ---------------------------------------------------------------------


def _numbers(text: str) -> set[str]:
    """Numbers in the text, with ``.``/``,`` removed (``4.000`` == ``4,000`` == ``4000``)."""
    return {re.sub(r"[.,]", "", n) for n in _NUMBER_RE.findall(text)}


def fact_matches(alternative: str, text: str) -> bool:
    """A numeric fact must appear as a whole number (``8`` does not match ``8000``);
    any other fact as a case-insensitive substring."""
    if _NUMERIC_FACT_RE.match(alternative):
        return re.sub(r"[.,]", "", alternative) in _numbers(text)
    return alternative.lower() in text.lower()


def fact_recall(facts: list[list[str]], text: str) -> float:
    """Fraction of fact groups found in ``text`` (1.0 when there are no facts)."""
    if not facts:
        return 1.0
    found = sum(any(fact_matches(alt, text) for alt in group) for group in facts)
    return found / len(facts)


def is_refusal(answer: str) -> bool:
    """Does the answer say that the information is not available?"""
    if answer.strip() == FALLBACK_ANSWER:
        return True
    low = answer.lower().replace("’", "'")
    # "non so" needs word boundaries: as a plain substring it matches inside "non sono".
    return any(signal in low for signal in REFUSAL_SIGNALS) or bool(_NON_SO_RE.search(low))


def hallucinated_claims(answer: str, corpus: str) -> list[str]:
    """Numbers / part codes / cited files in the answer that appear NOWHERE in the corpus.

    Reuses the output guardrail, but against the whole corpus: it catches what the
    linear chain invents (it has no guardrail) and anything that slipped past the agent.
    """
    return unsupported_claims(answer, corpus)


def behavior(answer: str, steps: list[str] | None) -> str:
    """What the system did: ``direct`` (agent small-talk route), ``refuse`` or ``answer``."""
    if steps and steps[0] == "route -> direct":
        return "direct"
    return "refuse" if is_refusal(answer) else "answer"


def is_correct(expected: str, facts: list[list[str]], answer: str, observed: str) -> bool:
    """The headline metric, per expected behavior.

    - answer: every fact group found (a hedged answer that still gives the facts counts);
    - refuse: the system said it does not know;
    - direct: the agent took the small-talk route.
    """
    if expected == "answer":
        return answer.strip() != FALLBACK_ANSWER and fact_recall(facts, answer) == 1.0
    return observed == expected


# --- Retrieval-level ------------------------------------------------------------------


def first_relevant_rank(facts: list[list[str]], chunks: list[str]) -> int | None:
    """1-based rank of the first chunk containing every fact group, or ``None``."""
    for rank, chunk in enumerate(chunks, start=1):
        if fact_recall(facts, chunk) == 1.0:
            return rank
    return None


# --- Trajectory-level (agent) ---------------------------------------------------------


def trajectory_stats(steps: list[str]) -> dict:
    """Count what the agent did, from its ``steps`` trace.

    LLM calls are reconstructed from the trace: 1 for routing, 1 per graded chunk, 1 per
    rewrite / generation / direct reply, and for each check 0 (deterministic guardrail
    failed: no judge called), 1 (grounded judge said no) or 2 (both judges ran).
    """
    llm_calls = 0
    for step in steps:
        if step.startswith(("route ->", "rewrite ->", "generate", "respond_direct")):
            llm_calls += 1
        elif step.startswith("grade ->"):
            graded = re.search(r"/(\d+)", step)  # "grade -> 2/4 relevant": 4 judge calls
            llm_calls += int(graded.group(1)) if graded else 0
        elif step.startswith("check ->"):
            if "(" in step:
                llm_calls += 0
            elif step == "check -> not_grounded":
                llm_calls += 1
            else:
                llm_calls += 2
    end = steps[-1] if steps else ""
    terminal = {"check -> useful": "useful", "fallback": "fallback", "respond_direct": "direct"}
    return {
        "n_steps": len(steps),
        "rewrites": sum(s.startswith("rewrite") for s in steps),
        "generations": sum(s.startswith("generate") for s in steps),
        "llm_calls": llm_calls,
        "terminal": terminal.get(end, end),
    }


EXPECTED_TERMINAL = {"answer": "useful", "refuse": "fallback", "direct": "direct"}


def path_ok(expected: str, terminal: str) -> bool:
    """Did the agent end where it should (answer -> useful, refuse -> fallback, ...)?"""
    return EXPECTED_TERMINAL[expected] == terminal


# --- Aggregation ----------------------------------------------------------------------


def _mean(values: Iterable[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def aggregate(records: list[dict]) -> dict:
    """Summary metrics over a list of per-item records (one system)."""
    answer = [r for r in records if r["expected"] == "answer"]
    refuse = [r for r in records if r["expected"] == "refuse"]
    direct = [r for r in records if r["expected"] == "direct"]
    summary = {
        "n": len(records),
        "accuracy": _mean(r["correct"] for r in records if r["correct"] is not None),
        "answer_accuracy": _mean(r["correct"] for r in answer),
        "fact_recall": _mean(r["fact_recall"] for r in answer),
        "refusal_accuracy": _mean(r["correct"] for r in refuse),
        "direct_accuracy": _mean(r["correct"] for r in direct if r["correct"] is not None),
        # answerable questions the system refused (a hedged answer with the facts is fine)
        "false_refusals": sum(r["behavior"] == "refuse" and not r["correct"] for r in answer),
        # the dangerous error: an answer given with confidence that is wrong (on an
        # answerable question, or on a question that should have been refused)
        "wrong_answers": sum(r["behavior"] == "answer" and r["correct"] is False for r in records),
        "hallucination_rate": _mean(bool(r["hallucinated"]) for r in records),
        "source_accuracy": _mean(r["source_ok"] for r in answer if r["source_ok"] is not None),
        "latency_s": _mean(r["latency_s"] for r in records),
    }
    agent = [r for r in records if r.get("trajectory")]
    if agent:
        summary["path_accuracy"] = _mean(r["trajectory"]["path_ok"] for r in agent)
        summary["llm_calls"] = _mean(r["trajectory"]["llm_calls"] for r in agent)
        summary["rewrites"] = _mean(r["trajectory"]["rewrites"] for r in agent)
    retrieval = [r for r in answer if r.get("retrieval_rank", "n/a") != "n/a"]
    if retrieval:
        summary["retrieval_hit_rate"] = _mean(r["retrieval_rank"] is not None for r in retrieval)
        summary["retrieval_mrr"] = _mean(
            1 / r["retrieval_rank"] if r["retrieval_rank"] else 0.0 for r in retrieval
        )
    return summary
