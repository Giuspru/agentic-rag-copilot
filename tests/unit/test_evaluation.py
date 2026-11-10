"""Evaluation metrics, dataset quality and the CI gate (no LLM)."""

import pytest

from copilot.agent.nodes import FALLBACK_ANSWER
from copilot.evaluation import metrics
from copilot.evaluation.dataset import load_dataset
from copilot.evaluation.runner import corpus_text
from copilot.evaluation.gate import check_gate, load_thresholds


@pytest.mark.parametrize(
    ("facts", "text", "recall"),
    [
        ([["8"]], "Tighten to 8 Nm", 1.0),
        ([["8"]], "every 8000 hours", 0.0),  # whole numbers only
        ([["0.05"]], "max 0,05 mm", 1.0),  # Italian decimal comma
        ([["4000"]], "ogni 4.000 ore", 1.0),  # Italian thousands separator
        ([["nlgi", "lithium"], ["15"]], "Lithium grease, 15 g", 1.0),
        ([["nlgi"], ["15"]], "NLGI 2", 0.5),
        ([], "anything", 1.0),
    ],
)
def test_fact_recall(facts, text, recall):
    assert metrics.fact_recall(facts, text) == recall


@pytest.mark.parametrize(
    ("answer", "refusal"),
    [
        (FALLBACK_ANSWER, True),
        ("I don't know. The context only discusses pumps.", True),
        ("Non lo so.", True),
        ("Le clip di centraggio non sono state rimosse.", False),  # "non so-no" is not a refusal
        ("Non so cosa significa in questo contesto.", True),
        ("Non so.", True),
        ("Use NLGI 2 grease.", False),
    ],
)
def test_is_refusal(answer, refusal):
    assert metrics.is_refusal(answer) is refusal


def test_is_correct_per_behavior():
    assert metrics.is_correct("answer", [["15"]], "15 grams", "answer")
    assert not metrics.is_correct("answer", [["15"]], FALLBACK_ANSWER, "refuse")
    assert metrics.is_correct("refuse", [], FALLBACK_ANSWER, "refuse")
    assert not metrics.is_correct("refuse", [], "Paris.", "answer")
    assert metrics.is_correct("direct", [], "Hello!", "direct")


def test_trajectory_stats_counts_llm_calls():
    steps = [
        "route -> retrieve",
        "retrieve -> 4 chunks for 'q'",
        "grade -> 2/4 relevant",
        "generate (attempt 1)",
        "check -> not_grounded (5000)",
        "generate (attempt 2)",
        "check -> not_grounded",
        "generate (attempt 3)",
        "check -> useful",
    ]
    stats = metrics.trajectory_stats(steps)
    # route 1 + grade 4 + generate 3 + checks (0 deterministic + 1 + 2)
    assert stats["llm_calls"] == 1 + 4 + 3 + 0 + 1 + 2
    assert stats["terminal"] == "useful" and stats["rewrites"] == 0
    assert metrics.path_ok("answer", "useful") and not metrics.path_ok("refuse", "useful")


def test_first_relevant_rank():
    chunks = ["noise", "grease NLGI 2", "15 grams of NLGI grease"]
    assert metrics.first_relevant_rank([["nlgi"], ["15"]], chunks) == 3
    assert metrics.first_relevant_rank([["6309"]], chunks) is None


def test_dataset_is_valid_and_answerable_facts_are_in_the_manual():
    """Data quality: a typo in an expected fact would make every system 'fail' that item."""
    items = load_dataset()
    corpus = corpus_text()
    assert {i.split for i in items} == {"dev", "test", "holdout"}
    for item in items:
        if item.expected == "answer":
            assert item.facts and item.source, item.id
            assert metrics.fact_recall(item.facts, corpus) == 1.0, f"{item.id}: fact not in corpus"
        else:
            assert not item.facts, item.id


def _report(**summary):
    return {"systems": {"agent": {"summary": {"split": {"test": summary}}}}}


def test_gate_passes_and_fails():
    thresholds = load_thresholds()
    good = {key.split("_", 1)[1]: bound for key, bound in thresholds["agent"].items()}
    assert check_gate(_report(**good), thresholds) == []
    metric = next(k for k in thresholds["agent"] if k.startswith("min_"))
    bad = {**good, metric.split("_", 1)[1]: -1}
    assert check_gate(_report(**bad), thresholds) != []
