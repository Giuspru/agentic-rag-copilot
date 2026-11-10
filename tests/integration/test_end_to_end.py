"""End-to-end tests against the real local models (skipped when Ollama is down).

LLM output is not fully deterministic across machines, so assertions are *semantic and
lenient*: they check the facts that matter (a part number, a fallback, a route), not
exact wording. Quality is measured by the evaluation, not here: these tests check that
the pieces work together.
"""

import pytest
from fastapi.testclient import TestClient

from copilot.agent.graph import run_agent
from copilot.agent.nodes import FALLBACK_ANSWER
from copilot.api.main import app
from copilot.config import settings
from copilot.evaluation.dataset import load_dataset
from copilot.evaluation.runner import run
from copilot.llm.embeddings import get_embeddings
from copilot.llm.provider import get_llm
from copilot.rag.chain import answer
from copilot.rag.ingestion import ingest
from copilot.rag.retriever import get_retriever
from copilot.rag.vectorstore import get_vectorstore

pytestmark = pytest.mark.ollama


@pytest.fixture(scope="module", autouse=True)
def index():
    """Make sure the vector store is populated (CI starts from an empty one)."""
    if get_vectorstore()._collection.count() == 0:
        ingest()


def test_linear_chain_answers_and_refuses():
    in_corpus = answer("What grease should I use for the bearings?").lower()
    assert "nlgi" in in_corpus or "lithium" in in_corpus
    out_of_corpus = answer("What is the correct tire pressure for a road bicycle?").lower()
    assert "know" in out_of_corpus or "not" in out_of_corpus


def test_agent_small_talk_takes_the_direct_route():
    result = run_agent("Hi! What can you do?")
    assert result["steps"] == ["route -> direct", "respond_direct"]
    assert result["sources"] == []


def test_agent_answers_from_the_manual():
    result = run_agent("What is the part number of the mechanical seal?")
    assert "SP-CP2000-SEAL" in result["answer"]
    assert result["sources"] == ["centrifugal-pump-maintenance-manual.md"]
    assert result["steps"][-1] == "check -> useful"


def test_agent_falls_back_out_of_corpus():
    result = run_agent("What is the correct tire pressure for a road bicycle?")
    assert result["answer"] == FALLBACK_ANSWER
    assert result["sources"] == []
    assert any(step.startswith("rewrite") for step in result["steps"])


def test_agent_handles_italian_via_rewrite():
    result = run_agent("quale grasso devo usare per i cuscinetti?")
    assert any(step.startswith("rewrite") for step in result["steps"])
    assert "nlgi" in result["answer"].lower()


def test_api_chat_end_to_end():
    response = TestClient(app).post("/chat", json={"question": "What grease should I use?"})
    assert response.status_code == 200
    assert response.json()["sources"] == ["centrifugal-pump-maintenance-manual.md"]


def test_api_ready_with_real_dependencies():
    assert TestClient(app).get("/ready").status_code == 200


def test_timeout_is_504_without_retry(monkeypatch):
    monkeypatch.setattr(settings, "llm_timeout", 0.001)
    for cached in (get_llm, get_embeddings, get_vectorstore, get_retriever):
        cached.cache_clear()
    try:
        response = TestClient(app).post("/chat", json={"question": "What grease?"})
        assert response.status_code == 504
        assert response.json()["error"] == "llm_timeout"
    finally:
        monkeypatch.undo()
        for cached in (get_llm, get_embeddings, get_vectorstore, get_retriever):
            cached.cache_clear()


def test_evaluation_smoke():
    """The evaluation pipeline runs end to end on a few items of each kind."""
    items = [i for i in load_dataset() if i.id in {"dev-04", "test-28", "test-32"}]
    report = run(items, systems=("agent",), progress=lambda line: None)
    summary = report["systems"]["agent"]["summary"]["all"]
    assert summary["n"] == 3
    assert summary["hallucination_rate"] == 0.0
