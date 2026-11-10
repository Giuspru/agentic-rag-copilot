"""Shared pytest fixtures.

Two kinds of tests:
  - ``tests/unit``: no LLM, no network. They run anywhere in seconds (CI on every push).
  - ``tests/integration``: marked ``@pytest.mark.ollama``. They need a running Ollama with
    the models pulled, and are skipped automatically when it is not reachable.
"""

from unittest.mock import patch

import httpx
import pytest

from copilot.agent import graph as graph_module
from copilot.config import settings


def _ollama_available() -> bool:
    try:
        return httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0).is_success
    except httpx.HTTPError:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip the ``ollama`` tests when Ollama is down, instead of failing them."""
    if _ollama_available():
        return
    skip = pytest.mark.skip(reason=f"Ollama not reachable at {settings.ollama_base_url}")
    for item in items:
        if "ollama" in item.keywords:
            item.add_marker(skip)


# --- A graph with fake nodes: the real wiring (edges, budgets, retry), no LLM ----------


def fake_route(state):
    return {
        "route": "retrieve",
        "query": state["question"],
        "rewrites": 0,
        "generations": 0,
        "steps": ["route"],
    }


def fake_retrieve(state):
    return {"documents": ["chunk"], "steps": ["retrieve"]}


def fake_grade(state):
    return {"steps": ["grade"]}


def fake_generate(state):
    return {"generation": "draft", "generations": state["generations"] + 1, "steps": ["generate"]}


def fake_check(state):
    """Always the longest path: not grounded until the last attempt, then not useful."""
    if state["generations"] < settings.max_generations:
        return {"verdict": "not_grounded", "steps": ["check"]}
    return {"verdict": "not_useful", "steps": ["check"]}


def fake_rewrite(state):
    return {"rewrites": state["rewrites"] + 1, "generations": 0, "steps": ["rewrite"]}


FAKE_NODES = {
    "route_question": fake_route,
    "retrieve": fake_retrieve,
    "grade_documents": fake_grade,
    "generate": fake_generate,
    "check_answer": fake_check,
    "rewrite_query": fake_rewrite,
}


@pytest.fixture
def fake_nodes():
    return dict(FAKE_NODES)


@pytest.fixture
def build_fake_agent():
    """Factory: compile the real graph with fake nodes (overridable per test)."""

    def build(**overrides):
        graph_module.get_agent.cache_clear()
        with patch.multiple(graph_module, **{**FAKE_NODES, **overrides}):
            agent = graph_module.get_agent()
        graph_module.get_agent.cache_clear()  # never leave a fake graph in the cache
        return agent

    return build
