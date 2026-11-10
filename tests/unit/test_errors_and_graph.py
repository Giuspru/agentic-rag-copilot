"""Domain errors, retry policy, budgets and recursion limit (no LLM: fake nodes)."""

import httpx
import pytest
from langgraph.errors import GraphRecursionError
from langgraph.graph import END
from ollama import ResponseError

from copilot.agent.graph import decide_after_check, decide_after_grade, max_graph_steps
from copilot.config import settings
from copilot.errors import (
    LLMTimeoutError,
    LLMUnavailableError,
    is_transient,
    llm_errors,
)


@pytest.mark.parametrize(
    ("exc", "transient"),
    [
        (ConnectionError("down"), True),
        (httpx.ConnectError("refused"), True),
        (ResponseError("boom", 500), True),
        (ResponseError("not found", 404), False),
        (httpx.ReadTimeout("slow"), False),
        (ValueError("bug"), False),
    ],
)
def test_is_transient(exc, transient):
    assert is_transient(exc) is transient


@pytest.mark.parametrize(
    ("exc", "domain_error", "status"),
    [
        (httpx.ConnectError("refused"), LLMUnavailableError, 503),
        (ConnectionError("down"), LLMUnavailableError, 503),
        (ResponseError("model 'x' not found", 404), LLMUnavailableError, 503),
        (httpx.ReadTimeout("slow"), LLMTimeoutError, 504),
    ],
)
def test_llm_errors_translates(exc, domain_error, status):
    with pytest.raises(domain_error) as caught, llm_errors():
        raise exc
    assert caught.value.status_code == status
    assert caught.value.__cause__ is exc


def test_llm_errors_lets_bugs_through():
    with pytest.raises(ValueError), llm_errors():
        raise ValueError("a bug is not an Ollama error")


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"documents": ["chunk"], "rewrites": 0}, "generate"),
        ({"documents": [], "rewrites": 0}, "rewrite_query"),
        ({"documents": [], "rewrites": settings.max_rewrites}, "fallback"),
    ],
)
def test_decide_after_grade(state, expected):
    assert decide_after_grade(state) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"verdict": "useful", "generations": 1, "rewrites": 0}, END),
        ({"verdict": "not_grounded", "generations": 1, "rewrites": 0}, "generate"),
        (
            {"verdict": "not_grounded", "generations": settings.max_generations, "rewrites": 0},
            "fallback",
        ),
        ({"verdict": "not_useful", "generations": 1, "rewrites": 0}, "rewrite_query"),
        (
            {"verdict": "not_useful", "generations": 1, "rewrites": settings.max_rewrites},
            "fallback",
        ),
    ],
)
def test_decide_after_check(state, expected):
    assert decide_after_check(state) == expected


def test_worst_case_path_matches_max_graph_steps(build_fake_agent):
    agent = build_fake_agent()
    final = agent.invoke({"question": "q"}, config={"recursion_limit": max_graph_steps() + 1})
    assert len(final["steps"]) == max_graph_steps()
    assert final["steps"][-1] == "fallback"


def test_recursion_limit_is_tight(build_fake_agent):
    with pytest.raises(GraphRecursionError):
        build_fake_agent().invoke({"question": "q"}, config={"recursion_limit": max_graph_steps()})


def test_a_looping_bug_stops_fast(build_fake_agent):
    calls = {"n": 0}

    def buggy_rewrite(state):  # forgets to increment 'rewrites': the loop never ends
        calls["n"] += 1
        return {"generations": 0, "steps": ["rewrite"]}

    agent = build_fake_agent(rewrite_query=buggy_rewrite)
    with pytest.raises(GraphRecursionError):
        agent.invoke({"question": "q"}, config={"recursion_limit": max_graph_steps() + 1})
    assert calls["n"] <= 5


def test_transient_errors_are_retried(build_fake_agent, fake_nodes):
    fake_route = fake_nodes["route_question"]
    attempts = {"n": 0}

    def flaky_route(state):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise httpx.ConnectError("Ollama restarting")
        return fake_route(state)

    final = build_fake_agent(route_question=flaky_route).invoke(
        {"question": "q"}, config={"recursion_limit": max_graph_steps() + 1}
    )
    assert attempts["n"] == settings.llm_max_attempts == 3
    assert final["steps"][-1] == "fallback"


def test_bugs_are_not_retried(build_fake_agent):
    attempts = {"n": 0}

    def buggy_route(state):
        attempts["n"] += 1
        raise ValueError("bug")

    with pytest.raises(ValueError):
        build_fake_agent(route_question=buggy_route).invoke({"question": "q"})
    assert attempts["n"] == 1
