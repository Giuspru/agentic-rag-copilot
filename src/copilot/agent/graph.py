"""Agent graph — wires the nodes into a self-corrective RAG (CRAG / Self-RAG style).

    START -> route --direct--> respond_direct -> END
               |
            retrieve
               v
     +----> retrieve -> grade --relevant chunks--> generate -> check --useful--> END
     |                    |                           ^          |
     |                    | none relevant             |          | not grounded:
     +------ rewrite <----+                           +----------+ retry, stricter
               ^                                                 |
               +------------------- not useful ------------------+

    Out of retries at any point -> fallback ("I don't know") -> END

The difference from the linear chain is in the arrows that go *back*: the agent judges
its own retrieval and its own answer, and corrects itself. Every loop is bounded by a
counter in the state (``max_rewrites``, ``max_generations``), so the graph always ends.

Nodes do the work (``nodes.py``); the small ``decide_*`` functions below only read the
state and name the next node — that is all a conditional edge is.

Robustness (phase 3):
  - every node that calls Ollama has a ``RetryPolicy``: transient errors (connection
    dropped, server error) are retried with exponential backoff. This is safe because
    nodes are *idempotent*: they read the state and return an update, so running one
    twice has no side effect;
  - the ``recursion_limit`` is computed from the budgets (``max_graph_steps``), so a
    looping bug stops after a few steps instead of LangGraph's default of 10007;
  - ``run_agent`` validates the input and translates Ollama errors into domain errors.
"""

import logging
import time
from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from copilot.agent.nodes import (
    check_answer,
    fallback,
    generate,
    grade_documents,
    respond_direct,
    retrieve,
    rewrite_query,
    route_question,
)
from copilot.agent.state import AgentState
from copilot.config import settings
from copilot.errors import is_transient, llm_errors
from copilot.guardrails.validators import validate_question

logger = logging.getLogger(__name__)

# --- Conditional edges ----------------------------------------------------------------


def decide_after_route(state: AgentState) -> str:
    return "retrieve" if state["route"] == "retrieve" else "respond_direct"


def decide_after_grade(state: AgentState) -> str:
    """Relevant chunks -> answer; none -> rewrite the query, while the budget lasts."""
    if state["documents"]:
        return "generate"
    if state["rewrites"] < settings.max_rewrites:
        return "rewrite_query"
    return "fallback"


def decide_after_check(state: AgentState) -> str:
    """Useful -> done; not grounded -> regenerate; off the point -> search again."""
    verdict = state["verdict"]
    if verdict == "useful":
        return END
    if verdict == "not_grounded" and state["generations"] < settings.max_generations:
        return "generate"
    if verdict == "not_useful" and state["rewrites"] < settings.max_rewrites:
        return "rewrite_query"
    return "fallback"


# --- Graph ----------------------------------------------------------------------------


def max_graph_steps() -> int:
    """Worst-case number of node executions for one question, given the budgets.

    The longest path: route, then ``max_rewrites + 1`` search rounds, each made of
    retrieve + grade + ``max_generations`` x (generate + check), a rewrite between
    rounds, and finally the fallback. With the default budgets (2, 2) that is 22.

    LangGraph stops any run longer than ``recursion_limit`` with a
    ``GraphRecursionError``. Its default is very high (10007 in LangGraph 1.x; it was 25
    in older versions): if a bug ever broke a budget — say, a rewrite that forgets to
    increment its counter — the agent would loop through thousands of LLM calls (hours)
    before stopping. A limit computed from the budgets is the tightest one that never
    cuts a legitimate run: a looping bug fails fast and loudly instead.
    """
    rounds = settings.max_rewrites + 1
    per_round = 2 + 2 * settings.max_generations
    return 1 + rounds * per_round + settings.max_rewrites + 1


def _retry_on(exc: BaseException) -> bool:
    """Retry predicate of the nodes: transient errors only, and leave a trace in the log."""
    if is_transient(exc):
        logger.warning("Transient error from Ollama, retrying the node: %r", exc)
        return True
    return False


@lru_cache
def get_agent() -> CompiledStateGraph:
    """Build and compile the agent graph (built once, then reused)."""
    graph = StateGraph(AgentState)

    # Wait 0.5s, then 1s, ... between attempts (exponential backoff with jitter).
    retry = RetryPolicy(max_attempts=settings.llm_max_attempts, retry_on=_retry_on)

    graph.add_node("route_question", route_question, retry_policy=retry)
    graph.add_node("respond_direct", respond_direct, retry_policy=retry)
    graph.add_node("retrieve", retrieve, retry_policy=retry)  # embeddings call Ollama too
    graph.add_node("grade_documents", grade_documents, retry_policy=retry)
    graph.add_node("rewrite_query", rewrite_query, retry_policy=retry)
    graph.add_node("generate", generate, retry_policy=retry)
    graph.add_node("check_answer", check_answer, retry_policy=retry)
    graph.add_node("fallback", fallback)  # no I/O: nothing to retry

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges(
        "route_question", decide_after_route, ["retrieve", "respond_direct"]
    )
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents", decide_after_grade, ["generate", "rewrite_query", "fallback"]
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "check_answer")
    graph.add_conditional_edges(
        "check_answer", decide_after_check, [END, "generate", "rewrite_query", "fallback"]
    )
    graph.add_edge("respond_direct", END)
    graph.add_edge("fallback", END)

    return graph.compile()


def run_agent(question: str, include_contexts: bool = False) -> dict:
    """Run the agent on a question.

    Returns the answer, the source documents it is grounded in (empty when the agent
    didn't answer from the manuals), and the trace of steps taken. With
    ``include_contexts`` it also returns ``contexts``: the text of the chunks the answer
    was generated from — the evaluation needs them, the API does not.

    Raises ``InputRejectedError`` (before any LLM call) if the input guardrail refuses
    the question, and ``LLMUnavailableError`` / ``LLMTimeoutError`` if Ollama fails
    beyond what the retries can absorb.
    """
    question = validate_question(question)
    start = time.perf_counter()
    with llm_errors():
        final = get_agent().invoke(
            {"question": question}, config={"recursion_limit": max_graph_steps() + 1}
        )
    sources = sorted({doc.metadata["source"] for doc in final.get("documents", [])})
    logger.info(
        "Agent answered in %.1fs with %d steps: %s",
        time.perf_counter() - start,
        len(final["steps"]),
        " | ".join(final["steps"]),
    )
    result = {"answer": final["generation"], "sources": sources, "steps": final["steps"]}
    if include_contexts:
        result["contexts"] = [doc.page_content for doc in final.get("documents", [])]
    return result
