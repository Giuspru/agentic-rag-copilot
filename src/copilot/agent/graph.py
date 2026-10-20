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
"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

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


@lru_cache
def get_agent() -> CompiledStateGraph:
    """Build and compile the agent graph (built once, then reused)."""
    graph = StateGraph(AgentState)

    graph.add_node("route_question", route_question)
    graph.add_node("respond_direct", respond_direct)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("check_answer", check_answer)
    graph.add_node("fallback", fallback)

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


def run_agent(question: str) -> dict:
    """Run the agent on a question.

    Returns the answer, the source documents it is grounded in (empty when the agent
    didn't answer from the manuals), and the trace of steps taken.
    """
    final = get_agent().invoke({"question": question})
    sources = sorted({doc.metadata["source"] for doc in final.get("documents", [])})
    return {"answer": final["generation"], "sources": sources, "steps": final["steps"]}
