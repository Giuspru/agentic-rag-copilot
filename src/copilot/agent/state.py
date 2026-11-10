"""Agent state — the data that flows through the graph.

In LangGraph every node receives the current state and returns only the keys it wants
to change; the graph merges that partial update into the state before calling the next
node. The state is therefore the agent's working memory for one question: what was
asked, what is being searched, what was found, what was answered, and how many retries
are left.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    """Working memory of the agent for a single question.

    ``total=False`` because the state fills up along the way: at the start only
    ``question`` exists, the other keys are written by the nodes.
    """

    question: str  # the user's original question (never modified)
    query: str  # the search query: starts as the question, may be rewritten
    route: Literal["retrieve", "direct"]
    documents: list[Document]  # chunks that survived grading (only relevant ones)
    generation: str  # the current answer
    verdict: Literal["useful", "not_grounded", "not_useful"]
    rewrites: int  # query rewrites done so far
    generations: int  # answer attempts done so far

    # Trace of what the agent did, one line per node. ``Annotated[..., operator.add]``
    # is a *reducer*: instead of replacing the list, each node's update is appended.
    steps: Annotated[list[str], operator.add]
