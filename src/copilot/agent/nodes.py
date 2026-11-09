"""Agent nodes — the units of work of the self-corrective RAG graph.

Each node is a plain function ``state -> partial update``: it reads what it needs from
the state, does one job, and returns only the keys it changes (plus a line for the
``steps`` trace). Nodes do the work; *which* node runs next is decided in ``graph.py``.

Several nodes use the LLM as a **judge** rather than as a writer: routing the question,
grading each retrieved chunk, checking the answer. Most judges use structured output
(``with_structured_output``): the model must reply with JSON matching a Pydantic schema,
so the graph branches on a reliable ``"yes"``/``"no"`` instead of parsing free text.

The exception is the groundedness judge, measured in phase 3: with the JSON schema it
rejected 5 correct answers out of 14 (even answers copied verbatim from the manual);
with the same prompt answering in plain text, 14/14 — and it still rejected all 14
invented answers. On a 3B model, constrained decoding was hurting the judgment. Its
plain-text reply is parsed conservatively: anything but a clear "yes" counts as "no".
"""

import re
from typing import Literal

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from copilot.agent.state import AgentState
from copilot.guardrails.groundedness import unsupported_claims
from copilot.llm.provider import get_llm
from copilot.rag.chain import SYSTEM_PROMPT, format_docs
from copilot.rag.retriever import get_retriever

# The last line of defense: when the agent cannot find support in the manuals, it says
# so with a fixed message instead of letting the LLM improvise.
FALLBACK_ANSWER = (
    "I don't know: the maintenance manuals don't contain enough information to answer "
    "this question."
)


# --- Structured outputs of the LLM judges -------------------------------------------


class RouteDecision(BaseModel):
    """Does the question need the manuals?"""

    route: Literal["retrieve", "direct"] = Field(
        description="'retrieve' for any question that needs facts; 'direct' only for "
        "greetings, thanks, or questions about the assistant itself."
    )


class BinaryScore(BaseModel):
    """A yes/no judgment."""

    binary_score: Literal["yes", "no"]


# --- Prompts --------------------------------------------------------------------------

# Few-shot examples make the 3B model's routing reliable, including on Italian messages.
ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route messages for a maintenance assistant that answers from equipment "
            "maintenance manuals.\n"
            "- 'direct': the message is ONLY a greeting, a thank-you, small talk, or asks "
            "what the assistant can do. No information is requested.\n"
            "- 'retrieve': the message asks for any information (facts, procedures, "
            "values, parts, troubleshooting), in any language and on any topic.\n\n"
            "Examples:\n"
            "'Hello!' -> direct\n"
            "'Ciao, cosa sai fare?' -> direct\n"
            "'Thanks, that helped' -> direct\n"
            "'How do I replace the seal?' -> retrieve\n"
            "'la pompa vibra' -> retrieve\n"
            "'What is the capital of France?' -> retrieve",
        ),
        ("human", "{question}"),
    ]
)

DIRECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a maintenance assistant that answers questions using equipment "
            "maintenance manuals. Reply briefly and politely to the user's message. "
            "Never state technical facts, numbers or procedures here: if the user asks "
            "for information, invite them to ask a question about the manuals.",
        ),
        ("human", "{question}"),
    ]
)

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether a retrieved document excerpt is relevant "
            "to a user question.\n"
            "If the excerpt contains information that helps answer the question, grade "
            "it 'yes'. Otherwise grade it 'no'.\n"
            "Be strict: sharing a few generic words (e.g. 'pump') is NOT enough.",
        ),
        ("human", "Excerpt:\n{document}\n\nQuestion: {question}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite search queries for a vector database of equipment maintenance "
            "manuals (written in English). The previous query did not retrieve relevant "
            "excerpts.\n"
            "Rewrite it as a short, specific English search query (translate it if the "
            "question is in another language) using the technical "
            "terms a maintenance manual would use (e.g. 'excessive vibration causes' "
            "instead of 'it shakes'). Reply with the query only, no explanations.",
        ),
        ("human", "Original question: {question}\nPrevious query: {query}"),
    ]
)

# Same contract as the linear chain (SYSTEM_PROMPT), plus the English search query as a
# hint: it helps the 3B model understand questions asked in another language.
GENERATE_HUMAN = "Context:\n{context}\n\nQuestion: {question}\n(The question in English: {query})"

GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT + "\n- Answer in the same language as the question."),
        ("human", GENERATE_HUMAN),
    ]
)

# Used when the previous answer was judged not grounded. At temperature 0 the same
# prompt would reproduce the same answer, so the retry must change the instructions,
# not just ask again.
STRICT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PROMPT + "\n- Answer in the same language as the question."
            "\n- WARNING: a previous draft contained claims not supported by the "
            "context. Use only statements you can point to in the context and copy "
            "numbers and part codes exactly as written.",
        ),
        ("human", GENERATE_HUMAN),
    ]
)

GROUNDED_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader checking whether an answer is supported by a set of facts "
            "from a maintenance manual.\n"
            "Grade 'yes' if the answer's technical content (causes, actions, numbers, "
            "part codes) can be found in the facts, even if it is paraphrased, "
            "summarized, or written in another language.\n"
            "Grade 'no' only if the answer states a number, part code, cause or action "
            "that does NOT appear in the facts.\n"
            "Ignore citations, section numbers and file names: they are not claims.",
        ),
        ("human", "Facts:\n{context}\n\nAnswer: {generation}\n\nReply with one word: yes or no."),
    ]
)

ANSWERS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader checking whether an answer addresses a question. The "
            "question and the answer may be in different languages.\n"
            "Grade 'yes' if the answer gives information about the same problem or "
            "component the question asks about.\n"
            "Grade 'no' if the answer is about a different problem or component, or if "
            "it only says it does not know.",
        ),
        ("human", "Question: {question}\n\nAnswer: {generation}"),
    ]
)


# --- Chains ---------------------------------------------------------------------------
# Composing a chain is cheap (the LLM itself is cached by ``get_llm()``), so these are
# built on demand. They can't use ``@lru_cache``: prompt templates are not hashable.


def _judge(prompt: ChatPromptTemplate, schema: type[BaseModel]) -> Runnable:
    """Prompt + LLM constrained to reply with JSON matching ``schema``."""
    return prompt | get_llm().with_structured_output(schema)


def _writer(prompt: ChatPromptTemplate) -> Runnable:
    """Prompt + LLM returning plain text."""
    return prompt | get_llm() | StrOutputParser()


_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_yes_no(text: str) -> Literal["yes", "no"]:
    """First "yes"/"no" word of a free-text reply; no clear "yes" -> "no" (prudence)."""
    match = _YES_NO_RE.search(text)
    return "yes" if match and match.group(1).lower() == "yes" else "no"


def _yes_no_judge(prompt: ChatPromptTemplate) -> Runnable:
    """Prompt + LLM replying in plain text, parsed to "yes"/"no"."""
    return _writer(prompt) | parse_yes_no


# --- Nodes ----------------------------------------------------------------------------


def route_question(state: AgentState) -> AgentState:
    """Decide whether the question needs the manuals or is just conversation."""
    decision = _judge(ROUTE_PROMPT, RouteDecision).invoke({"question": state["question"]})
    return {
        "route": decision.route,
        "query": state["question"],
        "rewrites": 0,
        "generations": 0,
        "steps": [f"route -> {decision.route}"],
    }


def respond_direct(state: AgentState) -> AgentState:
    """Answer small talk without retrieval (and without stating any fact)."""
    generation = _writer(DIRECT_PROMPT).invoke({"question": state["question"]})
    return {"generation": generation, "documents": [], "steps": ["respond_direct"]}


def retrieve(state: AgentState) -> AgentState:
    """Fetch candidate chunks for the current query (original or rewritten)."""
    documents = get_retriever().invoke(state["query"])
    return {
        "documents": documents,
        "steps": [f"retrieve -> {len(documents)} chunks for {state['query']!r}"],
    }


def grade_documents(state: AgentState) -> AgentState:
    """Keep only the chunks the LLM judges relevant to the search query.

    This is the "corrective" step of CRAG: irrelevant chunks are noise in the prompt and
    an invitation to hallucinate, so they are dropped before generation.

    Chunks are graded against the *query*, not the original question: the query is the
    agent's English statement of what it is looking for. A question in Italian gets no
    relevant chunk at first, the rewrite translates it, and the next grading is done in
    the same language as the manuals — which a 3B model judges far more reliably.
    """
    grader = _judge(GRADE_PROMPT, BinaryScore)

    def is_relevant(doc: Document) -> bool:
        grade = grader.invoke({"document": doc.page_content, "question": state["query"]})
        return grade.binary_score == "yes"

    relevant = [doc for doc in state["documents"] if is_relevant(doc)]
    return {
        "documents": relevant,
        "steps": [f"grade -> {len(relevant)}/{len(state['documents'])} relevant"],
    }


def rewrite_query(state: AgentState) -> AgentState:
    """Reformulate the search query to retrieve better chunks on the next attempt.

    New chunks deserve a fresh answer budget, so the generation counter is reset.
    """
    query = _writer(REWRITE_PROMPT).invoke({"question": state["question"], "query": state["query"]})
    query = query.strip().strip('"')
    return {
        "query": query,
        "rewrites": state["rewrites"] + 1,
        "generations": 0,
        "steps": [f"rewrite -> {query!r}"],
    }


def generate(state: AgentState) -> AgentState:
    """Write the answer from the relevant chunks only.

    A retry after a failed groundedness check uses the stricter prompt.
    """
    prompt = GENERATE_PROMPT if state["generations"] == 0 else STRICT_PROMPT
    generation = _writer(prompt).invoke(
        {
            "context": format_docs(state["documents"]),
            "question": state["question"],
            "query": state["query"],
        }
    )
    attempt = state["generations"] + 1
    return {
        "generation": generation,
        "generations": attempt,
        "steps": [f"generate (attempt {attempt})"],
    }


def check_answer(state: AgentState) -> AgentState:
    """Self-RAG check: is the answer grounded in the chunks, and does it answer?

    Three checks, cheapest first, because they catch different failures:
      1. deterministic (guardrail): every number, part code and cited file in the answer
         must appear in the chunks — exact facts are where the LLM judge is weakest;
      2. LLM judge "grounded" (plain-text yes/no, see the module docstring): no
         unsupported cause or action, paraphrase allowed;
      3. LLM judge "answers": a grounded answer can still miss the point.
    As in grading, the last judge compares the answer with the English query:
    cross-language judgments are where a 3B model is least reliable.
    """
    context = format_docs(state["documents"])
    unsupported = unsupported_claims(state["generation"], context)
    if unsupported:
        listed = ", ".join(unsupported[:5])
        return {"verdict": "not_grounded", "steps": [f"check -> not_grounded ({listed})"]}

    grounded = _yes_no_judge(GROUNDED_PROMPT).invoke(
        {"context": context, "generation": state["generation"]}
    )
    if grounded == "no":
        verdict = "not_grounded"
    else:
        answers = _judge(ANSWERS_PROMPT, BinaryScore).invoke(
            {"question": state["query"], "generation": state["generation"]}
        )
        verdict = "useful" if answers.binary_score == "yes" else "not_useful"
    return {"verdict": verdict, "steps": [f"check -> {verdict}"]}


def fallback(state: AgentState) -> AgentState:
    """Out of retries: admit not knowing rather than answer without support."""
    return {"generation": FALLBACK_ANSWER, "documents": [], "steps": ["fallback"]}
