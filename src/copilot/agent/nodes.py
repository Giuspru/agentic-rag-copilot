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

# Phase 4, measured on 83 auto-labeled (question, chunk) pairs of the dev split: the
# former last line "Be strict: sharing a few generic words is NOT enough" rejected no
# extra irrelevant chunk in English, and in Italian it dropped 2 of 3 relevant chunks
# while keeping 2 irrelevant ones. Without it: relevant kept 18/22 (was 16/22),
# irrelevant dropped 54/61 (was 52/61). Two more findings from the same bench: a
# plain-text yes/no reply (better for the groundedness judge) collapses this judge to
# 2/19, and so does moving the question *before* the excerpt. The question stays last.
GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether a retrieved document excerpt is relevant "
            "to a user question.\n"
            "If the excerpt contains information that helps answer the question, grade "
            "it 'yes'. Otherwise grade it 'no'.",
        ),
        ("human", "Excerpt:\n{document}\n\nQuestion: {question}"),
    ]
)

# Italian -> English equivalents of the manual's terms. The 3B model mistranslates some
# of them ("cuscinetti" -> "bushings" instead of "bearings", "giunto" -> "bearing"), and
# then retrieval, grading and checks all work on the wrong subject. Putting the whole
# glossary in the prompt made it worse: the model copied it into unrelated queries
# ("mechanical seal" everywhere, English 15/16). So the code looks the terms up in the
# question and passes only the ones present: English questions get no hint at all.
# Phase 4, dev + didactic questions: Italian 9/10 -> 10/10, English unchanged (16/16).
TERM_HINTS = {
    r"\bcuscinett[oi]\b": "bearing",
    r"\btenut[ae]\b": "seal",
    r"\bgiunt[oi]\b": "coupling",
    r"\bgirant[ei]\b": "impeller",
    r"\banell[oi] di usura\b": "wear ring",
    r"\bguarnizion[ei]\b": "gasket",
    r"\bprevalenza\b": "head",
    r"\bportata\b": "flow",
    r"\badescar\w*": "prime",
    r"\blato (?:accoppiamento|comando)\b": "drive end",
    r"\blato opposto\b": "non-drive end",
}


def term_hints(question: str) -> str:
    """The glossary entries that appear in the question, as a line for the prompt."""
    low = question.lower()
    found = [
        f"{match.group(0)} = {english}"
        for pattern, english in TERM_HINTS.items()
        if (match := re.search(pattern, low))
    ]
    return "\nTerms: " + "; ".join(found) if found else ""


# The former version had an example, "e.g. 'excessive vibration causes' instead of 'it
# shakes'": the 3B model copied it into unrelated queries ("part number for excessive
# vibration causes in a mechanical seal"), 4 times on 21 dev questions. Without the
# example, and with "keep the subject / invent nothing", the rewritten query retrieves
# the answer chunk for 21/21 dev questions (was 20/21), with no copied example.
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite search queries for a vector database of equipment maintenance "
            "manuals (written in English). The previous query did not retrieve relevant "
            "excerpts.\n"
            "Rewrite it as a short, specific English search query (translate it if the "
            "question is in another language). Keep the subject of the question; use the "
            "technical terms a maintenance manual would use for that same subject. "
            "Do not add topics the question does not mention, and never invent numbers, "
            "models or part codes. Reply with the query only, no explanations.",
        ),
        ("human", "Original question: {question}\nPrevious query: {query}{term_hints}"),
    ]
)

# Same contract as the linear chain (SYSTEM_PROMPT), plus the English search query as a
# hint: it helps the 3B model understand questions asked in another language.
GENERATE_HUMAN = "Context:\n{context}\n\nQuestion: {question}\n(The question in English: {query})"

# The deterministic guardrail rejects any number that is not in the excerpts, so a unit
# conversion (40 °C -> "104 °F") is a guaranteed failure: the writer is told the rule.
ANSWER_RULES = (
    "\n- Answer in the same language as the question."
    "\n- Copy numbers and units exactly as written in the context; never convert units."
)

GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT + ANSWER_RULES),
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
            SYSTEM_PROMPT
            + ANSWER_RULES
            + "\n- WARNING: a previous draft contained claims not supported by the "
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
    query = _writer(REWRITE_PROMPT).invoke(
        {
            "question": state["question"],
            "query": state["query"],
            "term_hints": term_hints(state["question"]),
        }
    )
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
    verdict: Literal["useful", "not_grounded", "not_useful"]
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
