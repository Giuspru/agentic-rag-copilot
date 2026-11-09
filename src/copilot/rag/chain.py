"""RAG chain — where retrieval and generation are wired into one answer.

This closes the linear RAG loop:

    question -> retriever -> chunks -> prompt (chunks as context) -> LLM -> answer

The chain is built with LCEL (LangChain Expression Language): small ``Runnable`` pieces
composed with the ``|`` operator, each feeding its output to the next. The prompt is our
first, simplest guardrail against hallucination: it tells the model to answer *only* from
the retrieved context and to admit when the answer is not there.
"""

from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from copilot.errors import llm_errors
from copilot.llm.provider import get_llm
from copilot.rag.retriever import get_retriever

# The system message defines behavior; keeping the rules here (not scattered in code)
# makes the assistant's "contract" easy to read and tune.
SYSTEM_PROMPT = (
    "You are a maintenance assistant. Answer the user's question using ONLY the "
    "context below, which are excerpts from equipment maintenance manuals.\n"
    "- If the context does not contain the answer, say you don't know. Never invent "
    "facts, numbers, or part codes.\n"
    "- Be concise and practical.\n"
    "- End the answer with the source document, written as (source: <file name>), "
    "copying the exact file name shown after 'Source:' in the context."
)
# The citation rule once read "e.g. (source: manual.md)": the 3B model copied the example
# and cited a file that does not exist in 5 answers out of 14. Without a format it
# sometimes forgot to cite. A placeholder format plus "copy the exact name": 15/15 cited,
# 15/15 with the right file name (phase 3, see guardrails/groundedness.py).

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def format_docs(docs) -> str:
    """Turn retrieved Document objects into a single context string for the prompt.

    Each chunk is prefixed with its source file so the model can cite it, and chunks are
    separated by blank lines so they read as distinct passages.
    """
    return "\n\n".join(
        f"[Source: {doc.metadata['source']}]\n{doc.page_content}" for doc in docs
    )


@lru_cache
def get_rag_chain() -> Runnable:
    """Build the linear RAG chain (built once, then reused).

    Reading the LCEL below: the input (the question string) is fed to a dict that builds
    the two template variables in parallel — ``context`` by running the retriever and
    formatting its chunks, ``question`` by passing the input through unchanged. That dict
    fills the PROMPT, the PROMPT goes to the LLM, and StrOutputParser extracts the plain
    text of the reply.
    """
    return (
        {
            "context": get_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | get_llm()
        | StrOutputParser()
    )


def answer(question: str) -> str:
    """Convenience wrapper: run the RAG chain on a question and return the answer text.

    Ollama errors are translated into domain errors, as in the agent (see ``errors.py``).
    """
    with llm_errors():
        return get_rag_chain().invoke(question)
