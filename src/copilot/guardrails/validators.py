"""Input guardrail — clean the question and refuse obvious prompt injections.

Runs before the agent, so a rejected question costs nothing: no LLM call is made.

Two jobs:
  1. **Normalization**: Unicode NFKC (so look-alike characters become plain ones),
     control characters removed, whitespace collapsed. A question made only of spaces
     becomes empty and is refused.
  2. **Prompt-injection screening**: a short list of patterns typical of attempts to
     override the assistant's instructions ("ignore previous instructions", "reveal your
     system prompt", the model's own special tokens...), in English and Italian.

The pattern list is a *first line* of defense, not the defense: rephrasings will get
through. The real protection is the architecture — the agent has no tools and no data
beyond the manuals, it answers only from retrieved excerpts, and every answer is checked
for groundedness. The filter's job is to stop the cheap attacks early and visibly.
"""

import re
import unicodedata

from copilot.errors import InputRejectedError

MAX_QUESTION_LENGTH = 2000  # same limit as the API schema, for callers that skip the API

_INJECTION_PATTERNS = [
    # English
    # "instructions" alone is a normal word in a maintenance manual ("show me the
    # instructions for seal replacement"): the patterns require the assistant's own
    # instructions (your / previous / all...) to avoid blocking legitimate questions.
    (
        r"\b(ignore|disregard|forget|override)\b.{0,20}\b(previous|prior|above|earlier|all|"
        r"your)\b.{0,20}\b(instructions?|rules|prompts?|guidelines)\b"
    ),
    (
        r"\b(reveal|show|print|repeat|leak)\b.{0,30}\b(your|initial|hidden)\s+"
        r"(prompt|instructions|rules)\b"
    ),
    r"\bsystem prompt\b",
    r"\byou are now\b",
    r"\bpretend (to be|you are)\b",
    r"\bjailbreak\b",
    # Italian (\w* accepts attached pronouns: "mostrami", "ignorale")
    r"\b(ignora|dimentica|trascura)\w*\b.{0,20}\b(tutte|tue)\b.{0,20}\b(istruzioni|regole)\b",
    (
        r"\b(ignora|dimentica|trascura)\w*\b.{0,20}\b(istruzioni|regole|prompt)\s+"
        r"(precedenti|iniziali|di sistema)\b"
    ),
    (
        r"\b(mostra|rivela|stampa|ripeti)\w*\b.{0,30}\b(tue|tuo|iniziali|nascost[ei])\s+"
        r"(prompt|istruzioni|regole)\b"
    ),
    r"\bprompt di sistema\b",
    r"\b(da ora|adesso) sei\b",
    r"\bfai finta di essere\b",
    # Chat-template special tokens (Llama 3, ChatML, Llama 2)
    r"<\|[a-z_]+\|>",
    r"\[/?INST\]",
    r"<<SYS>>",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)


def normalize(text: str) -> str:
    """NFKC-normalize, drop control characters, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(" " if unicodedata.category(ch) == "Cc" else ch for ch in text)
    return " ".join(text.split())


def validate_question(question: str) -> str:
    """Return the normalized question, or raise ``InputRejectedError``."""
    question = normalize(question)
    if not question:
        raise InputRejectedError("The question is empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise InputRejectedError(
            f"The question is too long (max {MAX_QUESTION_LENGTH} characters)."
        )
    if _INJECTION_RE.search(question):
        raise InputRejectedError(
            "The question was blocked by the input guardrail: it looks like an attempt to "
            "change the assistant's instructions. Please ask about the maintenance manuals."
        )
    return question
