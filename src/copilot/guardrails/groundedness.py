"""Output guardrail — a deterministic groundedness check on the answer.

The LLM judge in ``check_answer`` asks "is this answer supported by the excerpts?". It
is good at paraphrase and bad at exactness: a 3B model can easily accept "every 5000
hours" when the manual says 4000. This module checks precisely the facts where an error
is most dangerous in a maintenance context, and where a string comparison is reliable:

  - **numbers** (intervals, temperatures, torques, quantities...);
  - **part codes** (e.g. ``SP-CP2000-SEAL``);
  - **cited file names**: each must be exactly one of the ``[Source: ...]`` documents of
    the excerpts (``manual.md`` is not ``centrifugal-pump-maintenance-manual.md``).

Every such token in the answer must appear in the excerpts, otherwise it is reported as
unsupported. It runs *before* the LLM judges: it is instant, free, and when it fails the
judges need not be called at all.

Numbers are compared after removing ``.`` and ``,``, so ``4000``, ``4,000`` and the
Italian ``4.000`` all match, and so do ``0.05`` and ``0,05``. The comparison is
intentionally lenient in that direction (a rare false match is acceptable); what it must
never do is let through a number that is not in the manual at all.
"""

import re

_SOURCE_RE = re.compile(r"\[Source: ([^\]]+)\]")  # the headers written by format_docs()
_FILE_RE = re.compile(r"\b[\w-]+(?:\.[\w-]+)*\.(?:md|pdf|txt)\b", re.IGNORECASE)
_CODE_RE = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*(?:-[A-Z0-9]+)+\b")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
# Numbered-list markers ("1. ", "2) ") are formatting, not claims.
_LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)


def _numbers(text: str) -> set[str]:
    return {re.sub(r"[.,]", "", n) for n in _NUMBER_RE.findall(text)}


def unsupported_claims(answer: str, context: str) -> list[str]:
    """Return the numbers, part codes and file names in ``answer`` absent from ``context``.

    An empty list means the check passed.
    """
    unsupported: list[str] = []

    given = set(_SOURCE_RE.findall(context))
    files = _FILE_RE.findall(answer)
    unsupported += [f for f in dict.fromkeys(files) if f not in given]

    # File names are checked above; remove them so "manual-v2.md" adds no number "2".
    claims = _FILE_RE.sub(" ", answer)
    unsupported += [c for c in dict.fromkeys(_CODE_RE.findall(claims)) if c not in context]

    claims = _LIST_MARKER_RE.sub(" ", _CODE_RE.sub(" ", claims))
    known = _numbers(context)
    for raw in dict.fromkeys(_NUMBER_RE.findall(claims)):
        if re.sub(r"[.,]", "", raw) not in known:
            unsupported.append(raw)

    return unsupported
