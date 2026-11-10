"""The evaluation dataset — questions with the behavior and the facts we expect.

Each line of ``eval/dataset.jsonl`` is one ``EvalItem``. Three expected behaviors:

  - ``answer``: the manual contains the answer. ``facts`` lists what the answer must
    contain: a list of *groups*, every group must match, and a group matches if any of
    its alternatives does (``[["nlgi", "lithium"], ["15"]]`` = (nlgi OR lithium) AND 15);
  - ``refuse``: the manual does not contain the answer (off-topic, or on-topic but not
    covered): the copilot must say it does not know;
  - ``direct``: small talk: the agent must answer without retrieval.

Three splits, to keep the evaluation honest:

  - ``dev``: the questions used to *tune* prompts (phase 3 and 4). Scores on them are
    optimistic by construction — the prompts were chosen because they did well there;
  - ``test``: held out from tuning. It measured the phase-3 agent (the baseline); its
    failures were then *inspected* for error analysis, so after phase-4 changes it is no
    longer strictly unseen. It is the split the CI gate checks;
  - ``holdout``: written before any phase-4 change and evaluated once, at the end: the
    unbiased estimate of the final agent.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from copilot.config import PROJECT_ROOT

DATASET_PATH = PROJECT_ROOT / "eval" / "dataset.jsonl"


class EvalItem(BaseModel):
    id: str
    split: Literal["dev", "test", "holdout"]
    category: Literal[
        "fact", "multi_section", "italian", "out_of_corpus", "unanswerable", "smalltalk"
    ]
    question: str
    expected: Literal["answer", "refuse", "direct"]
    facts: list[list[str]] = []
    source: str | None = None
    reference: str = ""


def load_dataset(path: Path = DATASET_PATH, split: str = "all") -> list[EvalItem]:
    """Load (and validate) the dataset; ``split`` is ``dev``, ``test``, ``holdout`` or ``all``."""
    with path.open(encoding="utf-8") as f:
        items = [EvalItem(**json.loads(line)) for line in f if line.strip()]
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ids in the evaluation dataset.")
    return [item for item in items if split == "all" or item.split == split]
