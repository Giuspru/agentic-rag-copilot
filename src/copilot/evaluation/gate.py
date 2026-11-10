"""Quality gate — turn the evaluation into a pass/fail check for CI.

``eval/thresholds.json`` names a split and, for the agent, a set of bounds:
``min_<metric>`` (lower bound) or ``max_<metric>`` (upper bound) on the summary of that
split. The thresholds are set below the measured baseline with a margin: a small model
is not perfectly reproducible across machines (CPU vs Apple GPU), and the gate must catch
regressions, not noise.
"""

import json
from pathlib import Path

from copilot.config import PROJECT_ROOT

THRESHOLDS_PATH = PROJECT_ROOT / "eval" / "thresholds.json"


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_gate(report: dict, thresholds: dict) -> list[str]:
    """Return the violated thresholds (an empty list means the gate passed)."""
    summary = report["systems"]["agent"]["summary"]["split"].get(thresholds["split"])
    if summary is None:
        return [f"no '{thresholds['split']}' questions were evaluated"]
    failures = []
    for key, bound in thresholds["agent"].items():
        kind, metric = key.split("_", 1)
        value = summary.get(metric)
        if value is None:
            failures.append(f"{metric}: not measured")
        elif (kind == "min" and value < bound) or (kind == "max" and value > bound):
            failures.append(f"{metric} = {value} ({kind} {bound})")
    return failures
