"""Evaluate the copilot on the labeled dataset (``eval/dataset.jsonl``).

Usage:

    python scripts/evaluate.py                      # agent + linear, all questions
    python scripts/evaluate.py --system agent       # one system only
    python scripts/evaluate.py --split test         # only the held-out questions
    python scripts/evaluate.py --ids test-18 dev-02 # specific questions (debugging)
    python scripts/evaluate.py --system agent --gate  # CI: fail if below thresholds
    python scripts/evaluate.py --rescore eval/results/report.json  # new metrics, no LLM

Writes ``report.json`` (every answer, step and metric) and ``report.md`` (summary) to
``eval/results/`` (or ``--out``). With ``--gate`` the agent's scores on the test split
are checked against ``eval/thresholds.json`` and the exit code is 1 if any is missed:
that is what turns the evaluation into a quality gate in CI.
"""

import argparse
import json
import sys
from pathlib import Path

from copilot.config import PROJECT_ROOT
from copilot.evaluation.dataset import load_dataset
from copilot.evaluation.gate import check_gate, load_thresholds
from copilot.evaluation.runner import SYSTEMS, rescore, run, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--system", choices=[*SYSTEMS, "both"], default="both")
    parser.add_argument("--split", choices=["dev", "test", "holdout", "all"], default="all")
    parser.add_argument("--ids", nargs="*", help="evaluate only these item ids")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval" / "results")
    parser.add_argument("--gate", action="store_true", help="exit 1 if below thresholds")
    parser.add_argument(
        "--rescore", type=Path, help="re-score the answers saved in this report.json (no LLM)"
    )
    args = parser.parse_args()

    items = load_dataset(split=args.split)
    if args.ids:
        items = [item for item in items if item.id in set(args.ids)]
    systems = SYSTEMS if args.system == "both" else (args.system,)

    if args.rescore:
        report = rescore(json.loads(args.rescore.read_text(encoding="utf-8")), load_dataset())
    else:
        report = run(items, systems, progress=lambda line: print(line, flush=True))
    json_path, md_path = write_report(report, args.out)
    print(f"\nReport: {md_path}\n        {json_path}\n")
    print(md_path.read_text(encoding="utf-8").split("## Accuracy per category")[0])

    if args.gate:
        if "agent" not in report["systems"]:
            print("GATE: --gate needs the agent to be evaluated.")
            return 2
        failures = check_gate(report, load_thresholds())
        if failures:
            print("GATE FAILED:\n  - " + "\n  - ".join(failures))
            return 1
        print("GATE PASSED: every threshold in eval/thresholds.json is met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
