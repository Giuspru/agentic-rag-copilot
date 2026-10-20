"""Before/after: the linear RAG chain vs the self-corrective agent, side by side.

Usage:

    python scripts/compare.py                      # the built-in demo questions
    python scripts/compare.py "your question" ...  # your own questions

For each question it prints the linear chain's answer, then the agent's answer with the
path it took through the graph — which shows *why* the two can differ.
"""

import sys
import time

from copilot.agent.graph import run_agent
from copilot.rag.chain import answer

DEMO_QUESTIONS = [
    "What grease should I use for the bearings?",  # in the manual: both should answer
    "What is the tightening torque for the impeller nut?",  # not in the manual
    "la pompa perde dalla tenuta, cosa faccio?",  # Italian: needs rewrite + retries
]


def main() -> None:
    questions = sys.argv[1:] or DEMO_QUESTIONS
    for question in questions:
        print("=" * 80)
        print("Q:", question)

        start = time.perf_counter()
        linear = answer(question)
        print(f"\n[LINEAR] ({time.perf_counter() - start:.0f}s)\n{linear}")

        start = time.perf_counter()
        result = run_agent(question)
        print(f"\n[AGENT] ({time.perf_counter() - start:.0f}s)\n{result['answer']}")
        print("sources:", result["sources"])
        print("path:")
        for step in result["steps"]:
            print("  -", step)
        print()


if __name__ == "__main__":
    main()
