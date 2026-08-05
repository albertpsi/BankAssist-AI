"""Run the Lab 6 golden dataset against the real, running application.

Requires a real OPENAI_API_KEY (and, for RAG cases, a populated Pinecone
index) — this is the live counterpart to `pytest`'s stubbed evaluation
tests. If `AGENTOPS_ENABLED=true`, each case's chat/resume calls show up as
a session in the AgentOps dashboard, correlated to this report by trace id
(Lab 6 §17).

    python scripts/run_evaluation.py [--out evaluation/report.md]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.executor import GraphExecutor  # noqa: E402
from evaluation.report import render_markdown  # noqa: E402
from evaluation.runner import run_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("evaluation/report.md"), help="Where to write the report."
    )
    args = parser.parse_args()

    executor = GraphExecutor()
    report = run_evaluation(executor)
    markdown = render_markdown(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
