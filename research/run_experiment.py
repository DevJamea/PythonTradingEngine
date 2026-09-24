"""Master entry point: python -m research.run_experiment."""
from __future__ import annotations

from .runners.report import build_report
from .runners.run_all import main


def run() -> None:
    summary = main()
    path = build_report()
    print(f"report written: {path}")
    print(f"verdicts: {summary['verdicts']}")
    print(f"candidates: {summary['candidates']}")


if __name__ == "__main__":
    run()
