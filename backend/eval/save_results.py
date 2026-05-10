"""
Pretty-print evaluation CSVs under backend/results/.
No API server required.

Run from backend/:
    python eval/save_results.py
"""

from __future__ import annotations

import csv
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_BACKEND = _EVAL_DIR.parent
_RESULTS = _BACKEND / "results"


def print_csv(filepath: Path, title: str) -> None:
    if not filepath.exists():
        print(f"  ⚠ Not found: {filepath}")
        return

    with open(filepath, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("  ⚠ Empty file")
        return

    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")

    headers = list(rows[0].keys())
    print("  " + " | ".join(h[:18].ljust(18) for h in headers))
    print("  " + "-+-".join("-" * 18 for _ in headers))

    for row in rows:
        print("  " + " | ".join(str(row[h])[:18].ljust(18) for h in headers))

    print()


def main() -> None:
    print("\n═══ Clinical RAG — Evaluation Results ═══")
    print(f"Results dir: {_RESULTS}")
    print_csv(_RESULTS / "retrieval_metrics.csv", "Retrieval: Precision@K / Recall@K / F1@K (disease category)")
    print_csv(_RESULTS / "generation_scores.csv", "Generation: coherence / relevance / factual_grounding (1-5)")


if __name__ == "__main__":
    main()
