#!/usr/bin/env python3
"""Summarize baseline-vs-candidate quality retention from quality summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate_summary).read_text(encoding="utf-8"))
    rows = []
    for benchmark, base_row in baseline.get("by_benchmark", {}).items():
        cand_row = candidate.get("by_benchmark", {}).get(benchmark)
        if not cand_row:
            continue
        base_score = base_row.get("score_mean")
        cand_score = cand_row.get("score_mean")
        retention = compute_retention(cand_score, base_score)
        delta_pp = None
        if cand_score is not None and base_score is not None:
            delta_pp = 100.0 * (cand_score - base_score)
        rows.append(
            {
                "benchmark": benchmark,
                "baseline_score": base_score,
                "candidate_score": cand_score,
                "quality_retention": retention,
                "delta_percentage_points": delta_pp,
                "baseline_items": base_row.get("scored_items"),
                "candidate_items": cand_row.get("scored_items"),
            }
        )
    output = {
        "formula": "quality_retention = candidate_score / baseline_score",
        "delta_percentage_points_formula": "100 * (candidate_score - baseline_score)",
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


def compute_retention(candidate_score: Any, baseline_score: Any) -> float | None:
    if candidate_score is None or baseline_score in (None, 0):
        return None
    return float(candidate_score) / float(baseline_score)


if __name__ == "__main__":
    main()
