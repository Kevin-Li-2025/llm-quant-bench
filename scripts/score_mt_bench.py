#!/usr/bin/env python3
"""Score MT-Bench-style generations with an OpenAI-compatible judge.

The script supports two modes:

1. Single-answer scoring: pass --samples and a judge endpoint.
2. Pairwise scoring with order swapping: pass --baseline-samples and
   --candidate-samples. The judge is asked twice per question, once in each
   answer order, and the summary reports candidate win/tie/loss counts.

This is intentionally lightweight. For official MT-Bench leaderboard numbers,
use the FastChat judge scripts and judge model/version expected by that setup.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_quant_bench.client import ModelConfig, OpenAIChatClient  # noqa: E402


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-api-key")
    parser.add_argument("--samples", type=Path, help="Candidate MT-Bench samples.jsonl.")
    parser.add_argument("--baseline-samples", type=Path)
    parser.add_argument("--candidate-samples", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    judge = OpenAIChatClient(
        ModelConfig(
            name="judge",
            base_url=args.judge_base_url,
            model=args.judge_model,
            api_key=args.judge_api_key,
            temperature=0.0,
            max_tokens=args.judge_max_tokens,
            timeout_s=args.timeout_s,
        )
    )

    if args.samples:
        rows = run_single_score(args.samples, judge, args.concurrency)
        mode = "single"
    elif args.baseline_samples and args.candidate_samples:
        rows = run_pairwise_score(
            args.baseline_samples,
            args.candidate_samples,
            judge,
            args.concurrency,
        )
        mode = "pairwise"
    else:
        raise SystemExit("Use either --samples or both --baseline-samples and --candidate-samples.")

    judgments_path = args.out / "judgments.jsonl"
    with judgments_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize_judgments(rows, mode)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.out / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def run_single_score(samples_path: Path, judge: OpenAIChatClient, concurrency: int) -> list[dict[str, Any]]:
    samples = read_samples(samples_path)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(score_single_item, sample, judge) for sample in samples]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["item_id"])
    for row in rows:
        row["benchmark_elapsed_s"] = time.perf_counter() - started
    return rows


def run_pairwise_score(
    baseline_path: Path,
    candidate_path: Path,
    judge: OpenAIChatClient,
    concurrency: int,
) -> list[dict[str, Any]]:
    baseline = {row["item_id"]: row for row in read_samples(baseline_path)}
    candidate = {row["item_id"]: row for row in read_samples(candidate_path)}
    common_ids = sorted(set(baseline) & set(candidate))
    pairs = [(baseline[item_id], candidate[item_id]) for item_id in common_ids]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(score_pair_item, base, cand, judge) for base, cand in pairs]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["item_id"])
    return rows


def score_single_item(sample: dict[str, Any], judge: OpenAIChatClient) -> dict[str, Any]:
    prompt = single_judge_prompt(sample)
    result = judge.generate(prompt, stream=False)
    parsed = parse_judge_json(result.text)
    score = parsed.get("score")
    return {
        "item_id": sample["item_id"],
        "category": sample.get("task"),
        "ok": result.ok and isinstance(score, (int, float)),
        "score": float(score) if isinstance(score, (int, float)) else None,
        "rationale": parsed.get("rationale"),
        "judge_response": result.text,
        "judge_error": result.error,
    }


def score_pair_item(base: dict[str, Any], cand: dict[str, Any], judge: OpenAIChatClient) -> dict[str, Any]:
    forward = judge.generate(pairwise_judge_prompt(base, cand, swapped=False), stream=False)
    reverse = judge.generate(pairwise_judge_prompt(base, cand, swapped=True), stream=False)
    forward_parsed = parse_judge_json(forward.text)
    reverse_parsed = parse_judge_json(reverse.text)
    forward_winner = normalize_winner(forward_parsed.get("winner"), swapped=False)
    reverse_winner = normalize_winner(reverse_parsed.get("winner"), swapped=True)
    winner = combine_winners(forward_winner, reverse_winner)
    return {
        "item_id": cand["item_id"],
        "category": cand.get("task"),
        "ok": forward.ok and reverse.ok and winner is not None,
        "winner": winner,
        "forward_winner": forward_winner,
        "reverse_winner": reverse_winner,
        "judge_consistent": forward_winner == reverse_winner,
        "forward_response": forward.text,
        "reverse_response": reverse.text,
        "forward_error": forward.error,
        "reverse_error": reverse.error,
    }


def read_samples(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def single_judge_prompt(sample: dict[str, Any]) -> str:
    return (
        "You are an impartial MT-Bench evaluator. Score the assistant answer from 1 to 10.\n"
        "Use 10 for an excellent answer, 5 for a partially useful answer, and 1 for a wrong or unusable answer.\n"
        "Return only JSON with keys score and rationale.\n\n"
        f"Question:\n{sample.get('prompt', '')}\n\n"
        f"Assistant answer:\n{sample.get('response', '')}"
    )


def pairwise_judge_prompt(base: dict[str, Any], cand: dict[str, Any], *, swapped: bool) -> str:
    answer_a = cand.get("response", "") if swapped else base.get("response", "")
    answer_b = base.get("response", "") if swapped else cand.get("response", "")
    return (
        "You are an impartial pairwise MT-Bench evaluator. Compare Answer A and Answer B for the question.\n"
        "Return only JSON with key winner, where winner is A, B, or tie, plus a short rationale.\n\n"
        f"Question:\n{cand.get('prompt', '')}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}"
    )


def parse_judge_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(text)
        if not match:
            return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def normalize_winner(value: Any, *, swapped: bool) -> str | None:
    winner = str(value or "").strip().lower()
    if winner in {"tie", "draw", "equal"}:
        return "tie"
    if winner in {"a", "answer a"}:
        return "candidate" if swapped else "baseline"
    if winner in {"b", "answer b"}:
        return "baseline" if swapped else "candidate"
    if winner in {"candidate", "baseline"}:
        return winner
    return None


def combine_winners(forward: str | None, reverse: str | None) -> str | None:
    if forward is None and reverse is None:
        return None
    if forward == reverse:
        return forward
    if forward == "tie" or reverse == "tie":
        return "tie"
    if forward is None:
        return reverse
    if reverse is None:
        return forward
    return "tie"


def summarize_judgments(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    summary: dict[str, Any] = {
        "mode": mode,
        "items": len(rows),
        "ok": len(ok_rows),
        "failed": len(rows) - len(ok_rows),
    }
    if mode == "single":
        scores = [float(row["score"]) for row in ok_rows if row.get("score") is not None]
        summary["score_mean"] = statistics.fmean(scores) if scores else None
        summary["score_min"] = min(scores) if scores else None
        summary["score_max"] = max(scores) if scores else None
    else:
        wins = sum(1 for row in ok_rows if row.get("winner") == "candidate")
        ties = sum(1 for row in ok_rows if row.get("winner") == "tie")
        losses = sum(1 for row in ok_rows if row.get("winner") == "baseline")
        summary.update(
            {
                "candidate_wins": wins,
                "ties": ties,
                "candidate_losses": losses,
                "candidate_win_tie_rate": (wins + ties) / len(ok_rows) if ok_rows else None,
                "judge_consistency_rate": (
                    sum(1 for row in ok_rows if row.get("judge_consistent")) / len(ok_rows)
                    if ok_rows
                    else None
                ),
            }
        )
    return summary


def render_report(summary: dict[str, Any]) -> str:
    lines = ["# MT-Bench Judge Report", "", f"- Mode: {summary['mode']}", f"- Items: {summary['items']}", f"- OK: {summary['ok']}", f"- Failed: {summary['failed']}"]
    if summary["mode"] == "single":
        lines.append(f"- Mean judge score: {summary.get('score_mean')}")
    else:
        lines.extend(
            [
                f"- Candidate wins: {summary.get('candidate_wins')}",
                f"- Ties: {summary.get('ties')}",
                f"- Candidate losses: {summary.get('candidate_losses')}",
                f"- Candidate win/tie rate: {summary.get('candidate_win_tie_rate')}",
                f"- Judge consistency rate: {summary.get('judge_consistency_rate')}",
            ]
        )
    lines.extend(["", "This is an endpoint-based judge pass. For official MT-Bench reporting, use the judge model and prompts required by the FastChat MT-Bench setup."])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
