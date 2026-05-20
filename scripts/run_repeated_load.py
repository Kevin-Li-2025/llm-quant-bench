#!/usr/bin/env python3
"""Run fixed-shape load tests repeatedly and summarize confidence intervals."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_repeats_ci import render_report, summarize_repeats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--concurrencies", nargs="+", required=True, type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--stream-usage", action="store_true")
    parser.add_argument("--sleep-between-s", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.requests is None and args.duration_seconds is None:
        raise SystemExit("Use --requests or --duration-seconds.")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": str(args.config),
        "dataset": str(args.dataset),
        "concurrencies": args.concurrencies,
        "repeats": args.repeats,
        "requests": args.requests,
        "duration_seconds": args.duration_seconds,
        "stream": not args.no_stream,
        "stream_usage": args.stream_usage,
        "created_at_unix": time.time(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for concurrency in args.concurrencies:
        for repeat_index in range(1, args.repeats + 1):
            run_dir = args.out / f"c{concurrency}" / f"r{repeat_index}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-m",
                "llm_quant_bench",
                "load",
                "--config",
                str(args.config),
                "--dataset",
                str(args.dataset),
                "--out",
                str(run_dir),
                "--concurrency",
                str(concurrency),
            ]
            if args.requests is not None:
                cmd.extend(["--requests", str(args.requests)])
            if args.duration_seconds is not None:
                cmd.extend(["--duration-seconds", str(args.duration_seconds)])
            if args.no_stream:
                cmd.append("--no-stream")
            if args.stream_usage:
                cmd.append("--stream-usage")
            run_logged(cmd, run_dir / "run.log")
            summary = json.loads((run_dir / "load_summary.json").read_text(encoding="utf-8"))
            summary["repeat_index"] = repeat_index
            summary["concurrency"] = concurrency
            rows.append(summary)
            if args.sleep_between_s > 0:
                time.sleep(args.sleep_between_s)

    (args.out / "run_summaries.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    ci_summary = summarize_repeats(rows, group_by="concurrency")
    (args.out / "repeated_load_summary.json").write_text(
        json.dumps(ci_summary, indent=2),
        encoding="utf-8",
    )
    (args.out / "repeated_load_report.md").write_text(render_report(ci_summary), encoding="utf-8")
    print(json.dumps(ci_summary, indent=2))
    return 0


def run_logged(cmd: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
