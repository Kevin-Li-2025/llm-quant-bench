#!/usr/bin/env python3
"""Run a documented LongBench subset against an 8K OpenAI-compatible endpoint.

This script can either use an already-running endpoint or start a vLLM command.
It intentionally keeps service control explicit: pass --serve-command if you
want the script to start the server, and pass --stop-existing-pid-file if you
want it to stop a known earlier service first.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
QUALITY_SCRIPT = REPO_ROOT / "scripts" / "run_quality_eval.py"


DEFAULT_TASKS = [
    "multifieldqa_en",
    "hotpotqa",
    "qasper",
    "passage_count",
    "lcc",
    "gov_report",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="qwen72b-awq-l20")
    parser.add_argument("--longbench-dir", required=True, type=Path)
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--max-per-task", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--longbench-max-context-chars", type=int, default=24000)
    parser.add_argument("--longbench-max-tokens", type=int, default=128)
    parser.add_argument("--serve-command", nargs="+")
    parser.add_argument("--serve-shell-command")
    parser.add_argument("--serve-log", type=Path)
    parser.add_argument("--started-pid-file", type=Path)
    parser.add_argument("--stop-existing-pid-file", type=Path)
    parser.add_argument("--health-timeout-s", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_url": args.base_url,
        "model": args.model,
        "tasks": args.tasks,
        "max_per_task": args.max_per_task,
        "concurrency": args.concurrency,
        "longbench_max_context_chars": args.longbench_max_context_chars,
        "longbench_max_tokens": args.longbench_max_tokens,
        "serve_command": args.serve_command,
        "serve_shell_command": args.serve_shell_command,
        "notes": "Use an 8K service configuration; 1024-context serving is invalid for LongBench claims.",
    }
    (args.out / "longbench_8k_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    eval_cmd = [
        sys.executable,
        str(QUALITY_SCRIPT),
        "--out",
        str(args.out / "eval"),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--benchmarks",
        "longbench",
        "--longbench-dir",
        str(args.longbench_dir),
        "--longbench-tasks",
        *args.tasks,
        "--max-per-task",
        str(args.max_per_task),
        "--concurrency",
        str(args.concurrency),
        "--longbench-max-context-chars",
        str(args.longbench_max_context_chars),
        "--longbench-max-tokens",
        str(args.longbench_max_tokens),
        "--sample-log-every",
        "20",
    ]
    (args.out / "eval_command.json").write_text(json.dumps(eval_cmd, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps({"manifest": manifest, "eval_command": eval_cmd}, indent=2))
        return 0

    if args.stop_existing_pid_file:
        stop_pid_file(args.stop_existing_pid_file)

    server_process = None
    serve_command = args.serve_command
    if args.serve_shell_command:
        serve_command = ["bash", "-lc", args.serve_shell_command]
    if serve_command:
        server_process = start_service(serve_command, args.serve_log, args.started_pid_file)
        wait_for_health(args.base_url, args.health_timeout_s)
    else:
        wait_for_health(args.base_url, 30.0)

    run_logged(eval_cmd, args.out / "eval.log")

    if server_process and server_process.poll() is None:
        server_process.terminate()
    return 0


def stop_pid_file(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    pid = int(text)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def start_service(
    command: list[str],
    log_path: Path | None,
    pid_file: Path | None,
) -> subprocess.Popen[str]:
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
    else:
        log_handle = subprocess.DEVNULL
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if pid_file:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(process.pid), encoding="utf-8")
    return process


def wait_for_health(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    url = base_url.rstrip("/") + "/models"
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(5)
    raise TimeoutError(f"Endpoint did not become healthy at {url}: {last_error}")


def run_logged(cmd: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
