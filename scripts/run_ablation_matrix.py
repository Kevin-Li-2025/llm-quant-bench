#!/usr/bin/env python3
"""Run runtime or quantization ablation matrices from a JSON manifest.

The manifest can describe vLLM, SGLang, llama.cpp/GGUF, AWQ, GPTQ, FP8, or any
other endpoint as long as it exposes OpenAI-compatible chat completions.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summaries: list[dict[str, Any]] = []
    for experiment in manifest.get("experiments", []):
        if args.dry_run:
            summaries.append(dry_run_summary(experiment))
            continue
        summaries.append(run_experiment(experiment, args.out))

    output = {"experiments": summaries}
    (args.out / "ablation_summary.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    (args.out / "ablation_report.md").write_text(render_report(output), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


def run_experiment(experiment: dict[str, Any], root_out: Path) -> dict[str, Any]:
    name = experiment["name"]
    out_dir = root_out / safe_name(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    process = None
    try:
        stop_pid_file(experiment.get("stop_existing_pid_file"))
        if experiment.get("serve_command"):
            process = start_service(experiment["serve_command"], out_dir / "serve.log")
        wait_for_health(experiment.get("health_url"), float(experiment.get("health_timeout_s", 600)))
        load_summary = run_load(experiment, out_dir)
        quality_summary = run_optional_quality(experiment, out_dir)
        return {
            "name": name,
            "runtime": experiment.get("runtime"),
            "quant": experiment.get("quant"),
            "model": experiment.get("model", {}).get("model"),
            "status": "ok",
            "load_summary": load_summary,
            "quality_summary": quality_summary,
        }
    finally:
        if process and process.poll() is None:
            process.terminate()


def dry_run_summary(experiment: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": experiment.get("name"),
        "runtime": experiment.get("runtime"),
        "quant": experiment.get("quant"),
        "serve_command": experiment.get("serve_command"),
        "load": experiment.get("load"),
        "quality": experiment.get("quality"),
        "status": "dry_run",
    }


def run_load(experiment: dict[str, Any], out_dir: Path) -> dict[str, Any] | None:
    load = experiment.get("load")
    if not load:
        return None
    cmd = [
        "python3",
        "-m",
        "llm_quant_bench",
        "load",
        "--config",
        write_load_config(experiment, out_dir),
        "--dataset",
        load["dataset"],
        "--out",
        str(out_dir / "load"),
        "--concurrency",
        str(load.get("concurrency", 1)),
    ]
    if load.get("requests") is not None:
        cmd.extend(["--requests", str(load["requests"])])
    if load.get("duration_seconds") is not None:
        cmd.extend(["--duration-seconds", str(load["duration_seconds"])])
    if load.get("stream_usage", False):
        cmd.append("--stream-usage")
    if not load.get("stream", True):
        cmd.append("--no-stream")
    run_logged(cmd, out_dir / "load.log")
    summary_path = out_dir / "load" / "load_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None


def run_optional_quality(experiment: dict[str, Any], out_dir: Path) -> dict[str, Any] | None:
    quality = experiment.get("quality")
    if not quality:
        return None
    cmd = [
        "python3",
        "scripts/run_quality_eval.py",
        "--out",
        str(out_dir / "quality"),
        "--base-url",
        experiment["model"]["base_url"],
        "--model",
        experiment["model"]["model"],
        "--benchmarks",
        *quality["benchmarks"],
        "--concurrency",
        str(quality.get("concurrency", 1)),
    ]
    if quality.get("cmmlu_dir"):
        cmd.extend(["--cmmlu-dir", quality["cmmlu_dir"]])
    if quality.get("longbench_dir"):
        cmd.extend(["--longbench-dir", quality["longbench_dir"]])
    if quality.get("longbench_tasks"):
        cmd.extend(["--longbench-tasks", *quality["longbench_tasks"]])
    if quality.get("max_per_task") is not None:
        cmd.extend(["--max-per-task", str(quality["max_per_task"])])
    if quality.get("limit") is not None:
        cmd.extend(["--limit", str(quality["limit"])])
    run_logged(cmd, out_dir / "quality.log")
    summary_path = out_dir / "quality" / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None


def write_load_config(experiment: dict[str, Any], out_dir: Path) -> str:
    model = dict(experiment["model"])
    data = {
        "baseline": model,
        "candidate": model,
        "targets": {},
    }
    path = out_dir / "load_config.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def wait_for_health(url: str | None, timeout_s: float) -> None:
    if not url:
        return
    deadline = time.monotonic() + timeout_s
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


def start_service(command: list[str], log_path: Path) -> subprocess.Popen[str]:
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)


def stop_pid_file(path_value: str | None) -> None:
    if not path_value:
        return
    path = Path(path_value)
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    try:
        os.kill(int(text), signal.SIGTERM)
    except ProcessLookupError:
        return


def run_logged(cmd: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Ablation Matrix Report",
        "",
        "| Name | Runtime | Quant | Status | Output tok/s | Success Rate | Quality |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in summary["experiments"]:
        load = row.get("load_summary") or {}
        quality = row.get("quality_summary") or {}
        output_tps = (load.get("tokens") or {}).get("output_token_throughput")
        success = (load.get("requests") or {}).get("success_rate")
        quality_score = None
        if quality.get("by_benchmark"):
            scores = [
                item.get("score_mean")
                for item in quality["by_benchmark"].values()
                if item.get("score_mean") is not None
            ]
            quality_score = sum(scores) / len(scores) if scores else None
        lines.append(
            f"| {row.get('name')} | {row.get('runtime')} | {row.get('quant')} | {row.get('status')} | "
            f"{fmt(output_tps)} | {fmt(success)} | {fmt(quality_score)} |"
        )
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in name)


if __name__ == "__main__":
    raise SystemExit(main())
