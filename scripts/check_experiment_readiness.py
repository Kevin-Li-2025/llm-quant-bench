#!/usr/bin/env python3
"""Check whether an experiment matrix has the required endpoints, files, and runtimes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--check-health", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [
        check_experiment(experiment, check_health=args.check_health)
        for experiment in manifest.get("experiments", [])
    ]
    summary = {
        "manifest": str(args.manifest),
        "ready": sum(1 for row in rows if row["status"] == "ready"),
        "blocked": sum(1 for row in rows if row["status"] == "blocked"),
        "experiments": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.out.with_suffix(".md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["blocked"] == 0 else 1


def check_experiment(experiment: dict[str, Any], *, check_health: bool) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    requirements = infer_requirements(experiment)

    for command in requirements["commands"]:
        if shutil.which(command) is None:
            failures.append(f"missing command: {command}")

    for module in requirements["python_modules"]:
        if not python_module_exists(module):
            failures.append(f"missing python module: {module}")

    for path in requirements["paths"]:
        if not Path(path).exists():
            failures.append(f"missing path: {path}")

    for env_name in requirements["env"]:
        if not env_is_set(env_name):
            failures.append(f"missing environment variable: {env_name}")

    if check_health and experiment.get("health_url"):
        if not health_ok(str(experiment["health_url"])):
            warnings.append(f"health check failed: {experiment['health_url']}")

    return {
        "name": experiment.get("name"),
        "runtime": experiment.get("runtime"),
        "quant": experiment.get("quant"),
        "status": "blocked" if failures else "ready",
        "failures": failures,
        "warnings": warnings,
        "requirements": requirements,
    }


def infer_requirements(experiment: dict[str, Any]) -> dict[str, list[str]]:
    explicit = experiment.get("requires") or {}
    commands = list(explicit.get("commands") or [])
    modules = list(explicit.get("python_modules") or [])
    paths = list(explicit.get("paths") or [])
    env = list(explicit.get("env") or [])

    runtime = str(experiment.get("runtime") or "").lower()
    command_text = " ".join(str(part) for part in experiment.get("serve_command") or [])
    if "vllm" in runtime or "vllm" in command_text:
        if not has_command(commands, "vllm"):
            commands.append("vllm")
    if "sglang" in runtime or "sglang" in command_text:
        if "sglang" not in modules:
            modules.append("sglang")
    if "llama.cpp" in runtime or "llama-server" in command_text:
        if "llama-server" not in commands:
            commands.append("llama-server")

    for token in command_text.split():
        cleaned = token.strip("'\"")
        if cleaned.startswith("/") and not cleaned.startswith("/v1"):
            if any(cleaned.endswith(suffix) for suffix in (".json", ".jsonl", ".gguf")) or "/models/" in cleaned:
                paths.append(cleaned)

    return {
        "commands": sorted(set(commands)),
        "python_modules": sorted(set(modules)),
        "paths": sorted(set(paths)),
        "env": sorted(set(env)),
    }


def has_command(commands: list[str], executable_name: str) -> bool:
    return any(Path(command).name == executable_name for command in commands)


def python_module_exists(module: str) -> bool:
    code = f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec({module!r}) else 1)"
    return subprocess.run([sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def env_is_set(name: str) -> bool:
    import os

    return bool(os.environ.get(name))


def health_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status < 500
    except (urllib.error.URLError, TimeoutError):
        return False


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Experiment Readiness",
        "",
        f"- Ready: {summary['ready']}",
        f"- Blocked: {summary['blocked']}",
        "",
        "| Experiment | Runtime | Quant | Status | Failures |",
        "|---|---|---|---|---|",
    ]
    for row in summary["experiments"]:
        failures = "; ".join(row["failures"]) if row["failures"] else ""
        lines.append(
            f"| {row.get('name')} | {row.get('runtime')} | {row.get('quant')} | {row.get('status')} | {failures} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
