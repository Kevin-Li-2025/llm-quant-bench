#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute energy and token-per-joule metrics from a load summary and nvidia-smi power log."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--power-log", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    samples = read_power_samples(args.power_log)
    if len(samples) < 2:
        raise SystemExit("power log must contain at least two samples")

    energy_j = integrate_power_joules(samples)
    duration_s = samples[-1][0] - samples[0][0]
    avg_power_w = energy_j / duration_s if duration_s > 0 else None
    output_tokens = int(summary["tokens"]["output_tokens"])
    prompt_tokens = int(summary["tokens"].get("prompt_tokens") or 0)
    total_tokens = output_tokens + prompt_tokens

    result = {
        "summary_path": str(args.summary),
        "power_log_path": str(args.power_log),
        "power_samples": len(samples),
        "power_duration_s": duration_s,
        "avg_power_w": avg_power_w,
        "energy_j": energy_j,
        "energy_wh": energy_j / 3600,
        "output_tokens": output_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "output_tokens_per_joule": safe_div(output_tokens, energy_j),
        "total_tokens_per_joule": safe_div(total_tokens, energy_j),
        "joules_per_output_token": safe_div(energy_j, output_tokens),
        "joules_per_total_token": safe_div(energy_j, total_tokens),
        "output_tokens_per_kwh": safe_div(output_tokens, energy_j / 3_600_000),
        "total_tokens_per_kwh": safe_div(total_tokens, energy_j / 3_600_000),
    }

    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def read_power_samples(path: Path) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            timestamp = parse_time(row[0].strip())
            try:
                power_w = float(row[1])
            except ValueError:
                continue
            samples.append((timestamp, power_w))
    samples.sort(key=lambda item: item[0])
    return samples


def parse_time(value: str) -> float:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(timezone.utc).timestamp()


def integrate_power_joules(samples: list[tuple[float, float]]) -> float:
    energy_j = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        dt = max(0.0, t1 - t0)
        energy_j += ((p0 + p1) / 2.0) * dt
    return energy_j


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())
