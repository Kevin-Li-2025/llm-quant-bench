from __future__ import annotations

import concurrent.futures
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .client import ModelConfig, OpenAIChatClient, estimate_tokens
from .metrics import (
    compare_candidate_to_baseline,
    geometric_usability_score,
    quantile,
    safe_ratio,
    score_with_rules,
    similarity,
)


@dataclass
class BenchmarkTargets:
    min_quality_retention: float = 0.98
    min_success_rate: float = 0.995
    max_weakly_scored_rate: float = 0.0
    max_judge_inconsistency_rate: float = 0.05
    max_judge_unvalidated_rate: float = 0.0
    max_p95_ttft_s: float | None = None
    max_p95_inter_token_latency_s: float | None = None
    max_p95_time_per_output_token_s: float | None = None
    min_p05_tokens_per_second: float | None = None
    target_context_tokens: int = 8192
    min_context_success_rate: float = 0.99

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BenchmarkTargets":
        data = data or {}
        return cls(
            min_quality_retention=float(data.get("min_quality_retention", 0.98)),
            min_success_rate=float(data.get("min_success_rate", 0.995)),
            max_weakly_scored_rate=float(data.get("max_weakly_scored_rate", 0.0)),
            max_judge_inconsistency_rate=float(
                data.get("max_judge_inconsistency_rate", 0.05)
            ),
            max_judge_unvalidated_rate=float(
                data.get("max_judge_unvalidated_rate", 0.0)
            ),
            max_p95_ttft_s=(
                None
                if data.get("max_p95_ttft_s") is None
                else float(data["max_p95_ttft_s"])
            ),
            max_p95_inter_token_latency_s=(
                None
                if data.get("max_p95_inter_token_latency_s") is None
                else float(data["max_p95_inter_token_latency_s"])
            ),
            max_p95_time_per_output_token_s=(
                None
                if data.get("max_p95_time_per_output_token_s") is None
                else float(data["max_p95_time_per_output_token_s"])
            ),
            min_p05_tokens_per_second=(
                None
                if data.get("min_p05_tokens_per_second") is None
                else float(data["min_p05_tokens_per_second"])
            ),
            target_context_tokens=int(data.get("target_context_tokens", 8192)),
            min_context_success_rate=float(data.get("min_context_success_rate", 0.99)),
        )


@dataclass
class BenchmarkConfig:
    baseline: ModelConfig
    candidate: ModelConfig
    targets: BenchmarkTargets
    judge: ModelConfig | None = None

    @classmethod
    def from_path(cls, path: Path) -> "BenchmarkConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        judge_data = data.get("judge") or {}
        judge = None
        if judge_data.get("enabled"):
            judge = ModelConfig.from_dict(judge_data, "judge")
        return cls(
            baseline=ModelConfig.from_dict(data["baseline"], "baseline"),
            candidate=ModelConfig.from_dict(data["candidate"], "candidate"),
            targets=BenchmarkTargets.from_dict(data.get("targets")),
            judge=judge,
        )


def load_dataset(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample.setdefault("id", f"line-{line_no}")
            sample.setdefault("category", "default")
            if "prompt" not in sample:
                raise ValueError(f"{path}:{line_no} is missing required field 'prompt'")
            samples.append(sample)
    if not samples:
        raise ValueError(f"{path} does not contain any benchmark samples")
    return samples


def run_benchmark(
    *,
    config: BenchmarkConfig,
    dataset: list[dict[str, Any]],
    out_dir: Path,
    repeats: int,
    concurrency: int,
    stream: bool,
    duration_s: float | None = None,
) -> dict[str, Any]:
    if not dataset:
        raise ValueError("dataset must contain at least one sample")
    if duration_s is None and repeats < 1:
        raise ValueError("repeats must be >= 1 when duration_s is not set")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be > 0")

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    report_path = out_dir / "report.md"
    raw_records: list[dict[str, Any]] = []

    started = time.time()
    monotonic_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        with results_path.open("w", encoding="utf-8") as handle:
            repeat = 1
            while True:
                if duration_s is None and repeat > repeats:
                    break
                if duration_s is not None and time.monotonic() - monotonic_started >= duration_s:
                    break

                futures = [
                    pool.submit(_run_one_pair, config, sample, repeat, stream)
                    for sample in dataset
                ]
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    raw_records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                repeat += 1

    finished = time.time()
    summary = summarize_records(
        raw_records,
        config.targets,
        benchmark_duration_s=finished - started,
    )
    summary["started_at_epoch_s"] = started
    summary["finished_at_epoch_s"] = finished
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(render_report(summary, config.targets), encoding="utf-8")
    return summary


def summarize_file(results_path: Path, targets: BenchmarkTargets) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return summarize_records(records, targets)


def summarize_records(
    records: list[dict[str, Any]],
    targets: BenchmarkTargets,
    benchmark_duration_s: float | None = None,
) -> dict[str, Any]:
    total = len(records)
    candidate = [record["candidate"] for record in records]
    baseline = [record["baseline"] for record in records]

    candidate_ok = [item for item in candidate if item["ok"]]
    baseline_ok = [item for item in baseline if item["ok"]]
    wins = sum(1 for record in records if record["comparison"]["result"] == "win")
    ties = sum(1 for record in records if record["comparison"]["result"] == "tie")
    losses = sum(1 for record in records if record["comparison"]["result"] == "lose")

    baseline_quality = _avg([item["quality_score"] for item in baseline])
    candidate_quality = _avg([item["quality_score"] for item in candidate])
    quality_retention_raw = safe_ratio(candidate_quality, baseline_quality)
    quality_retention = min(quality_retention_raw, 1.0)
    stability_rate = safe_ratio(len(candidate_ok), total)
    context_records = [
        record
        for record in records
        if int(record.get("context_tokens") or 0) >= targets.target_context_tokens
    ]
    context_success_rate = (
        safe_ratio(
            sum(1 for record in context_records if record["candidate"]["ok"]),
            len(context_records),
        )
        if context_records
        else 1.0
    )
    performance_attainment_rate = _performance_attainment(candidate_ok, targets)
    usability_score = geometric_usability_score(
        quality_retention=quality_retention,
        stability_rate=stability_rate,
        performance_attainment_rate=performance_attainment_rate,
        context_success_rate=context_success_rate,
    )

    candidate_ttft = _present([item.get("ttft_s") for item in candidate_ok])
    candidate_itl = _present(
        [item.get("inter_token_latency_s") for item in candidate_ok]
    )
    candidate_tpot = _present(
        [item.get("time_per_output_token_s") for item in candidate_ok]
    )
    candidate_tps = _present([item.get("tokens_per_second") for item in candidate_ok])
    candidate_latency = _present([item.get("latency_s") for item in candidate_ok])
    judged_records = [
        record for record in records if record["comparison"].get("judge_used")
    ]
    inconsistent_judges = [
        record
        for record in judged_records
        if record["comparison"].get("judge_consistent") is False
    ]
    unvalidated_judges = [
        record
        for record in judged_records
        if record["comparison"].get("judge_consistent") is None
    ]
    weakly_scored_records = [
        record
        for record in records
        if not record["candidate"].get("objective_score")
        and not record["comparison"].get("judge_used")
    ]
    objective_rule_rate = safe_ratio(
        sum(1 for record in records if record["candidate"].get("objective_score")),
        total,
    )
    judge_rate = safe_ratio(len(judged_records), total)
    weakly_scored_rate = safe_ratio(len(weakly_scored_records), total)
    judge_inconsistency_rate = (
        safe_ratio(len(inconsistent_judges), len(judged_records))
        if judged_records
        else 0.0
    )
    judge_unvalidated_rate = (
        safe_ratio(len(unvalidated_judges), len(judged_records))
        if judged_records
        else 0.0
    )
    total_candidate_output_tokens = sum(
        int(item.get("output_tokens") or 0) for item in candidate_ok
    )
    output_token_throughput = (
        total_candidate_output_tokens / benchmark_duration_s
        if benchmark_duration_s and benchmark_duration_s > 0
        else None
    )
    request_throughput = (
        len(candidate_ok) / benchmark_duration_s
        if benchmark_duration_s and benchmark_duration_s > 0
        else None
    )

    return {
        "total_pairs": total,
        "baseline": {
            "success_rate": safe_ratio(len(baseline_ok), total),
            "quality_score": baseline_quality,
        },
        "candidate": {
            "success_rate": stability_rate,
            "quality_score": candidate_quality,
            "severe_error_rate": safe_ratio(
                sum(1 for item in candidate if item.get("severe_error")), total
            ),
            "p50_ttft_s": quantile(candidate_ttft, 0.50),
            "p95_ttft_s": quantile(candidate_ttft, 0.95),
            "p50_inter_token_latency_s": quantile(candidate_itl, 0.50),
            "p95_inter_token_latency_s": quantile(candidate_itl, 0.95),
            "p50_time_per_output_token_s": quantile(candidate_tpot, 0.50),
            "p95_time_per_output_token_s": quantile(candidate_tpot, 0.95),
            "p50_latency_s": quantile(candidate_latency, 0.50),
            "p95_latency_s": quantile(candidate_latency, 0.95),
            "p05_tokens_per_second": quantile(candidate_tps, 0.05),
            "p50_tokens_per_second": quantile(candidate_tps, 0.50),
            "output_token_throughput": output_token_throughput,
            "request_throughput": request_throughput,
        },
        "comparison": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_tie_rate": safe_ratio(wins + ties, total),
            "loss_rate": safe_ratio(losses, total),
            "quality_retention_raw": quality_retention_raw,
            "quality_retention_capped": quality_retention,
        },
        "service": {
            "stability_rate": stability_rate,
            "performance_attainment_rate": performance_attainment_rate,
            "context_success_rate": context_success_rate,
            "usability_score": usability_score,
            "benchmark_duration_s": benchmark_duration_s,
        },
        "scoring": {
            "objective_rule_rate": objective_rule_rate,
            "judge_rate": judge_rate,
            "weakly_scored_rate": weakly_scored_rate,
            "judge_inconsistency_rate": judge_inconsistency_rate,
            "judge_unvalidated_rate": judge_unvalidated_rate,
        },
        "by_category": _summarize_by_category(records),
        "targets": asdict(targets),
        "passed": {
            "quality_retention": quality_retention >= targets.min_quality_retention,
            "success_rate": stability_rate >= targets.min_success_rate,
            "weakly_scored_rate": weakly_scored_rate
            <= targets.max_weakly_scored_rate,
            "judge_inconsistency_rate": judge_inconsistency_rate
            <= targets.max_judge_inconsistency_rate,
            "judge_unvalidated_rate": judge_unvalidated_rate
            <= targets.max_judge_unvalidated_rate,
            "context_success_rate": context_success_rate
            >= targets.min_context_success_rate,
            "p95_ttft_s": (
                True
                if targets.max_p95_ttft_s is None
                else (
                    quantile(candidate_ttft, 0.95) is not None
                    and quantile(candidate_ttft, 0.95) <= targets.max_p95_ttft_s
                )
            ),
            "p95_inter_token_latency_s": (
                True
                if targets.max_p95_inter_token_latency_s is None
                else (
                    quantile(candidate_itl, 0.95) is not None
                    and quantile(candidate_itl, 0.95)
                    <= targets.max_p95_inter_token_latency_s
                )
            ),
            "p95_time_per_output_token_s": (
                True
                if targets.max_p95_time_per_output_token_s is None
                else (
                    quantile(candidate_tpot, 0.95) is not None
                    and quantile(candidate_tpot, 0.95)
                    <= targets.max_p95_time_per_output_token_s
                )
            ),
            "p05_tokens_per_second": (
                True
                if targets.min_p05_tokens_per_second is None
                else (
                    quantile(candidate_tps, 0.05) is not None
                    and quantile(candidate_tps, 0.05)
                    >= targets.min_p05_tokens_per_second
                )
            ),
        },
    }


def render_report(summary: dict[str, Any], targets: BenchmarkTargets) -> str:
    candidate = summary["candidate"]
    comparison = summary["comparison"]
    service = summary["service"]
    scoring = summary["scoring"]
    passed = summary["passed"]

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.2f}%"

    def num(value: float | None, suffix: str = "") -> str:
        return "n/a" if value is None else f"{value:.3f}{suffix}"

    lines = [
        "# Quantized LLM Benchmark Report",
        "",
        "## Headline",
        "",
        f"- Total prompt pairs: {summary['total_pairs']}",
        f"- Quality retention: {pct(comparison['quality_retention_capped'])} "
        f"(raw {pct(comparison['quality_retention_raw'])})",
        f"- Win + tie rate: {pct(comparison['win_tie_rate'])}",
        f"- Candidate success rate: {pct(candidate['success_rate'])}",
        f"- Context success rate: {pct(service['context_success_rate'])}",
        f"- Performance attainment rate: {pct(service['performance_attainment_rate'])}",
        f"- Composite usability score: {pct(service['usability_score'])}",
        f"- Weakly scored rate: {pct(scoring['weakly_scored_rate'])}",
        "",
        "## Candidate Serving",
        "",
        f"- p95 TTFT: {num(candidate['p95_ttft_s'], 's')}",
        f"- p95 inter-token latency: {num(candidate['p95_inter_token_latency_s'], 's')}",
        f"- p95 time per output token: {num(candidate['p95_time_per_output_token_s'], 's')}",
        f"- p95 latency: {num(candidate['p95_latency_s'], 's')}",
        f"- p05 output speed: {num(candidate['p05_tokens_per_second'], ' tok/s')}",
        f"- p50 output speed: {num(candidate['p50_tokens_per_second'], ' tok/s')}",
        f"- Output token throughput: {num(candidate['output_token_throughput'], ' tok/s')}",
        f"- Request throughput: {num(candidate['request_throughput'], ' req/s')}",
        f"- Severe error rate: {pct(candidate['severe_error_rate'])}",
        "",
        "Note: throughput in this report is observed during paired baseline/candidate evaluation. Use vLLM bench or GenAI-Perf for standalone serving capacity.",
        "",
        "## Scoring Confidence",
        "",
        f"- Objective rule rate: {pct(scoring['objective_rule_rate'])}",
        f"- Judge rate: {pct(scoring['judge_rate'])}",
        f"- Weakly scored rate: {pct(scoring['weakly_scored_rate'])}",
        f"- Judge inconsistency rate: {pct(scoring['judge_inconsistency_rate'])}",
        f"- Judge unvalidated rate: {pct(scoring['judge_unvalidated_rate'])}",
        "",
        "## Category Breakdown",
        "",
        "| Category | Pairs | Quality Retention | Win + Tie | Success | Severe Errors |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            "| {category} | {pairs} | {quality} | {win_tie} | {success} | {severe} |".format(
                category=category,
                pairs=item["pairs"],
                quality=pct(item["quality_retention_capped"]),
                win_tie=pct(item["win_tie_rate"]),
                success=pct(item["success_rate"]),
                severe=pct(item["severe_error_rate"]),
            )
            for category, item in sorted(summary.get("by_category", {}).items())
        ],
        "",
        "## Pass/Fail",
        "",
        f"- Quality retention >= {targets.min_quality_retention:.3f}: {_mark(passed['quality_retention'])}",
        f"- Success rate >= {targets.min_success_rate:.3f}: {_mark(passed['success_rate'])}",
        f"- Weakly scored rate <= {targets.max_weakly_scored_rate:.3f}: {_mark(passed['weakly_scored_rate'])}",
        f"- Judge inconsistency rate <= {targets.max_judge_inconsistency_rate:.3f}: {_mark(passed['judge_inconsistency_rate'])}",
        f"- Judge unvalidated rate <= {targets.max_judge_unvalidated_rate:.3f}: {_mark(passed['judge_unvalidated_rate'])}",
        f"- Context success >= {targets.min_context_success_rate:.3f}: {_mark(passed['context_success_rate'])}",
        f"- p95 TTFT target: {_mark(passed['p95_ttft_s'])}",
        f"- p95 inter-token latency target: {_mark(passed['p95_inter_token_latency_s'])}",
        f"- p95 time per output token target: {_mark(passed['p95_time_per_output_token_s'])}",
        f"- p05 tokens/s target: {_mark(passed['p05_tokens_per_second'])}",
        "",
        "## Formulas",
        "",
        "- Quality retention = min(mean(candidate_quality_score) / mean(baseline_quality_score), 1.0).",
        "- Win + tie rate = (candidate_wins + ties) / total_pairs.",
        "- Success rate = successful_candidate_requests / total_candidate_requests.",
        "- Weakly scored rate = samples with only non-empty scoring and no judge / total_pairs.",
        "- Judge inconsistency rate = position-swapped judge disagreements / judged_pairs.",
        "- Judge unvalidated rate = judged pairs where only one answer order returned a valid verdict / judged_pairs.",
        "- Inter-token latency approximates streaming chunk spacing; time per output token = decode_seconds / output_tokens.",
        "- Performance attainment = requests meeting TTFT, inter-token latency, time-per-output-token, and tokens/s targets / successful_candidate_requests.",
        "- Output token throughput = total candidate output tokens / benchmark wall-clock seconds.",
        "- Request throughput = successful candidate requests / benchmark wall-clock seconds.",
        "- Context success = successful long-context candidate requests / total long-context requests.",
        "- Composite usability = quality_retention * success_rate * performance_attainment * context_success.",
        "- Perplexity = exp(-mean(token_logprob)); PPL delta % = (candidate_ppl - baseline_ppl) / baseline_ppl * 100.",
    ]
    return "\n".join(lines) + "\n"


def _run_one_pair(
    config: BenchmarkConfig,
    sample: dict[str, Any],
    repeat: int,
    stream: bool,
) -> dict[str, Any]:
    baseline_client = OpenAIChatClient(config.baseline)
    candidate_client = OpenAIChatClient(config.candidate)
    prompt = str(sample["prompt"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        baseline_future = pool.submit(baseline_client.generate, prompt, stream=stream)
        candidate_future = pool.submit(candidate_client.generate, prompt, stream=stream)
        baseline_result = baseline_future.result()
        candidate_result = candidate_future.result()

    baseline_quality = score_with_rules(sample, baseline_result.text, ok=baseline_result.ok)
    candidate_quality = score_with_rules(
        sample, candidate_result.text, ok=candidate_result.ok
    )

    comparison = {
        "result": compare_candidate_to_baseline(
            baseline_quality.score,
            candidate_quality.score,
            baseline_result.text,
            candidate_result.text,
        ),
        "baseline_candidate_similarity": similarity(
            baseline_result.text,
            candidate_result.text,
        ),
        "judge_used": False,
    }

    if config.judge:
        comparison = (
            _judge_comparison(
                config.judge,
                sample,
                baseline_result.text,
                candidate_result.text,
            )
            or comparison
        )

    context_tokens = sample.get("context_tokens")
    if context_tokens is None:
        context_tokens = estimate_tokens(prompt)

    return {
        "sample_id": sample["id"],
        "category": sample.get("category", "default"),
        "repeat": repeat,
        "context_tokens": context_tokens,
        "baseline": _pack_generation(baseline_result, baseline_quality),
        "candidate": _pack_generation(candidate_result, candidate_quality),
        "comparison": comparison,
    }


def _judge_comparison(
    judge: ModelConfig,
    sample: dict[str, Any],
    baseline_text: str,
    candidate_text: str,
) -> dict[str, Any] | None:
    judge_client = OpenAIChatClient(judge)
    first = _judge_once(
        judge_client,
        sample,
        answer_a=baseline_text,
        answer_b=candidate_text,
    )
    swapped = _judge_once(
        judge_client,
        sample,
        answer_a=candidate_text,
        answer_b=baseline_text,
    )
    if first is None and swapped is None:
        return None

    first_result = (
        _winner_to_candidate_result(first["winner"], candidate_is_a=False)
        if first
        else None
    )
    swapped_result = (
        _winner_to_candidate_result(swapped["winner"], candidate_is_a=True)
        if swapped
        else None
    )
    if first_result is None:
        return {
            "result": swapped_result,
            "reason": swapped.get("reason", ""),
            "judge_used": True,
            "judge_position_swap": True,
            "judge_consistent": None,
            "first_result": None,
            "swapped_result": swapped_result,
        }
    if swapped_result is None:
        return {
            "result": first_result,
            "reason": first.get("reason", ""),
            "judge_used": True,
            "judge_position_swap": True,
            "judge_consistent": None,
            "first_result": first_result,
            "swapped_result": None,
        }
    if first_result == swapped_result:
        return {
            "result": first_result,
            "reason": first.get("reason", ""),
            "judge_used": True,
            "judge_position_swap": True,
            "judge_consistent": True,
            "first_result": first_result,
            "swapped_result": swapped_result,
        }
    return {
        "result": "tie",
        "reason": "Judge verdict changed after swapping answer positions.",
        "judge_used": True,
        "judge_position_swap": True,
        "judge_consistent": False,
        "first_result": first_result,
        "swapped_result": swapped_result,
    }


def _judge_once(
    judge_client: OpenAIChatClient,
    sample: dict[str, Any],
    *,
    answer_a: str,
    answer_b: str,
) -> dict[str, str] | None:
    prompt = f"""You are comparing two answers to the same user prompt.

Return strict JSON only:
{{"winner":"A|B|tie","reason":"short reason"}}

User prompt:
{sample["prompt"]}

Answer A:
{answer_a}

Answer B:
{answer_b}

Rules:
- Choose A only if A is materially better for the user's task.
- Choose B only if B is materially better for the user's task.
- Choose tie if they are equivalent, or if differences are only style, wording, or length.
- Do not prefer an answer because of position, verbosity, tone, or formatting unless it affects task correctness.
- Penalize factual errors, reasoning errors, broken formatting, unsafe behavior, and missing required content.
"""
    result = judge_client.generate(prompt, stream=False)
    if not result.ok:
        return None
    try:
        payload = json.loads(_strip_code_fence(result.text))
    except json.JSONDecodeError:
        return None
    winner = str(payload.get("winner", "")).lower()
    if winner not in {"a", "b", "tie"}:
        return None
    return {
        "winner": winner,
        "reason": str(payload.get("reason", "")),
    }


def _winner_to_candidate_result(winner: str, *, candidate_is_a: bool) -> str:
    if winner == "tie":
        return "tie"
    candidate_won = (winner == "a" and candidate_is_a) or (
        winner == "b" and not candidate_is_a
    )
    return "win" if candidate_won else "lose"


def _pack_generation(result: Any, score: Any) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "error": result.error,
        "text": result.text,
        "latency_s": result.latency_s,
        "ttft_s": result.ttft_s,
        "inter_token_latency_s": result.inter_token_latency_s,
        "time_per_output_token_s": result.time_per_output_token_s,
        "output_tokens": result.output_tokens,
        "prompt_tokens": result.prompt_tokens,
        "tokens_per_second": result.tokens_per_second,
        "quality_score": score.score,
        "checks": score.checks,
        "severe_error": score.severe_error,
        "objective_score": score.objective,
    }


def _performance_attainment(
    successful_candidate_results: list[dict[str, Any]],
    targets: BenchmarkTargets,
) -> float:
    if not successful_candidate_results:
        return 0.0
    passed = 0
    for item in successful_candidate_results:
        ok = True
        if targets.max_p95_ttft_s is not None:
            ttft = item.get("ttft_s")
            ok = ok and ttft is not None and ttft <= targets.max_p95_ttft_s
        if targets.max_p95_inter_token_latency_s is not None:
            itl = item.get("inter_token_latency_s")
            ok = (
                ok
                and itl is not None
                and itl <= targets.max_p95_inter_token_latency_s
            )
        if targets.max_p95_time_per_output_token_s is not None:
            tpot = item.get("time_per_output_token_s")
            ok = (
                ok
                and tpot is not None
                and tpot <= targets.max_p95_time_per_output_token_s
            )
        if targets.min_p05_tokens_per_second is not None:
            tps = item.get("tokens_per_second")
            ok = ok and tps is not None and tps >= targets.min_p05_tokens_per_second
        if ok:
            passed += 1
    return safe_ratio(passed, len(successful_candidate_results))


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _present(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _summarize_by_category(records: list[dict[str, Any]]) -> dict[str, Any]:
    categories = sorted({str(record.get("category", "default")) for record in records})
    out: dict[str, Any] = {}
    for category in categories:
        subset = [record for record in records if record.get("category", "default") == category]
        baseline_quality = _avg(
            [record["baseline"]["quality_score"] for record in subset]
        )
        candidate_quality = _avg(
            [record["candidate"]["quality_score"] for record in subset]
        )
        quality_retention_raw = safe_ratio(candidate_quality, baseline_quality)
        pairs = len(subset)
        wins = sum(1 for record in subset if record["comparison"]["result"] == "win")
        ties = sum(1 for record in subset if record["comparison"]["result"] == "tie")
        candidate_ok = sum(1 for record in subset if record["candidate"]["ok"])
        severe = sum(1 for record in subset if record["candidate"].get("severe_error"))
        out[category] = {
            "pairs": pairs,
            "quality_retention_raw": quality_retention_raw,
            "quality_retention_capped": min(quality_retention_raw, 1.0),
            "win_tie_rate": safe_ratio(wins + ties, pairs),
            "success_rate": safe_ratio(candidate_ok, pairs),
            "severe_error_rate": safe_ratio(severe, pairs),
        }
    return out


def _mark(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()
