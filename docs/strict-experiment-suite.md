# Strict Experiment Suite

This document tracks the evidence still needed before making stronger research
claims about single-L20 70B serving.

## Current Blockers

The current L20 host has only the AWQ checkpoint staged:

- `/home/hhai/models/Qwen2.5-72B-Instruct-AWQ`

No BF16/FP16 baseline endpoint, judge endpoint, SGLang runtime, llama.cpp
runtime, GGUF checkpoint, GPTQ checkpoint, or FP8 checkpoint is currently
available on that host. Those missing prerequisites block the corresponding
claims; they should not be inferred from the AWQ run.

## Evidence Checklist

| Evidence | Status | Entry Point | Blocking Condition |
|---|---|---|---|
| BF16/FP16 baseline vs AWQ retention | ready to run once baseline exists | `scripts/run_quality_retention.py` | needs matched BF16/FP16 endpoint |
| MT-Bench judge score | ready to run once judge exists | `scripts/score_mt_bench.py` | needs judge endpoint/key, preferably GPT-4-class |
| LongBench official-metric score | completed on the current 60-sample subset: 16.16% macro/micro | `scripts/score_longbench_official.py` | leaderboard claims still require exact official repo revision |
| vLLM repeated load with CI | runnable on current AWQ endpoint | `scripts/run_repeated_load.py` | needs vLLM service running |
| vLLM vs SGLang vs llama.cpp | scaffolded | `scripts/run_ablation_matrix.py` | needs SGLang, llama.cpp, and GGUF checkpoint |
| AWQ vs GPTQ vs FP8 | scaffolded | `scripts/run_ablation_matrix.py` | needs GPTQ and FP8 checkpoints/endpoints |
| Multi-run confidence intervals | runnable for any repeated summaries | `scripts/summarize_repeats_ci.py` | needs repeated runs per condition |

## Commands

Readiness check:

```bash
python3 scripts/check_experiment_readiness.py \
  --manifest examples/full_research_matrix.example.json \
  --out runs/research-matrix-readiness/readiness.json
```

LongBench official-metric postprocess from the existing 8K samples:

```bash
python3 scripts/score_longbench_official.py \
  --samples runs/qwen72b-awq-l20/quality-eval-20260520T084323Z/current-longbench-8k/eval/samples.jsonl \
  --out runs/qwen72b-awq-l20/quality-eval-20260520T084323Z/current-longbench-8k/official-metrics
```

Repeated fixed-shape load with confidence intervals:

```bash
python3 scripts/run_repeated_load.py \
  --config runs/qwen72b-awq-l20/config.fixed512x256.json \
  --dataset runs/qwen72b-awq-l20/fixed-512in-256out.jsonl \
  --out runs/qwen72b-awq-l20/repeated-fixed512x256-vllm-awq \
  --concurrencies 1 4 8 16 \
  --repeats 3 \
  --requests 80 \
  --stream-usage
```

BF16/FP16 retention once a baseline endpoint is available:

```bash
python3 scripts/run_quality_retention.py \
  --out runs/qwen72b-bf16-vs-awq-retention \
  --baseline-base-url "$BASELINE_BASE_URL" \
  --baseline-model "$BASELINE_MODEL" \
  --candidate-base-url http://127.0.0.1:8001/v1 \
  --candidate-model qwen72b-awq-l20 \
  --benchmarks mmlu cmmlu gsm8k \
  --cmmlu-dir /home/hhai/quality-eval-data/cmmlu/test \
  --concurrency 8
```

MT-Bench judge once a judge endpoint is available:

```bash
python3 scripts/score_mt_bench.py \
  --out runs/qwen72b-awq-l20/mt-bench-judge \
  --samples runs/qwen72b-awq-l20/quality-eval-20260520T084323Z/current-mt-bench-generation/samples.jsonl \
  --judge-base-url "$JUDGE_BASE_URL" \
  --judge-model "$JUDGE_MODEL" \
  --judge-api-key "$JUDGE_API_KEY" \
  --concurrency 4
```

## Reporting Rules

- Report AWQ absolute quality separately from BF16/FP16 retention.
- Report MT-Bench generation success separately from judge score.
- Report LongBench dependency-light official-metric scores separately from
  official leaderboard scores.
- For repeated load results, report mean, standard deviation, and two-sided 95%
  CI over run-level summaries.
- For runtime and quantization ablations, hold model family, prompt shape,
  decoding settings, concurrency, and measurement window constant.
