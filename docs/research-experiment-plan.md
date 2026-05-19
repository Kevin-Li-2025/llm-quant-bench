# Research Experiment Plan

This document tracks the experiments needed for a credible technical report on single-L20 70B-class quantized serving.

## Current 24h Soak

Status: started.

- Run directory: `/home/hhai/llm-quant-bench/runs/qwen72b-awq-l20/soak-24h-fixed512x256-c10-20260519T034856Z`
- Model: `Qwen/Qwen2.5-72B-Instruct-AWQ`
- Runtime: vLLM `0.8.5.post1`
- GPU: 1x NVIDIA L20
- Serving config: `awq_marlin`, `max_model_len=1024`, `max_num_seqs=48`, `max_num_batched_tokens=4096`
- Workload: fixed-shape ~512 input / 256 output
- Concurrency: 10
- Duration: 86,400 seconds
- Load output: `load/load_summary.json`
- Power log: `gpu_power.csv`

After completion, compute energy with:

```bash
python3 scripts/summarize_energy.py \
  --summary /home/hhai/llm-quant-bench/runs/qwen72b-awq-l20/soak-24h-fixed512x256-c10-20260519T034856Z/load/load_summary.json \
  --power-log /home/hhai/llm-quant-bench/runs/qwen72b-awq-l20/soak-24h-fixed512x256-c10-20260519T034856Z/gpu_power.csv \
  --out /home/hhai/llm-quant-bench/runs/qwen72b-awq-l20/soak-24h-fixed512x256-c10-20260519T034856Z/energy_summary.json
```

Report:

- total requests
- success rate
- failed requests and error classes
- output tokens/s
- p95 TTFT
- p95 latency
- average GPU power
- energy Wh
- output tokens/J
- total tokens/J
- CUDA OOM count from vLLM logs

## Quality Retention

Goal: compare a high-quality baseline against the L20 AWQ candidate.

Required baseline:

- FP16/BF16 Qwen2.5-72B-Instruct endpoint, or a trusted hosted Qwen2.5-72B-Instruct endpoint.
- The current L20 cannot host FP16/BF16 72B locally, and the current machine has only 69GB free disk, so the baseline must be external or multi-GPU.

Benchmarks:

| Benchmark | Purpose | Candidate Command Shape |
|---|---|---|
| MMLU / MMLU-Pro | general knowledge and reasoning | `lm-evaluation-harness` local OpenAI-compatible endpoint |
| CMMLU / C-Eval | Chinese knowledge and reasoning | OpenCompass preferred |
| GSM8K / MATH | math reasoning | `lm-evaluation-harness` or OpenCompass |
| MT-Bench / Arena-Hard | instruction following and preference | judge with answer-order swap |
| LongBench / LongBench v2 | long-context behavior | run separate 8K/16K service config |

Acceptance metrics:

- quality retention = candidate score / baseline score
- pass if task-level retention is at least 0.98 for business-critical categories
- report all drops over 2 percentage points individually
- keep judge inconsistency under 5% for preference benchmarks

## Energy

Primary formula:

```text
energy_j = integral(power_watts dt)
output_tokens_per_joule = output_tokens / energy_j
total_tokens_per_joule = (prompt_tokens + output_tokens) / energy_j
joules_per_output_token = energy_j / output_tokens
```

For the current setup, `gpu_power.csv` is sampled every 10 seconds using `nvidia-smi`. This measures GPU board power, not full-system wall power.

## More Models

Current disk status leaves about 69GB free. A single 70B AWQ checkpoint usually needs about 35-45GB, so only one additional model can be staged safely unless older artifacts are removed.

Candidate order:

| Priority | Model | Quant | Why |
|---:|---|---|---|
| 1 | `casperhansen/llama-3.3-70b-instruct-awq` or another Llama 3.3 70B AWQ | AWQ | strong multilingual 70B baseline, available as AWQ |
| 2 | `Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ` | AWQ | reasoning-oriented 70B distilled model, available as AWQ |
| 3 | Qwen3 70B-class model | AWQ/GPTQ/FP8 if available | Qwen3 AWQ docs exist, but an official dense 70B-class AWQ checkpoint must be verified before download |

Do not download all candidates at once on the current disk.

## More Runtimes

| Runtime | Status | Notes |
|---|---|---|
| vLLM | complete for Qwen2.5-72B AWQ | current best result uses AWQ Marlin |
| SGLang | not installed | install in a separate env and test AWQ support |
| llama.cpp / GGUF | not installed | requires a GGUF 70B Q4 checkpoint; results are not directly comparable to AWQ Marlin |

Runtime comparisons should use the same fixed-shape workload:

- ~512 input / 256 output
- concurrency 1, 4, 8, 10, 16
- streaming or equivalent token timing
- GPU power monitor

## Ablations

| Ablation | Current Status | Notes |
|---|---|---|
| AWQ Marlin | complete | best fixed-shape point: c16, 127.70 output tok/s |
| AWQ non-Marlin | partially tested | slower than AWQ Marlin in earlier runs |
| GPTQ | not tested | requires compatible 70B GPTQ checkpoint |
| FP8 | not tested | NVIDIA NIM L20 profiles for Qwen2.5 72B use 4 or 8 L20 GPUs, not single L20 |
| max concurrency | tested | c24 fixed-shape saturated and regressed versus c16 |
| max context | tested at c1 | 4096 and 8192 context rows completed without OOM |

## Paper Positioning

This should be written as a systems benchmark or empirical technical report, not as a new quantization algorithm paper.

Claim that is supported:

```text
Single-L20 serving of Qwen2.5-72B AWQ is feasible, measurable, and stable under fixed-shape load, with c10 reaching 108.70 output tok/s and c16 reaching 127.70 output tok/s for ~512 input / 256 output.
```

Claims that are not yet supported:

- lossless quality
- superiority over A100/H100
- generalization to all 70B models
- production SLA without 24h soak completion
