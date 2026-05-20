# L20 Qwen2.5-72B AWQ Throughput Notes

This note records one concrete single-GPU serving run for `Qwen/Qwen2.5-72B-Instruct-AWQ` on an NVIDIA L20.

## Environment

- GPU: NVIDIA L20, 46 GB VRAM
- Serving stack: vLLM `0.8.5.post1`
- Model: `Qwen/Qwen2.5-72B-Instruct-AWQ`
- Quantization: `awq_marlin`
- Serving mode: OpenAI-compatible `/v1/chat/completions`
- Test shape: short golden prompts, streaming responses, candidate-only load test
- Context mode: `--max-model-len 1024`

The main throughput sweep is a short-context configuration. Separate 4096-context and 8192-context c1 runs are included below to make the single-L20 70B claim concrete.

## Experiment Matrix

| Model | Quant | GPU | Context | Concurrency | Success Rate | p95 TTFT | tok/s | OOM |
|---|---|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-72B-Instruct | Q4 AWQ / AWQ Marlin | L20 48GB (46GB usable) | 8192 | 1 | 100% (3/3) | 11.03s | 6.16 output tok/s | No |
| Qwen2.5-72B-Instruct | Q4 AWQ / AWQ Marlin | L20 48GB (46GB usable) | 4096 | 1 | 100% (5/5) | 5.28s | 10.02 output tok/s | No |
| Qwen2.5-72B-Instruct | Q4 AWQ / AWQ Marlin | L20 48GB (46GB usable) | 1024 | 10 | 100% (36740/36740) | 6.61s | 108.84 output tok/s | No |
| Qwen2.5-72B-Instruct | Q4 AWQ / AWQ Marlin | L20 48GB (46GB usable) | 1024 | 16 | 100% (1177/1177) | 0.14s | 245.91 output tok/s | No |
| Qwen2.5-72B-Instruct | Q4 AWQ / AWQ Marlin | L20 48GB (46GB usable) | 1024 | 32 | 100% (408/408) | 2.17s | 390.08 output tok/s | No |
| Qwen2.5-72B-Instruct | Q4 AWQ / AWQ Marlin | L20 48GB (46GB usable) | 1024 | 48 | 100% (2333/2333) | 0.24s | 488.63 output tok/s | No |
| Qwen2.5-72B-Instruct | Q4 AWQ / AWQ Marlin | L20 48GB (46GB usable) | 1024 | 64 | 100% (461/461) | 3.39s | 432.63 output tok/s | No |

The 4096-context row used `--max-model-len 4096`, `--max-num-seqs 1`, `--max-num-batched-tokens 4096`, a 3,875-token prompt by tokenizer count, and `max_tokens=128`. vLLM reported:

```text
GPU KV cache size: 10,848 tokens
Maximum concurrency for 4,096 tokens per request: 2.65x
```

The load-test summary was:

```text
successful requests: 5 / 5
prompt tokens: 19,520 total
output tokens: 170 total
output token throughput: 10.02 tok/s
p95 TTFT: 5.28s
p95 latency: 7.29s
p05 per-request decode speed: 16.89 tok/s
errors: {}
OOM: no CUDA OOM found in the vLLM log
```

The 8192-context row used `--max-model-len 8192`, `--max-num-seqs 1`, `--max-num-batched-tokens 2048`, a 7,514-token prompt by tokenizer count, and `max_tokens=128`. vLLM reported:

```text
GPU KV cache size: 12,320 tokens
Maximum concurrency for 8,192 tokens per request: 1.50x
```

The load-test summary was:

```text
successful requests: 3 / 3
prompt tokens: 22,629 total
output tokens: 123 total
output token throughput: 6.16 tok/s
p95 TTFT: 11.03s
p95 latency: 13.54s
p05 per-request decode speed: 16.36 tok/s
errors: {}
OOM: no CUDA OOM found in the vLLM log
```

## Best Throughput Configuration

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /home/hhai/models/Qwen2.5-72B-Instruct-AWQ \
  --host 0.0.0.0 --port 8001 \
  --served-model-name qwen72b-awq-l20 \
  --quantization awq_marlin \
  --dtype half \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.98 \
  --max-num-seqs 48 \
  --max-num-batched-tokens 4096 \
  --enforce-eager \
  --swap-space 1 \
  --disable-log-requests \
  --trust-remote-code
```

vLLM reported:

```text
GPU KV cache size: 10,848 tokens
Maximum concurrency for 1,024 tokens per request: 10.59x
```

## Fixed-Shape 512x256 Benchmark

This run is intended for fairer comparison with external serving benchmarks.

- Prompt set: 128 unique prompts
- Raw tokenizer prompt length: 498 tokens
- Server-side prompt usage after chat formatting: 527 tokens on average
- Output length: fixed 256 tokens
- Sampling controls: `max_tokens=256`, `min_tokens=256`, `ignore_eos=true`, `temperature=0`
- Endpoint mode: streaming with `stream_options.include_usage=true`
- Serving config: the same 1024-context `awq_marlin` service listed above

| Shape | Concurrency | Requests | Success Rate | p95 TTFT | p95 Latency | Output tok/s | Req/s | p05 Decode tok/s | OOM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ~512 in / 256 out | 1 | 8 | 100% | 0.76s | 15.88s | 16.21 | 0.063 | 16.92 | No |
| ~512 in / 256 out | 4 | 32 | 100% | 2.94s | 18.63s | 57.10 | 0.223 | 14.99 | No |
| ~512 in / 256 out | 8 | 64 | 100% | 6.06s | 22.20s | 93.38 | 0.365 | 12.13 | No |
| ~512 in / 256 out | 10 | 80 | 100% | 6.71s | 23.69s | 108.70 | 0.425 | 11.26 | No |
| ~512 in / 256 out | 16 | 128 | 100% | 11.01s | 36.29s | 127.70 | 0.499 | 7.66 | No |
| ~512 in / 256 out | 24 | 120 | 100% | 37.26s | 64.08s | 120.51 | 0.471 | 5.25 | No |

The c10 row is the closest match to the GigaGPU 10-concurrent-user public table. The c16 fixed-shape run used 45.3 GB of the 46.1 GB visible VRAM during sampling, with about 97% GPU utilization. No CUDA OOM or request errors were observed in these runs. The c24 saturation run did not improve throughput over c16 and had much worse tail latency.

## Repeated Fixed-Shape CI

The fixed-shape c1/c4/c8/c16 conditions were repeated three times each after the
initial screening runs.

- Run directory: `/home/hhai/llm-quant-bench/runs/qwen72b-awq-l20/repeated-fixed512x256-vllm-awq-20260520T120138Z`
- Workload: same ~512 input / 256 output prompt set
- Repeats: 3 per concurrency
- CI method: two-sided 95% CI over run-level summaries using Student t critical values
- Error check: zero vLLM log matches for CUDA OOM, OutOfMemory, Traceback, ERROR, or Killed

| Concurrency | Runs | Success Rate | Output tok/s | Req/s | p95 TTFT | p95 Latency | p95 TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 100% +/- 0.00% | 16.57 +/- 1.21 | 0.065 +/- 0.005 | 0.54s +/- 1.90s | 15.66s +/- 1.94s | 0.059s +/- 0.000s |
| 4 | 3 | 100% +/- 0.00% | 55.75 +/- 3.54 | 0.218 +/- 0.014 | 3.01s +/- 0.17s | 18.70s +/- 0.16s | 0.070s +/- 0.000s |
| 8 | 3 | 100% +/- 0.00% | 93.26 +/- 0.23 | 0.364 +/- 0.001 | 6.05s +/- 0.06s | 22.16s +/- 0.05s | 0.082s +/- 0.000s |
| 16 | 3 | 100% +/- 0.00% | 127.22 +/- 12.68 | 0.497 +/- 0.050 | 11.91s +/- 0.15s | 34.95s +/- 1.00s | 0.125s +/- 0.001s |

The repeated c16 mean is essentially the same as the earlier single c16
screening result, 127.22 versus 127.70 output tok/s. The c8 repeat is extremely
tight, while the c16 output-throughput CI is wider because the third c16 run
dropped to 121.32 output tok/s after two roughly 130 output tok/s runs.

## 24h Fixed-Shape Soak

The c10 fixed-shape workload was run for 24 hours on the same vLLM service.

- Run directory: `/home/hhai/llm-quant-bench/runs/qwen72b-awq-l20/soak-24h-fixed512x256-c10-20260519T034856Z`
- Load summary: `load/load_summary.json`
- Power summary: `energy_summary.json`
- Power samples: 8,618 samples at approximately 10-second intervals
- Power measurement scope: GPU board power from `nvidia-smi`, not full-system wall power
- Error check: zero vLLM log matches for CUDA OOM, OutOfMemory, Traceback, ERROR, or Killed

| Shape | Concurrency | Duration | Requests | Success Rate | p95 TTFT | p95 Latency | Output tok/s | Req/s | p05 Decode tok/s | OOM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ~512 in / 256 out | 10 | 24.00h | 36740 | 100% | 6.61s | 23.54s | 108.84 | 0.425 | 11.26 | No |

| Metric | Value |
|---|---:|
| Prompt tokens | 19,361,980 |
| Output tokens | 9,405,440 |
| Total tokens | 28,767,420 |
| Average GPU power | 330.15 W |
| GPU energy | 7.92 kWh |
| Output tokens per joule | 0.330 |
| Total tokens per joule | 1.008 |
| Joules per output token | 3.033 |

The day-long run closely matched the short c10 fixed-shape screening run: 108.84 output tok/s over 24 hours versus 108.70 output tok/s in the short run. This is the strongest stability result in the current repo.

## External Comparisons

These comparisons are directional because serving throughput depends on hardware, prompt length, output length, concurrency, batching, kernels, and sampling settings.

| Source | Hardware | Model / Quant | Shape | Reported Result | Note |
|---|---|---|---|---:|---|
| This repo | 1x L20 48GB | Qwen2.5-72B AWQ Marlin | ~512 input / 256 output, c8 | 93.38 output tok/s | Fixed-shape aggregate throughput, 64/64 success. |
| This repo | 1x L20 48GB | Qwen2.5-72B AWQ Marlin | ~512 input / 256 output, c10 | 108.70 output tok/s | Closest local match to 10-concurrent-user public tables. |
| This repo | 1x L20 48GB | Qwen2.5-72B AWQ Marlin | ~512 input / 256 output, c10, 24h soak | 108.84 output tok/s | Sustained result with 36,740/36,740 successful requests. |
| This repo | 1x L20 48GB | Qwen2.5-72B AWQ Marlin | ~512 input / 256 output, c16 | 127.70 output tok/s | Higher throughput, 128/128 success, p95 latency 36.29s. |
| [GigaGPU Apr 2026](https://gigagpu.com/tokens-sec-benchmark-update-april-2026/) | 1x RTX 3090 | Qwen 2.5 72B Q4 | 512 input / 256 output, 10 concurrent users | 32 tok/s | Similar fixed-shape public table. |
| [GigaGPU Apr 2026](https://gigagpu.com/tokens-sec-benchmark-update-april-2026/) | 1x RTX 5090 | Qwen 2.5 72B Q4 | 512 input / 256 output, 10 concurrent users | 58-82 tok/s | This L20 run is above that published 5090 range. |
| [GigaGPU Apr 2026](https://gigagpu.com/tokens-sec-benchmark-update-april-2026/) | 1x RTX 6000 Pro | Qwen 2.5 72B Q4 | 512 input / 256 output, 10 concurrent users | 45 tok/s | Different GPU and runtime details. |
| [Qwen official speed benchmark](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html) | 1x A100 80GB | Qwen2.5-72B AWQ, Transformers | input 1 / 6144 / 14336, 2048 output, batch size 1 | 11.50 / 8.17 / 5.57 tok/s | Useful single-request reference, not aggregate serving throughput. |
| [Qwen official speed benchmark](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html) | 2x A100 80GB | Qwen2.5-72B AWQ, vLLM | input 1 / 6144 / 14336 / 30720, 2048 output, batch size 1 | 44.30 / 40.67 / 36.63 / 30.02 tok/s | Official vLLM baseline uses 2 A100s and batch size 1. |
| [NVIDIA NIM supported models](https://docs.nvidia.com/nim/large-language-models/1.14.0/supported-models.html) | L20 | Qwen2.5 72B Instruct FP8 | Optimized profiles | 4 or 8 GPUs | NVIDIA's optimized L20 profile is multi-GPU; this run demonstrates a single-L20 AWQ path. |

Bottom line: the fixed-shape c8/c16 numbers are strong versus public single-GPU Q4 tables, and the c10 24h soak shows day-long sustained serving stability. These should be described as aggregate serving throughput, not single-request speed. The long-context c1 rows are capacity evidence. None of these results prove lossless quality retention.

## Load Test Command

```bash
python3 -m llm_quant_bench load \
  --config runs/qwen72b-awq-l20/config.load.json \
  --dataset examples/golden_set.jsonl \
  --out runs/qwen72b-awq-l20/load-c48-300s-awq-marlin-mbt4096 \
  --concurrency 48 \
  --duration-seconds 300 \
  --stream-usage
```

## Results

| Config | Duration | Success | Output tok/s | Req/s | p95 latency | p95 TTFT | p05 decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| c16 / mbt4096 | 300s | 1177 / 1177 | 245.91 | 3.83 | 8.09s | 0.14s | 9.00 |
| c20 / mbt4096 | 60s | 282 / 282 | 268.92 | 4.16 | 8.58s | 1.80s | 15.17 |
| c24 / mbt4096 | 60s | 329 / 329 | 315.27 | 4.87 | 8.83s | 1.97s | 14.76 |
| c32 / mbt4096 | 60s | 408 / 408 | 390.08 | 6.04 | 9.56s | 2.17s | 13.61 |
| c48 / mbt4096 | 60s | 485 / 485 | 460.29 | 7.13 | 12.35s | 2.70s | 10.48 |
| c64 / mbt4096 | 60s | 461 / 461 | 432.63 | 6.71 | 18.15s | 3.39s | 7.13 |
| c48 / mbt4096 | 300s | 2333 / 2333 | 488.63 | 7.56 | 12.35s | 0.24s | 10.47 |

## Interpretation

`c48 / max_num_batched_tokens=4096` was the best tested short-context throughput point. It sustained 488.63 output tokens/s for five minutes with zero request failures.

`c64` did not improve throughput and degraded latency, so it is past the useful concurrency point for this workload.

For lower latency, `c32 / max_num_batched_tokens=4096` is a more balanced setting. It reached 390.08 output tokens/s in the 60-second screening run with lower p95 latency than c48.

For long-context tests, use a separate configuration. A previous 8K validation used `--max-model-len 8192`, lower concurrency, and successfully handled a 7,496-token prompt, but that is not the same workload as this throughput run.
