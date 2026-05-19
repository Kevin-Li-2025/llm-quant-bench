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

The main throughput sweep is a short-context configuration. A separate 4096-context c1 run is included below to make the single-L20 70B claim concrete.

## Minimum Experiment Matrix

| Model | Quant | GPU | Context | Concurrency | Success Rate | p95 TTFT | tok/s | OOM |
|---|---|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-72B-Instruct | Q4 AWQ / AWQ Marlin | L20 48GB (46GB usable) | 4096 | 1 | 100% (5/5) | 5.28s | 10.02 output tok/s | No |

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
