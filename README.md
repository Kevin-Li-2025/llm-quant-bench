# LLM Quant Bench

Benchmark toolkit for quantized LLM quality retention, throughput, latency, stability, and sustained serving usability.

The original motivation is simple: verify whether a compressed 70B-class model can stay usable on a single NVIDIA L20 GPU. The toolkit compares a baseline model and a quantized candidate on the same golden prompts, records quality and serving metrics, and writes reproducible `summary.json` and `report.md` outputs.

It works with any OpenAI-compatible `/v1/chat/completions` server, including vLLM, TGI, llama.cpp server, and custom gateways.

## Scope

This project is enough for internal regression testing and for building evidence that a quantized 70B model is usable on constrained hardware. It is not, by itself, enough to claim that a quantized model is lossless.

Public evaluation systems generally require broad coverage, multiple metrics, reproducibility, and explicit limits. HELM emphasizes broad coverage, multi-metric measurement, and standardization. EleutherAI `lm-evaluation-harness` provides standard academic tasks. OpenCompass provides large benchmark configurations. LongBench focuses on long-context understanding. The MT-Bench and Chatbot Arena paper discusses LLM-as-judge biases such as position bias, verbosity bias, and self-enhancement bias. NVIDIA GenAI-Perf tracks serving metrics such as TTFT, ITL, request latency, output token throughput, and request throughput.

Recommended positioning:

- Business golden-set regression.
- Baseline-vs-quantized quality retention.
- 24-72 hour soak tests.
- Real single-GPU serving SLA validation.

Recommended external benchmarks to add:

- General capability: `lm-evaluation-harness`, for example MMLU/MMLU-Pro, GPQA, GSM8K/MATH, ARC, HellaSwag, and TruthfulQA.
- Chinese and broad benchmark coverage: OpenCompass, for example C-Eval, CMMLU, AGIEval, BBH, GSM8K, MATH, and HumanEval.
- Long context: LongBench or LongBench v2, plus needle-in-a-haystack or custom 8K/16K/32K retrieval and summarization sets.
- Instruction following and preference: MT-Bench, AlpacaEval, and Arena-Hard. Use answer-order swapping for judge runs and sample human review when possible.
- Code: HumanEval, MBPP, BigCodeBench, and real repository edit tasks.
- Serving capacity: vLLM `bench serve` or NVIDIA GenAI-Perf with fixed request-rate, concurrency, input-length, and output-length matrices for TTFT, ITL, TPOT, throughput, and goodput.

References:

- [HELM by Stanford CRFM](https://crfm.stanford.edu/2022/11/17/helm.html)
- [HELM GitHub](https://github.com/stanford-crfm/helm)
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [OpenCompass](https://github.com/open-compass/opencompass)
- [LongBench paper](https://arxiv.org/abs/2308.14508)
- [MT-Bench and Chatbot Arena paper](https://arxiv.org/abs/2306.05685)
- [NVIDIA GenAI-Perf metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2500/user-guide/docs/perf_analyzer/genai-perf/README.html)
- [vLLM serving benchmark](https://docs.vllm.ai/en/latest/cli/bench/serve.html)

## Quick Start

Run an end-to-end smoke test without a real model:

```bash
python3 -m llm_quant_bench demo --out runs/demo
```

The demo starts a local mock OpenAI-compatible server and writes:

- `runs/demo/results.jsonl`
- `runs/demo/summary.json`
- `runs/demo/report.md`

If this command fails, the problem is inside the benchmark tool. If this command works but a real 70B model fails, the issue is usually the model-serving endpoint, OpenAI-compatible protocol behavior, GPU memory, timeout settings, or concurrency settings.

Prepare two services:

- `baseline`: an FP16/BF16 70B model, or another high-quality reference model that you trust.
- `candidate`: the quantized model under test, such as AWQ, GPTQ, GGUF, or EXL2 Q4.

Edit [examples/config.example.json](examples/config.example.json), then run:

```bash
python3 -m llm_quant_bench run \
  --config examples/config.example.json \
  --dataset examples/golden_set.jsonl \
  --out runs/l20-70b-q4 \
  --repeats 3 \
  --concurrency 1
```

Outputs:

- `runs/l20-70b-q4/results.jsonl`: raw baseline and candidate results for each prompt.
- `runs/l20-70b-q4/summary.json`: machine-readable aggregate metrics.
- `runs/l20-70b-q4/report.md`: Markdown report ready for internal review.

If the server does not support streaming:

```bash
python3 -m llm_quant_bench run \
  --config examples/config.example.json \
  --dataset examples/golden_set.jsonl \
  --out runs/l20-70b-q4 \
  --no-stream
```

For OpenAI-compatible servers that support streaming usage metadata, add:

```bash
python3 -m llm_quant_bench run \
  --config examples/config.example.json \
  --dataset examples/golden_set.jsonl \
  --out runs/l20-70b-q4 \
  --stream-usage
```

This sends `stream_options.include_usage=true` so prompt and completion token counts can come from the server instead of tokenizer-free estimates.

## Candidate Load Test

The `run` command requests both the baseline and candidate, so its throughput only describes the benchmark run itself. To measure candidate-only serving capacity, use `load`:

```bash
python3 -m llm_quant_bench load \
  --config examples/config.example.json \
  --dataset examples/golden_set.jsonl \
  --out runs/l20-70b-q4-load-c1 \
  --concurrency 1 \
  --requests 100
```

Run for a fixed duration:

```bash
python3 -m llm_quant_bench load \
  --config examples/config.example.json \
  --dataset examples/golden_set.jsonl \
  --out runs/l20-70b-q4-load-c4-10m \
  --concurrency 4 \
  --duration-seconds 600
```

If the endpoint supports OpenAI streaming usage events, add `--stream-usage` to collect prompt token totals during load tests.

Outputs:

- `load_results.jsonl`
- `load_summary.json`
- `load_report.md`

This command only load-tests an already running candidate endpoint. It does not load, quantize, or optimize the 70B model. For real 70B serving on an L20, first serve the quantized model with vLLM, llama.cpp, TGI, TensorRT-LLM, or another inference stack.

Run a long soak test:

```bash
python3 -m llm_quant_bench run \
  --config examples/config.example.json \
  --dataset examples/golden_set.jsonl \
  --out runs/l20-70b-q4-soak-24h \
  --duration-seconds 86400 \
  --concurrency 1
```

## Golden Set Format

Each line is one JSON object. The only required field is `prompt`:

```json
{"id":"json-001","category":"format","severity":"critical","prompt":"Return only valid JSON...","require_json":true,"contains_all":["answer"]}
```

Supported scoring fields:

- `expected`: exact match after normalization.
- `expected_regex`: regular-expression match.
- `contains_all`: output must contain all listed strings.
- `require_json`: output must be valid JSON.
- `expected_number` and `number_tolerance`: numeric answer with tolerance.
- `reference` and `min_similarity`: text similarity against a reference answer.
- `min_chars` and `max_chars`: length constraints.
- `severity: critical|high|normal`: failures on important samples count as severe errors.
- `context_tokens`: explicit context-length marker used for long-context success rate.

Samples with no objective scoring rule and no judge are only checked for non-empty output. They are counted in `weakly_scored_rate`. For external claims such as "near-lossless", target `weakly_scored_rate = 0`.

## Metrics

The goal is not only to prove that a 70B model can load. The goal is to prove that quality stays close to the baseline and that the service remains stable over time.

Quality retention:

```text
quality_retention_raw = mean(candidate_quality_score) / mean(baseline_quality_score)
quality_retention = min(quality_retention_raw, 1.0)
```

Win/tie rate:

```text
win_tie_rate = (candidate_wins + ties) / total_pairs
loss_rate = candidate_losses / total_pairs
```

Stability:

```text
success_rate = successful_candidate_requests / total_candidate_requests
severe_error_rate = severe_candidate_errors / total_candidate_requests
weakly_scored_rate = samples_with_only_non_empty_scoring_and_no_judge / total_pairs
judge_inconsistency_rate = position_swapped_judge_disagreements / judged_pairs
judge_unvalidated_rate = judged_pairs_with_only_one_valid_order / judged_pairs
```

Performance attainment:

```text
performance_attainment =
  candidate_requests_meeting_ttft_itl_tpot_and_tokens_per_second_targets
  / successful_candidate_requests
```

Streaming performance:

```text
time_per_output_token = decode_seconds / output_tokens
inter_token_latency ~= mean_time_between_streaming_chunks
output_token_throughput = total_candidate_output_tokens / benchmark_wall_clock_seconds
request_throughput = successful_candidate_requests / benchmark_wall_clock_seconds
```

Long-context success rate:

```text
context_success_rate =
  successful_candidate_requests_with_context_tokens >= target_context_tokens
  / total_candidate_requests_with_context_tokens >= target_context_tokens
```

Composite usability:

```text
usability_score =
  quality_retention
  * success_rate
  * performance_attainment
  * context_success_rate
```

Perplexity:

```text
PPL = exp(-mean(log p(token_i)))
PPL_delta_percent = (candidate_ppl - baseline_ppl) / baseline_ppl * 100
```

If an external tool has already produced token logprobs, compute perplexity directly:

```bash
python3 -m llm_quant_bench ppl \
  --baseline-logprobs baseline_logprobs.json \
  --candidate-logprobs candidate_logprobs.json
```

`baseline_logprobs.json` can be a JSON array or:

```json
{"token_logprobs":[-0.1,-0.3,-0.2]}
```

## Suggested Gates

A practical definition of "near-lossless sustained usability" for a single-L20 70B Q4 deployment could be:

- `quality_retention >= 0.98`
- `win_tie_rate >= 0.95`
- `success_rate >= 0.995`
- `weakly_scored_rate = 0`
- `judge_inconsistency_rate <= 0.05`
- `judge_unvalidated_rate = 0`
- `severe_error_rate <= 0.005`
- 8K or 16K long-context `context_success_rate >= 0.99`
- 0 OOM or CUDA crashes in a 24-72 hour soak test
- `p95_ttft_s` and `p05_tokens_per_second` meet the product SLA

For stronger evidence, expand `examples/golden_set.jsonl` to 200-1000 samples covering your real traffic: RAG, code, JSON tool calls, long-document summarization, math reasoning, safety refusals, multi-turn dialogue, and high-frequency support questions.

## External Benchmark Commands

`lm-evaluation-harness` is useful for general capability checks:

```bash
lm-eval run \
  --model local-chat-completions \
  --model_args model=quantized-70b,base_url=http://localhost:8001/v1/chat/completions,num_concurrent=1,max_retries=3 \
  --tasks mmlu_pro,gpqa,gsm8k_cot,mathqa,arc_challenge,hellaswag,truthfulqa_mc2 \
  --apply_chat_template \
  --batch_size auto \
  --output_path runs/external/lm-eval-l20-70b-q4
```

Task names change across `lm-evaluation-harness` versions. Run `lm-eval ls tasks` first to verify the tasks available in your environment.

vLLM serving benchmark is useful for serving-capacity checks:

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url http://localhost:8001 \
  --endpoint /v1/chat/completions \
  --model quantized-70b \
  --dataset-name random \
  --input-len 8192 \
  --output-len 512 \
  --num-prompts 200 \
  --request-rate 1
```

GenAI-Perf is useful for a more standardized serving report:

```bash
genai-perf profile \
  -m quantized-70b \
  --service-kind openai \
  --endpoint-type chat \
  --streaming \
  --url localhost:8001 \
  --synthetic-input-tokens-mean 8192 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 512 \
  --output-tokens-stddev 0 \
  --num-prompts 200 \
  --concurrency 1
```
