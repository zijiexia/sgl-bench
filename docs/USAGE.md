# sgl-bench usage guide

A lightweight, **client-side** performance benchmark for any OpenAI-compatible /
SGLang HTTP endpoint. It never loads a model engine, so it installs without
torch/CUDA/sglang and runs anywhere — your laptop, a CI box, a login node.

- [Install](#install)
- [Mental model: client-only](#mental-model-client-only)
- [Subcommands](#subcommands)
  - [serve](#serve) · [latency](#latency) · [throughput](#throughput) · [list-datasets](#list-datasets)
- [Datasets](#datasets)
- [Recipes](#recipes)
- [Getting trustworthy numbers](#getting-trustworthy-numbers)
- [Interpreting the metrics](#interpreting-the-metrics)
- [Custom model architectures](#custom-model-architectures)

---

## Install

Not on PyPI yet — install from source:

```bash
git clone https://github.com/zijiexia/sgl-bench && cd sgl-bench
pip install -e .                      # base: aiohttp, requests, numpy, transformers, tqdm, tabulate, pydantic
pip install -e '.[multimodal]'        # + pillow, pybase64, datasets   (for image / mmmu datasets)
pip install -e '.[longbench]'         # + pandas, datasets             (for longbench_v2)
# once published: pip install git+https://github.com/zijiexia/sgl-bench
```

No torch, no CUDA, no sglang. Verify it stays light:

```bash
python -c "import sys, sgl_bench; assert 'torch' not in sys.modules; print('ok')"
```

---

## Mental model: client-only

Every subcommand is a **pure HTTP client** and **requires `--base-url`** (or
`--host`/`--port`) pointing at an **already-running** server. You start the
server separately; sgl-bench drives load and measures from the client side.

The numbers are **end-to-end / user-visible**: they include network, server-side
tokenize/detokenize, scheduler queueing, and SSE streaming on top of raw model
compute. They are *not* the same as in-process engine benchmarks
(`sglang.bench_one_batch`, `vllm bench latency`) which measure compute in
isolation and need a GPU. For benchmarking a *deployment*, the client numbers
are the right (SLO-relevant) ones. sgl-bench cannot produce per-layer/kernel
timings or profiler traces.

> Naming follows vLLM/sglang convention (`serve` / `latency` / `throughput`).
> In vLLM/sglang those latter two run the engine in-process; **here they are all
> client-mode**, hitting a running server over HTTP.

---

## Subcommands

Common endpoint flags (all subcommands): `--base-url URL` (e.g.
`http://localhost:30000`), `--backend {sglang,vllm,openai,...}` (default
`sglang`), `--model NAME` / `--tokenizer NAME` (HF id or local path).

### serve

Online serving benchmark: drive a controlled request-rate / concurrency and
measure TTFT, ITL, TPOT, throughput, and latency percentiles. Delegates verbatim
to the vendored `bench_serving`, so every upstream flag works.

Key flags: `--dataset-name`, `--num-prompts N`, `--request-rate {inf|N}`
(req/s; `inf` = send all at once), `--max-concurrency N` (cap in-flight
requests), `--random-input-len`, `--random-output-len`, `--random-range-ratio`
(1.0 = fixed length), `--seed`, `--flush-cache`, `--warmup-requests N`.

```bash
sgl-bench serve --base-url http://localhost:30000 --model deepseek-ai/DeepSeek-V4-Flash \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 \
    --random-range-ratio 1.0 --num-prompts 16 --max-concurrency 16 --flush-cache
```

Example tail (DeepSeek-V4-Flash, 4×B200):

```
============ Serving Benchmark Result ============
Successful requests:                     16
Benchmark duration (s):                  9.09
Total input tokens:                      131072
Total generated tokens:                  16384
Output token throughput (tok/s):         1803.22
Total token throughput (tok/s):          16228.95
Accept length:                           2.74          # MTP speculative accept length
Mean TTFT (ms):                          1653.31
Median TTFT (ms):                        1689.44
Mean TPOT (ms):                          6.69
Mean ITL (ms):                           6.69
==================================================
```

### latency

Single-batch latency sweep (client-mode "one-batch"): for each
`(batch_size, input_len, output_len)` combination, fire one batch and report
its latency, prefill/decode throughput, TTFT, and accept length. Reuses the
vendored `run_one_case`; **the tokenizer is auto-resolved from the server**
(`/v1/models` for vllm, `/server_info` for sglang) so you usually don't even
pass `--model`.

Key flags: `--batch-size N [N ...]`, `--input-len N [N ...]`, `--output-len N
[N ...]` (swept as a cross-product), `--dataset-name {random,random-ids,mmmu,
generated-shared-prefix}`, `--skip-warmup`, `--tokenizer` (override),
`--temperature`, `--client-stream-interval`.

```bash
sgl-bench latency --base-url http://localhost:30000 --backend sglang \
    --batch-size 1 8 16 --input-len 4096 --output-len 128 --dataset-name random
```

```
|   bs |   in |   out |   latency(s) |   in_tput |   out_tput |   ttft(s) |   acc_len |
|------|------|-------|--------------|-----------|------------|-----------|-----------|
|    8 | 4096 |   128 |        1.39  |   45410.2 |     1538.1 |     0.722 |      2.67 |
|   16 | 4096 |   128 |        2.20  |   45897.5 |     2642.9 |     1.428 |      2.78 |
```

### throughput

Saturation / max throughput — a thin preset over `serve` that forces
`--request-rate inf` (send as fast as the server drains). Combine with
`--max-concurrency` to bound in-flight requests. All `serve` flags apply.

```bash
sgl-bench throughput --base-url http://localhost:30000 --model deepseek-ai/DeepSeek-V4-Flash \
    --dataset-name random --random-input-len 4096 --random-output-len 256 \
    --num-prompts 64 --max-concurrency 32
```

### list-datasets

Enumerate available `--dataset-name` values, marking each `(vendored)` /
`(user)` and which optional extra it needs:

```bash
sgl-bench list-datasets
```

---

## Datasets

| name | source | use |
|---|---|---|
| `random`, `random-ids` | synthetic random tokens (no shared prefix) | length-controlled load; ~0% prefix-cache hit |
| `generated-shared-prefix` | shared system prompt + unique question | **cache-hit** scenarios (see recipes) |
| `sharegpt` | real ShareGPT conversations (HF) | realistic mixed lengths |
| `custom`, `openai` | your JSONL / OpenAI-format file (`--dataset-path`) | replay real traffic |
| `mooncake` | Mooncake trace | timestamped arrival replay |
| `image`, `mmmu` | multimodal (needs `[multimodal]`) | VLM serving |
| `longbench_v2` | long-context (needs `[longbench]`) | long-context serving |
| *(your own)* | a module in `sgl_bench/datasets/` | see the add-a-dataset skill |

Heavy datasets are gated: a bare install never imports PIL/pandas; asking for
`mmmu` without `[multimodal]` prints `pip install 'sgl-bench[multimodal]'`.

---

## Recipes

All hit a running server at `:30000`. Set `M=deepseek-ai/DeepSeek-V4-Flash`.

**8k in / 1k out, concurrency 1 and 16:**

```bash
for mc in 1 16; do
  sgl-bench serve --base-url http://localhost:30000 --model $M \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 \
    --random-range-ratio 1.0 --num-prompts $((mc==1?8:32)) --max-concurrency $mc --flush-cache
done
```

**32k in / 1k out:** same, with `--random-input-len 32768`.

**32k with cache hit (shared 30k prefix + 1k unique question, ~88% prefix reuse):**

```bash
sgl-bench serve --base-url http://localhost:30000 --model $M \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 2 --gsp-prompts-per-group 8 \
  --gsp-system-prompt-len 30720 --gsp-question-len 1024 --gsp-output-len 1024 \
  --max-concurrency 16 --flush-cache
```

**Single-batch latency sweep:**

```bash
sgl-bench latency --base-url http://localhost:30000 --backend sglang \
  --batch-size 1 4 16 --input-len 2048 8192 --output-len 256
```

---

## Getting trustworthy numbers

These are real lessons from benchmarking DeepSeek-V4 on B200 — they matter a lot:

- **Warm up.** The *first* run after the server starts — or the first time a new
  batch shape (e.g. a new `--max-concurrency` / `--batch-size`) is exercised —
  pays a one-time cost (CUDA-graph capture, allocator growth). It inflates that
  run's TTFT/latency (median stays sane; **mean and P99 spike**). Discard a
  warmup run, or rely on `--warmup-requests`.
- **Comparing two tools / configs? Control for order.** Whichever you run *first*
  eats the cold-start and looks slower. Interleave runs (A,B,A,B) or warm the
  server first, then compare. (Verified: sgl-bench and sglang's own
  `bench_serving` produce identical numbers once warm.)
- **Benchmark from close to the server.** Network latency (localhost ≈ 0,
  cross-region ≈ tens of ms) dominates TTFT/ITL. Run from the same host/DC.
- **Don't let the client be the bottleneck.** At very high token rates a single
  client can't drain SSE fast enough — inflating ITL, capping throughput. Use
  enough concurrency; for extreme throughput, multiple client processes.
- **Apples-to-apples:** fix lengths with `--random-range-ratio 1.0`, pin
  `--seed`, and `--flush-cache` for a clean cold-cache run (omit it / use the
  shared-prefix dataset for cache-hit scenarios). `--flush-cache` picks the
  endpoint from `--backend`: `/flush_cache` for sglang, `/reset_prefix_cache`
  for vllm — the latter is only mounted when the vllm server was started with
  `VLLM_SERVER_DEV_MODE=1`. If the flush is rejected, sgl-bench prints a
  warning rather than quietly measuring a warm cache.

---

## Interpreting the metrics

- **TTFT** (time to first token) — latency until the first streamed token;
  includes queueing. The headline for interactive UX.
- **ITL** (inter-token latency) / **TPOT** (time per output token) — steady-state
  decode speed per token. `1000 / TPOT` ≈ per-stream tok/s.
- **Output token throughput** — generated tok/s across all requests (system
  capacity for decode). **Total token throughput** includes input tokens.
- **Accept length** (sglang + speculative/MTP models) — average tokens accepted
  per decode step; >1 means speculative decoding is helping. `latency` reports
  `acc_len` per case (`-1` = not reported, e.g. very short outputs).
- **Percentiles (P90/P99)** — tail behavior; watch these for SLOs, not just means.

---

## Custom model architectures

For bleeding-edge archs not yet in `transformers` (e.g. `deepseek_v4`), vanilla
`AutoTokenizer.from_pretrained` can fail to parse the model config. sgl-bench
falls back to importing `sglang` **iff it is already installed** in the env (it
registers such configs), purely to load the tokenizer — still not a hard
dependency. On a box that serves the model with sglang, this is automatic.
Standard models (Llama, Qwen, …) need nothing extra. You can always pass
`--tokenizer <hf-id-or-local-path>` explicitly.
