# sgl-bench

Lightweight, **client-side** performance benchmark for any OpenAI-compatible /
[SGLang](https://github.com/sgl-project/sglang) HTTP endpoint.

```bash
pip install sgl-bench          # no torch, no CUDA, no sglang
sgl-bench serve --backend sglang --base-url http://localhost:30000 \
    --dataset-name random --num-prompts 100
```

## What this is (and isn't)

sgl-bench is a **pure HTTP client**. It never loads a model engine, so it
installs without torch/CUDA/sglang and runs anywhere — your laptop, a CI box.
**Every subcommand needs `--base-url` (or `--host`/`--port`) pointing at an
already-running server.** You start the server separately; sgl-bench just
drives load and measures.

| subcommand | measures |
|---|---|
| `sgl-bench serve` | online serving under a controlled request-rate / concurrency: TTFT, ITL, throughput, latency percentiles |
| `sgl-bench latency` | single-batch latency sweep (client-mode "one-batch") — *added in M3* |
| `sgl-bench throughput` | saturation / max throughput (`--request-rate inf`) — *added in M3* |
| `sgl-bench list-datasets` | available `--dataset-name` values |

> Naming follows vLLM/sglang convention. Note that in vLLM/sglang, `latency`
> and `throughput` run the engine **in-process** (needs a GPU). Here they are
> **client-mode** — they hit a running server over HTTP. sgl-bench has no
> in-process mode by design.

## Client-mode measurement semantics — read this

Client-mode numbers measure the **end-to-end, user-visible** experience:
network + server-side tokenize/detokenize + scheduler queueing + SSE streaming,
on top of the raw model compute. They are **not** the same as the in-process
engine benchmarks (`sglang.bench_one_batch` / `vllm bench latency`), which
measure compute in isolation:

- client TTFT / ITL are **higher** (they include network + queueing + streaming),
- client throughput is **≤** the engine's in-process ceiling (the client/network
  can become the bottleneck).

For benchmarking a *deployment*, these are the right (SLO-relevant) numbers. To
get clean, comparable results: benchmark from **close to the server**, make sure
a single client isn't the bottleneck, use `--ignore-eos` + fixed output lengths
for apples-to-apples, and warm up first. sgl-bench cannot give per-layer/kernel
timing or profiler traces — for that you need the in-process tools (which require
the full sglang install).

## Architecture

The benchmark engine is **vendored verbatim from sglang** (`bench_serving` +
its `benchmark/datasets/` layer), with the sglang-runtime couplings cut so it
installs standalone. The slice is pinned at a SHA in
`sgl_bench/_vendored/sglang/SOURCES.yaml`; to upgrade, bump it and run
`python scripts/sync_vendored.py`. Your own datasets live in `sgl_bench/datasets/`
and register without touching vendored code.

## License

Apache-2.0. Vendored sglang sources (themselves adapted from vLLM) are also
Apache-2.0; see `NOTICE`.
