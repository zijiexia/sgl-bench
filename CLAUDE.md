# sgl-bench — project context for Claude Code

Lightweight, **client-side** performance benchmark for any OpenAI-compatible /
SGLang HTTP endpoint. Point it at an already-running server; get latency /
throughput numbers. Installs without torch, CUDA, or sglang.

## Core architectural principle

**The benchmark engine is vendored verbatim from
[sgl-project/sglang](https://github.com/sgl-project/sglang)** (`bench_serving`
+ the client core of `bench_one_batch_server_internal` + the
`benchmark/datasets/` layer). sgl-bench contributes only the thin transport/CLI
shell, the dataset-overlay registry, and the vendoring machinery that cuts the
sglang-runtime couplings so the slice installs standalone.

This mirrors the sibling repo **sgl-eval** (accuracy harness vendored from
NeMo-Skills): same SOURCES.yaml + sync + audit playbook, different upstream.

Enforced by:
- `sgl_bench/_vendored/sglang/SOURCES.yaml` pins the upstream SHA and records
  every file's source path + the rewrites that cut its sglang couplings.
- `scripts/audit_vendored.py` (run by `tests/test_vendor_audit.py`) fails if any
  vendored file still has an untranslated `from sglang...` / `import sglang...`.
- `tests/test_smoke_import.py` fails if a bare `import sgl_bench` pulls in
  torch / sglang / PIL / pandas — the whole point is staying lightweight.

## Client-mode, always

Every subcommand is a pure HTTP client and needs `--base-url`. sgl-bench never
loads an engine. The numbers are **end-to-end / user-visible** (network +
server tokenize/detokenize + scheduler queueing + streaming), which is what you
want for a deployment — but they differ from in-process engine benchmarks
(`sglang.bench_one_batch`): client TTFT/ITL are higher, client throughput ≤ the
engine ceiling. See README "measurement semantics". sgl-bench cannot do
per-layer/kernel timing or profiler traces (that needs the in-process engine).

## File layout

```
sgl_bench/
├── cli.py                 # `sgl-bench` entry: serve | latency | throughput | list-datasets
├── registry.py            # dataset overlay: @register_dataset + vendored base + extras
├── transport/             # SE-owned glue (the only non-vendored runtime code)
│   ├── _base.py           # Subcommand base (subclass discovery) + runpy delegator
│   ├── serve.py           # online serving -> delegates to vendored bench_serving CLI
│   ├── latency.py         # client-mode one-batch sweep (own loop over run_one_case)
│   └── throughput.py      # serve preset: --request-rate inf
├── datasets/              # USER-owned dataset extension layer (NOT vendored)
│   └── jsonl_prompts.py   # the canonical "add a dataset" example
└── _vendored/sglang/      # DO NOT hand-edit.
    ├── SOURCES.yaml        # the manifest (pinned SHA + per-file rewrites)
    ├── bench_serving.py    # online-serving client engine (upstream benchmark/serving.py)
    ├── one_batch_client.py # client primitives from upstream benchmark/one_batch_server.py
    ├── _net.py             # hand-slice of srt/utils/network.py (NetworkAddress + resolve_*)
    └── benchmark/{utils.py, datasets/*}
```

## Editing rules

- **Never hand-edit anything under `sgl_bench/_vendored/`.** To change vendored
  content, edit `SOURCES.yaml` and run `python scripts/sync_vendored.py`.
  Exception: `_vendored/sglang/_net.py` is a hand-curated slice (upstream
  `network.py` top-imports zmq/psutil); re-verify it by hand on a SHA bump.
- The sglang couplings are cut by SOURCES.yaml rewrites at well-known seams:
  `FAKE_BOOTSTRAP_HOST` inlined, `resolve_base_url`/`resolve_host_port` →
  `_net`, srt tokenizer/processor branches replaced with `transformers`, the
  srt chat encoder replaced with the HF chat template, `get_dataset`
  re-pointed to `sgl_bench.registry`, engine/CI functions dropped from
  `one_batch_client`. A new `from sglang.srt...` import appearing upstream
  will fail the audit — handle it with a new rewrite.
- New transport / CLI features are SE code, fine to add directly.

## Adding a dataset

Drop a module in `sgl_bench/datasets/`, decorate a `BaseDataset` subclass with
`@register_dataset("name", add_cli_args=...)`, return `List[DatasetRow]` from
`load()`. No reinstall, no vendored edits. Copy
`sgl_bench/datasets/jsonl_prompts.py` as the template. User datasets overlay
vendored ones on a name clash.

## Testing

```bash
.venv/bin/pytest                       # smoke + overlay + dispatch + audit
.venv/bin/python scripts/audit_vendored.py
```

`live`-marked tests need a running endpoint and are skipped by default.

## Upgrading the vendored slice

Bump `synced_from_sha` in SOURCES.yaml (prefer a release tag's commit) and run
`python scripts/sync_vendored.py`. If sync raises "replacement target not
found", upstream moved a seam — update that rewrite. Then run `pytest` (the
audit + smoke tests catch leaks/regressions).

Before re-targeting a rewrite, check whether upstream has **absorbed** it —
that has happened twice already (per-backend `--flush-cache` routing, and the
`reasoning`/`reasoning_content` TTFT fix). Delete such a rewrite rather than
moving it to the new seam: the local delta should only ever be coupling cuts.
Upstream also renames things (`autobench` → `agentic-trace` at v0.5.19), so
diff the CLI surface and call out user-visible changes in the bump commit.
