---
name: add-dataset
description: Add a new dataset to sgl-bench so it can be driven by `serve` / `latency` / `throughput` via `--dataset-name`. Use when the user asks to "add a dataset", "support <dataset>", "benchmark with my own prompts", "register a dataset", or "add a prompt source" inside the sgl-bench repo.
---

# Adding a dataset

sgl-bench has two layers of datasets:

- **User datasets** (the common case): a module you drop in `sgl_bench/datasets/`.
  Registered at import via `@register_dataset`, it overlays the vendored set —
  **no reinstall, no edits to vendored code.** This is what this skill does by
  default.
- **Vendored datasets**: synced from sglang under `_vendored/sglang/benchmark/
  datasets/`. Only add one here if the dataset already exists upstream and you
  want to track it (see §5).

First decide which you need. If the user wants to benchmark *their own* prompts /
a synthetic generator / a new file format → user dataset (§1–§4). If they're
asking for a dataset sglang already ships that we haven't vendored → §5.

## 1. The contract

A dataset is a `BaseDataset` subclass that turns CLI args into a list of
`DatasetRow`. Both come from the vendored common module:

```python
from sgl_bench._vendored.sglang.benchmark.datasets.common import BaseDataset, DatasetRow
```

`DatasetRow(prompt, prompt_len, output_len, ...)`:
- `prompt`: str | list[str] | list[dict] — what gets sent to the endpoint.
- `prompt_len`: int — input token count (use the passed tokenizer to compute it).
- `output_len`: int — target generation length for this request.
- optional: `image_data`, `timestamp` (arrival replay), `routing_key`,
  `extra_request_body` (per-request API params).

`BaseDataset` requires two methods:
- `from_args(cls, args) -> BaseDataset` — read your CLI flags off the argparse
  `Namespace` and construct the instance.
- `load(self, tokenizer, model_id=None) -> list[DatasetRow]` — build the rows.

## 2. Create the module

Copy the canonical example and adapt it — it is the template:

```bash
cp sgl_bench/datasets/jsonl_prompts.py sgl_bench/datasets/<name>.py
```

Shape (keep it light — no torch/heavy imports at module top, or a bare
`import sgl_bench` stops being lightweight and the smoke test fails):

```python
import argparse
from dataclasses import dataclass
from typing import Any, List, Optional

from sgl_bench._vendored.sglang.benchmark.datasets.common import BaseDataset, DatasetRow
from sgl_bench.registry import register_dataset


def _add_cli_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("<name> dataset")
    g.add_argument("--<name>-foo", type=str, default=None, help="...")


@register_dataset("<name>", add_cli_args=_add_cli_args)
@dataclass
class MyDataset(BaseDataset):
    foo: Optional[str] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "MyDataset":
        return cls(foo=getattr(args, "<name>_foo", None))

    def load(self, tokenizer: Any, model_id: Optional[str] = None) -> List[DatasetRow]:
        rows: List[DatasetRow] = []
        for prompt, out_len in _your_source(self.foo):
            rows.append(DatasetRow(
                prompt=prompt,
                prompt_len=len(tokenizer.encode(prompt)),
                output_len=out_len,
            ))
        if not rows:
            raise SystemExit("<name>: produced no rows (check your inputs)")
        return rows
```

Notes:
- Decorator order matters: `@register_dataset(...)` **outside** `@dataclass`
  (dataclass runs first, then the resulting class is registered).
- `add_cli_args` is optional — omit it if your dataset only needs existing flags
  (e.g. `--num-prompts`, `--random-output-len`). When present, its flags are
  injected into the parser for every subcommand (vLLM-style arg groups).
- Prefix your flags with the dataset name (`--<name>-...`) to avoid clashes.
- A user dataset registered under a vendored name (e.g. `"random"`) **overrides**
  the vendored one — useful, but intentional only.
- Heavy deps (PIL, pandas, an HF `datasets` download): import them lazily inside
  `load()`, not at module top, and document the extra needed.

## 3. Verify it autoloads

No reinstall needed (editable install). It registers on import via the
`sgl_bench.datasets` autoload:

```bash
sgl-bench list-datasets        # <name> appears, tagged (user)
```

Against a running server:

```bash
sgl-bench serve --base-url http://localhost:30000 --backend sglang \
    --dataset-name <name> [--<name>-foo ...] --num-prompts 8 --max-concurrency 8
```

## 4. Add a test

Mirror `tests/test_example_dataset.py`: ensure the module imports (registers),
`registry.resolve("<name>")` returns `source="user"`, and `get_dataset` produces
the expected rows with a fake tokenizer (`.encode = lambda s: s.split()`).

```bash
pytest -q
python scripts/audit_vendored.py     # unaffected (user layer isn't vendored)
```

## 5. (Alternative) Vendor a new dataset from sglang

Only if the dataset already lives in sglang's `benchmark/datasets/` and you want
to track upstream. Add it to `sgl_bench/_vendored/sglang/SOURCES.yaml` under
`files:` (the global `from sglang.benchmark.` rewrite covers its imports), then:

```bash
python scripts/sync_vendored.py
python scripts/audit_vendored.py
```

If it pulls heavy deps (PIL/pandas/HF-datasets), **keep it out of the vendored
`DATASET_MAPPING`** (so a bare install stays light) and register it lazily in
`sgl_bench/registry.py:_VENDORED_EXTRAS` with its `needs_extra` group, mirroring
`image`/`mmmu`/`longbench_v2`. Add the extra's deps to `pyproject.toml`.

## 6. Commit

One commit. Title: `add <name> dataset`. Contents: the new
`sgl_bench/datasets/<name>.py` (+ test), or for §5 the `SOURCES.yaml` entry +
synced files + any `_VENDORED_EXTRAS`/pyproject change. Nothing else.
