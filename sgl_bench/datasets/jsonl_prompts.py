"""Example user dataset -- the canonical "add a dataset" recipe.

Benchmark with *your own* prompts from a JSONL file: one object per line,
``{"prompt": "...", "output_len": 128}`` (``output_len`` optional, falls back
to ``--jsonl-output-len``).

    sgl-bench serve --base-url http://localhost:30000 \
        --dataset-name jsonl-prompts --jsonl-prompts-path prompts.jsonl

This whole file IS the extension mechanism: drop a module in sgl_bench/datasets/,
decorate a BaseDataset subclass with @register_dataset, and it's available to
every subcommand -- no reinstall, no edits to vendored code. To add your own,
copy this file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, List, Optional

from sgl_bench._vendored.sglang.benchmark.datasets.common import BaseDataset, DatasetRow
from sgl_bench.registry import register_dataset


def _add_cli_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("jsonl-prompts dataset")
    g.add_argument(
        "--jsonl-prompts-path",
        type=str,
        default=None,
        help='Path to a JSONL file, one {"prompt": "...", "output_len"?: int} per line.',
    )
    g.add_argument(
        "--jsonl-output-len",
        type=int,
        default=128,
        help="Default output length when a row omits output_len (default: 128).",
    )


@register_dataset("jsonl-prompts", add_cli_args=_add_cli_args)
@dataclass
class JsonlPromptsDataset(BaseDataset):
    path: Optional[str] = None
    default_output_len: int = 128

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "JsonlPromptsDataset":
        return cls(
            path=getattr(args, "jsonl_prompts_path", None),
            default_output_len=getattr(args, "jsonl_output_len", 128),
        )

    def load(self, tokenizer: Any, model_id: Optional[str] = None) -> List[DatasetRow]:
        if not self.path:
            raise SystemExit(
                "--jsonl-prompts-path is required for --dataset-name jsonl-prompts"
            )
        rows: List[DatasetRow] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj["prompt"]
                output_len = int(obj.get("output_len", self.default_output_len))
                prompt_len = len(tokenizer.encode(prompt))
                rows.append(
                    DatasetRow(prompt=prompt, prompt_len=prompt_len, output_len=output_len)
                )
        if not rows:
            raise SystemExit(f"no prompts found in {self.path}")
        return rows
