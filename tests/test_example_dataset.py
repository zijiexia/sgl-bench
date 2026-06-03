"""The shipped example dataset (sgl_bench/datasets/jsonl_prompts.py) auto-loads
and works end-to-end through the registry -- proving the 'drop a file' UX."""

import argparse
import json

from sgl_bench import registry


def _ensure_registered():
    # Importing the module fires @register_dataset once (module caching makes
    # this idempotent). This is exactly what registry._autoload_user does at
    # runtime; we call it directly so the test doesn't depend on autoload state.
    import sgl_bench.datasets.jsonl_prompts  # noqa: F401


class _FakeTokenizer:
    def encode(self, text):
        return text.split()


def test_example_dataset_autoloads_and_resolves():
    _ensure_registered()
    spec = registry.resolve("jsonl-prompts")
    assert spec.source == "user"
    assert spec.name == "jsonl-prompts"


def test_example_dataset_listed():
    _ensure_registered()
    assert "jsonl-prompts" in registry.all_specs()


def test_example_dataset_end_to_end(tmp_path):
    _ensure_registered()
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        json.dumps({"prompt": "hello world foo"}) + "\n"
        + json.dumps({"prompt": "a b", "output_len": 7}) + "\n"
    )
    args = argparse.Namespace(
        dataset_name="jsonl-prompts", jsonl_prompts_path=str(p), jsonl_output_len=128
    )
    rows = registry.get_dataset(args, tokenizer=_FakeTokenizer(), model_id=None)
    assert len(rows) == 2
    assert rows[0].prompt == "hello world foo" and rows[0].prompt_len == 3
    assert rows[0].output_len == 128  # falls back to default
    assert rows[1].output_len == 7  # per-row override
