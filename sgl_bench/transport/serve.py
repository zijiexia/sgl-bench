"""``sgl-bench serve`` -- online serving benchmark (TTFT/ITL/throughput under
a controlled request-rate / concurrency). Pure HTTP client; needs ``--base-url``
(or ``--host``/``--port``) pointing at an already-running server.

Delegates verbatim to the vendored ``bench_serving`` CLI, so every upstream
flag (``--backend``, ``--dataset-name``, ``--num-prompts``, ``--request-rate``,
``--max-concurrency``, all per-dataset flags) works unchanged.
"""

from __future__ import annotations

from typing import List

from sgl_bench.transport._base import Subcommand, run_vendored_main

_VENDORED = "sgl_bench._vendored.sglang.bench_serving"


class ServeSubcommand(Subcommand):
    name = "serve"
    help = "Online serving benchmark under load (TTFT/ITL/throughput). Needs --base-url."

    def run(self, argv: List[str]) -> int:
        return run_vendored_main(_VENDORED, argv)
