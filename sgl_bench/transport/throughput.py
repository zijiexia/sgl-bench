"""``sgl-bench throughput`` -- saturation / max-throughput.

This is the client-mode equivalent of an "offline throughput" run: it's just
``serve`` with ``--request-rate inf`` (send as fast as the server drains),
typically combined with ``--max-concurrency``. No new engine -- a thin preset
over the vendored bench_serving CLI. Needs ``--base-url``.
"""

from __future__ import annotations

from typing import List

from sgl_bench.transport._base import Subcommand, run_vendored_main

_VENDORED = "sgl_bench._vendored.sglang.bench_serving"


def _ensure_request_rate_inf(argv: List[str]) -> List[str]:
    """Force the saturation regime unless the user set a rate explicitly."""
    if any(a == "--request-rate" or a.startswith("--request-rate=") for a in argv):
        return list(argv)
    return ["--request-rate", "inf", *argv]


class ThroughputSubcommand(Subcommand):
    name = "throughput"
    help = "Saturation / max throughput (request-rate=inf). Needs --base-url."

    def run(self, argv: List[str]) -> int:
        return run_vendored_main(_VENDORED, _ensure_request_rate_inf(argv))
