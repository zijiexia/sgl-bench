"""Subcommand base + discovery, plus the vendored-``__main__`` delegator.

Subcommands are discovered by subclassing ``Subcommand`` (vLLM-style, see
vllm/entrypoints/cli/benchmark) rather than a hand-maintained dispatch dict:
drop a module in ``sgl_bench/transport/`` defining a ``Subcommand`` subclass
and it shows up automatically.
"""

from __future__ import annotations

import importlib
import pkgutil
import runpy
import sys
from typing import Dict, List


class Subcommand:
    """One ``sgl-bench <name>`` subcommand. Subclasses set ``name``/``help``
    and implement ``run(argv)``; ``argv`` is everything after the subcommand."""

    name: str = ""
    help: str = ""

    def run(self, argv: List[str]) -> int:
        raise NotImplementedError


def run_vendored_main(module: str, argv: List[str]) -> int:
    """Execute a vendored module's ``if __name__ == "__main__"`` block with
    ``argv``, reusing the upstream argparse + run loop verbatim -- so we never
    duplicate the ~80 bench_serving flags. The vendored parser reads
    ``sys.argv[1:]``; we set it, run, and restore."""
    old_argv = sys.argv
    sys.argv = [module.rsplit(".", 1)[-1], *argv]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return 0
    finally:
        sys.argv = old_argv


def discover() -> Dict[str, Subcommand]:
    """Import every ``sgl_bench.transport`` submodule so ``Subcommand``
    subclasses register, then return ``{name: instance}``."""
    import sgl_bench.transport as pkg

    for _finder, mod_name, _is_pkg in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        if mod_name.rsplit(".", 1)[-1] != "_base":
            importlib.import_module(mod_name)
    return {c.name: c() for c in Subcommand.__subclasses__() if getattr(c, "name", "")}
