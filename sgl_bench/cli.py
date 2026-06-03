"""``sgl-bench`` CLI entry point.

A pure client-side performance benchmark: every subcommand talks HTTP to an
*already-running* OpenAI-compatible / sglang endpoint (``--base-url``); nothing
here loads a model engine.

Subcommands (discovered from ``sgl_bench.transport``):
  serve         online serving benchmark under load
  latency       single-batch latency sweep            (added in M3)
  throughput    saturation / max-throughput           (added in M3)
  list-datasets enumerate available datasets
"""

from __future__ import annotations

import sys
from typing import List, Optional

from sgl_bench.transport._base import discover


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = discover()

    if not argv or argv[0] in ("-h", "--help"):
        _print_usage(cmds)
        return 0 if argv else 2

    name, rest = argv[0], argv[1:]

    if name == "list-datasets":
        return _list_datasets()

    cmd = cmds.get(name)
    if cmd is None:
        print(f"sgl-bench: unknown subcommand {name!r}\n", file=sys.stderr)
        _print_usage(cmds)
        return 2
    return cmd.run(rest)


def _print_usage(cmds) -> None:
    print("usage: sgl-bench <subcommand> [options]\n")
    print("Client-side perf benchmark for a running endpoint (always needs --base-url).\n")
    print("subcommands:")
    width = max([len(n) for n in cmds] + [len("list-datasets")])
    for n in sorted(cmds):
        print(f"  {n:<{width}}  {cmds[n].help}")
    print(f"  {'list-datasets':<{width}}  List available --dataset-name values.")
    print("\nRun `sgl-bench <subcommand> --help` for per-subcommand flags.")


def _list_datasets() -> int:
    # Lazy import so a bare `import sgl_bench` never pulls the registry /
    # vendored dataset modules (and their deps) until this command is run.
    from sgl_bench.registry import all_specs

    specs = all_specs()
    width = max((len(n) for n in specs), default=0)
    print("available datasets (--dataset-name):")
    for name in sorted(specs):
        spec = specs[name]
        extra = f"  [needs {spec.needs_extra}]" if spec.needs_extra else ""
        print(f"  {name:<{width}}  ({spec.source}){extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
