"""Sync vendored sources from upstream repos based on SOURCES.yaml.

Usage:
    python scripts/sync_vendored.py                # all vendored packages
    python scripts/sync_vendored.py sglang         # one specific subpackage

For each manifest:
  1. Reads ``sgl_bench/_vendored/<pkg>/SOURCES.yaml``.
  2. Fetches each `src` file at `synced_from_sha` via `gh api`.
  3. Applies per-file ``replacements`` (textual {old, new}) -- run FIRST, on the
     original upstream text, so they can swap/remove import lines and code
     blocks before the prefix rewrites below see them.
  4. Applies global ``import_rewrites`` (textual, .py only).
  5. Removes top-level functions/methods listed in `drop_functions`.
  6. Removes ``from <module>`` blocks matching prefixes in `drop_imports`.
  7. Writes to `dst` with a provenance banner:
     - ``.py`` and ``.yaml`` get a ``#`` comment header.
     - Files marked ``binary: true`` are copied verbatim with no banner.

Idempotent: re-running with the same SHA yields the same files.

Adapted from sgl-eval's scripts/sync_vendored.py; the only addition is the
per-file ``replacements`` step (3), which lets a single capability cover both
"inline an srt constant", "re-point an srt import", and "delete a nested
``if`` block" without bespoke AST surgery.
"""

from __future__ import annotations

import argparse
import ast
import base64
import subprocess
from pathlib import Path
from typing import Iterable, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = ROOT / "sgl_bench" / "_vendored"

BANNER_PY = (
    "# Vendored from {repo}@{sha}\n"
    "# Source: {src}\n"
    "# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun\n"
    "# `python scripts/sync_vendored.py`.\n\n"
)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "packages",
        nargs="*",
        help="vendored subpackages to sync (default: all under sgl_bench/_vendored/)",
    )
    args = parser.parse_args(argv)

    pkgs = args.packages or [p.name for p in _vendored_packages()]
    if not pkgs:
        print("(no vendored packages found)")
        return 0

    rc = 0
    for name in pkgs:
        pkg_dir = VENDOR_ROOT / name
        manifest = pkg_dir / "SOURCES.yaml"
        if not manifest.exists():
            print(f"SKIP {name}: no SOURCES.yaml")
            rc = 1
            continue
        sync_one(pkg_dir, manifest)
    return rc


def _vendored_packages() -> Iterable[Path]:
    if not VENDOR_ROOT.exists():
        return []
    return [p for p in VENDOR_ROOT.iterdir() if p.is_dir() and (p / "SOURCES.yaml").exists()]


def sync_one(pkg_dir: Path, manifest: Path) -> None:
    spec = yaml.safe_load(manifest.read_text())
    repo = spec["upstream_repo"]
    sha = spec["synced_from_sha"]
    rewrites = spec.get("import_rewrites", {}) or {}
    files = spec.get("files", []) or []

    print(f"==> {pkg_dir.relative_to(ROOT)}  (upstream={repo}@{sha[:12]})")
    for entry in files:
        src = entry["src"]
        dst = pkg_dir / entry["dst"]
        is_binary = bool(entry.get("binary", False))
        replacements = entry.get("replacements", []) or []
        drop_fns = entry.get("drop_functions", []) or []
        drop_imports = entry.get("drop_imports", []) or []

        dst.parent.mkdir(parents=True, exist_ok=True)

        if is_binary:
            payload = fetch_bytes(repo, src, sha)
            dst.write_bytes(payload)
            print(f"    wrote {dst.relative_to(ROOT)}  ({len(payload)} bytes)")
            continue

        content = fetch(repo, src, sha)
        if dst.suffix == ".py":
            for rep in replacements:
                old, new = rep["old"], rep.get("new", "")
                if old not in content:
                    raise ValueError(
                        f"{src}: replacement target not found (upstream drift?):\n{old!r}"
                    )
                content = content.replace(old, new)
            for old, new in rewrites.items():
                content = content.replace(old, new)
            for fn in drop_fns:
                content = drop_function(content, fn)
            for prefix in drop_imports:
                content = drop_import(content, prefix)
            banner = BANNER_PY.format(repo=repo, sha=sha, src=src)
            dst.write_text(banner + content)
        elif dst.suffix in {".yaml", ".yml"}:
            banner = BANNER_PY.format(repo=repo, sha=sha, src=src)
            dst.write_text(banner + content)
        else:
            dst.write_text(content)
        print(f"    wrote {dst.relative_to(ROOT)}")


def fetch(repo: str, src: str, sha: str) -> str:
    out = subprocess.check_output(
        ["gh", "api", f"repos/{repo}/contents/{src}?ref={sha}", "--jq", ".content"]
    )
    return base64.b64decode(out).decode()


def fetch_bytes(repo: str, src: str, sha: str) -> bytes:
    out = subprocess.check_output(
        ["gh", "api", f"repos/{repo}/contents/{src}?ref={sha}", "--jq", ".content"]
    )
    return base64.b64decode(out)


def drop_function(content: str, name: str) -> str:
    """Remove a top-level def or class method named `name` from `content`."""
    tree = ast.parse(content)
    spans: list[tuple[int, int]] = []

    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
        ):
            spans.append((node.lineno, node.end_lineno or node.lineno))

    if not spans:
        return content

    return _drop_line_ranges(content, spans)


def drop_import(content: str, module_prefix: str) -> str:
    """Remove ``from <module>...`` and ``import <module>...`` blocks where
    the module path starts with ``module_prefix``."""
    tree = ast.parse(content)
    spans: list[tuple[int, int]] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module == module_prefix or node.module.startswith(module_prefix + ".")
            ):
                spans.append((node.lineno, node.end_lineno or node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_prefix or alias.name.startswith(module_prefix + "."):
                    spans.append((node.lineno, node.end_lineno or node.lineno))
                    break

    if not spans:
        return content
    return _drop_line_ranges(content, spans)


def _drop_line_ranges(content: str, spans: list[tuple[int, int]]) -> str:
    lines = content.split("\n")
    drop = set()
    for start, end in spans:
        for i in range(start - 1, end):
            drop.add(i)
    return "\n".join(line for idx, line in enumerate(lines) if idx not in drop)


if __name__ == "__main__":
    raise SystemExit(main())
