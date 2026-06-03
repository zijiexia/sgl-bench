"""Static check: no untranslated upstream imports under ``sgl_bench/_vendored/``.

Each vendored package's SOURCES.yaml declares ``forbid_import_prefixes`` (e.g.
``[sglang]``); any ``from <prefix>...`` / ``import <prefix>...`` that survived
the sync rewrites is a leak and fails the audit. Run by
tests/test_vendor_audit.py and manually as a CI invariant.

Generalized from sgl-eval's single hard-coded ``nemo_skills`` regex so one
script can guard multiple upstreams. The word-boundary regex matches bare
``sglang`` / ``sglang.`` but NOT ``sgl_bench._vendored.sglang`` (those lines
start with ``sgl_bench``), so the legitimate re-pointed imports pass.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import yaml

ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = ROOT / "sgl_bench" / "_vendored"


def _forbidden_prefixes() -> List[str]:
    prefixes: List[str] = []
    for manifest in VENDOR_ROOT.rglob("SOURCES.yaml"):
        spec = yaml.safe_load(manifest.read_text()) or {}
        prefixes.extend(spec.get("forbid_import_prefixes", []) or [])
    return sorted(set(prefixes))


def _pattern(prefixes: List[str]) -> "re.Pattern[str] | None":
    if not prefixes:
        return None
    alt = "|".join(re.escape(p) for p in prefixes)
    return re.compile(rf"^\s*(?:from|import)\s+(?:{alt})(?:\.|\s|$)", re.MULTILINE)


def find_untranslated() -> List[tuple[Path, list[str]]]:
    bad: list[tuple[Path, list[str]]] = []
    if not VENDOR_ROOT.exists():
        return bad
    pattern = _pattern(_forbidden_prefixes())
    if pattern is None:
        return bad
    for py in VENDOR_ROOT.rglob("*.py"):
        text = py.read_text()
        hits = pattern.findall(text)
        if hits:
            bad.append((py, hits))
    return bad


def main() -> int:
    bad = find_untranslated()
    if bad:
        print("Untranslated upstream imports found:")
        for path, hits in bad:
            print(f"  {path.relative_to(ROOT)}: {len(hits)} occurrence(s)")
        return 1
    print("OK: all _vendored imports translated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
