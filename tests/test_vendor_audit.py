"""Drift guard: no untranslated upstream (`sglang`) imports survive under
sgl_bench/_vendored/. Wraps scripts/audit_vendored.py (which is not an
installed package, so we load it by path)."""

import importlib.util
from pathlib import Path


def _load_audit():
    path = Path(__file__).resolve().parent.parent / "scripts" / "audit_vendored.py"
    spec = importlib.util.spec_from_file_location("audit_vendored", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_untranslated_vendored_imports():
    audit = _load_audit()
    bad = audit.find_untranslated()
    assert bad == [], "untranslated upstream imports: " + ", ".join(
        f"{p.name}:{hits}" for p, hits in bad
    )
