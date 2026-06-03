"""The whole point of sgl-bench: it stays lightweight. A bare import must not
drag in torch / PIL / pandas / sglang. These guard the import surface."""

import subprocess
import sys


def _import_in_subprocess(stmt: str) -> set[str]:
    """Run `stmt` in a fresh interpreter, return the set of imported top-level
    modules. A subprocess avoids pytest's own already-imported modules."""
    code = f"import sys; {stmt}; print('\\n'.join(sorted(m.split('.')[0] for m in sys.modules)))"
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    return set(out.split())


def test_bare_import_is_light():
    mods = _import_in_subprocess("import sgl_bench")
    for heavy in ("torch", "sglang", "PIL", "pandas", "datasets", "vllm"):
        assert heavy not in mods, f"bare `import sgl_bench` pulled in {heavy!r}"


def test_cli_discovery_is_light():
    # Discovering subcommands must not import the vendored bench engine or any
    # heavy dataset module either -- serve delegates lazily via runpy.
    mods = _import_in_subprocess(
        "from sgl_bench.transport._base import discover; discover()"
    )
    for heavy in ("torch", "sglang", "PIL", "pandas"):
        assert heavy not in mods, f"discover() pulled in {heavy!r}"


def test_serve_subcommand_registered():
    from sgl_bench.transport._base import discover

    assert "serve" in discover()


def test_net_slice_to_url():
    from sgl_bench._vendored.sglang._net import NetworkAddress

    assert NetworkAddress("127.0.0.1", 30000).to_url() == "http://127.0.0.1:30000"
    assert NetworkAddress("::1", 30000).to_url() == "http://[::1]:30000"
