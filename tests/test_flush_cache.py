"""``--flush-cache`` routes to the backend's own reset endpoint: sglang keeps
``/flush_cache``, vllm gets ``/reset_prefix_cache``. A rejected flush warns
loudly instead of silently benchmarking against a warm cache."""

import pytest

from sgl_bench import cache


@pytest.mark.parametrize(
    "backend,endpoint",
    [
        ("sglang", "/flush_cache"),
        ("sglang-native", "/flush_cache"),
        ("sglang-oai", "/flush_cache"),
        ("sglang-oai-chat", "/flush_cache"),
        ("vllm", "/reset_prefix_cache"),
        ("vllm-chat", "/reset_prefix_cache"),
        # Backends with no flush endpoint keep the sglang default.
        ("lmdeploy", "/flush_cache"),
        ("trt", "/flush_cache"),
    ],
)
def test_endpoint_per_backend(backend, endpoint):
    assert cache.flush_cache_endpoint(backend) == endpoint


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


def _patch_post(monkeypatch, status_code=200):
    calls = []

    def fake_post(url, headers=None, timeout=None):
        calls.append((url, headers))
        return _Resp(status_code)

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    return calls


def test_flush_cache_posts_backend_endpoint(monkeypatch):
    calls = _patch_post(monkeypatch)

    assert cache.flush_cache("http://h:8000", "vllm", headers={"a": "b"}) is True
    assert cache.flush_cache("http://h:30000", "sglang") is True

    assert calls == [
        ("http://h:8000/reset_prefix_cache", {"a": "b"}),
        ("http://h:30000/flush_cache", None),
    ]


def test_failed_flush_warns_and_returns_false(monkeypatch, capsys):
    _patch_post(monkeypatch, status_code=404)

    assert cache.flush_cache("http://h:8000", "vllm") is False
    out = capsys.readouterr().out
    assert "/reset_prefix_cache" in out
    assert "VLLM_SERVER_DEV_MODE=1" in out


def test_connection_error_does_not_raise(monkeypatch, capsys):
    import requests

    def boom(*a, **kw):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)

    assert cache.flush_cache("http://h:30000", "sglang") is False
    assert "/flush_cache" in capsys.readouterr().out


def test_vendored_bench_serving_uses_the_helper():
    """The SOURCES.yaml rewrite must survive a re-sync -- if upstream moves the
    seam, sync raises; if someone reverts it, this catches the regression."""
    from pathlib import Path

    src = Path(cache.__file__).with_name("_vendored") / "sglang" / "bench_serving.py"
    text = src.read_text()
    assert "from sgl_bench.cache import flush_cache" in text
    assert 'requests.post(base_url + "/flush_cache"' not in text
