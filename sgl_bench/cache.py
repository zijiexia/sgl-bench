"""Backend-aware prefix-cache flushing -- the ``--flush-cache`` seam.

The reset endpoint is not the same across servers:

* sglang exposes ``POST /flush_cache``;
* vllm exposes ``POST /reset_prefix_cache``, and only mounts it when the
  server was started with ``VLLM_SERVER_DEV_MODE=1``.

The vendored one-batch client (``one_batch_client.run_one_case``) already
branches on the backend upstream; this module is the same rule for the
online-serving path, wired into ``bench_serving`` by a SOURCES.yaml rewrite
(see ``_vendored/sglang/SOURCES.yaml``) so no vendored file is hand-edited.

Backends with no flush endpoint at all (``trt``, ``gserver``, ``truss``, ...)
keep the sglang default and simply get a warning if the POST is rejected --
a silently-unflushed cache would quietly invalidate a cold-cache measurement.
"""

from __future__ import annotations

from typing import Dict, Optional

SGLANG_FLUSH_ENDPOINT = "/flush_cache"
VLLM_FLUSH_ENDPOINT = "/reset_prefix_cache"

# Only vllm names its endpoint differently today; ``vllm``/``vllm-chat`` both
# hit the same server, so match on the prefix rather than the exact name.
_VLLM_BACKEND_PREFIX = "vllm"


def flush_cache_endpoint(backend: str) -> str:
    """Return the cache-reset HTTP path for ``backend`` (a bench backend name
    from ``ASYNC_REQUEST_FUNCS``: ``sglang``, ``sglang-oai``, ``vllm``, ...)."""
    if (backend or "").startswith(_VLLM_BACKEND_PREFIX):
        return VLLM_FLUSH_ENDPOINT
    return SGLANG_FLUSH_ENDPOINT


def flush_cache(
    base_url: str,
    backend: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 60.0,
) -> bool:
    """POST the backend's cache-reset endpoint. Returns True on success and
    warns (without raising) otherwise -- a failed flush should not kill a
    benchmark run, but it must not pass for a cold cache either."""
    import requests

    endpoint = flush_cache_endpoint(backend)
    try:
        response = requests.post(base_url + endpoint, headers=headers, timeout=timeout)
    except Exception as e:  # network/connection level
        print(f"Warning: cache flush POST {endpoint} failed: {e}")
        return False

    if response.status_code != 200:
        hint = ""
        if endpoint == VLLM_FLUSH_ENDPOINT:
            hint = " (vllm only mounts it with VLLM_SERVER_DEV_MODE=1 set on the server)"
        print(
            f"Warning: cache flush POST {endpoint} returned "
            f"{response.status_code}{hint}; the cache was NOT flushed."
        )
        return False
    return True
