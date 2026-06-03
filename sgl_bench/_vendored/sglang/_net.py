# Hand-curated minimal slice of sgl-project/sglang
# python/sglang/srt/utils/network.py @ e5b8e3a66aa6052d86905869a7dd7c816c8401f7
#
# Only NetworkAddress (+ the _wrap/_is_ipv6 helpers it needs for to_url) is
# vendored: bench_serving uses NetworkAddress(host, port).to_url() and nothing
# else. The upstream module top-imports zmq + psutil, which we deliberately do
# not pull in -- hence this is a manual slice, NOT auto-synced by
# scripts/sync_vendored.py. Re-verify by hand when bumping the pinned SHA.

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass


def _is_ipv6(host: str) -> bool:
    """Check whether *host* is a valid IPv6 address (without brackets)."""
    try:
        ipaddress.IPv6Address(host)
        return True
    except ValueError:
        return False


def _wrap(host: str) -> str:
    """Wrap an IPv6 address in brackets; pass IPv4/hostname through."""
    return f"[{host}]" if _is_ipv6(host) else host


@dataclass(frozen=True)
class NetworkAddress:
    host: str
    port: int

    def __post_init__(self):
        # Auto-strip IPv6 brackets so callers can pass "[::1]" or "::1"
        if self.host.startswith("[") and self.host.endswith("]"):
            object.__setattr__(self, "host", self.host[1:-1])

    @property
    def is_ipv6(self) -> bool:
        return _is_ipv6(self.host)

    @property
    def family(self) -> socket.AddressFamily:
        return socket.AF_INET6 if self.is_ipv6 else socket.AF_INET

    def to_url(self, scheme: str = "http") -> str:
        """``http://127.0.0.1:30000`` or ``http://[::1]:30000``."""
        return f"{scheme}://{_wrap(self.host)}:{self.port}"

    def to_tcp(self) -> str:
        """``tcp://`` endpoint for ZMQ / torch distributed."""
        return self.to_url("tcp")

    def to_host_port_str(self) -> str:
        """``host:port`` string for gRPC listen address, session IDs, logs."""
        return f"{_wrap(self.host)}:{self.port}"

    def to_bind_tuple(self) -> tuple[str, int]:
        """Raw ``(host, port)`` tuple for ``socket.bind()`` / ``socket.connect()``."""
        return (self.host, self.port)
