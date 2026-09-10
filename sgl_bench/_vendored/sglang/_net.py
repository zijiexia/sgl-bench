# Hand-curated minimal slice of sgl-project/sglang
# python/sglang/srt/utils/network.py @ 0bcd822377da7b5718e674eaf9c870d349424dd1
#
# Only NetworkAddress (+ the _wrap/_is_ipv6 helpers it needs) and the two
# resolve_* wrappers are vendored -- that is everything bench_serving imports.
# The upstream module top-imports zmq + psutil, which we deliberately do not
# pull in -- hence this is a manual slice, NOT auto-synced by
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


def resolve_base_url(base_url: str, host: str, port: int) -> str:
    """Base URL a client sends to: ``base_url`` if set, else ``http://host:port``
    (IPv6-correct via :class:`NetworkAddress`)."""
    if base_url:
        return base_url
    return NetworkAddress(host, port).to_url()


def resolve_host_port(base_url: str, host: str, port: int) -> str:
    """Like :func:`resolve_base_url` but returns the scheme-less ``host:port``
    form (for gRPC-style endpoints): ``base_url`` if set, else ``host:port``
    (IPv6-correct via :class:`NetworkAddress`)."""
    if base_url:
        return base_url
    return NetworkAddress(host, port).to_host_port_str()
