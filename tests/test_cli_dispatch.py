"""Subcommand discovery + arg wiring, without touching the network."""

import pytest

from sgl_bench import cli
from sgl_bench.transport._base import discover


def test_all_three_subcommands_discovered():
    cmds = discover()
    assert {"serve", "latency", "throughput"} <= set(cmds)


def test_throughput_injects_request_rate_inf():
    from sgl_bench.transport.throughput import _ensure_request_rate_inf

    assert _ensure_request_rate_inf(["--base-url", "x"]) == [
        "--request-rate", "inf", "--base-url", "x",
    ]


def test_throughput_preserves_explicit_request_rate():
    from sgl_bench.transport.throughput import _ensure_request_rate_inf

    argv = ["--request-rate", "8", "--base-url", "x"]
    assert _ensure_request_rate_inf(argv) == argv
    assert _ensure_request_rate_inf(["--request-rate=8"]) == ["--request-rate=8"]


def test_latency_parser_builds_and_parses_sweep_flags():
    from sgl_bench.transport.latency import _build_parser

    args = _build_parser().parse_args(
        ["--base-url", "http://h:1/v1", "--batch-size", "1", "8", "--input-len", "1024"]
    )
    assert args.base_url == "http://h:1/v1"
    assert tuple(args.batch_size) == (1, 8)
    assert tuple(args.input_len) == (1024,)


def test_latency_requires_base_url():
    from sgl_bench.transport.latency import _main

    with pytest.raises(SystemExit):
        _main(["--batch-size", "1"])  # base-url defaults to "" -> must error before any network


def test_cli_no_args_prints_usage():
    assert cli.main([]) == 2


def test_cli_unknown_subcommand():
    assert cli.main(["bogus-subcommand"]) == 2
