"""``sgl-bench latency`` -- single-batch latency sweep (client-mode "one-batch").

Pure HTTP client; requires ``--base-url`` pointing at a running server. Reuses
the vendored ``run_one_case`` primitive + ``BenchArgs`` flags, but owns the
sweep loop: the upstream orchestrator (``run_benchmark_internal``) embeds the
engine launch + GitHub-CI reporting, which we dropped when vendoring.

This mirrors the *client* paths of upstream ``run_benchmark_internal``:
tokenizer + capacity thresholds come from the server (``/v1/models`` for vllm,
``/server_info`` for sglang); pass ``--tokenizer`` to override for a generic
OpenAI endpoint.

Heavy imports are deferred into ``run()`` so subcommand discovery stays cheap.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

from sgl_bench.transport._base import Subcommand

_DEFAULT_TIMEOUT = 600


class LatencySubcommand(Subcommand):
    name = "latency"
    help = "Single-batch latency sweep (client-mode one-batch). Needs --base-url."

    def run(self, argv: List[str]) -> int:
        return _main(argv)


def _build_parser() -> argparse.ArgumentParser:
    from sgl_bench._vendored.sglang.one_batch_client import BenchArgs

    p = argparse.ArgumentParser(prog="sgl-bench latency", description=LatencySubcommand.help)
    BenchArgs.add_cli_args(p)  # --base-url, --batch-size, --input-len, --output-len, ...
    p.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HF model id/dir for client-side tokenization. If omitted, resolved "
        "from the server (/v1/models for vllm, /server_info for sglang).",
    )
    return p


def _resolve_tokenizer_and_thresholds(args) -> Tuple[object, Optional[str], float, float]:
    """Mirror the client-mode tokenizer + skip-threshold resolution from
    upstream run_benchmark_internal. Generic / vllm endpoints get no skip
    guards (thresholds = inf); only sglang's /server_info exposes capacity."""
    import requests

    from sgl_bench._vendored.sglang.benchmark.utils import get_processor, get_tokenizer

    load = get_processor if args.dataset_name == "mmmu" else get_tokenizer
    tok_capacity = float("inf")
    max_running = float("inf")

    if args.tokenizer:
        model_name = None if args.backend == "sglang" else args.tokenizer
        return load(args.tokenizer), model_name, tok_capacity, max_running

    if args.backend == "vllm":
        r = requests.get(args.base_url + "/v1/models", timeout=_DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            raise SystemExit("no models on /v1/models; pass --tokenizer <hf-model-id>")
        model_name = data[0]["id"]
        return load(model_name), model_name, tok_capacity, max_running

    # sglang: /server_info carries tokenizer_path + capacity thresholds.
    try:
        info = requests.get(args.base_url + "/server_info", timeout=_DEFAULT_TIMEOUT).json()
        states = info.get("internal_states", [{}]) or [{}]
        path = (
            info.get("tokenizer_path")
            or states[0].get("tokenizer_path")
            or info.get("prefill", [{}])[0].get("tokenizer_path")
        )
        if not path:
            raise RuntimeError("no tokenizer_path in /server_info")
        tokenizer = load(path)
        dp_size = states[0].get("dp_size") or 1
        mr = states[0].get("effective_max_running_requests_per_dp", -1)
        if mr and mr > 0:
            max_running = mr * dp_size
        tok_capacity = sum(
            states[i].get("memory_usage", {}).get("token_capacity", 1_000_000_000)
            for i in range(min(dp_size, len(states)))
        )
        return tokenizer, None, tok_capacity, max_running
    except Exception as e:
        raise SystemExit(
            f"could not resolve tokenizer from {args.base_url}/server_info ({e}).\n"
            f"Pass --tokenizer <hf-model-id> (required for non-sglang endpoints)."
        )


def _main(argv: List[str]) -> int:
    import itertools
    import random

    import numpy as np
    from tabulate import tabulate

    from sgl_bench._vendored.sglang.one_batch_client import (
        run_one_case,
        should_skip_due_to_max_running_requests,
        should_skip_due_to_token_capacity,
    )

    args = _build_parser().parse_args(argv)
    if not args.base_url:
        raise SystemExit("`sgl-bench latency` needs --base-url pointing at a running server.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer, model_name, tok_capacity, max_running = _resolve_tokenizer_and_thresholds(args)

    common = dict(
        temperature=args.temperature,
        return_logprob=args.return_logprob,
        stream_interval=args.client_stream_interval,
        input_len_step_percentage=args.input_len_step_percentage,
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        parallel_batch=args.parallel_batch,
        backend=args.backend,
        model_name=model_name,
        fake_prefill=args.fake_prefill,
        lora_name=args.lora_name,
        lora_request_distribution=args.lora_request_distribution,
        lora_zipf_alpha=args.lora_zipf_alpha,
        gsp_num_groups=args.gsp_num_groups,
        gsp_system_prompt_len=args.gsp_system_prompt_len,
        gsp_question_len=args.gsp_question_len,
        gsp_output_len=args.gsp_output_len,
    )

    if not args.skip_warmup:
        print("=" * 8 + " Warmup " + "=" * 8)
        for bs in sorted(set(args.batch_size)):
            run_one_case(
                args.base_url, batch_size=bs, input_len=1024, output_len=16,
                run_name="", result_filename="", **common,
            )
        print("=" * 8 + " Warmup End " + "=" * 8 + "\n")

    results = []
    for bs, il, ol in itertools.product(args.batch_size, args.input_len, args.output_len):
        if should_skip_due_to_max_running_requests(
            bs, max_running
        ) or should_skip_due_to_token_capacity(bs, il, ol, tok_capacity):
            continue
        results.append(
            run_one_case(
                args.base_url, bs, il, ol,
                run_name=args.run_name, result_filename=args.result_filename,
                cache_hit_rate=args.cache_hit_rate, **common,
            )
        )

    _print_summary(results, tabulate)
    return 0


def _print_summary(results, tabulate) -> None:
    if not results:
        print("(no cases run)")
        return
    rows = [
        [
            r.batch_size, r.input_len, r.output_len, round(r.latency, 3),
            round(r.input_throughput, 1), round(r.output_throughput, 1),
            round(r.last_ttft, 3), round(r.acc_length, 2),
        ]
        for r in results
    ]
    headers = ["bs", "in", "out", "latency(s)", "in_tput", "out_tput", "ttft(s)", "acc_len"]
    print("\n" + tabulate(rows, headers=headers, tablefmt="github"))
