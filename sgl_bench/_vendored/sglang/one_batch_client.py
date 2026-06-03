# Vendored from sgl-project/sglang@e5b8e3a66aa6052d86905869a7dd7c816c8401f7
# Source: python/sglang/test/bench_one_batch_server_internal.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

import argparse
import dataclasses
import itertools
import json
import multiprocessing
import os
import random
import re
import time
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests
from pydantic import BaseModel
from tabulate import tabulate
from transformers import AutoProcessor, PreTrainedTokenizer

from sgl_bench.registry import get_dataset
from sgl_bench._vendored.sglang.benchmark.utils import get_processor, get_tokenizer
def run_profile(*args, **kwargs):
    raise RuntimeError(
        "profiling needs the in-process engine; unavailable in sgl-bench client mode"
    )
FAKE_BOOTSTRAP_HOST = "2.2.2.2"

DEFAULT_TIMEOUT = 600


def get_cache_tokens_from_metrics(url: str) -> Optional[tuple]:
    """
    Get cached_tokens_total and prompt_tokens_total from Prometheus /metrics endpoint.
    Returns (cached_tokens_total, prompt_tokens_total) or None if metrics are not available.
    """
    try:
        response = requests.get(url + "/metrics", timeout=5)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            return None

        # Parse Prometheus text format
        # Looking for: sglang:cached_tokens_total{...} <value>
        #              sglang:prompt_tokens_total{...} <value>
        cached_tokens_total = 0.0
        prompt_tokens_total = 0.0

        for line in response.text.split("\n"):
            if line.startswith("sglang:cached_tokens_total{"):
                match = re.search(
                    r"sglang:cached_tokens_total\{[^}]*\}\s+([\d.eE+-]+)", line
                )
                if match:
                    cached_tokens_total += float(match.group(1))
            elif line.startswith("sglang:prompt_tokens_total{"):
                match = re.search(
                    r"sglang:prompt_tokens_total\{[^}]*\}\s+([\d.eE+-]+)", line
                )
                if match:
                    prompt_tokens_total += float(match.group(1))

        return (cached_tokens_total, prompt_tokens_total)
    except Exception as e:
        print(f"Warning: Failed to get cache tokens from metrics: {e}")
        return None


def calculate_cache_hit_rate(
    before: Optional[tuple], after: Optional[tuple]
) -> Optional[float]:
    """
    Calculate cache hit rate from before/after metrics snapshots.
    Returns cached_tokens_delta / prompt_tokens_delta for the benchmark run.
    """
    if before is None or after is None:
        return None

    cached_delta = after[0] - before[0]
    prompt_delta = after[1] - before[1]

    if prompt_delta > 0:
        return cached_delta / prompt_delta
    return None


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    temperature: float = 0.0
    return_logprob: bool = False
    client_stream_interval: int = 1
    input_len_step_percentage: float = 0.0
    base_url: str = ""
    skip_warmup: bool = False
    show_report: bool = False
    profile: bool = False
    profile_activities: Tuple[str] = ("CPU", "GPU")
    profile_start_step: Optional[int] = None
    profile_steps: int = 5
    profile_by_stage: bool = False
    profile_prefix: Optional[str] = None
    profile_output_dir: Optional[str] = None
    dataset_path: str = ""
    dataset_name: str = "random"
    gsp_num_groups: int = 1
    gsp_system_prompt_len: int = 2048
    gsp_question_len: int = 128
    gsp_output_len: int = 256
    parallel_batch: bool = False
    result_filename: str = "result.jsonl"
    pydantic_result_filename: Optional[str] = None
    append_to_github_summary: bool = True
    seed: int = 42
    cache_hit_rate: float = 0.0
    backend: str = "sglang"
    fake_prefill: bool = False
    server_args_for_metrics: Optional[List[str]] = None
    lora_name: Optional[List[str]] = None
    lora_request_distribution: str = "uniform"
    lora_zipf_alpha: float = 1.1
    enable_multi_batch: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument("--temperature", type=float, default=BenchArgs.temperature)
        parser.add_argument("--return-logprob", action="store_true")
        parser.add_argument(
            "--client-stream-interval",
            type=int,
            default=BenchArgs.client_stream_interval,
        )
        parser.add_argument(
            "--input-len-step-percentage",
            type=float,
            default=BenchArgs.input_len_step_percentage,
        )
        parser.add_argument("--base-url", type=str, default=BenchArgs.base_url)
        parser.add_argument("--skip-warmup", action="store_true")
        parser.add_argument("--show-report", action="store_true")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument(
            "--profile-activities",
            type=str,
            nargs="+",
            default=("CPU", "GPU"),
            choices=["CPU", "GPU", "XPU"],
            help="Profiler activities: CPU, GPU, XPU. use torch profiler.",
        )
        parser.add_argument(
            "--profile-start-step",
            type=int,
            default=BenchArgs.profile_start_step,
            help="Start profiling after this many forward steps. Useful for warmup.",
        )
        parser.add_argument(
            "--profile-steps", type=int, default=BenchArgs.profile_steps
        )
        parser.add_argument("--profile-by-stage", action="store_true")
        parser.add_argument(
            "--profile-prefix",
            type=str,
            default=BenchArgs.profile_prefix,
        )
        parser.add_argument(
            "--profile-output-dir",
            type=str,
            default=BenchArgs.profile_output_dir,
        )
        parser.add_argument(
            "--dataset-path",
            type=str,
            default=BenchArgs.dataset_path,
            help="Path to the dataset.",
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default=BenchArgs.dataset_name,
            choices=["mmmu", "random", "random-ids", "generated-shared-prefix"],
            help="Name of the dataset to benchmark on.",
        )
        parser.add_argument(
            "--gsp-num-groups",
            type=int,
            default=BenchArgs.gsp_num_groups,
            help="Number of shared prefix groups. batch_size requests are distributed across groups.",
        )
        parser.add_argument(
            "--gsp-system-prompt-len",
            type=int,
            default=BenchArgs.gsp_system_prompt_len,
            help="Length of the shared system prompt in tokens per group.",
        )
        parser.add_argument(
            "--gsp-question-len",
            type=int,
            default=BenchArgs.gsp_question_len,
            help="Length of the unique question suffix in tokens per request.",
        )
        parser.add_argument(
            "--gsp-output-len",
            type=int,
            default=BenchArgs.gsp_output_len,
            help="Output length in tokens for generated-shared-prefix requests.",
        )
        parser.add_argument("--parallel-batch", action="store_true")
        parser.add_argument(
            "--result-filename",
            type=str,
            default=BenchArgs.result_filename,
            help="Store the results line by line in the JSON Line format to this file.",
        )
        parser.add_argument(
            "--pydantic-result-filename",
            type=str,
            default=BenchArgs.pydantic_result_filename,
            help="Store the results as pydantic models in the JSON format to this file.",
        )
        parser.add_argument(
            "--no-append-to-github-summary",
            action="store_false",
            dest="append_to_github_summary",
            help="Disable appending the output of this run to github ci summary",
        )
        parser.add_argument("--seed", type=int, default=BenchArgs.seed)
        parser.add_argument(
            "--cache-hit-rate",
            type=float,
            default=BenchArgs.cache_hit_rate,
            help="Cache hit rate for benchmarking (0.0-1.0). "
            "0.0 means no cache hits (flush all), 0.4 means 40%% of input tokens are cached.",
        )
        parser.add_argument(
            "--backend",
            type=str,
            default=BenchArgs.backend,
            choices=["sglang", "vllm"],
            help="Backend server type (sglang or vllm).",
        )
        parser.add_argument(
            "--fake-prefill",
            action="store_true",
            default=BenchArgs.fake_prefill,
            help="Enable fake prefill mode for decode-only benchmarking. "
            "Use with a decode server running --disaggregation-transfer-backend fake "
            "to benchmark pure decode performance without a real prefill node.",
        )
        parser.add_argument(
            "--server-args-for-metrics",
            type=str,
            nargs="*",
            default=None,
            help="Server launch arguments to record in metrics output (for tracking configurations).",
        )
        parser.add_argument(
            "--lora-name",
            type=str,
            nargs="*",
            default=BenchArgs.lora_name,
            help="Name(s) of pre-loaded LoRA adapter(s) to apply to the batch "
            "(sent as `lora_path` in the SGLang /generate payload). Requires "
            "the server to be launched with --enable-lora and --lora-paths "
            "<name>=<path> for every name listed here. Pass one name to apply "
            "a single adapter to every prompt, or multiple names to sample a "
            "per-prompt adapter per --lora-request-distribution.",
        )
        parser.add_argument(
            "--lora-request-distribution",
            type=str,
            default=BenchArgs.lora_request_distribution,
            choices=["uniform", "distinct", "skewed"],
            help="How to sample a LoRA adapter per prompt when more than one "
            "is listed in --lora-name. Mirrors bench_serving.py. "
            "'uniform' picks uniformly at random, 'distinct' round-robins so "
            "consecutive prompts get different adapters, 'skewed' samples "
            "from a Zipf distribution over --lora-name (alpha controls the "
            "skew; see --lora-zipf-alpha).",
        )
        parser.add_argument(
            "--lora-zipf-alpha",
            type=float,
            default=BenchArgs.lora_zipf_alpha,
            help="Zipf exponent for 'skewed' LoRA sampling: the number of "
            "requests to adapter i is alpha times the number to adapter i+1. "
            "Must be > 1. Only used when --lora-request-distribution=skewed.",
        )
        parser.add_argument(
            "--enable-multi-batch",
            action="store_true",
            help=(
                "Allow --batch-size to exceed the server's "
                "effective_max_running_requests_per_dp * dp_size. The surplus "
                "requests are queued by the scheduler and promoted as slots "
                "free, so the batch is served as multiple sequential batches "
                "at the running-batch cap. Useful for stabilizing throughput "
                "measurements: driving more total prompts through a "
                "fixed running batch amortizes per-request prefill and "
                "first-step transients into steady-state decode. "
                "NOTE: only `overall_throughput` (= total_tokens / wall_time) "
                "is meaningful in this mode; input_throughput, "
                "output_throughput, last_ttft, and ITL assume one-shot "
                "batching and will be misleading."
            ),
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


class BenchOneCaseResult(BaseModel):
    run_name: str
    batch_size: int
    input_len: int
    output_len: int
    latency: float
    input_throughput: float
    output_throughput: float
    overall_throughput: float
    last_ttft: float
    last_gen_throughput: float
    acc_length: float
    cache_hit_rate: Optional[float] = None
    profile_link: Optional[str] = None

    def dump_to_jsonl(self, result_filename: str):
        with open(result_filename, "a") as fout:
            res = {
                "run_name": self.run_name,
                "batch_size": self.batch_size,
                "input_len": self.input_len,
                "output_len": self.output_len,
                "latency": round(self.latency, 4),
                "input_throughput": round(self.input_throughput, 2),
                "output_throughput": round(self.output_throughput, 2),
                "overall_throughput": round(self.overall_throughput, 2),
                "last_ttft": round(self.last_ttft, 4),
                "last_gen_throughput": round(self.last_gen_throughput, 2),
                "acc_length": round(self.acc_length, 2),
                "cache_hit_rate": (
                    round(self.cache_hit_rate, 4)
                    if self.cache_hit_rate is not None
                    else None
                ),
            }
            fout.write(json.dumps(res) + "\n")






def _warmup_cache(
    url: str,
    input_ids: List[List[int]],
    input_len: int,
    cache_hit_rate: float,
    dataset_name: str = "random",
    image_data: Optional[List] = None,
    backend: str = "sglang",
    model_name: Optional[str] = None,
):
    """Warm up the cache by sending prefix tokens to populate the radix/prefix cache.

    Args:
        url: Server URL
        input_ids: List of input token id lists
        input_len: Length of input tokens
        cache_hit_rate: Fraction of input tokens to cache (0.0-1.0)
        dataset_name: Name of the dataset (used to determine if image data should be included)
        image_data: Optional image data for VLM models
        backend: Backend server type ("sglang" or "vllm")
        model_name: Model name (required for vllm backend)
    """
    cached_token_len = int(input_len * cache_hit_rate)
    if cached_token_len <= 0:
        return

    print(
        f"Warming up cache with {cache_hit_rate*100:.1f}% hit rate "
        f"({cached_token_len} tokens per request)"
    )
    # Create prefix input_ids for cache warming
    cache_warmup_input_ids = [ids[:cached_token_len] for ids in input_ids]

    if backend == "vllm":
        cache_warmup_payload = {
            "model": model_name,
            "prompt": cache_warmup_input_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        gen_url = url + "/v1/completions"
    else:
        cache_warmup_payload = {
            "input_ids": cache_warmup_input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 1,  # Minimal output, just to populate cache
                "ignore_eos": True,
            },
            "stream": False,
        }
        if dataset_name == "mmmu" and image_data is not None:
            # include image data in cache warmup
            cache_warmup_payload["image_data"] = image_data
        gen_url = url + "/generate"

    warmup_response = requests.post(
        gen_url,
        json=cache_warmup_payload,
        timeout=DEFAULT_TIMEOUT,
    )
    warmup_response.raise_for_status()
    print("Cache warmup completed")


def _flush_cache_with_retry(url: str, endpoint: str, max_retries: int = 3):
    """Post to a cache flush endpoint with retries on failure."""
    for attempt in range(max_retries):
        response = requests.post(url + endpoint, timeout=DEFAULT_TIMEOUT)
        if response.status_code == 200:
            return
        if attempt < max_retries - 1:
            time.sleep(2)
        else:
            response.raise_for_status()


def run_one_case(
    url: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    temperature: float,
    return_logprob: bool,
    stream_interval: int,
    input_len_step_percentage: float,
    run_name: str,
    result_filename: str,
    tokenizer: PreTrainedTokenizer | AutoProcessor,
    profile: bool = False,
    profile_activities: Tuple[str] = ("CPU", "GPU"),
    profile_start_step: Optional[int] = None,
    profile_steps: int = BenchArgs.profile_steps,
    profile_by_stage: bool = False,
    profile_prefix: Optional[str] = BenchArgs.profile_prefix,
    profile_output_dir: Optional[str] = BenchArgs.profile_output_dir,
    dataset_name: str = BenchArgs.dataset_name,
    dataset_path: str = BenchArgs.dataset_path,
    parallel_batch: bool = False,
    cache_hit_rate: float = BenchArgs.cache_hit_rate,
    backend: str = "sglang",
    model_name: Optional[str] = None,
    gsp_num_groups: int = BenchArgs.gsp_num_groups,
    gsp_system_prompt_len: int = BenchArgs.gsp_system_prompt_len,
    gsp_question_len: int = BenchArgs.gsp_question_len,
    gsp_output_len: int = BenchArgs.gsp_output_len,
    fake_prefill: bool = False,
    lora_name: Optional[List[str]] = None,
    lora_request_distribution: str = BenchArgs.lora_request_distribution,
    lora_zipf_alpha: float = BenchArgs.lora_zipf_alpha,
):
    if backend == "vllm":
        # You need to have export VLLM_SERVER_DEV_MODE=1 in your environment to use this endpoint.
        _flush_cache_with_retry(url, "/reset_prefix_cache")
    else:
        _flush_cache_with_retry(url, "/flush_cache")

    # Load input token ids via bench_serving.get_dataset
    supported_datasets = ("random", "random-ids", "mmmu", "generated-shared-prefix")
    if dataset_name not in supported_datasets:
        raise ValueError(
            f"Unsupported dataset for batch benchmark: {dataset_name}. "
            f"Supported: {supported_datasets}"
        )

    actual_gsp_groups = min(gsp_num_groups, batch_size)
    dataset_args = SimpleNamespace(
        dataset_name=dataset_name,
        num_prompts=batch_size,
        random_input_len=input_len,
        random_output_len=output_len,
        random_range_ratio=1.0,
        dataset_path=dataset_path,
        tokenize_prompt=dataset_name not in ("mmmu", "generated-shared-prefix"),
        backend=backend,
        seed=BenchArgs.seed,
        gsp_num_groups=actual_gsp_groups,
        gsp_prompts_per_group=(batch_size + actual_gsp_groups - 1) // actual_gsp_groups,
        gsp_system_prompt_len=gsp_system_prompt_len,
        gsp_question_len=gsp_question_len,
        gsp_output_len=gsp_output_len,
    )
    tok_inner = getattr(tokenizer, "tokenizer", tokenizer)
    dataset_model_id = model_name or getattr(tok_inner, "name_or_path", None)
    input_requests = get_dataset(dataset_args, tokenizer, model_id=dataset_model_id)

    if dataset_name == "generated-shared-prefix":
        input_requests = input_requests[:batch_size]
        input_ids = [tokenizer.encode(req.prompt) for req in input_requests]
        input_len = sum(len(ids) for ids in input_ids) // len(input_ids)
        output_len = gsp_output_len
        image_data = None
    elif dataset_name == "mmmu":
        input_ids = [tok_inner.encode(req.prompt) for req in input_requests]
        image_data = [req.image_data for req in input_requests]
    else:
        input_ids = [req.prompt for req in input_requests]
        image_data = None

    # Build payload based on backend
    if backend == "vllm":
        payload = {
            "model": model_name,
            "prompt": input_ids,
            "max_tokens": output_len,
            "temperature": temperature,
            "stream": True,
            "ignore_eos": True,
        }
        if return_logprob:
            payload["logprobs"] = 1
        gen_url = url + "/v1/completions"
    else:
        # Load sampling parameters
        use_structured_outputs = False
        if use_structured_outputs:
            texts = []
            for _ in range(batch_size):
                texts.append(
                    "Human: What is the capital city of france? can you give as many trivial information as possible about that city? answer in json.\n"
                    * 50
                    + "Assistant:"
                )
            json_schema = "$$ANY$$"
        else:
            json_schema = None

        payload = {
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": output_len,
                "ignore_eos": True,
                "json_schema": json_schema,
                "stream_interval": stream_interval,
            },
            "return_logprob": return_logprob,
            "stream": True,
            **({"parallel_batch": parallel_batch} if parallel_batch else {}),
        }
        payload["input_ids"] = input_ids
        if image_data is not None:
            payload["image_data"] = image_data
        if fake_prefill:
            payload["bootstrap_host"] = FAKE_BOOTSTRAP_HOST
            payload["bootstrap_room"] = 0
        if lora_name:
            # SGLang /generate accepts lora_path as either a string (applied
            # to every prompt) or a list matching the batch size (per-prompt
            # adapter). See io_struct.GenerateReqInput._normalize_lora_path.
            if len(lora_name) == 1:
                payload["lora_path"] = lora_name[0]
            elif lora_request_distribution == "uniform":
                payload["lora_path"] = [
                    random.choice(lora_name) for _ in range(batch_size)
                ]
            elif lora_request_distribution == "distinct":
                payload["lora_path"] = [
                    lora_name[i % len(lora_name)] for i in range(batch_size)
                ]
            elif lora_request_distribution == "skewed":
                weights = np.array([lora_zipf_alpha**-i for i in range(len(lora_name))])
                probs = weights / np.sum(weights)
                payload["lora_path"] = list(
                    np.random.choice(lora_name, size=batch_size, p=probs)
                )
            else:
                raise ValueError(
                    f"Unexpected lora_request_distribution: "
                    f"{lora_request_distribution!r}"
                )
        gen_url = url + "/generate"

    # Warm up cache if cache_hit_rate > 0.0
    if cache_hit_rate > 0.0:
        _warmup_cache(
            url=url,
            input_ids=input_ids,
            input_len=input_len,
            cache_hit_rate=cache_hit_rate,
            dataset_name=dataset_name,
            image_data=image_data,
            backend=backend,
            model_name=model_name,
        )

    # Turn on profiler
    profile_link = None
    if profile:
        profile_link: str = run_profile(
            url=url,
            num_steps=profile_steps,
            activities=profile_activities,
            output_dir=profile_output_dir,
            profile_by_stage=profile_by_stage,
            profile_prefix=profile_prefix,
            start_step=profile_start_step,
        )

    # Get metrics before the request (for cache hit rate calculation)
    metrics_before = get_cache_tokens_from_metrics(url)

    # Run the request
    tic = time.perf_counter()
    response = requests.post(
        gen_url,
        json=payload,
        stream=True,
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()

    # Get the TTFT of the last request in the batch
    last_ttft = 0.0
    if backend == "vllm":
        # Parse OpenAI-compatible streaming format from vLLM
        first_token_indices = set()
        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                data_str = chunk[5:].strip()
                if data_str == "[DONE]":
                    break
                data = json.loads(data_str)
                if "error" in data:
                    raise RuntimeError(f"Request has failed. {data}.")
                for choice in data.get("choices", []):
                    idx = choice["index"]
                    if idx not in first_token_indices:
                        first_token_indices.add(idx)
                        if len(first_token_indices) == batch_size:
                            last_ttft = time.perf_counter() - tic
    else:
        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                data = json.loads(chunk[5:].strip("\n"))
                if "error" in data:
                    raise RuntimeError(f"Request has failed. {data}.")

                assert (
                    data["meta_info"]["finish_reason"] is None
                    or data["meta_info"]["finish_reason"]["type"] == "length"
                )
                if data["meta_info"]["completion_tokens"] == 1:
                    last_ttft = time.perf_counter() - tic

    # Compute metrics
    latency = time.perf_counter() - tic
    input_throughput = batch_size * input_len / last_ttft
    output_throughput = batch_size * output_len / (latency - last_ttft)
    overall_throughput = batch_size * (input_len + output_len) / latency

    if backend == "vllm":
        # vLLM does not expose these metrics via API
        last_gen_throughput = -1
        acc_length = -1
    else:
        response = requests.get(url + "/server_info", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        server_info = response.json()
        internal_state = server_info.get("internal_states", [{}])
        last_gen_throughput = internal_state[0].get("last_gen_throughput", None) or -1
        acc_length = internal_state[0].get("avg_spec_accept_length", None) or -1

    # Calculate cache hit rate from before/after metrics delta
    metrics_after = get_cache_tokens_from_metrics(url)
    metrics_cache_hit_rate = calculate_cache_hit_rate(metrics_before, metrics_after)

    # Print results
    print(f"batch size: {batch_size}")
    print(f"input_len: {input_len}")
    print(f"output_len: {output_len}")
    print(f"latency: {latency:.2f} s")
    print(f"input throughput: {input_throughput:.2f} tok/s")
    if output_len != 1:
        print(f"output throughput: {output_throughput:.2f} tok/s")
    print(f"last_ttft: {last_ttft:.2f} s")
    print(f"last generation throughput: {last_gen_throughput:.2f} tok/s")
    if acc_length > 0:
        print(f"acc_length: {acc_length:.2f} ")
    if metrics_cache_hit_rate is not None:
        print(f"cache hit rate: {metrics_cache_hit_rate:.4f}")

    # Dump results
    result = BenchOneCaseResult(
        run_name=run_name,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        latency=latency,
        input_throughput=input_throughput,
        output_throughput=output_throughput,
        overall_throughput=overall_throughput,
        last_ttft=last_ttft,
        last_gen_throughput=last_gen_throughput,
        acc_length=acc_length,
        cache_hit_rate=metrics_cache_hit_rate,
        profile_link=profile_link,
    )

    # Save and return the results
    if result_filename:
        result.dump_to_jsonl(result_filename)

    return result


def should_skip_due_to_token_capacity(
    batch_size, input_len, output_len, skip_token_capacity_threshold
):
    if batch_size * (input_len + output_len) > skip_token_capacity_threshold:
        print(
            "=" * 8
            + f"Skip benchmark {batch_size=} * ({input_len=} + {output_len=}) = {batch_size * (input_len + output_len)} > {skip_token_capacity_threshold=} due to kv cache limit."
            + "=" * 8
        )
        return True
    return False


def should_skip_due_to_max_running_requests(
    batch_size, skip_max_running_requests_threshold
):
    if batch_size > skip_max_running_requests_threshold:
        print(
            "=" * 8
            + f"Skip benchmark {batch_size=} > {skip_max_running_requests_threshold=} due to max running requests limit."
            + "=" * 8
        )
        return True
    return False




