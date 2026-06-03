# Vendored from sgl-project/sglang@e5b8e3a66aa6052d86905869a7dd7c816c8401f7
# Source: python/sglang/benchmark/datasets/__init__.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

from typing import Dict, Type

from sgl_bench._vendored.sglang.benchmark.datasets.autobench import AutoBenchmarkDataset
from sgl_bench._vendored.sglang.benchmark.datasets.common import BaseDataset, DatasetRow
from sgl_bench._vendored.sglang.benchmark.datasets.custom import CustomDataset
from sgl_bench._vendored.sglang.benchmark.datasets.generated_shared_prefix import (
    GeneratedSharedPrefixDataset,
)
from sgl_bench._vendored.sglang.benchmark.datasets.mooncake import MooncakeDataset
from sgl_bench._vendored.sglang.benchmark.datasets.openai_dataset import OpenAIDataset
from sgl_bench._vendored.sglang.benchmark.datasets.random import RandomDataset
from sgl_bench._vendored.sglang.benchmark.datasets.sharegpt import ShareGPTDataset
from sgl_bench._vendored.sglang.benchmark.datasets.speed_bench import SpeedBenchDataset

DATASET_MAPPING: Dict[str, Type[BaseDataset]] = {
    "autobench": AutoBenchmarkDataset,
    "sharegpt": ShareGPTDataset,
    "custom": CustomDataset,
    "openai": OpenAIDataset,
    # TODO: "random" vs "random-ids" should be a flag (e.g. --random-source=sharegpt|integers),
    # not two separate dataset names sharing the same class.
    "random": RandomDataset,
    "random-ids": RandomDataset,
    "generated-shared-prefix": GeneratedSharedPrefixDataset,
    "mooncake": MooncakeDataset,
    "speed-bench": SpeedBenchDataset,
}


def get_dataset(args, tokenizer, model_id=None):
    dataset_name = args.dataset_name
    if dataset_name.startswith("random") and dataset_name not in DATASET_MAPPING:
        dataset_name = "random-ids"

    if dataset_name not in DATASET_MAPPING:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    dataset_cls = DATASET_MAPPING[dataset_name]
    dataset = dataset_cls.from_args(args)
    return dataset.load(tokenizer=tokenizer, model_id=model_id)


__all__ = [
    "DATASET_MAPPING",
    "DatasetRow",
    "get_dataset",
]
