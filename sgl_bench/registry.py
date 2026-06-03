"""Merged dataset registry -- the extension hook.

The vendored ``DATASET_MAPPING`` (synced from sglang) is the read-only base
layer. Users add their own datasets by dropping a module in
``sgl_bench/datasets/`` and decorating a ``BaseDataset`` subclass with
``@register_dataset(...)``; those overlay the base layer (user wins on a name
clash) without ever editing vendored code.

The vendored ``bench_serving`` is rewritten at sync time to call this module's
``get_dataset`` instead of its own, so both layers are visible to every run.
This mirrors sgl-eval's ``registry.py`` (decorator + ``pkgutil`` autoload).
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type


@dataclass
class DatasetSpec:
    name: str
    cls: Type
    add_cli_args: Optional[Callable[[argparse.ArgumentParser], None]] = None
    needs_extra: Optional[str] = None  # "multimodal" | "longbench" | None
    source: str = "user"  # "user" | "vendored"


_USER: Dict[str, DatasetSpec] = {}
_autoloaded = False

# Vendored datasets gated behind an optional dependency group. They are NOT in
# the vendored DATASET_MAPPING (datasets/__init__.py never imports them), so a
# bare install stays free of PIL/pandas/HF-datasets; the registry lazy-imports
# the module only when the dataset is actually resolved.
#   name -> (extra, module, class)
_VENDORED_EXTRAS: Dict[str, tuple] = {
    "image": ("multimodal", "sgl_bench._vendored.sglang.benchmark.datasets.image", "ImageDataset"),
    "mmmu": ("multimodal", "sgl_bench._vendored.sglang.benchmark.datasets.mmmu", "MMMUDataset"),
    "longbench_v2": (
        "longbench",
        "sgl_bench._vendored.sglang.benchmark.datasets.longbench_v2",
        "LongBenchV2Dataset",
    ),
}


def register_dataset(
    name: str,
    *,
    add_cli_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
    needs_extra: Optional[str] = None,
) -> Callable[[Type], Type]:
    """Decorator: register a ``BaseDataset`` subclass under ``name``.

    ``add_cli_args(parser)`` (optional) lets the dataset contribute its own CLI
    flags, vLLM-style. ``needs_extra`` names an optional dependency group to
    check before use.
    """

    def deco(cls: Type) -> Type:
        if name in _USER:
            raise ValueError(f"dataset {name!r} already registered by the user layer")
        _USER[name] = DatasetSpec(name, cls, add_cli_args, needs_extra, source="user")
        return cls

    return deco


def _autoload_user() -> None:
    """Import every module under ``sgl_bench.datasets`` so ``@register_dataset``
    decorators fire. Idempotent."""
    global _autoloaded
    if _autoloaded:
        return
    _autoloaded = True
    try:
        pkg = importlib.import_module("sgl_bench.datasets")
    except ModuleNotFoundError:
        return
    for _finder, mod_name, is_pkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        if not is_pkg:
            importlib.import_module(mod_name)


def _vendored_map() -> Dict[str, Type]:
    # Imported lazily so a bare `import sgl_bench` / `import sgl_bench.registry`
    # never pulls the vendored dataset modules (and their deps).
    from sgl_bench._vendored.sglang.benchmark.datasets import DATASET_MAPPING

    return DATASET_MAPPING


def resolve(name: str) -> DatasetSpec:
    """Resolve a ``--dataset-name`` to a spec. User datasets overlay vendored
    ones. Preserves sglang's ``random*`` -> ``random-ids`` fallback."""
    _autoload_user()
    if name in _USER:
        return _USER[name]
    vmap = _vendored_map()
    key = name
    if name.startswith("random") and name not in vmap:
        key = "random-ids"
    if key in vmap:
        return DatasetSpec(name=key, cls=vmap[key], source="vendored")
    if name in _VENDORED_EXTRAS:
        extra, module, clsname = _VENDORED_EXTRAS[name]
        # Probe the representative dep first -- catches modules (e.g.
        # longbench_v2) that import their heavy deps lazily inside load().
        _require_extra(extra, name)
        try:
            mod = importlib.import_module(module)
        except ImportError as e:
            missing = getattr(e, "name", extra)
            raise SystemExit(
                f"dataset {name!r} needs the optional '{extra}' dependencies "
                f"(missing: {missing}).\n  pip install 'sgl-bench[{extra}]'"
            )
        return DatasetSpec(
            name=name, cls=getattr(mod, clsname), needs_extra=extra, source="vendored"
        )
    available = ", ".join(sorted(set(vmap) | set(_VENDORED_EXTRAS) | set(_USER)))
    raise ValueError(f"Unknown dataset: {name!r}. Available: {available}")


def all_specs() -> Dict[str, DatasetSpec]:
    """Every known dataset: vendored base overlaid with user registrations."""
    _autoload_user()
    out: Dict[str, DatasetSpec] = {
        n: DatasetSpec(n, c, source="vendored") for n, c in _vendored_map().items()
    }
    # Advertise extra-gated datasets without importing them (cls left None).
    for name, (extra, _module, _clsname) in _VENDORED_EXTRAS.items():
        out[name] = DatasetSpec(name=name, cls=None, needs_extra=extra, source="vendored")
    out.update(_USER)  # user overlays vendored on name clash
    return out


def get_dataset(args: argparse.Namespace, tokenizer, model_id=None):
    """Drop-in replacement for the vendored ``get_dataset`` (the vendored
    bench_serving is rewritten to call this), with the user overlay applied."""
    spec = resolve(args.dataset_name)
    if spec.needs_extra:
        _require_extra(spec.needs_extra, args.dataset_name)
    dataset = spec.cls.from_args(args)
    return dataset.load(tokenizer=tokenizer, model_id=model_id)


def add_user_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Let user datasets contribute CLI flags. Invoked from the vendored
    bench_serving ``__main__`` right before ``parse_args`` (injected at sync).
    Vendored datasets already declare their flags inline upstream."""
    _autoload_user()
    for spec in _USER.values():
        if spec.add_cli_args is not None:
            spec.add_cli_args(parser)


_EXTRA_PROBE = {"multimodal": "PIL", "longbench": "pandas"}


def _require_extra(extra: str, dataset_name: str) -> None:
    probe = _EXTRA_PROBE.get(extra)
    if probe is None:
        return
    try:
        importlib.import_module(probe)
    except ImportError:
        raise SystemExit(
            f"dataset {dataset_name!r} needs the optional '{extra}' dependencies.\n"
            f"  pip install 'sgl-bench[{extra}]'"
        )
