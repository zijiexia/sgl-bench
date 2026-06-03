"""The dataset overlay: user datasets register without touching vendored code,
overlay the vendored base on name clash, and the vendored ``random*`` fallback
survives. Mirrors the plan's M2 acceptance criteria."""

import argparse

import pytest

from sgl_bench import registry


@pytest.fixture(autouse=True)
def _clean_user_registry():
    """Each test starts with an empty user layer and forces re-autoload."""
    saved = dict(registry._USER)
    registry._USER.clear()
    registry._autoloaded = True  # skip importing sgl_bench.datasets in tests
    try:
        yield
    finally:
        registry._USER.clear()
        registry._USER.update(saved)
        registry._autoloaded = False


class _FakeDataset:
    def __init__(self, **kw):
        self.kw = kw

    @classmethod
    def from_args(cls, args):
        return cls(path=getattr(args, "fake_path", None))

    def load(self, tokenizer, model_id=None):
        return [("fake-row", self.kw)]


def test_vendored_datasets_still_resolve():
    # A vendored light dataset resolves with source="vendored".
    spec = registry.resolve("random")
    assert spec.source == "vendored"
    assert spec.name == "random"


def test_random_fallback_preserved():
    # sglang's `random*` -> `random-ids` fallback for unknown random variants.
    spec = registry.resolve("random-some-unknown-variant")
    assert spec.source == "vendored"
    assert spec.name == "random-ids"


def test_user_dataset_registers_and_resolves():
    registry.register_dataset("my-trace")(_FakeDataset)
    spec = registry.resolve("my-trace")
    assert spec.source == "user"
    assert spec.cls is _FakeDataset


def test_user_overlays_vendored_on_name_clash():
    # Registering under a vendored name (e.g. "random") wins.
    registry.register_dataset("random")(_FakeDataset)
    spec = registry.resolve("random")
    assert spec.source == "user"
    assert spec.cls is _FakeDataset


def test_all_specs_merges_both_layers():
    registry.register_dataset("my-trace")(_FakeDataset)
    specs = registry.all_specs()
    assert specs["my-trace"].source == "user"
    assert specs["random"].source == "vendored"
    assert "sharegpt" in specs  # vendored light dataset present


def test_unknown_dataset_raises_with_available_list():
    with pytest.raises(ValueError, match="Unknown dataset"):
        registry.resolve("does-not-exist")


def test_add_user_dataset_args_injects_flags():
    def _args(p):
        p.add_argument("--fake-path", default=None)

    registry.register_dataset("my-trace", add_cli_args=_args)(_FakeDataset)
    parser = argparse.ArgumentParser()
    registry.add_user_dataset_args(parser)
    ns = parser.parse_args(["--fake-path", "/tmp/x"])
    assert ns.fake_path == "/tmp/x"


def test_extra_gated_datasets_advertised_without_import():
    # image/mmmu/longbench_v2 show up in the listing with their extra, and
    # listing them must NOT import PIL/pandas (cls stays None).
    import sys

    specs = registry.all_specs()
    for name, extra in (("image", "multimodal"), ("mmmu", "multimodal"), ("longbench_v2", "longbench")):
        assert specs[name].needs_extra == extra
        assert specs[name].source == "vendored"
    assert "PIL" not in sys.modules and "pandas" not in sys.modules


def test_resolve_extra_without_dep_gives_friendly_error():
    # The dev venv has no [multimodal]/[longbench] deps, so resolving these
    # must raise a clear "pip install sgl-bench[...]" message, not ImportError.
    with pytest.raises(SystemExit, match="multimodal"):
        registry.resolve("image")
    with pytest.raises(SystemExit, match="longbench"):
        registry.resolve("longbench_v2")


def test_get_dataset_shim_drives_user_dataset():
    def _args(p):
        p.add_argument("--fake-path", default=None)

    registry.register_dataset("my-trace", add_cli_args=_args)(_FakeDataset)
    ns = argparse.Namespace(dataset_name="my-trace", fake_path="/tmp/x")
    rows = registry.get_dataset(ns, tokenizer=None, model_id=None)
    assert rows == [("fake-row", {"path": "/tmp/x"})]
