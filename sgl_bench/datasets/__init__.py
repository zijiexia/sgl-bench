"""User-owned dataset extension layer (first-class, NOT vendored).

Drop a module here, decorate a ``BaseDataset`` subclass with
``@register_dataset(...)`` (see ``sgl_bench.registry``), and it becomes
available to every subcommand -- no reinstall, no edits to vendored code.
Populated starting in M2/M5.
"""
