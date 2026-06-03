"""sgl-bench: a lightweight, client-side performance benchmark for any
OpenAI-compatible / sglang HTTP endpoint.

Pure HTTP client -- it never loads a model engine, so it installs without
torch/CUDA/sglang. Point it at an *already-running* server; every subcommand
needs ``--base-url``.
"""

__version__ = "0.0.1"
