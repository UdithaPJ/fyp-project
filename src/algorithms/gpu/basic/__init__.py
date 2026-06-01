"""
src/algorithms/gpu/basic/
=========================

GPU baseline algorithm implementations.

Purpose
-------
Used **exclusively** for benchmarking against the highly tuned
implementations in ``src/algorithms/gpu/cuda_optimized/``.  Every module
exposes one function::

    <algorithm>_gpu_baseline(graph_csr, params)  -> dict

Backend priority (per module)
-----------------------------
    1. cuGraph (RAPIDS)
    2. CuPy sparse
    3. RuntimeError

The baselines never silently fall back to CPU — that path lives in
``src/algorithms/cpu/`` and is selected explicitly via the runner's
``mode="cpu_single"`` / ``"cpu_multi"`` arguments.

Never imported by the web application.
"""

from __future__ import annotations

from ._utils import (
    BASELINE_MODE,
    CUGRAPH_AVAILABLE, CUDF_AVAILABLE, CUPY_AVAILABLE,
    cugraph_version, cupy_version,
)
from .bfs       import bfs_gpu_baseline
from .hits      import hits_gpu_baseline
from .louvain   import louvain_gpu_baseline
from .mcl       import mcl_gpu_baseline
from .pagerank  import pagerank_gpu_baseline
from .rwr       import rwr_gpu_baseline


__all__ = [
    "BASELINE_MODE",
    "CUGRAPH_AVAILABLE", "CUDF_AVAILABLE", "CUPY_AVAILABLE",
    "cugraph_version", "cupy_version",
    "pagerank_gpu_baseline",
    "bfs_gpu_baseline",
    "hits_gpu_baseline",
    "louvain_gpu_baseline",
    "rwr_gpu_baseline",
    "mcl_gpu_baseline",
]
