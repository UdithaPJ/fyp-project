"""
src/algorithms/gpu/basic/
=========================

GPU baseline algorithm implementations.

Purpose
-------
Used **exclusively** for benchmarking against the highly tuned
implementations in ``src/algorithms/gpu/cuda_optimized/``.  Every
module exposes one function::

    <algorithm>_gpu_baseline(graph_csr, params)  -> dict

Backend policy (strict — no silent fallbacks)
---------------------------------------------
Five algorithms use cuGraph (RAPIDS) exclusively:
    pagerank, bfs, hits, louvain, rwr  →  ``"gpu_baseline_cugraph"``

    Each module raises ``ImportError`` at **import time** if cuGraph/cuDF
    are not installed (see the module-level ``try: import cugraph`` blocks).

One algorithm (MCL) uses CuPy exclusively because cuGraph has no MCL:
    mcl  →  ``"gpu_baseline_cupy"``

    MCL does **not** import from ``_utils.py`` (which requires cuGraph)
    so it remains independently importable on a CuPy-only installation.

Package-level import handling
------------------------------
This ``__init__.py`` wraps the cuGraph-backed imports in a try/except so
that ``from src.algorithms.gpu.basic.mcl import mcl_gpu_baseline`` can
succeed on CuPy-only machines.  If cuGraph is absent, the five cuGraph
algorithm names in this namespace are replaced with stub callables that
raise ``ImportError`` with RAPIDS install instructions.  The underlying
module-level hard-fail still applies when the individual files are
imported directly.

Never imported by the web application.
"""

from __future__ import annotations

# Mode identifiers (independent of any GPU library)
BASELINE_MODE_CUGRAPH: str = "gpu_baseline_cugraph"
BASELINE_MODE_CUPY: str    = "gpu_baseline_cupy"

# ---------------------------------------------------------------------------
# cuGraph-backed baselines — hard-fail if RAPIDS is not installed.
# The try/except here exists solely to preserve MCL's independent
# importability; the individual module files still perform their own
# module-level ImportError raises.
# ---------------------------------------------------------------------------
try:
    from .bfs       import bfs_gpu_baseline
    from .hits      import hits_gpu_baseline
    from .louvain   import louvain_gpu_baseline
    from .pagerank  import pagerank_gpu_baseline
    from .rwr       import rwr_gpu_baseline
    _CUGRAPH_AVAILABLE = True
except ImportError as _cugraph_import_error:
    _CUGRAPH_AVAILABLE = False
    _err = _cugraph_import_error

    def _missing_cugraph_backend(*args, **kwargs):  # type: ignore[misc]
        """Stub: raises ImportError when called without RAPIDS installed."""
        raise ImportError(
            "src/algorithms/gpu/basic/ cuGraph algorithms require RAPIDS "
            "(cuGraph + cuDF).  Install via:\n"
            "  conda install -c rapidsai -c nvidia -c conda-forge "
            "rapids=24.02 python=3.10 cudatoolkit=11.8"
        ) from _err

    bfs_gpu_baseline      = _missing_cugraph_backend  # type: ignore[assignment]
    hits_gpu_baseline     = _missing_cugraph_backend  # type: ignore[assignment]
    louvain_gpu_baseline  = _missing_cugraph_backend  # type: ignore[assignment]
    pagerank_gpu_baseline = _missing_cugraph_backend  # type: ignore[assignment]
    rwr_gpu_baseline      = _missing_cugraph_backend  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# CuPy-backed baseline: MCL has no cuGraph equivalent.
# This import is independent of cuGraph availability.
# Wrapped so the package always loads even when CuPy is also absent.
# ---------------------------------------------------------------------------
try:
    from .mcl import mcl_gpu_baseline
    _CUPY_AVAILABLE = True
except ImportError as _cupy_import_error:
    _CUPY_AVAILABLE = False
    _cupy_err = _cupy_import_error

    def mcl_gpu_baseline(*args, **kwargs):   # type: ignore[misc]
        """Stub: raises ImportError when called without CuPy installed."""
        raise ImportError(
            "src/algorithms/gpu/basic/mcl.py requires CuPy "
            "(cuGraph has no MCL equivalent).  Install via:\n"
            "  pip install cupy-cuda12x   # or cupy-cuda11x"
        ) from _cupy_err


__all__ = [
    "BASELINE_MODE_CUGRAPH",
    "BASELINE_MODE_CUPY",
    "pagerank_gpu_baseline",
    "bfs_gpu_baseline",
    "hits_gpu_baseline",
    "louvain_gpu_baseline",
    "rwr_gpu_baseline",
    "mcl_gpu_baseline",
]
