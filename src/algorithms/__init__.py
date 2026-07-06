"""
src/algorithms/__init__.py
==========================

Builds the algorithm registry by dynamically wrapping each module in
``src/algorithms/gpu/cuda_optimized/`` with a thin adapter class that
satisfies the :class:`~src.algorithms.base.AlgorithmBase` interface
expected by ``src/runner/algorithm_runner.py``.

Why adapters instead of direct imports
---------------------------------------
The runner resolves algorithms via ``ALGORITHM_REGISTRY[name]`` and expects
each entry to provide:

  * ``NAME``             — string identifier
  * ``PARAM_SCHEMA``     — dict of default parameter values
  * ``gpu(csr, params)`` — staticmethod returning the standard result dict
  * ``validate_params``  — classmethod (inherited from AlgorithmBase)
  * ``describe``         — classmethod (inherited from AlgorithmBase)

The cuda_optimized modules expose ``_DEFAULT_PARAMS`` and ``_gpu()`` — the
adapter translates between the two conventions.

Result wrapping
---------------
``_gpu()`` in each cuda_optimized module returns::

    {"output": {<algorithm-specific keys>}, "extra_params": {...}}

The adapter's ``gpu()`` staticmethod unpacks this and wraps it into the
standard result dict::

    {
      "algorithm":      str,
      "mode":           "gpu",
      "execution_time": 0.0,    # overwritten by runner's BenchmarkTimer
      "num_nodes":      int,
      "num_edges":      int,
      "result":         dict,   # the "output" value from _gpu()
    }

CPU modes are NOT exposed here — they are benchmarking-only tools that live
in ``src/algorithms/cpu/`` and are never used by the webapp.
"""

from __future__ import annotations

import importlib
import logging

import scipy.sparse as sp

from .base import AlgorithmBase

# ---------------------------------------------------------------------------
# Per-algorithm metadata
# ---------------------------------------------------------------------------

# (registry_name, cuda_optimized module path, one-line description)
_ALGO_META: list[tuple[str, str, str]] = [
    (
        "pagerank",
        "src.algorithms.gpu.cuda_optimized.pagerank",
        "Rank regulators by global influence in the GRN (PageRank).",
    ),
    (
        "bfs",
        "src.algorithms.gpu.cuda_optimized.bfs",
        "Trace a regulatory cascade outward from a source TF (BFS).",
    ),
    (
        "louvain",
        "src.algorithms.gpu.cuda_optimized.louvain",
        "Detect communities of co-regulated genes (Louvain).",
    ),
    (
        "rwr",
        "src.algorithms.gpu.cuda_optimized.rwr",
        "Proximity walk from seed TFs (Random Walk with Restart).",
    ),
    (
        "hits",
        "src.algorithms.gpu.cuda_optimized.hits",
        "Identify hub TFs and authority target genes (HITS).",
    ),
    (
        "mcl",
        "src.algorithms.gpu.cuda_optimized.mcl",
        "Markov-clustering for tightly co-regulated modules (MCL).",
    ),
]


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

def _make_adapter(name: str, module_path: str, description: str) -> type[AlgorithmBase]:
    """
    Create an AlgorithmBase subclass that delegates ``gpu()`` to the
    corresponding cuda_optimized module's ``_gpu()`` function and
    ``gpu_baseline()`` to the matching module in
    ``src/algorithms/gpu/basic/``.

    The cuda_optimized module is imported eagerly so import errors surface
    at startup.  The baseline module is imported lazily inside the adapter
    so a missing RAPIDS/CuPy installation does not break the registry at
    import time — only at ``gpu_baseline()`` call time.
    """
    mod = importlib.import_module(module_path)
    defaults: dict = dict(getattr(mod, "_DEFAULT_PARAMS", {}))

    def _looks_like_standard_result(x: object) -> bool:
        return (
            isinstance(x, dict)
            and "algorithm" in x
            and "mode" in x
            and "execution_time" in x
            and "num_nodes" in x
            and "num_edges" in x
            and "result" in x
            and isinstance(x.get("result"), dict)
        )

    def _gpu_staticmethod(graph_csr: sp.csr_matrix, params: dict) -> dict:
        raw = mod._gpu(graph_csr, params)
        # Unpack {"output": ..., "extra_params": ...} wrapper produced by _gpu()
        result_data = raw["output"] if isinstance(raw, dict) and "output" in raw else raw

        # Some cuda_optimized modules already return the fully-standardised
        # result dict (with {algorithm, mode, execution_time, num_nodes, ...}).
        # Others return only the inner algorithm-specific payload.
        if _looks_like_standard_result(result_data):
            return result_data

        return {
            "algorithm":      name,
            "mode":           "gpu",
            "execution_time": 0.0,   # overwritten by BenchmarkTimer in the runner
            "num_nodes":      int(graph_csr.shape[0]),
            "num_edges":      int(graph_csr.nnz),
            "result":         result_data,
        }

    # ---- CPU adapters (cpu_single / cpu_multi) ----
    # MEMORY_FIX (Fix Cat. 3 verification): bind cpu_single / cpu_multi
    # so that run_algorithm(..., mode="cpu_single") works.  The CPU
    # functions live in src/algorithms/cpu/<single|multi>_threaded/{name}.py
    # and return only the inner result; we wrap into the standard
    # envelope here.
    def _make_cpu_dispatch(cpu_mode: str):
        def _cpu_staticmethod(graph_csr: sp.csr_matrix, params: dict) -> dict:
            cpu_mod_path = (
                f"src.algorithms.cpu."
                f"{'single_threaded' if cpu_mode == 'cpu_single' else 'multi_threaded'}"
                f".{name}"
            )
            cpu_mod = importlib.import_module(cpu_mod_path)
            fn = getattr(cpu_mod, f"_{cpu_mode}", None)
            if fn is None:
                raise AttributeError(
                    f"{cpu_mod_path} does not expose _{cpu_mode}(graph_csr, params)"
                )
            raw = fn(graph_csr, params)
            result_data = raw["output"] if isinstance(raw, dict) and "output" in raw else raw
            if _looks_like_standard_result(result_data):
                return result_data
            return {
                "algorithm":      name,
                "mode":           cpu_mode,
                "execution_time": 0.0,
                "num_nodes":      int(graph_csr.shape[0]),
                "num_edges":      int(graph_csr.nnz),
                "result":         result_data,
            }
        return _cpu_staticmethod

    _cpu_single_static = _make_cpu_dispatch("cpu_single")
    _cpu_multi_static  = _make_cpu_dispatch("cpu_multi")

    # ---- Baseline (gpu_baseline) adapter ----
    # Imported lazily so a missing cuGraph / CuPy install does not break
    # the registry at module-import time.  Only fails when the user
    # actually calls run_algorithm(..., mode="gpu_baseline").
    baseline_module_path = f"src.algorithms.gpu.basic.{name}"
    baseline_fn_name     = f"{name}_gpu_baseline"

    def _gpu_baseline_staticmethod(graph_csr: sp.csr_matrix, params: dict) -> dict:
        try:
            baseline_mod = importlib.import_module(baseline_module_path)
        except ImportError as exc:
            raise RuntimeError(
                f"gpu_baseline implementation for '{name}' is not "
                f"available: cannot import {baseline_module_path} ({exc})."
            ) from exc
        fn = getattr(baseline_mod, baseline_fn_name, None)
        if fn is None:
            raise RuntimeError(
                f"Module {baseline_module_path} does not expose "
                f"{baseline_fn_name}(graph_csr, params)."
            )
        result_data = fn(graph_csr, params)
        if _looks_like_standard_result(result_data):
            return result_data
        return {
            "algorithm":      name,
            "mode":           "gpu_baseline",
            "execution_time": 0.0,
            "num_nodes":      int(graph_csr.shape[0]),
            "num_edges":      int(graph_csr.nnz),
            "result":         result_data,
        }

    cls: type[AlgorithmBase] = type(
        name.upper(),
        (AlgorithmBase,),
        {
            "__doc__":      description,
            "NAME":         name,
            "PARAM_SCHEMA": defaults,
            "cpu_single":   staticmethod(_cpu_single_static),
            "cpu_multi":    staticmethod(_cpu_multi_static),
            "gpu":          staticmethod(_gpu_staticmethod),
            "gpu_baseline": staticmethod(_gpu_baseline_staticmethod),
        },
    )
    return cls


# ---------------------------------------------------------------------------
# Build registry
# ---------------------------------------------------------------------------

ALGORITHM_REGISTRY: dict[str, type[AlgorithmBase]] = {
    name: _make_adapter(name, path, desc)
    for name, path, desc in _ALGO_META
}

# Named exports for any code that does ``from src.algorithms import PageRank``
PageRank = ALGORITHM_REGISTRY["pagerank"]
BFS      = ALGORITHM_REGISTRY["bfs"]
Louvain  = ALGORITHM_REGISTRY["louvain"]
RWR      = ALGORITHM_REGISTRY["rwr"]
HITS     = ALGORITHM_REGISTRY["hits"]
MCL      = ALGORITHM_REGISTRY["mcl"]

__all__ = [
    "AlgorithmBase",
    "ALGORITHM_REGISTRY",
    "PageRank",
    "BFS",
    "Louvain",
    "RWR",
    "HITS",
    "MCL",
]
