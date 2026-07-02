"""
src/algorithms/cpu/multi_threaded/louvain.py
============================================

Louvain — cpu_multi mode, GraphBLAS-era replacement.

.. note::
   SuiteSparse:GraphBLAS does **not** ship a native Louvain primitive,
   and Louvain Phase 1 (a sequential greedy node-by-node modularity
   maximization) does not vectorize cleanly under GraphBLAS semirings —
   any node-by-node implementation in python-graphblas would be slower
   than the scipy single-threaded version because of per-call kernel
   launch overhead.

   To preserve the ``cpu_multi`` mode label without misleading benchmark
   results we **delegate to** ``louvain_cpu_single`` and annotate the
   result note so downstream readers can see what happened.  The previous
   ``ProcessPoolExecutor`` Phase-1 implementation was producing
   non-deterministic community assignments and was usually slower than
   the sequential version anyway, so this fall-through is a strict
   improvement.

For the deterministic single-thread variant see
``src.algorithms.cpu.single_threaded.louvain``.
"""

from __future__ import annotations

import warnings

import scipy.sparse as sp

from src.algorithms.cpu.multi_threaded._graphblas_utils import (
    _GRAPHBLAS_AVAILABLE,
    _configure_threads,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "min_delta_q":       1e-4,
    "max_levels":        10,
    "resolution":        1.0,
    "max_phase1_passes": 100,
}


def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# Public implementation — delegates to cpu_single
# ---------------------------------------------------------------------------

def louvain_cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict,
    n_workers: int | None = None,
) -> dict:
    """Run Louvain via the scipy single-thread implementation.

    GraphBLAS configuration (thread count) is still applied so that any
    subsequent scipy MKL calls inside ``louvain_cpu_single`` can benefit
    from the right OpenMP thread setting on the host.
    """
    if _GRAPHBLAS_AVAILABLE:
        _configure_threads(n_workers)

    warnings.warn(
        "louvain_cpu_multi: SuiteSparse:GraphBLAS has no native Louvain "
        "primitive; delegating to scipy cpu_single.",
        UserWarning, stacklevel=2,
    )
    # Lazy import — keep this module light when Louvain is not requested.
    from src.algorithms.cpu.single_threaded.louvain import louvain_cpu_single

    result = louvain_cpu_single(graph_csr, params)
    existing_note = result.get("note") or ""
    extra = (
        "cpu_multi backend: GraphBLAS has no Louvain primitive — "
        "delegated to scipy cpu_single."
    )
    result["note"] = (existing_note + " | " + extra).strip(" |")
    return result


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict | None = None,
    n_workers: int | None = None,
    **_,
) -> dict:
    """Benchmark runner entry-point for cpu_multi mode."""
    p = _merge_params(params)
    return {
        "output":       louvain_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "backend": "graphblas_fallback_scipy"},
    }
