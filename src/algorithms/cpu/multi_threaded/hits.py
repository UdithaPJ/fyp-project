"""
src/algorithms/cpu/multi_threaded/hits.py
=========================================

HITS — GraphBLAS-backed cpu_multi implementation.

Two SpMVs per iteration (``A^T @ h`` and ``A @ a``) executed through
SuiteSparse:GraphBLAS's parallel ``plus_times`` SpMV.  L2 normalisation
remains in numpy because it is a trivial O(N) operation.

For the deterministic single-thread variant see
``src.algorithms.cpu.single_threaded.hits``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import hits_pack_result as _pack_result
from src.algorithms.cpu.multi_threaded._graphblas_utils import (
    _configure_threads,
    _from_scipy,
    _require_graphblas,
    _vec_from_np,
    _vec_to_np,
    gb,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "max_iter":  100,
    "tolerance": 1e-6,
}


def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def hits_cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict,
    n_workers: int | None = None,
) -> dict:
    """HITS using SuiteSparse:GraphBLAS for both SpMVs per iteration."""
    _require_graphblas()
    n_threads = _configure_threads(n_workers)

    p        = _merge_params(params)
    max_iter = int(p["max_iter"])
    tol      = float(p["tolerance"])

    n = int(graph_csr.shape[0])
    if n == 0:
        return _pack_result(np.zeros(0, dtype=np.float32),
                            np.zeros(0, dtype=np.float32), 0, True)

    A_gb  = _from_scipy(graph_csr, dtype=np.float32)
    A_T_gb = A_gb.T.new()

    h = np.ones(n, dtype=np.float32)
    a = np.ones(n, dtype=np.float32)
    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        h_old, a_old = h, a

        # ---- a_new = A^T @ h_old, L2-normalised --------------------------
        h_gb = _vec_from_np(h_old, dtype=gb.dtypes.FP32)
        a_new_gb = A_T_gb.mxv(h_gb, gb.semiring.plus_times).new()
        a_new = _vec_to_np(a_new_gb, n, dtype=np.float32, fill=0.0)
        norm_a = float(np.linalg.norm(a_new))
        if norm_a > 0.0:
            a_new = a_new / norm_a

        # ---- h_new = A @ a_new, L2-normalised ----------------------------
        a_new_v = _vec_from_np(a_new, dtype=gb.dtypes.FP32)
        h_new_gb = A_gb.mxv(a_new_v, gb.semiring.plus_times).new()
        h_new = _vec_to_np(h_new_gb, n, dtype=np.float32, fill=0.0)
        norm_h = float(np.linalg.norm(h_new))
        if norm_h > 0.0:
            h_new = h_new / norm_h

        if float(np.linalg.norm(h_new - h_old)
                 + np.linalg.norm(a_new - a_old)) < tol:
            converged = True
            a, h = a_new, h_new
            break

        a, h = a_new, h_new

    result = _pack_result(h, a, iteration, converged)
    result["note"] = f"graphblas SuiteSparse nthreads={n_threads}"
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
    """Benchmark runner entry-point for cpu_multi mode (now GraphBLAS-backed)."""
    p = _merge_params(params)
    return {
        "output":       hits_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "backend": "graphblas"},
    }
