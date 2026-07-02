"""
src/algorithms/cpu/multi_threaded/pagerank.py
=============================================

PageRank — GraphBLAS-backed cpu_multi implementation.

The mode string ``cpu_multi`` is preserved for benchmark continuity, but
the implementation underneath is now SuiteSparse:GraphBLAS via
``python-graphblas`` — no ProcessPoolExecutor, no pickle IPC, no
BLAS oversubscription.  See ``_graphblas_utils.py`` for the rationale.

Algorithm
---------
Standard power iteration with network-type-aware dangling redistribution.
The transition matrix ``M`` (column-stochastic) is built via GraphBLAS
SpGEMM ``D_inv @ A`` then transposed; the per-iteration SpMV uses the
``plus_times`` semiring through SuiteSparse's parallel SpMV kernel.

For the deterministic single-thread variant see
``src.algorithms.cpu.single_threaded.pagerank``.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import _split_top_nodes
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
    "damping":   0.85,
    "max_iter":  100,
    "tolerance": 1e-6,
}


def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def pagerank_cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict,
    n_workers: int | None = None,
) -> dict:
    """PageRank using SuiteSparse:GraphBLAS for the SpMV iteration.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
    params    : dict
        damping   (float, default 0.85)
        max_iter  (int,   default 100)
        tolerance (float, default 1e-6)
        network_type (str) — "grn" | "ppi" | "mirna"
    n_workers : int | None
        OpenMP thread count passed to SuiteSparse.  ``None`` uses
        ``os.cpu_count()``.

    Returns
    -------
    dict with keys: scores, iterations, converged, top_regulators,
                    top_targets, note
    """
    _require_graphblas()
    n_threads = _configure_threads(n_workers)

    p            = _merge_params(params)
    d            = float(p["damping"])
    max_iter     = int(p["max_iter"])
    tol          = float(p["tolerance"])
    network_type = str(p.get("network_type", "grn")).lower()

    n = int(graph_csr.shape[0])
    if n == 0:
        return {
            "scores": [], "iterations": 0, "converged": True,
            "top_regulators": [], "top_targets": [],
            "note": "graphblas SuiteSparse (empty graph)",
        }

    # ---- Build column-stochastic transition matrix M_T -------------------
    # M[j, i] = A[i, j] / out_degree(i)
    # We build (D_inv @ A) which is row-normalised, then transpose.
    A_gb = _from_scipy(graph_csr, dtype=np.float32)
    out_deg_gb = A_gb.reduce_rowwise(gb.monoid.plus).new()
    out_deg = _vec_to_np(out_deg_gb, n, dtype=np.float32, fill=0.0)

    dangling_mask = out_deg == 0
    active_mask = ~dangling_mask
    n_active = int(active_mask.sum())

    inv_deg = np.zeros(n, dtype=np.float32)
    inv_deg[active_mask] = 1.0 / out_deg[active_mask]
    D_inv = gb.ss.diag(_vec_from_np(inv_deg, dtype=gb.dtypes.FP32))
    M_T = (D_inv @ A_gb).T.new()

    teleport_per_node = (1.0 - d) / n

    if n_active == 0:
        warnings.warn(
            "no outgoing-edge nodes found, using uniform dangling redistribution",
            UserWarning, stacklevel=2,
        )
        eligible_mask = np.ones(n, dtype=bool)
    elif network_type == "ppi":
        eligible_mask = np.ones(n, dtype=bool)
    else:  # grn / mirna
        eligible_mask = active_mask

    n_eligible = int(eligible_mask.sum())

    PR = np.full(n, 1.0 / n, dtype=np.float32)
    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        PR_old = PR

        # ---- SpMV via SuiteSparse:GraphBLAS -----------------------------
        PR_gb = _vec_from_np(PR_old, dtype=gb.dtypes.FP32)
        result_gb = M_T.mxv(PR_gb, gb.semiring.plus_times).new()
        result = _vec_to_np(result_gb, n, dtype=np.float32, fill=0.0)

        # ---- Network-type-aware dangling redistribution -----------------
        dangling_mass = d * float(PR_old[dangling_mask].sum())
        dangling_contrib = np.zeros(n, dtype=np.float32)
        if n_eligible > 0:
            dangling_contrib[eligible_mask] = dangling_mass / n_eligible
        else:
            dangling_contrib[:] = dangling_mass / n

        PR = d * result + dangling_contrib + teleport_per_node

        if float(np.abs(PR - PR_old).sum()) < tol:
            converged = True
            break

    top_reg, top_tgt = _split_top_nodes(PR, out_deg)
    return {
        "scores":         PR.tolist(),
        "iterations":     iteration,
        "converged":      converged,
        "top_regulators": top_reg,
        "top_targets":    top_tgt,
        "note":           f"graphblas SuiteSparse nthreads={n_threads}",
    }


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
        "output":       pagerank_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "backend": "graphblas"},
    }
