"""
src/algorithms/cpu/multi_threaded/mcl.py
=========================================

Markov Clustering (MCL) — GraphBLAS-backed cpu_multi.

This is the algorithm where the GraphBLAS approach delivers the largest
speedup: MCL is dominated by **SpGEMM** for the expansion step
(``M = M @ M``), and SuiteSparse:GraphBLAS implements cache-tiled,
hash-based parallel SpGEMM that consistently beats scipy by 3–10× on
biological-scale graphs.

Pipeline per iteration
----------------------
1. Save ``M_old`` (host-side scipy CSR — needed for the cheap
   element-wise Frobenius diff at the end of the loop).
2. **Expansion** via SuiteSparse SpGEMM ``M_gb = M_gb @ M_gb``.
3. **Inflation** via SuiteSparse element-wise power.
4. **Column normalisation** via reduce_columnwise + diagonal-scale
   SpGEMM.
5. **Prune** via SuiteSparse ``select(">", threshold)``.
6. Convert back to scipy for convergence check + (post-loop) cluster
   extraction.

Symmetrization + self-loop preconditioning happens once on scipy.

For the single-thread variant see
``src.algorithms.cpu.single_threaded.mcl``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import _extract_clusters
from src.algorithms.cpu.multi_threaded._graphblas_utils import (
    _configure_threads,
    _from_scipy,
    _require_graphblas,
    _to_scipy,
    _vec_from_np,
    _vec_to_np,
    gb,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "expansion":       2,
    "inflation":       2.0,
    "prune_threshold": 1e-3,
    "max_iter":        100,
    "convergence_tol": 1e-4,
}


def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def mcl_cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict,
    n_workers: int | None = None,
) -> dict:
    """MCL using SuiteSparse:GraphBLAS SpGEMM for the expansion step.

    Inflation, column normalisation, and pruning also go through
    SuiteSparse; only the cheap Frobenius-norm convergence check and the
    final attractor cluster extraction run on scipy.
    """
    _require_graphblas()
    n_threads = _configure_threads(n_workers)

    p   = _merge_params(params)
    e   = int(p["expansion"])
    r   = float(p["inflation"])
    thr = float(p["prune_threshold"])
    cap = int(p["max_iter"])
    tol = float(p["convergence_tol"])

    n = int(graph_csr.shape[0])
    if n == 0:
        return {
            "cluster_assignments": [], "num_clusters": 0,
            "iterations": 0, "converged": True,
            "note": "graphblas SuiteSparse (empty graph)",
        }

    # ---- Preprocess: symmetrize + binarize + self-loops + col-norm ------
    graph_sym = graph_csr + graph_csr.T
    graph_sym.data = np.ones_like(graph_sym.data, dtype=np.float32)
    eye = sp.eye(n, format="csr", dtype=np.float32)
    M_sp = (graph_sym + eye).astype(np.float32)
    col_sums = np.asarray(M_sp.sum(axis=0)).flatten().astype(np.float32)
    col_sums[col_sums == 0] = 1.0
    M_sp = (M_sp @ sp.diags(1.0 / col_sums, format="csr",
                            dtype=np.float32)).astype(np.float32)

    M_gb = _from_scipy(M_sp, dtype=np.float32)

    converged = False
    iteration = 0

    for iteration in range(1, cap + 1):
        # Save scipy snapshot for cheap Frobenius diff (also serves as
        # the previous-iter matrix should we early-exit).
        M_old_sp = _to_scipy(M_gb, "csr")

        # ---- Expansion: M = M^e via SuiteSparse SpGEMM ------------------
        for _ in range(e - 1):
            M_gb = M_gb.mxm(M_gb, gb.semiring.plus_times).new()

        # ---- Inflation: element-wise power ------------------------------
        M_gb = M_gb.apply(gb.binary.pow, right=r).new()

        # ---- Column normalise via SuiteSparse reduce + diag-scale -------
        col_sums_gb = M_gb.reduce_columnwise(gb.monoid.plus).new()
        col_sums_np = _vec_to_np(col_sums_gb, n, dtype=np.float32, fill=0.0)
        col_sums_np[col_sums_np == 0] = 1.0
        inv_col = (1.0 / col_sums_np).astype(np.float32)
        D_inv_gb = gb.ss.diag(_vec_from_np(inv_col, dtype=gb.dtypes.FP32))
        M_gb = M_gb.mxm(D_inv_gb, gb.semiring.plus_times).new()

        # ---- Prune small entries (sparsify) -----------------------------
        M_gb = M_gb.select(">", thr).new()

        # ---- Convergence check on scipy ---------------------------------
        M_sp = _to_scipy(M_gb, "csr")
        diff = M_sp - M_old_sp
        frob = float(np.sqrt((diff.data.astype(np.float64) ** 2).sum()))
        if frob < tol:
            converged = True
            break

    # ---- Cluster extraction (scipy, reuses existing helper) -------------
    M_final = _to_scipy(M_gb, "csr")
    labels = _extract_clusters(M_final)

    return {
        "cluster_assignments": labels.tolist(),
        "num_clusters":        int(labels.max() + 1) if labels.size > 0 else 0,
        "iterations":          iteration,
        "converged":           converged,
        "note": (
            f"graphblas SuiteSparse nthreads={n_threads}. "
            "Graph was symmetrized for MCL. Original edge directions not preserved."
        ),
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
        "output":       mcl_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "backend": "graphblas"},
    }
