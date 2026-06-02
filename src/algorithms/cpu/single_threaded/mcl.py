"""
src/algorithms/cpu/single_threaded/mcl.py
==========================================

Markov Clustering (MCL) — single-threaded CPU implementation only.

All matrix operations (SpGEMM, element-wise power, column sums) run on a
single CPU thread via scipy.  The inflation step uses ``_inflate_serial``,
which is the straightforward element-wise power followed by column
renormalisation.

The directed GRN is symmetrized before MCL runs (see module docstring of
the original cpu/mcl.py for the biological rationale).

Shared helpers imported from src.algorithms.common.helpers:
    _add_self_loops, _col_normalize, _expand, _prune,
    _frobenius_diff, _extract_clusters

Exclusive to this file:
    _inflate_serial — serial element-wise inflation step

For the parallel inflation variant see:
    src.algorithms.cpu.multi_threaded.mcl
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import (
    _add_self_loops,
    _col_normalize,
    _expand,
    _extract_clusters,
    _frobenius_diff,
    _prune,
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# Inflation step — serial (exclusive to single-threaded mode)
# ---------------------------------------------------------------------------

def _inflate_serial(M: sp.csr_matrix, r: float) -> sp.csr_matrix:
    """Inflation step (serial): element-wise power r, then column-renormalise."""
    # MEMORY_FIX (C-5/M-8): keep everything float32 and mutate in place.
    # Old code: tocsc().astype(float64) → +120 MB transient per iter on a
    # 15 M-edge graph.  Then the `M_csc @ inv` SpGEMM allocates another
    # full output matrix that is only used to renormalise.  Both are
    # avoidable: scale data column-by-column via indptr ranges.
    M_csc = M.tocsc()
    if M_csc.dtype != np.float32:
        M_csc = M_csc.astype(np.float32)
    np.power(M_csc.data, r, out=M_csc.data)
    indptr = M_csc.indptr
    data = M_csc.data
    for j in range(M_csc.shape[1]):
        s, e = int(indptr[j]), int(indptr[j + 1])
        if s == e:
            continue
        col_sum = float(data[s:e].sum())
        if col_sum > 0.0:
            data[s:e] /= col_sum
    return M_csc.tocsr()


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def mcl_cpu_single(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """
    MCL — single-threaded CPU implementation using scipy sparse operations.

    Detects co-regulated gene modules in a GRN.  The directed input graph is
    symmetrized before MCL runs.  All matrix operations run on a single CPU
    thread.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix — directed GRN adjacency (CSR).
    params    : dict
        expansion       (int,   default 2)    — matrix power per iteration
        inflation       (float, default 2.0)  — inflation exponent
        prune_threshold (float, default 1e-3) — entries below this are zeroed
        max_iter        (int,   default 100)  — hard iteration cap
        convergence_tol (float, default 1e-4) — Frobenius-norm convergence threshold

    Returns
    -------
    dict with keys: cluster_assignments, num_clusters, iterations, converged, note
    """
    p   = _merge_params(params)
    e   = int(p["expansion"])
    r   = float(p["inflation"])
    thr = float(p["prune_threshold"])
    cap = int(p["max_iter"])
    tol = float(p["convergence_tol"])

    graph_sym = graph_csr + graph_csr.T
    graph_sym.data = np.ones_like(graph_sym.data)

    M = _add_self_loops(graph_sym.astype(np.float64))
    M = _col_normalize(M)

    converged = False
    for iteration in range(1, cap + 1):
        M_old = M.copy()
        M = _expand(M, e)
        M = _inflate_serial(M, r)
        M = _prune(M, thr)

        if _frobenius_diff(M, M_old) < tol:
            converged = True
            break

    labels = _extract_clusters(M)
    return {
        "cluster_assignments": labels.tolist(),
        "num_clusters":        int(labels.max() + 1),
        "iterations":          iteration,
        "converged":           converged,
        "note": "Graph was symmetrized for MCL. Original edge directions not preserved.",
    }


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _cpu_single(
    graph_csr: sp.csr_matrix,
    params: dict | None = None,
    **_,
) -> dict:
    """Benchmark runner entry-point for cpu_single mode."""
    p = _merge_params(params)
    return {"output": mcl_cpu_single(graph_csr, p), "extra_params": p}
