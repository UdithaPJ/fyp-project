"""
src/algorithms/cpu/multi_threaded/mcl.py
=========================================

Markov Clustering (MCL) — multi-process CPU implementation only.

The inflation step (embarrassingly parallel over columns) is distributed
via multiprocessing.Pool.  Expansion (SpGEMM) remains serial because
scipy's SpGEMM already exploits BLAS-level parallelism.

The directed GRN is symmetrized before MCL runs (see module docstring of
the original cpu/mcl.py for the biological rationale).

Shared helpers imported from src.algorithms.common.helpers:
    _add_self_loops, _col_normalize, _expand, _prune,
    _frobenius_diff, _extract_clusters

Exclusive to this file:
    _inflate_col_chunk — per-column-chunk inflation worker
    _inflate_parallel  — dispatches column chunks to multiprocessing.Pool

For the single-threaded variant see:
    src.algorithms.cpu.single_threaded.mcl
"""

from __future__ import annotations

from multiprocessing import Pool

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
# Module-level workers — must be at module scope for multiprocessing.Pool pickle
# ---------------------------------------------------------------------------

def _inflate_col_chunk(args: tuple) -> np.ndarray:
    """
    Inflate a contiguous block of CSC columns.

    Receives (data, local_indptr, r) where:
      data         — non-zero values for columns in this chunk
      local_indptr — column pointer array re-zeroed to this chunk's start
      r            — inflation exponent
    Returns the inflated data array (same length as input data).
    """
    data, local_indptr, r = args
    new_data = data.astype(np.float64).copy()
    n_cols = len(local_indptr) - 1
    for j in range(n_cols):
        s, e = int(local_indptr[j]), int(local_indptr[j + 1])
        if s == e:
            continue
        col = new_data[s:e]
        col **= r
        col_sum = col.sum()
        if col_sum > 0.0:
            col /= col_sum
        new_data[s:e] = col
    return new_data


def _inflate_parallel(
    M:         sp.csr_matrix,
    r:         float,
    n_workers: int,
) -> sp.csr_matrix:
    """Inflation step parallelised over column chunks via multiprocessing.Pool."""
    M_csc      = M.tocsc().astype(np.float64)
    n          = M_csc.shape[1]
    chunk_size = max(1, (n + n_workers - 1) // n_workers)

    args_list:  list[tuple]             = []
    col_ranges: list[tuple[int, int]]   = []

    for start in range(0, n, chunk_size):
        end   = min(start + chunk_size, n)
        ptr_s = int(M_csc.indptr[start])
        ptr_e = int(M_csc.indptr[end])
        local_indptr = (M_csc.indptr[start:end + 1] - M_csc.indptr[start]).copy()
        args_list.append((M_csc.data[ptr_s:ptr_e].copy(), local_indptr, r))
        col_ranges.append((ptr_s, ptr_e))

    with Pool(processes=n_workers) as pool:
        results = pool.map(_inflate_col_chunk, args_list)

    new_data = M_csc.data.copy().astype(np.float64)
    for chunk_data, (ptr_s, ptr_e) in zip(results, col_ranges):
        new_data[ptr_s:ptr_e] = chunk_data

    M_new_csc = sp.csc_matrix(
        (new_data, M_csc.indices.copy(), M_csc.indptr.copy()),
        shape=M_csc.shape,
    )
    return M_new_csc.tocsr()


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def mcl_cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict,
    n_workers: int = 4,
) -> dict:
    """
    MCL — multi-process CPU implementation.

    Detects co-regulated gene modules in a GRN.  The directed input graph is
    symmetrized before MCL runs.  The inflation step (column-wise) is
    distributed across worker processes; expansion remains serial.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
    params    : dict
        expansion       (int,   default 2)
        inflation       (float, default 2.0)
        prune_threshold (float, default 1e-3)
        max_iter        (int,   default 100)
        convergence_tol (float, default 1e-4)
    n_workers : int — number of worker processes for inflation

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
        M = _inflate_parallel(M, r, n_workers)
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

def _cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict | None = None,
    n_workers: int = 4,
    **_,
) -> dict:
    """Benchmark runner entry-point for cpu_multi mode."""
    p = _merge_params(params)
    return {
        "output":       mcl_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "n_workers": n_workers},
    }
