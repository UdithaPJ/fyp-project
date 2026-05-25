"""
src/algorithms/cpu/multi_threaded/hits.py
=========================================

HITS — multi-process CPU implementation only.

Both the A^T·h (authority) and A·a (hub) matrix-vector products are
parallelised using ProcessPoolExecutor with row-range partitioning.
Row chunks for both matrices are built once before the iteration loop
(only the score vectors change each iteration).

The L2 normalisation is applied after gathering partial results on the
main process — it is a trivial O(N) serial step.

Shared helpers imported from src.algorithms.common.helpers:
    _l2_normalize, _top_k, hits_pack_result  (imported as _pack_result)

Exclusive to this file:
    _spmv_hits_chunk — row-chunk SpMV worker
    _build_chunks    — pre-builds chunk argument tuples
    _parallel_spmv   — runs a pre-chunked SpMV via ProcessPoolExecutor

For the single-threaded variant see:
    src.algorithms.cpu.single_threaded.hits
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import (
    _l2_normalize,
    _top_k,
    hits_pack_result as _pack_result,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "max_iter":  100,
    "tolerance": 1e-6,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# Module-level SpMV worker — must be at module scope for ProcessPoolExecutor
# ---------------------------------------------------------------------------

def _spmv_hits_chunk(args: tuple) -> np.ndarray:
    """
    ProcessPoolExecutor worker: SpMV for a contiguous block of rows.

    Receives (data, indices, local_indptr, vec, n_cols) where local_indptr
    is re-zeroed to this chunk's start.  Returns the partial result vector.
    """
    data, indices, local_indptr, vec, n_cols = args
    n_rows_chunk = len(local_indptr) - 1
    M_chunk = sp.csr_matrix(
        (data, indices, local_indptr),
        shape=(n_rows_chunk, n_cols),
    )
    return (M_chunk @ vec).astype(np.float64)


def _build_chunks(M: sp.csr_matrix, n_workers: int) -> list[tuple]:
    """Pre-build row-chunk argument tuples for ProcessPoolExecutor."""
    N      = M.shape[0]
    n_cols = M.shape[1]
    chunk_size = max(1, (N + n_workers - 1) // n_workers)
    chunks = []
    for start in range(0, N, chunk_size):
        end   = min(start + chunk_size, N)
        ptr_s = int(M.indptr[start])
        ptr_e = int(M.indptr[end])
        local_indptr = (M.indptr[start:end + 1] - M.indptr[start]).copy()
        chunks.append((
            M.data[ptr_s:ptr_e].copy(),
            M.indices[ptr_s:ptr_e].copy(),
            local_indptr,
            None,     # placeholder — vector injected per-iteration
            n_cols,
        ))
    return chunks


def _parallel_spmv(
    chunks:    list[tuple],
    vec:       np.ndarray,
    n_workers: int,
) -> np.ndarray:
    """Run a pre-chunked sparse matrix-vector product in parallel."""
    args_list = [(c[0], c[1], c[2], vec, c[4]) for c in chunks]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        parts = list(ex.map(_spmv_hits_chunk, args_list))
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def hits_cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict,
    n_workers: int = 4,
) -> dict:
    """
    HITS — multi-process CPU implementation.

    Both the A^T·h and A·a products are parallelised via row-range
    partitioning.  Row chunks for both matrices are built once before
    the iteration loop; only the score vectors change per iteration.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix — directed GRN adjacency.
    params    : dict
        max_iter  (int,   default 100)
        tolerance (float, default 1e-6)
    n_workers : int

    Returns
    -------
    dict with keys: hub_scores, authority_scores, iterations, converged,
                    top_hubs, top_authorities, hub_authority_overlap
    """
    p        = _merge_params(params)
    max_iter = int(p["max_iter"])
    tol      = float(p["tolerance"])

    N   = graph_csr.shape[0]
    A   = graph_csr.astype(np.float64)
    A_T = A.T.tocsr()

    chunks_A_T = _build_chunks(A_T, n_workers)
    chunks_A   = _build_chunks(A,   n_workers)

    h         = np.ones(N, dtype=np.float64)
    a         = np.ones(N, dtype=np.float64)
    converged = False

    for iteration in range(1, max_iter + 1):
        h_old, a_old = h, a

        a_new = _l2_normalize(_parallel_spmv(chunks_A_T, h_old, n_workers))
        h_new = _l2_normalize(_parallel_spmv(chunks_A,   a_new, n_workers))

        if np.linalg.norm(h_new - h_old) + np.linalg.norm(a_new - a_old) < tol:
            converged = True
            a, h = a_new, h_new
            break

        a, h = a_new, h_new

    return _pack_result(h, a, iteration, converged)


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
        "output":       hits_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "n_workers": n_workers},
    }
