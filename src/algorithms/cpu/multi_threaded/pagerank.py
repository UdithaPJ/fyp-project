"""
src/algorithms/cpu/multi_threaded/pagerank.py
=============================================

PageRank — multi-process CPU implementation only.

The SpMV step (M @ PR) — the bottleneck in each iteration — is split into
row-range chunks and dispatched to a ProcessPoolExecutor.  Each worker
reconstructs a CSR sub-matrix for its rows and performs a local SpMV.
The partial results are concatenated to form the full output vector.

Dangling redistribution and convergence checks remain serial (trivial cost).

Shared helpers imported from src.algorithms.common.helpers:
    _build_transition_matrix, _split_top_nodes

For the single-threaded variant see:
    src.algorithms.cpu.single_threaded.pagerank
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import (
    _build_transition_matrix,
    _split_top_nodes,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "damping":   0.85,
    "max_iter":  100,
    "tolerance": 1e-6,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# Module-level worker — must be at module scope for ProcessPoolExecutor pickle
# ---------------------------------------------------------------------------

def _spmv_row_chunk(args: tuple) -> np.ndarray:
    """
    ProcessPoolExecutor worker: compute CSR SpMV for a contiguous row range.

    Receives (data, indices, indptr, pr, n_cols) where indptr is re-zeroed
    to the chunk's local start.  Returns the partial result vector.
    """
    data, indices, indptr, pr, n_cols = args
    n_rows_chunk = len(indptr) - 1
    M_chunk = sp.csr_matrix(
        (data, indices, indptr),
        shape=(n_rows_chunk, n_cols),
    )
    return (M_chunk @ pr).astype(np.float64)


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def pagerank_cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict,
    n_workers: int = 4,
) -> dict:
    """
    PageRank — multi-process CPU implementation.

    The SpMV step (M @ PR) is split into row-range chunks and dispatched
    to a ProcessPoolExecutor.  Each worker reconstructs a CSR sub-matrix
    for its rows and performs a local SpMV; partial results are concatenated.

    Dangling redistribution and convergence checks remain serial.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
    params    : dict
        damping   (float, default 0.85)
        max_iter  (int,   default 100)
        tolerance (float, default 1e-6)
    n_workers : int — number of parallel worker processes

    Returns
    -------
    dict with keys: scores, iterations, converged, top_regulators, top_targets
    """
    p        = _merge_params(params)
    d        = float(p["damping"])
    max_iter = int(p["max_iter"])
    tol      = float(p["tolerance"])

    N = graph_csr.shape[0]
    M, dangling_mask = _build_transition_matrix(graph_csr)
    teleport_per_node = (1.0 - d) / N

    out_degrees = np.asarray(graph_csr.sum(axis=1)).flatten()
    active_mask = out_degrees > 0
    n_active    = int(active_mask.sum())
    if n_active == 0:
        warnings.warn(
            "Warning: no outgoing-edge nodes found, using uniform dangling redistribution",
            UserWarning,
            stacklevel=2,
        )

    # Pre-build chunk argument templates (data/indices/indptr slices of M)
    chunk_size  = max(1, (N + n_workers - 1) // n_workers)
    chunk_specs: list[tuple] = []

    for start in range(0, N, chunk_size):
        end        = min(start + chunk_size, N)
        ptr_s      = int(M.indptr[start])
        ptr_e      = int(M.indptr[end])
        local_indptr = (M.indptr[start:end + 1] - M.indptr[start]).copy()
        chunk_specs.append((
            M.data[ptr_s:ptr_e].copy(),
            M.indices[ptr_s:ptr_e].copy(),
            local_indptr,
            None,   # placeholder — PR is filled per-iteration below
            N,
        ))

    PR        = np.full(N, 1.0 / N, dtype=np.float64)
    converged = False

    for iteration in range(1, max_iter + 1):
        PR_old = PR

        dangling_mass = d * float(PR_old[dangling_mask].sum())
        dangling_contrib = np.zeros(N, dtype=np.float64)
        if n_active > 0:
            dangling_contrib[active_mask] = dangling_mass / n_active
        else:
            dangling_contrib[:] = dangling_mass / N

        args_list = [
            (spec[0], spec[1], spec[2], PR_old, spec[4])
            for spec in chunk_specs
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            partial_results = list(executor.map(_spmv_row_chunk, args_list))

        spmv_result = np.concatenate(partial_results)
        PR = d * spmv_result + dangling_contrib + teleport_per_node

        if np.abs(PR - PR_old).sum() < tol:
            converged = True
            break

    top_reg, top_tgt = _split_top_nodes(PR, out_degrees)
    return {
        "scores":         PR.tolist(),
        "iterations":     iteration,
        "converged":      converged,
        "top_regulators": top_reg,
        "top_targets":    top_tgt,
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
        "output":       pagerank_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "n_workers": n_workers},
    }
