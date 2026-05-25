"""
algorithms/mcl.py — Markov Clustering (MCL) for GRN Community Detection
========================================================================

Biological Context
------------------
MCL models gene regulatory interactions as a random walk on a directed GRN.
Tightly co-regulated or functionally coupled genes form "flow pools" that
trap probability mass during iterated expansion and inflation steps.  The
resulting clusters correspond to co-regulated gene modules — groups of genes
that share common transcription factor (TF) regulators or participate in the
same regulatory pathway (e.g. E2F target genes, NF-κB regulon members).

Directed GRN → undirected symmetrization
-----------------------------------------
MCL requires an undirected (symmetric) stochastic matrix.  Because GRNs are
directed (TF → gene), each function symmetrizes the input as a first step:

    A_sym = binarize(A + A^T)

Mutual regulation (TF ↔ gene feedback loops) contributes edges from both
directions, creating a stronger undirected connection between those two nodes.
One-way TF → gene edges appear once.  The symmetrization is explicit and
logged in the returned ``note`` field so downstream consumers know the
original edge directions were not preserved.

Algorithm Overview
------------------
1. Add self-loops to the adjacency matrix (ensures irreducibility).
2. Column-normalise to produce a column-stochastic (Markov) matrix M.
3. Iterate until convergence:
       a. Expansion  — M = M^e        (spreads flow across the graph)
       b. Inflation  — M[i,j] ^= r, then column-renormalise
                                       (sharpens strong flows, kills weak ones)
       c. Pruning    — zero out entries below prune_threshold
       d. Check ||M_new − M_old||_F < convergence_tol
4. Extract clusters from attractor columns (non-zero diagonal nodes).

Parameter Guide
---------------
expansion       (int,   default 2)    Matrix power per iteration.
                                      Higher → longer random walks → larger clusters.
inflation       (float, default 2.0)  Inflation exponent.
                                      Range 1.4 (coarse) … 6.0 (fine-grained).
                                      2.0 is standard for most biological networks.
prune_threshold (float, default 1e-3) Flow values below this are zeroed.
                                      Increase to reduce memory at the cost of accuracy.
max_iter        (int,   default 100)  Hard iteration cap.
convergence_tol (float, default 1e-4) Early-stop Frobenius-norm threshold.

References
----------
van Dongen, S. (2000). A Cluster Algorithm for Graphs. CWI Technical Report.
"""

# ── GPU / CUDA-optimised implementation ──────────────────────────────────
# Source: biological_network_framework/algorithms/mcl.py
# Requires: cupy-cuda11x (or matching CUDA version), pycuda
# ──────────────────────────────────────────────────────────────────────────

import sys
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

# ---------------------------------------------------------------------------
# Optional CuPy import
# ---------------------------------------------------------------------------

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    cpsp = None
    _CUPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "expansion": 2,
    "inflation": 2.0,
    "prune_threshold": 1e-3,
    "max_iter": 100,
    "convergence_tol": 1e-4,
}

# For GPU: keep at most this many non-zero entries per column after each
# inflation step to bound VRAM growth on mid-range cards (RTX 2060 / 6 GB).
_GPU_TOPK: int = 512


# ---------------------------------------------------------------------------
# Shared helpers (CPU)
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


def _add_self_loops(M: sp.csr_matrix) -> sp.csr_matrix:
    """Add identity to ensure every node has at least one self-transition."""
    n = M.shape[0]
    eye = sp.eye(n, format="csr", dtype=M.dtype)
    result = M + eye
    result.sum_duplicates()
    return result


def _col_normalize(M: sp.csr_matrix) -> sp.csr_matrix:
    """Column-normalise M so each column sums to 1 (column-stochastic matrix)."""
    M_csc = M.tocsc().astype(np.float64)
    col_sums = np.asarray(M_csc.sum(axis=0)).flatten()
    col_sums[col_sums == 0.0] = 1.0          # guard isolated nodes
    inv = sp.diags(1.0 / col_sums, format="csr")
    return (M_csc @ inv).tocsr()


def _expand(M: sp.csr_matrix, e: int) -> sp.csr_matrix:
    """Expansion step: raise M to the integer power e (SpGEMM)."""
    result = M
    for _ in range(e - 1):
        result = result @ M
    return result


def _inflate_serial(M: sp.csr_matrix, r: float) -> sp.csr_matrix:
    """Inflation step (serial): element-wise power r, then column-renormalise."""
    M_csc = M.tocsc().astype(np.float64)
    M_csc.data **= r
    # Re-use column normalisation on the modified data
    col_sums = np.asarray(M_csc.sum(axis=0)).flatten()
    col_sums[col_sums == 0.0] = 1.0
    inv = sp.diags(1.0 / col_sums, format="csr")
    return (M_csc @ inv).tocsr()


def _prune(M: sp.csr_matrix, threshold: float) -> sp.csr_matrix:
    """Zero out entries below threshold and remove structural zeros."""
    M = M.copy()
    M.data[M.data < threshold] = 0.0
    M.eliminate_zeros()
    return M


def _frobenius_diff(A: sp.csr_matrix, B: sp.csr_matrix) -> float:
    """Frobenius norm of (A - B) for sparse matrices."""
    diff = A - B
    return float(np.sqrt(diff.data @ diff.data))


def _extract_clusters(M: sp.csr_matrix) -> np.ndarray:
    """
    Extract cluster labels from a converged MCL matrix.

    Strategy
    --------
    1. Identify attractor nodes: columns j where M[j,j] > 0.
    2. For each node i, assign it to the attractor j = argmax M[i, attractors].
    3. Fallback: weakly-connected components if no diagonal is non-zero.
    """
    n = M.shape[0]
    M_csr = M.tocsr()
    diag = np.asarray(M_csr.diagonal()).flatten()
    attractors = np.where(diag > 0)[0]

    if len(attractors) == 0:
        _, labels = csgraph.connected_components(
            M_csr, directed=False, connection="weak"
        )
        return labels.astype(np.int32)

    M_att = np.asarray(M_csr[:, attractors].todense())  # (n, K)
    assignments = np.argmax(M_att, axis=1).flatten()
    return assignments.astype(np.int32)


# ---------------------------------------------------------------------------
# Multiprocessing worker — must be module-level for pickle compatibility
# ---------------------------------------------------------------------------

def _inflate_col_chunk(args: tuple) -> np.ndarray:
    """
    Inflate a contiguous block of CSC columns.

    Receives a tuple (data, local_indptr, r) where:
      data        — non-zero values for columns in this chunk
      local_indptr — column pointer array re-zeroed to this chunk's start
      r           — inflation exponent
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


def _inflate_parallel(M: sp.csr_matrix, r: float, n_workers: int) -> sp.csr_matrix:
    """Inflation step parallelised over column chunks via multiprocessing."""
    M_csc = M.tocsc().astype(np.float64)
    n = M_csc.shape[1]
    chunk_size = max(1, (n + n_workers - 1) // n_workers)

    args_list = []
    col_ranges = []                                  # (start, end, ptr_s, ptr_e)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
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
# GPU helpers (CuPy)
# ---------------------------------------------------------------------------

def _col_normalize_gpu(M):
    """Column-normalise a CuPy CSC sparse matrix."""
    col_sums = cp.asarray(M.sum(axis=0)).flatten()
    col_sums[col_sums == 0.0] = 1.0
    inv = cpsp.diags(1.0 / col_sums)
    return M @ inv


def _inflate_gpu(M, r: float):
    """Inflation step on GPU: element-wise power then column-renormalise."""
    M_csc = M.tocsc()
    M_csc.data = M_csc.data ** r
    return _col_normalize_gpu(M_csc).tocsr()


def _topk_per_col_gpu(M, k: int):
    """
    Keep only the top-k non-zero values per column on GPU.
    Operates on a CuPy CSC matrix in-place on the data array.
    """
    M_csc = M.tocsc()
    indptr = M_csc.indptr
    data = M_csc.data
    n_cols = M_csc.shape[1]

    for j in range(n_cols):
        s, e = int(indptr[j]), int(indptr[j + 1])
        if (e - s) <= k:
            continue
        col_data = data[s:e]
        # Threshold = k-th largest value
        kth_val = float(cp.partition(col_data, -(k))[-(k)])
        data[s:e] = cp.where(col_data >= kth_val, col_data, 0.0)

    M_csc.eliminate_zeros()
    return M_csc.tocsr()


def _prune_gpu(M, threshold: float):
    M_copy = M.copy()
    M_copy.data[M_copy.data < threshold] = 0.0
    M_copy.eliminate_zeros()
    return M_copy


def _frobenius_diff_gpu(A, B) -> float:
    diff = A - B
    return float(cp.sqrt((diff.data ** 2).sum()).item())


def _extract_clusters_gpu(M) -> np.ndarray:
    """Transfer converged matrix to CPU and extract cluster labels."""
    M_cpu = sp.csr_matrix(
        (cp.asnumpy(M.data), cp.asnumpy(M.indices), cp.asnumpy(M.indptr)),
        shape=M.shape,
    )
    return _extract_clusters(M_cpu)


# ---------------------------------------------------------------------------
# Public implementations
# ---------------------------------------------------------------------------

def mcl_cpu_single(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """
    MCL — single-threaded CPU implementation using scipy sparse operations.

    Detects co-regulated gene modules in a GRN.  The directed input graph is
    symmetrized before MCL runs (see module docstring).  All matrix operations
    (SpGEMM, element-wise power, column sums) run on a single CPU thread.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
        Pre-processed directed GRN adjacency matrix (CSR format).
    params    : dict
        See module docstring for key descriptions.

    Returns
    -------
    dict with keys:
        cluster_assignments : list[int]  — 0-indexed cluster ID per node
        num_clusters        : int
        iterations          : int        — iterations until convergence/cap
        converged           : bool
        note                : str        — records symmetrization applied
    """
    p = _merge_params(params)
    e   = int(p["expansion"])
    r   = float(p["inflation"])
    thr = float(p["prune_threshold"])
    cap = int(p["max_iter"])
    tol = float(p["convergence_tol"])

    # GRN edges are directed. MCL requires an undirected graph. We symmetrize
    # here as an approximation — mutual regulation (TF↔gene feedback) appears
    # as a stronger connection since both directions contribute.
    graph_sym = graph_csr + graph_csr.T
    graph_sym.data = np.ones_like(graph_sym.data)  # binarize

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


def mcl_cpu_multi(
    graph_csr: sp.csr_matrix,
    params: dict,
    n_workers: int = 4,
) -> dict:
    """
    MCL — multi-threaded CPU implementation.

    Detects co-regulated gene modules in a GRN.  The directed input graph is
    symmetrized before MCL runs (see module docstring).  The inflation step
    (embarrassingly parallel over columns) is distributed via multiprocessing.Pool;
    expansion remains serial because scipy's SpGEMM already exploits BLAS.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
    params    : dict  — see module docstring
    n_workers : int   — number of worker processes for inflation

    Returns
    -------
    Same structure as mcl_cpu_single.
    """
    p = _merge_params(params)
    e   = int(p["expansion"])
    r   = float(p["inflation"])
    thr = float(p["prune_threshold"])
    cap = int(p["max_iter"])
    tol = float(p["convergence_tol"])

    # GRN edges are directed. MCL requires an undirected graph. We symmetrize
    # here as an approximation — mutual regulation (TF↔gene feedback) appears
    # as a stronger connection since both directions contribute.
    graph_sym = graph_csr + graph_csr.T
    graph_sym.data = np.ones_like(graph_sym.data)  # binarize

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


def mcl_gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """
    MCL — GPU-accelerated implementation via CuPy sparse (cupyx.scipy.sparse).

    The full MCL matrix pipeline (expansion, inflation, pruning) runs on the
    GPU.  Aggressive top-k column pruning (keep ≤ _GPU_TOPK entries per column)
    bounds VRAM growth on mid-range cards (RTX 2060 / 6 GB).  Cluster
    extraction transfers the final matrix to CPU.

    Falls back to mcl_cpu_single with a warning if CuPy is unavailable.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
    params    : dict  — see module docstring

    Returns
    -------
    Same structure as mcl_cpu_single.
    """
    if not _CUPY_AVAILABLE:
        warnings.warn(
            "CuPy is not available — falling back to mcl_cpu_single.",
            RuntimeWarning,
            stacklevel=2,
        )
        return mcl_cpu_single(graph_csr, params)

    p = _merge_params(params)
    e   = int(p["expansion"])
    r   = float(p["inflation"])
    thr = float(p["prune_threshold"])
    cap = int(p["max_iter"])
    tol = float(p["convergence_tol"])

    # GRN edges are directed. MCL requires an undirected graph. We symmetrize
    # here as an approximation — mutual regulation (TF↔gene feedback) appears
    # as a stronger connection since both directions contribute.
    graph_sym = graph_csr + graph_csr.T
    graph_sym.data = np.ones_like(graph_sym.data)  # binarize

    # Prepare symmetrized matrix on CPU, then upload to GPU
    M_cpu = _add_self_loops(graph_sym.astype(np.float64))
    M_cpu = _col_normalize(M_cpu)
    M_coo = M_cpu.tocoo()
    M = cpsp.csr_matrix(
        (
            cp.asarray(M_coo.data, dtype=cp.float64),
            (cp.asarray(M_coo.row), cp.asarray(M_coo.col)),
        ),
        shape=M_coo.shape,
    )

    converged = False
    for iteration in range(1, cap + 1):
        M_old = M.copy()

        # Expansion (SpGEMM on GPU)
        for _ in range(e - 1):
            M = M @ M

        # Inflation
        M = _inflate_gpu(M, r)

        # Top-k pruning to control VRAM, then threshold prune
        M = _topk_per_col_gpu(M, _GPU_TOPK)
        M = _prune_gpu(M, thr)

        if _frobenius_diff_gpu(M, M_old) < tol:
            converged = True
            break

    labels = _extract_clusters_gpu(M)

    # Free GPU memory before returning
    del M, M_old
    cp.get_default_memory_pool().free_all_blocks()

    return {
        "cluster_assignments": labels.tolist(),
        "num_clusters":        int(labels.max() + 1),
        "iterations":          iteration,
        "converged":           converged,
        "note": "Graph was symmetrized for MCL. Original edge directions not preserved.",
    }


# ---------------------------------------------------------------------------
# Runner interface — called by benchmark/runner.py as _cpu_single, etc.
# ---------------------------------------------------------------------------

def _cpu_single(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    """Benchmark runner entry-point for cpu_single mode."""
    p = _merge_params(params)
    result = mcl_cpu_single(graph_csr, p)
    return {"output": result, "extra_params": p}


def _cpu_multi(
    graph_csr: sp.csr_matrix,
    params: dict | None = None,
    n_workers: int = 4,
    **_,
) -> dict:
    """Benchmark runner entry-point for cpu_multi mode."""
    p = _merge_params(params)
    result = mcl_cpu_multi(graph_csr, p, n_workers=n_workers)
    return {"output": result, "extra_params": {**p, "n_workers": n_workers}}


def _gpu(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    """Benchmark runner entry-point for gpu mode."""
    p = _merge_params(params)
    result = mcl_gpu(graph_csr, p)
    return {"output": result, "extra_params": p}
