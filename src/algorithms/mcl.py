"""
src/algorithms/mcl.py — Markov Clustering for co-regulated gene modules
========================================================================

Biological context
------------------
MCL detects co-regulated gene modules by simulating a random walk on the
GRN: tightly co-regulated genes form "flow pools" that trap probability
mass during the expansion/inflation cycle.  The directed GRN is
symmetrised internally because MCL requires an undirected stochastic
matrix; mutual TF↔gene feedback contributes from both directions,
producing a stronger connection.  This is recorded in the ``note`` field.

Algorithm
---------
1. Symmetrise + add self-loops + column-normalise → Markov matrix M
2. Repeat until ||M_new − M_old||_F < tol:
   a. Expansion : M = M^e          (spreads flow)
   b. Inflation : M[i,j] **= r; renormalise columns  (sharpens)
   c. Prune     : zero entries below prune_threshold
3. Extract clusters from attractor columns (non-zero diagonal nodes).
"""

from __future__ import annotations

import time
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

from .base import AlgorithmBase

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    cpsp = None
    _CUPY_AVAILABLE = False


_GPU_TOPK = 512   # bound VRAM growth on mid-range cards

_SYMMETRIZE_NOTE = (
    "Graph was symmetrized for MCL. Original edge directions not preserved."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_self_loops(M: sp.csr_matrix) -> sp.csr_matrix:
    n = M.shape[0]
    eye = sp.eye(n, format="csr", dtype=M.dtype)
    out = M + eye
    out.sum_duplicates()
    return out


def _col_normalize(M: sp.csr_matrix) -> sp.csr_matrix:
    M_csc = M.tocsc().astype(np.float64)
    col_sums = np.asarray(M_csc.sum(axis=0)).flatten()
    col_sums[col_sums == 0.0] = 1.0
    inv = sp.diags(1.0 / col_sums, format="csr")
    return (M_csc @ inv).tocsr()


def _expand(M: sp.csr_matrix, e: int) -> sp.csr_matrix:
    out = M
    for _ in range(e - 1):
        out = out @ M
    return out


def _inflate_serial(M: sp.csr_matrix, r: float) -> sp.csr_matrix:
    M_csc = M.tocsc().astype(np.float64)
    M_csc.data **= r
    col_sums = np.asarray(M_csc.sum(axis=0)).flatten()
    col_sums[col_sums == 0.0] = 1.0
    inv = sp.diags(1.0 / col_sums, format="csr")
    return (M_csc @ inv).tocsr()


def _prune(M: sp.csr_matrix, threshold: float) -> sp.csr_matrix:
    M = M.copy()
    M.data[M.data < threshold] = 0.0
    M.eliminate_zeros()
    return M


def _frobenius_diff(A, B) -> float:
    diff = A - B
    return float(np.sqrt(diff.data @ diff.data))


def _extract_clusters(M: sp.csr_matrix) -> np.ndarray:
    n = M.shape[0]
    M_csr = M.tocsr()
    diag = np.asarray(M_csr.diagonal()).flatten()
    attractors = np.where(diag > 0)[0]
    if len(attractors) == 0:
        _, labels = csgraph.connected_components(M_csr, directed=False, connection="weak")
        return labels.astype(np.int32)
    M_att = np.asarray(M_csr[:, attractors].todense())
    return np.argmax(M_att, axis=1).astype(np.int32)


def _symmetrize_and_prep(graph_csr: sp.csr_matrix) -> sp.csr_matrix:
    sym = graph_csr + graph_csr.T
    sym.data = np.ones_like(sym.data)
    M = _add_self_loops(sym.astype(np.float64))
    return _col_normalize(M)


# ---------------------------------------------------------------------------
# Multiprocessing inflation worker (module-level for pickle)
# ---------------------------------------------------------------------------

def _inflate_col_chunk(args: tuple) -> np.ndarray:
    data, local_indptr, r = args
    new_data = data.astype(np.float64).copy()
    n_cols = len(local_indptr) - 1
    for j in range(n_cols):
        s, e = int(local_indptr[j]), int(local_indptr[j + 1])
        if s == e: continue
        col = new_data[s:e]
        col **= r
        cs = col.sum()
        if cs > 0.0:
            col /= cs
        new_data[s:e] = col
    return new_data


def _inflate_parallel(M: sp.csr_matrix, r: float, n_workers: int,
                      pool: Pool | None = None) -> sp.csr_matrix:
    M_csc = M.tocsc().astype(np.float64)
    n = M_csc.shape[1]
    chunk_size = max(1, (n + n_workers - 1) // n_workers)
    args_list, ranges = [], []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        ptr_s = int(M_csc.indptr[start])
        ptr_e = int(M_csc.indptr[end])
        local_indptr = (M_csc.indptr[start:end + 1] - M_csc.indptr[start]).copy()
        args_list.append((M_csc.data[ptr_s:ptr_e].copy(), local_indptr, r))
        ranges.append((ptr_s, ptr_e))

    if pool is not None:
        results = pool.map(_inflate_col_chunk, args_list)
    else:
        with Pool(processes=n_workers) as p:
            results = p.map(_inflate_col_chunk, args_list)

    new_data = M_csc.data.copy().astype(np.float64)
    for chunk_data, (ptr_s, ptr_e) in zip(results, ranges):
        new_data[ptr_s:ptr_e] = chunk_data

    out = sp.csc_matrix(
        (new_data, M_csc.indices.copy(), M_csc.indptr.copy()),
        shape=M_csc.shape,
    )
    return out.tocsr()


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _col_normalize_gpu(M):
    col_sums = cp.asarray(M.sum(axis=0)).flatten()
    col_sums[col_sums == 0.0] = 1.0
    inv = cpsp.diags(1.0 / col_sums)
    return M @ inv


def _inflate_gpu(M, r: float):
    M_csc = M.tocsc()
    M_csc.data = M_csc.data ** r
    return _col_normalize_gpu(M_csc).tocsr()


def _topk_per_col_gpu(M, k: int):
    M_csc = M.tocsc()
    indptr, data = M_csc.indptr, M_csc.data
    n_cols = M_csc.shape[1]
    for j in range(n_cols):
        s, e = int(indptr[j]), int(indptr[j + 1])
        if (e - s) <= k: continue
        col = data[s:e]
        kth_val = float(cp.partition(col, -(k))[-(k)])
        data[s:e] = cp.where(col >= kth_val, col, 0.0)
    M_csc.eliminate_zeros()
    return M_csc.tocsr()


def _prune_gpu(M, threshold: float):
    out = M.copy()
    out.data[out.data < threshold] = 0.0
    out.eliminate_zeros()
    return out


def _frobenius_diff_gpu(A, B) -> float:
    diff = A - B
    return float(cp.sqrt((diff.data ** 2).sum()).item())


def _extract_clusters_gpu(M) -> np.ndarray:
    M_cpu = sp.csr_matrix(
        (cp.asnumpy(M.data), cp.asnumpy(M.indices), cp.asnumpy(M.indptr)),
        shape=M.shape,
    )
    return _extract_clusters(M_cpu)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class MCL(AlgorithmBase):
    """Markov Clustering with symmetrisation for GRN modules."""

    NAME = "mcl"
    PARAM_SCHEMA = {
        "expansion":       2,
        "inflation":       2.0,
        "prune_threshold": 0.001,
        "max_iter":        100,
        "convergence_tol": 1e-4,
    }

    @staticmethod
    def cpu_single(graph_csr: sp.csr_matrix, params: dict) -> dict:
        t0 = time.perf_counter()
        e   = int(params.get("expansion",       2))
        r   = float(params.get("inflation",     2.0))
        thr = float(params.get("prune_threshold", 0.001))
        cap = int(params.get("max_iter",        100))
        tol = float(params.get("convergence_tol", 1e-4))

        M = _symmetrize_and_prep(graph_csr)
        converged = False
        iteration = 0
        for iteration in range(1, cap + 1):
            M_old = M.copy()
            M = _expand(M, e)
            M = _inflate_serial(M, r)
            M = _prune(M, thr)
            if _frobenius_diff(M, M_old) < tol:
                converged = True
                break

        labels = _extract_clusters(M)
        elapsed = time.perf_counter() - t0
        return MCL.build_result(
            mode="cpu_single", execution_time=elapsed, graph_csr=graph_csr,
            result_data={
                "cluster_assignments": labels.tolist(),
                "num_clusters":        int(labels.max() + 1),
                "iterations":          iteration,
                "converged":           converged,
                "note":                _SYMMETRIZE_NOTE,
            },
        )

    @staticmethod
    def cpu_multi(graph_csr: sp.csr_matrix, params: dict) -> dict:
        t0 = time.perf_counter()
        e   = int(params.get("expansion",       2))
        r   = float(params.get("inflation",     2.0))
        thr = float(params.get("prune_threshold", 0.001))
        cap = int(params.get("max_iter",        100))
        tol = float(params.get("convergence_tol", 1e-4))
        n_workers = int(params.get("n_workers", 4))

        M = _symmetrize_and_prep(graph_csr)
        converged = False
        iteration = 0
        # One Pool reused across all iterations (avoid spawn-per-iter on Windows)
        with Pool(processes=n_workers) as pool:
            for iteration in range(1, cap + 1):
                M_old = M.copy()
                M = _expand(M, e)
                M = _inflate_parallel(M, r, n_workers, pool=pool)
                M = _prune(M, thr)
                if _frobenius_diff(M, M_old) < tol:
                    converged = True
                    break

        labels = _extract_clusters(M)
        elapsed = time.perf_counter() - t0
        return MCL.build_result(
            mode="cpu_multi", execution_time=elapsed, graph_csr=graph_csr,
            result_data={
                "cluster_assignments": labels.tolist(),
                "num_clusters":        int(labels.max() + 1),
                "iterations":          iteration,
                "converged":           converged,
                "note":                _SYMMETRIZE_NOTE,
            },
        )

    @staticmethod
    def gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
        if not _CUPY_AVAILABLE:
            warnings.warn(
                "CuPy unavailable — falling back to mcl cpu_single.",
                RuntimeWarning, stacklevel=2,
            )
            r = MCL.cpu_single(graph_csr, params)
            r["mode"] = "gpu"
            return r

        t0 = time.perf_counter()
        e   = int(params.get("expansion",       2))
        r   = float(params.get("inflation",     2.0))
        thr = float(params.get("prune_threshold", 0.001))
        cap = int(params.get("max_iter",        100))
        tol = float(params.get("convergence_tol", 1e-4))
        topk = int(params.get("top_k_per_column", _GPU_TOPK))

        M_cpu = _symmetrize_and_prep(graph_csr)
        coo = M_cpu.tocoo()
        M = cpsp.csr_matrix(
            (cp.asarray(coo.data, dtype=cp.float64),
             (cp.asarray(coo.row), cp.asarray(coo.col))),
            shape=coo.shape,
        )

        converged = False
        iteration = 0
        for iteration in range(1, cap + 1):
            M_old = M.copy()
            for _ in range(e - 1):
                M = M @ M
            M = _inflate_gpu(M, r)
            M = _topk_per_col_gpu(M, topk)
            M = _prune_gpu(M, thr)
            if _frobenius_diff_gpu(M, M_old) < tol:
                converged = True
                break

        labels = _extract_clusters_gpu(M)
        del M, M_old
        cp.get_default_memory_pool().free_all_blocks()

        elapsed = time.perf_counter() - t0
        return MCL.build_result(
            mode="gpu", execution_time=elapsed, graph_csr=graph_csr,
            result_data={
                "cluster_assignments": labels.tolist(),
                "num_clusters":        int(labels.max() + 1),
                "iterations":          iteration,
                "converged":           converged,
                "note":                _SYMMETRIZE_NOTE,
            },
        )
