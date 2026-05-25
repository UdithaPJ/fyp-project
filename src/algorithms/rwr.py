"""
src/algorithms/rwr.py — Random Walk with Restart for GRN diffusion
===================================================================

Biological context
------------------
RWR from a seed set of TFs measures how regulatory influence diffuses
through the GRN with periodic restarts at the seeds — quantifying each
gene's *regulatory proximity* to the chosen master TFs.  Output:

  top_nodes — globally highest-scoring nodes (any role)
  top_tfs   — highest-scoring nodes within the seed set (which seeds
              accumulated the most influence after diffusion settled)

Transition matrix is column-stochastic with out-degree normalisation,
so the walk only follows real regulatory edges.  When a node lacks
outgoing edges, its column is zero and the restart vector dominates.
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

from .base import AlgorithmBase

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    cpsp = None
    _CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_transition(graph_csr: sp.csr_matrix) -> sp.csr_matrix:
    """Column-stochastic out-degree-normalised transition matrix."""
    out_deg = np.asarray(graph_csr.sum(axis=1)).flatten()
    safe = np.where(out_deg == 0, 1.0, out_deg)
    D_inv = sp.diags(1.0 / safe, format="csr")
    return (D_inv @ graph_csr).T.tocsr().astype(np.float64)


def _resolve_seeds(seed_nodes, N: int) -> np.ndarray:
    """Validate and normalise seed_nodes; default to node 0 if empty."""
    if not seed_nodes:
        return np.array([0], dtype=np.int64)
    flat: list[int] = []
    for s in seed_nodes:
        if isinstance(s, (list, tuple)):
            flat.extend(int(x) for x in s)
        else:
            flat.append(int(s))
    seeds = np.array(sorted(set(flat)), dtype=np.int64)
    bad = seeds[(seeds < 0) | (seeds >= N)]
    if len(bad) > 0:
        raise ValueError(f"RWR seed_nodes out of bounds [0,{N}): {bad.tolist()}")
    return seeds


def _spmv_chunk(args: tuple) -> np.ndarray:
    """Pickle-safe SpMV worker for ProcessPoolExecutor."""
    data, indices, indptr, vec, n_cols = args
    n_rows_chunk = len(indptr) - 1
    M_chunk = sp.csr_matrix((data, indices, indptr), shape=(n_rows_chunk, n_cols))
    return (M_chunk @ vec).astype(np.float64)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class RWR(AlgorithmBase):
    """Random Walk with Restart from a seed-TF set."""

    NAME = "rwr"
    PARAM_SCHEMA = {
        "restart_prob": 0.3,
        "max_iter":     100,
        "tolerance":    1e-6,
        "seed_nodes":   [],
    }

    @staticmethod
    def cpu_single(graph_csr: sp.csr_matrix, params: dict) -> dict:
        t0 = time.perf_counter()
        N = graph_csr.shape[0]
        r   = float(params.get("restart_prob", 0.3))
        cap = int(params.get("max_iter",     100))
        tol = float(params.get("tolerance",   1e-6))
        seeds = _resolve_seeds(params.get("seed_nodes", []), N)

        W = _build_transition(graph_csr)
        p0 = np.zeros(N, dtype=np.float64)
        p0[seeds] = 1.0 / len(seeds)
        p = p0.copy()

        iteration = 0
        for iteration in range(1, cap + 1):
            p_new = (1.0 - r) * (W @ p) + r * p0
            if np.abs(p_new - p).sum() < tol:
                p = p_new
                break
            p = p_new

        elapsed = time.perf_counter() - t0
        top_nodes = np.argsort(p)[::-1][:20].tolist()
        seed_scores = p[seeds]
        top_tfs = seeds[np.argsort(seed_scores)[::-1][:10]].tolist()

        return RWR.build_result(
            mode="cpu_single", execution_time=elapsed,
            graph_csr=graph_csr,
            result_data={
                "scores":     p.tolist(),
                "top_nodes":  top_nodes,
                "top_tfs":    top_tfs,
                "iterations": iteration,
            },
        )

    @staticmethod
    def cpu_multi(graph_csr: sp.csr_matrix, params: dict) -> dict:
        t0 = time.perf_counter()
        N = graph_csr.shape[0]
        r   = float(params.get("restart_prob", 0.3))
        cap = int(params.get("max_iter",     100))
        tol = float(params.get("tolerance",   1e-6))
        n_workers = int(params.get("n_workers", 4))
        seeds = _resolve_seeds(params.get("seed_nodes", []), N)

        W = _build_transition(graph_csr)
        chunk_size = max(1, (N + n_workers - 1) // n_workers)
        chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            ptr_s, ptr_e = int(W.indptr[start]), int(W.indptr[end])
            local_indptr = (W.indptr[start:end + 1] - W.indptr[start]).copy()
            chunks.append((
                W.data[ptr_s:ptr_e].copy(),
                W.indices[ptr_s:ptr_e].copy(),
                local_indptr, N,
            ))

        p0 = np.zeros(N, dtype=np.float64)
        p0[seeds] = 1.0 / len(seeds)
        p = p0.copy()
        iteration = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for iteration in range(1, cap + 1):
                args = [(c[0], c[1], c[2], p, c[3]) for c in chunks]
                parts = list(ex.map(_spmv_chunk, args))
                spmv = np.concatenate(parts)
                p_new = (1.0 - r) * spmv + r * p0
                if np.abs(p_new - p).sum() < tol:
                    p = p_new
                    break
                p = p_new

        elapsed = time.perf_counter() - t0
        top_nodes = np.argsort(p)[::-1][:20].tolist()
        seed_scores = p[seeds]
        top_tfs = seeds[np.argsort(seed_scores)[::-1][:10]].tolist()

        return RWR.build_result(
            mode="cpu_multi", execution_time=elapsed,
            graph_csr=graph_csr,
            result_data={
                "scores":     p.tolist(),
                "top_nodes":  top_nodes,
                "top_tfs":    top_tfs,
                "iterations": iteration,
            },
        )

    @staticmethod
    def gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
        if not _CUPY_AVAILABLE:
            warnings.warn(
                "CuPy unavailable — falling back to rwr cpu_single.",
                RuntimeWarning, stacklevel=2,
            )
            r = RWR.cpu_single(graph_csr, params)
            r["mode"] = "gpu"
            return r

        t0 = time.perf_counter()
        N = graph_csr.shape[0]
        r   = float(params.get("restart_prob", 0.3))
        cap = int(params.get("max_iter",     100))
        tol = float(params.get("tolerance",   1e-6))
        seeds = _resolve_seeds(params.get("seed_nodes", []), N)

        W_cpu = _build_transition(graph_csr)
        coo = W_cpu.tocoo()
        W_gpu = cpsp.csr_matrix(
            (cp.asarray(coo.data, dtype=cp.float64),
             (cp.asarray(coo.row), cp.asarray(coo.col))),
            shape=coo.shape,
        )
        seeds_gpu = cp.asarray(seeds)
        p0 = cp.zeros(N, dtype=cp.float64)
        p0[seeds_gpu] = 1.0 / len(seeds)
        p = p0.copy()
        iteration = 0
        for iteration in range(1, cap + 1):
            p_new = (1.0 - r) * (W_gpu @ p) + r * p0
            if float(cp.abs(p_new - p).sum()) < tol:
                p = p_new
                break
            p = p_new

        p_cpu = cp.asnumpy(p)
        del W_gpu, p0, p, seeds_gpu
        cp.get_default_memory_pool().free_all_blocks()

        elapsed = time.perf_counter() - t0
        top_nodes = np.argsort(p_cpu)[::-1][:20].tolist()
        seed_scores = p_cpu[seeds]
        top_tfs = seeds[np.argsort(seed_scores)[::-1][:10]].tolist()

        return RWR.build_result(
            mode="gpu", execution_time=elapsed,
            graph_csr=graph_csr,
            result_data={
                "scores":     p_cpu.tolist(),
                "top_nodes":  top_nodes,
                "top_tfs":    top_tfs,
                "iterations": iteration,
            },
        )
