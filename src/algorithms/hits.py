"""
src/algorithms/hits.py — HITS hubs & authorities for directed GRNs
===================================================================

Biological context
------------------
HITS exploits edge direction in ways symmetric algorithms cannot:

  hub score      — high for TFs that regulate many high-authority genes
                   (master regulators sitting atop regulatory hierarchies,
                   e.g. TP53 in DNA damage response)

  authority score — high for genes targeted by many high-hub TFs
                    (convergence points like CDKN1A receiving stress-response
                    input from multiple upstream TFs)

The directed adjacency is used AS-IS — symmetrising would erase the
regulator-vs-regulated distinction that is the entire point of HITS.

``hub_authority_overlap`` (intersection of top hubs and top authorities)
identifies *feedback regulators*: genes that are both downstream targets
AND themselves regulate others.  These are signal-amplification switches
(e.g. MYC, regulated by growth-factor TFs AND driving cell-cycle targets).
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


_TOP_K = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0.0 else v


def _top_k(scores: np.ndarray, k: int = _TOP_K) -> list[int]:
    return np.argsort(scores)[::-1][:k].tolist()


def _spmv_chunk(args: tuple) -> np.ndarray:
    data, indices, indptr, vec, n_cols = args
    n_rows_chunk = len(indptr) - 1
    M_chunk = sp.csr_matrix((data, indices, indptr), shape=(n_rows_chunk, n_cols))
    return (M_chunk @ vec).astype(np.float64)


def _build_chunks(M: sp.csr_matrix, n_workers: int) -> list[tuple]:
    N = M.shape[0]
    n_cols = M.shape[1]
    chunk_size = max(1, (N + n_workers - 1) // n_workers)
    out = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        ptr_s, ptr_e = int(M.indptr[start]), int(M.indptr[end])
        local_indptr = (M.indptr[start:end + 1] - M.indptr[start]).copy()
        out.append((
            M.data[ptr_s:ptr_e].copy(),
            M.indices[ptr_s:ptr_e].copy(),
            local_indptr, n_cols,
        ))
    return out


def _pack(h: np.ndarray, a: np.ndarray, iteration: int, converged: bool) -> dict:
    top_h = _top_k(h)
    top_a = _top_k(a)
    overlap = sorted(set(top_h) & set(top_a))
    return {
        "hub_scores":            h.tolist(),
        "authority_scores":      a.tolist(),
        "top_hubs":              top_h,
        "top_authorities":       top_a,
        "hub_authority_overlap": overlap,
        "iterations":            iteration,
        "converged":             converged,
    }


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class HITS(AlgorithmBase):
    """HITS — directed-graph hub & authority scores."""

    NAME = "hits"
    PARAM_SCHEMA = {
        "max_iter":  100,
        "tolerance": 1e-6,
    }

    @staticmethod
    def cpu_single(graph_csr: sp.csr_matrix, params: dict) -> dict:
        t0 = time.perf_counter()
        cap = int(params.get("max_iter",  100))
        tol = float(params.get("tolerance", 1e-6))

        N = graph_csr.shape[0]
        A   = graph_csr.astype(np.float64)
        A_T = A.T.tocsr()

        h = np.ones(N, dtype=np.float64)
        a = np.ones(N, dtype=np.float64)
        converged = False
        iteration = 0
        for iteration in range(1, cap + 1):
            h_old, a_old = h, a
            a_new = _l2_normalize(A_T @ h_old)
            h_new = _l2_normalize(A   @ a_new)
            if np.linalg.norm(h_new - h_old) + np.linalg.norm(a_new - a_old) < tol:
                converged = True
                a, h = a_new, h_new
                break
            a, h = a_new, h_new

        elapsed = time.perf_counter() - t0
        return HITS.build_result(
            mode="cpu_single", execution_time=elapsed,
            graph_csr=graph_csr, result_data=_pack(h, a, iteration, converged),
        )

    @staticmethod
    def cpu_multi(graph_csr: sp.csr_matrix, params: dict) -> dict:
        t0 = time.perf_counter()
        cap = int(params.get("max_iter",  100))
        tol = float(params.get("tolerance", 1e-6))
        n_workers = int(params.get("n_workers", 4))

        N = graph_csr.shape[0]
        A   = graph_csr.astype(np.float64)
        A_T = A.T.tocsr()
        chunks_A_T = _build_chunks(A_T, n_workers)
        chunks_A   = _build_chunks(A,   n_workers)

        h = np.ones(N, dtype=np.float64)
        a = np.ones(N, dtype=np.float64)
        converged = False
        iteration = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            def _spmv(chunks, vec):
                args = [(c[0], c[1], c[2], vec, c[3]) for c in chunks]
                parts = list(ex.map(_spmv_chunk, args))
                return np.concatenate(parts)

            for iteration in range(1, cap + 1):
                h_old, a_old = h, a
                a_new = _l2_normalize(_spmv(chunks_A_T, h_old))
                h_new = _l2_normalize(_spmv(chunks_A,   a_new))
                if np.linalg.norm(h_new - h_old) + np.linalg.norm(a_new - a_old) < tol:
                    converged = True
                    a, h = a_new, h_new
                    break
                a, h = a_new, h_new

        elapsed = time.perf_counter() - t0
        return HITS.build_result(
            mode="cpu_multi", execution_time=elapsed,
            graph_csr=graph_csr, result_data=_pack(h, a, iteration, converged),
        )

    @staticmethod
    def gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
        if not _CUPY_AVAILABLE:
            warnings.warn(
                "CuPy unavailable — falling back to hits cpu_single.",
                RuntimeWarning, stacklevel=2,
            )
            r = HITS.cpu_single(graph_csr, params)
            r["mode"] = "gpu"
            return r

        t0 = time.perf_counter()
        cap = int(params.get("max_iter",  100))
        tol = float(params.get("tolerance", 1e-6))

        N = graph_csr.shape[0]
        coo = graph_csr.astype(np.float64).tocoo()
        coo_T = graph_csr.T.tocsr().astype(np.float64).tocoo()
        A_gpu = cpsp.csr_matrix(
            (cp.asarray(coo.data, dtype=cp.float64),
             (cp.asarray(coo.row), cp.asarray(coo.col))),
            shape=coo.shape,
        )
        A_T_gpu = cpsp.csr_matrix(
            (cp.asarray(coo_T.data, dtype=cp.float64),
             (cp.asarray(coo_T.row), cp.asarray(coo_T.col))),
            shape=coo_T.shape,
        )

        h = cp.ones(N, dtype=cp.float64)
        a = cp.ones(N, dtype=cp.float64)
        converged = False
        iteration = 0
        for iteration in range(1, cap + 1):
            h_old, a_old = h, a
            a_raw = A_T_gpu @ h_old
            an = float(cp.linalg.norm(a_raw))
            a = a_raw / an if an > 0 else a_raw
            h_raw = A_gpu @ a
            hn = float(cp.linalg.norm(h_raw))
            h = h_raw / hn if hn > 0 else h_raw
            dh = float(cp.linalg.norm(h - h_old))
            da = float(cp.linalg.norm(a - a_old))
            if dh + da < tol:
                converged = True
                break

        h_cpu = cp.asnumpy(h)
        a_cpu = cp.asnumpy(a)
        del A_gpu, A_T_gpu, h, a
        cp.get_default_memory_pool().free_all_blocks()

        elapsed = time.perf_counter() - t0
        return HITS.build_result(
            mode="gpu", execution_time=elapsed,
            graph_csr=graph_csr, result_data=_pack(h_cpu, a_cpu, iteration, converged),
        )
