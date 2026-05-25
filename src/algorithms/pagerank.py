"""
src/algorithms/pagerank.py — PageRank for hub-gene / TF identification
=======================================================================

Biological context
------------------
In a directed Gene Regulatory Network (GRN), PageRank models *regulatory
influence* as forward flow along TF → gene edges.  Damping ``d = 0.85``
means each regulatory hop has an 85% chance of propagating, matching
empirically observed cascade depths of 4–6 hops in real GRNs.

Two distinct node classes are ranked separately:

  top_regulators — nodes with out-degree > 0 (TFs, intermediate regulators)
  top_targets    — nodes with out-degree == 0 (pure target genes)

Dangling node handling (GRN-specific)
--------------------------------------
Dangling mass (probability accumulated on pure-target / out-degree-zero
nodes) is redistributed ONLY to nodes with at least one outgoing edge.
This preserves the biological asymmetry between regulators and targets;
the standard "uniform over all N" redistribution artificially inflates
non-TF gene scores.
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


_TOP_REG = 15
_TOP_TGT = 15


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_transition_matrix(
    graph_csr: sp.csr_matrix,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Column-stochastic M and a dangling-node mask."""
    out_degrees = np.asarray(graph_csr.sum(axis=1)).flatten()
    dangling_mask = out_degrees == 0
    safe_deg = np.where(dangling_mask, 1.0, out_degrees)
    D_inv = sp.diags(1.0 / safe_deg, format="csr")
    M = (D_inv @ graph_csr).T.tocsr().astype(np.float64)
    return M, dangling_mask


def _split_top_nodes(
    scores: np.ndarray,
    out_degrees: np.ndarray,
    n_reg: int = _TOP_REG,
    n_tgt: int = _TOP_TGT,
) -> tuple[list[int], list[int]]:
    reg_idx = np.where(out_degrees > 0)[0]
    tgt_idx = np.where(out_degrees == 0)[0]
    top_reg = (
        reg_idx[np.argsort(scores[reg_idx])[::-1][:n_reg]].tolist()
        if len(reg_idx) else []
    )
    top_tgt = (
        tgt_idx[np.argsort(scores[tgt_idx])[::-1][:n_tgt]].tolist()
        if len(tgt_idx) else []
    )
    return top_reg, top_tgt


# ---------------------------------------------------------------------------
# Module-level worker (for ProcessPoolExecutor — must be picklable)
# ---------------------------------------------------------------------------

def _spmv_row_chunk(args: tuple) -> np.ndarray:
    data, indices, indptr, pr, n_cols = args
    n_rows_chunk = len(indptr) - 1
    M_chunk = sp.csr_matrix(
        (data, indices, indptr), shape=(n_rows_chunk, n_cols)
    )
    return (M_chunk @ pr).astype(np.float64)


# ---------------------------------------------------------------------------
# Core power iteration (shared between cpu_single and cpu_multi)
# ---------------------------------------------------------------------------

def _power_iteration(
    graph_csr: sp.csr_matrix,
    damping: float,
    max_iter: int,
    tolerance: float,
    spmv,                # callable: (PR_old) -> np.ndarray
) -> tuple[np.ndarray, int, bool, np.ndarray]:
    """Run the power iteration loop; return (PR, iterations, converged, out_deg)."""
    N = graph_csr.shape[0]
    M, dangling_mask = _build_transition_matrix(graph_csr)
    teleport_per_node = (1.0 - damping) / N

    out_degrees = np.asarray(graph_csr.sum(axis=1)).flatten()
    active_mask = out_degrees > 0
    n_active = int(active_mask.sum())
    if n_active == 0:
        warnings.warn(
            "PageRank: no outgoing-edge nodes found, using uniform dangling redistribution",
            UserWarning, stacklevel=2,
        )

    PR = np.full(N, 1.0 / N, dtype=np.float64)
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        PR_old = PR
        dangling_mass = damping * float(PR_old[dangling_mask].sum())
        contrib = np.zeros(N, dtype=np.float64)
        if n_active > 0:
            contrib[active_mask] = dangling_mass / n_active
        else:
            contrib[:] = dangling_mass / N

        spmv_result = spmv(M, PR_old)
        PR = damping * spmv_result + contrib + teleport_per_node

        if np.abs(PR - PR_old).sum() < tolerance:
            converged = True
            break

    return PR, iteration, converged, out_degrees


# ---------------------------------------------------------------------------
# Public algorithm class
# ---------------------------------------------------------------------------

class PageRank(AlgorithmBase):
    """PageRank with TF-only dangling redistribution for directed GRNs."""

    NAME = "pagerank"
    PARAM_SCHEMA = {
        "damping":   0.85,
        "max_iter":  100,
        "tolerance": 1e-6,
    }

    @staticmethod
    def cpu_single(graph_csr: sp.csr_matrix, params: dict) -> dict:
        t0 = time.perf_counter()
        d   = float(params.get("damping",   0.85))
        cap = int(params.get("max_iter",  100))
        tol = float(params.get("tolerance", 1e-6))

        def _spmv(M, PR_old):
            return M @ PR_old

        PR, iters, converged, out_deg = _power_iteration(
            graph_csr, d, cap, tol, _spmv
        )
        elapsed = time.perf_counter() - t0
        top_reg, top_tgt = _split_top_nodes(PR, out_deg)

        return PageRank.build_result(
            mode="cpu_single",
            execution_time=elapsed,
            graph_csr=graph_csr,
            result_data={
                "scores":         PR.tolist(),
                "top_regulators": top_reg,
                "top_targets":    top_tgt,
                "iterations":     iters,
                "converged":      converged,
            },
        )

    @staticmethod
    def cpu_multi(graph_csr: sp.csr_matrix, params: dict) -> dict:
        t0 = time.perf_counter()
        d   = float(params.get("damping",   0.85))
        cap = int(params.get("max_iter",  100))
        tol = float(params.get("tolerance", 1e-6))
        n_workers = int(params.get("n_workers", 4))

        # Pre-build chunk specs once
        N = graph_csr.shape[0]
        M_full, _ = _build_transition_matrix(graph_csr)
        chunk_size = max(1, (N + n_workers - 1) // n_workers)
        chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            ptr_s = int(M_full.indptr[start])
            ptr_e = int(M_full.indptr[end])
            local_indptr = (M_full.indptr[start:end + 1] - M_full.indptr[start]).copy()
            chunks.append((
                M_full.data[ptr_s:ptr_e].copy(),
                M_full.indices[ptr_s:ptr_e].copy(),
                local_indptr,
                N,
            ))

        with ProcessPoolExecutor(max_workers=n_workers) as ex:

            def _spmv(_M, PR_old):
                args = [(c[0], c[1], c[2], PR_old, c[3]) for c in chunks]
                parts = list(ex.map(_spmv_row_chunk, args))
                return np.concatenate(parts)

            PR, iters, converged, out_deg = _power_iteration(
                graph_csr, d, cap, tol, _spmv
            )

        elapsed = time.perf_counter() - t0
        top_reg, top_tgt = _split_top_nodes(PR, out_deg)

        return PageRank.build_result(
            mode="cpu_multi",
            execution_time=elapsed,
            graph_csr=graph_csr,
            result_data={
                "scores":         PR.tolist(),
                "top_regulators": top_reg,
                "top_targets":    top_tgt,
                "iterations":     iters,
                "converged":      converged,
            },
        )

    @staticmethod
    def gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
        if not _CUPY_AVAILABLE:
            warnings.warn(
                "CuPy unavailable — falling back to pagerank cpu_single.",
                RuntimeWarning, stacklevel=2,
            )
            r = PageRank.cpu_single(graph_csr, params)
            r["mode"] = "gpu"
            return r

        t0 = time.perf_counter()
        d   = float(params.get("damping",   0.85))
        cap = int(params.get("max_iter",  100))
        tol = float(params.get("tolerance", 1e-6))

        N = graph_csr.shape[0]
        M_cpu, dangling_mask = _build_transition_matrix(graph_csr)
        teleport = (1.0 - d) / N
        out_deg = np.asarray(graph_csr.sum(axis=1)).flatten()
        active_idx = np.where(out_deg > 0)[0]
        n_active = int(len(active_idx))

        coo = M_cpu.tocoo()
        M_gpu = cpsp.csr_matrix(
            (cp.asarray(coo.data, dtype=cp.float64),
             (cp.asarray(coo.row), cp.asarray(coo.col))),
            shape=coo.shape,
        )
        dang_idx_gpu   = cp.asarray(np.where(dangling_mask)[0]) if dangling_mask.any() else None
        active_idx_gpu = cp.asarray(active_idx) if n_active > 0 else None

        PR = cp.full(N, 1.0 / N, dtype=cp.float64)
        converged = False
        iteration = 0
        for iteration in range(1, cap + 1):
            PR_old = PR
            PR_new = cp.full(N, teleport, dtype=cp.float64)
            PR_new += d * (M_gpu @ PR_old)
            if dang_idx_gpu is not None:
                dangling_mass = float(PR_old[dang_idx_gpu].sum())
                if active_idx_gpu is not None:
                    PR_new[active_idx_gpu] += d * dangling_mass / n_active
                else:
                    PR_new += d * dangling_mass / N
            if float(cp.abs(PR_new - PR_old).sum()) < tol:
                converged = True
                PR = PR_new
                break
            PR = PR_new

        PR_cpu = cp.asnumpy(PR)
        del M_gpu, PR_old, PR_new, PR
        if dang_idx_gpu is not None: del dang_idx_gpu
        if active_idx_gpu is not None: del active_idx_gpu
        cp.get_default_memory_pool().free_all_blocks()

        elapsed = time.perf_counter() - t0
        top_reg, top_tgt = _split_top_nodes(PR_cpu, out_deg)

        return PageRank.build_result(
            mode="gpu",
            execution_time=elapsed,
            graph_csr=graph_csr,
            result_data={
                "scores":         PR_cpu.tolist(),
                "top_regulators": top_reg,
                "top_targets":    top_tgt,
                "iterations":     iteration,
                "converged":      converged,
            },
        )
