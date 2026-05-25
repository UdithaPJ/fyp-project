"""
algorithms/hits.py — HITS (Hyperlink-Induced Topic Search) for GRN Analysis
============================================================================

Biological Context — Why HITS suits directed GRNs
--------------------------------------------------
GRNs are *directed* networks: edges encode TF → target-gene regulatory events.
HITS exploits this directionality in a way symmetric algorithms cannot:

  Hub score   — high for nodes (TFs) that regulate many high-authority genes.
                A TF with a high hub score is a *master regulator*: it sits at
                the top of broad regulatory hierarchies.  Example: TP53 in the
                DNA-damage response network.

  Authority score — high for nodes (genes) that receive regulatory input from
                    many high-hub TFs.  A gene with a high authority score is a
                    *convergence point* of regulatory signals — often a key
                    pathway effector.  Example: CDKN1A, which is targeted by
                    multiple stress-response TFs.

The directed CSR matrix is used AS-IS (not symmetrized).  Symmetrising would
erase the distinction between "regulator" and "regulated", collapsing the
biological signal that separates TFs from their targets.

hub_authority_overlap interpretation
--------------------------------------
Nodes that appear in both top-hub and top-authority lists are *feedback
regulators*: genes that are both downstream targets of upstream TFs AND
themselves regulate other genes.  These are biologically significant —
they often represent signal amplification switches (e.g. MYC, which is
both regulated by growth-factor TFs and itself drives cell-cycle targets).

Validation workflow
-------------------
Cross-reference ``top_hubs`` against the TRRUST database (the source of the
preprocessed GRN).  TRRUST annotates known TFs; high overlap confirms that
HITS correctly ranks TFs above non-TF genes in hub score.  Residuals (hub
nodes not in TRRUST) are novel regulator candidates for wet-lab follow-up.

Algorithm
---------
1. Initialise h[i] = a[i] = 1.0 for all nodes.
2. Authority update : a  ← A^T h    (L2-normalise)
3. Hub update       : h  ← A  a     (L2-normalise)
4. Convergence      : ||Δh||₂ + ||Δa||₂ < tolerance

Parameter Guide
---------------
max_iter  (int,   default 100)   Hard iteration cap.
tolerance (float, default 1e-6)  Combined L2-norm convergence threshold.
"""

# ── GPU / CUDA-optimised implementation ──────────────────────────────────
# Source: biological_network_framework/algorithms/hits.py
# Requires: cupy-cuda11x (or matching CUDA version), pycuda
# ──────────────────────────────────────────────────────────────────────────

import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Optional CuPy
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
    "max_iter": 100,
    "tolerance": 1e-6,
}

_TOP_K: int = 15   # top hubs / authorities to return


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalise; return unchanged if norm is zero."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0.0 else v


def _top_k(scores: np.ndarray, k: int = _TOP_K) -> list[int]:
    return np.argsort(scores)[::-1][:k].tolist()


def _pack_result(
    hub: np.ndarray,
    auth: np.ndarray,
    iteration: int,
    converged: bool,
) -> dict:
    top_h = _top_k(hub)
    top_a = _top_k(auth)
    overlap = sorted(set(top_h) & set(top_a))
    return {
        "hub_scores":            hub.tolist(),
        "authority_scores":      auth.tolist(),
        "iterations":            iteration,
        "converged":             converged,
        "top_hubs":              top_h,
        "top_authorities":       top_a,
        "hub_authority_overlap": overlap,
    }


# ---------------------------------------------------------------------------
# Module-level SpMV worker — must be at module scope for ProcessPoolExecutor
# ---------------------------------------------------------------------------

def _spmv_hits_chunk(args: tuple) -> np.ndarray:
    """
    ProcessPoolExecutor worker: SpMV for a contiguous block of rows.

    Receives (data, indices, local_indptr, vec, n_cols) where local_indptr is
    re-zeroed to this chunk's start.  Returns the partial result vector.
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
    N = M.shape[0]
    n_cols = M.shape[1]
    chunk_size = max(1, (N + n_workers - 1) // n_workers)
    chunks = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        ptr_s = int(M.indptr[start])
        ptr_e = int(M.indptr[end])
        local_indptr = (M.indptr[start:end + 1] - M.indptr[start]).copy()
        chunks.append((
            M.data[ptr_s:ptr_e].copy(),
            M.indices[ptr_s:ptr_e].copy(),
            local_indptr,
            None,    # placeholder — vector injected per-iteration
            n_cols,
        ))
    return chunks


def _parallel_spmv(
    chunks: list[tuple],
    vec: np.ndarray,
    n_workers: int,
) -> np.ndarray:
    """Run a pre-chunked sparse matrix-vector product in parallel."""
    args_list = [
        (c[0], c[1], c[2], vec, c[4])
        for c in chunks
    ]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        parts = list(ex.map(_spmv_hits_chunk, args_list))
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# GPU helper
# ---------------------------------------------------------------------------

def _csr_to_gpu(m: sp.csr_matrix):
    """Upload a scipy CSR matrix to GPU as a CuPy CSR matrix."""
    coo = m.tocoo()
    return cpsp.csr_matrix(
        (
            cp.asarray(coo.data,  dtype=cp.float64),
            (cp.asarray(coo.row), cp.asarray(coo.col)),
        ),
        shape=coo.shape,
    )


# ---------------------------------------------------------------------------
# Public implementations
# ---------------------------------------------------------------------------

def hits_cpu_single(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """
    HITS — single-threaded CPU using scipy sparse SpMV on the directed graph.

    Uses the directed adjacency matrix directly (no symmetrisation).
    Two SpMV operations per iteration:
        authority update : a ← A^T · h   then L2-normalise
        hub      update  : h ← A   · a   then L2-normalise

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix — directed GRN adjacency.
    params    : dict — see module docstring.

    Returns
    -------
    dict with keys: hub_scores, authority_scores, iterations, converged,
                    top_hubs, top_authorities, hub_authority_overlap
    """
    p = _merge_params(params)
    max_iter = int(p["max_iter"])
    tol      = float(p["tolerance"])

    N   = graph_csr.shape[0]
    A   = graph_csr.astype(np.float64)
    A_T = A.T.tocsr()

    h = np.ones(N, dtype=np.float64)
    a = np.ones(N, dtype=np.float64)
    converged = False

    for iteration in range(1, max_iter + 1):
        h_old, a_old = h, a

        a_new = _l2_normalize(A_T @ h_old)
        h_new = _l2_normalize(A   @ a_new)

        if np.linalg.norm(h_new - h_old) + np.linalg.norm(a_new - a_old) < tol:
            converged = True
            a, h = a_new, h_new
            break

        a, h = a_new, h_new

    return _pack_result(h, a, iteration, converged)


def hits_cpu_multi(
    graph_csr: sp.csr_matrix,
    params: dict,
    n_workers: int = 4,
) -> dict:
    """
    HITS — multi-process CPU implementation.

    Both the A^T·h (authority) and A·a (hub) matrix-vector products are
    parallelised using ProcessPoolExecutor with row-range partitioning.
    Row chunks for both matrices are built once before the iteration loop
    (only the score vectors change each iteration).

    The L2 normalisation is applied after gathering partial results on the
    main process — it is a trivial O(N) serial step.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
    params    : dict
    n_workers : int

    Returns
    -------
    Same structure as hits_cpu_single.
    """
    p = _merge_params(params)
    max_iter = int(p["max_iter"])
    tol      = float(p["tolerance"])

    N   = graph_csr.shape[0]
    A   = graph_csr.astype(np.float64)
    A_T = A.T.tocsr()

    # Build row-chunk templates once; score vectors injected each iteration
    chunks_A_T = _build_chunks(A_T, n_workers)
    chunks_A   = _build_chunks(A,   n_workers)

    h = np.ones(N, dtype=np.float64)
    a = np.ones(N, dtype=np.float64)
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


def hits_gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """
    HITS — GPU-accelerated via CuPy sparse (cupyx.scipy.sparse).

    Both A and A^T are uploaded to GPU once before the loop.  All SpMV
    operations and L2 normalisations run on-device.  Only the final
    converged hub and authority vectors are transferred back to CPU.

    Falls back to hits_cpu_single with a warning if CuPy is unavailable.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
    params    : dict

    Returns
    -------
    Same structure as hits_cpu_single.
    """
    try:
        from src.benchmarking.benchmark import _ensure_cuda_context
        if not _ensure_cuda_context():
            raise RuntimeError("no CUDA context")
    except Exception:
        pass

    if not _CUPY_AVAILABLE:
        warnings.warn(
            "CuPy unavailable — falling back to hits_cpu_single.",
            RuntimeWarning,
            stacklevel=2,
        )
        return hits_cpu_single(graph_csr, params)

    p = _merge_params(params)
    max_iter = int(p["max_iter"])
    tol      = float(p["tolerance"])

    N = graph_csr.shape[0]
    A_gpu   = _csr_to_gpu(graph_csr.astype(np.float64))
    A_T_gpu = _csr_to_gpu(graph_csr.T.tocsr().astype(np.float64))

    h_gpu = cp.ones(N, dtype=cp.float64)
    a_gpu = cp.ones(N, dtype=cp.float64)
    converged = False

    for iteration in range(1, max_iter + 1):
        h_old, a_old = h_gpu, a_gpu

        # Authority update
        a_raw = A_T_gpu @ h_old
        a_norm = float(cp.linalg.norm(a_raw))
        a_gpu = a_raw / a_norm if a_norm > 0.0 else a_raw

        # Hub update
        h_raw = A_gpu @ a_gpu
        h_norm = float(cp.linalg.norm(h_raw))
        h_gpu = h_raw / h_norm if h_norm > 0.0 else h_raw

        dh = float(cp.linalg.norm(h_gpu - h_old))
        da = float(cp.linalg.norm(a_gpu - a_old))
        if dh + da < tol:
            converged = True
            break

    h_cpu = cp.asnumpy(h_gpu)
    a_cpu = cp.asnumpy(a_gpu)

    del A_gpu, A_T_gpu, h_gpu, a_gpu, h_old, a_old
    cp.get_default_memory_pool().free_all_blocks()

    return _pack_result(h_cpu, a_cpu, iteration, converged)


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _cpu_single(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    p = _merge_params(params)
    return {"output": hits_cpu_single(graph_csr, p), "extra_params": p}


def _cpu_multi(
    graph_csr: sp.csr_matrix,
    params: dict | None = None,
    n_workers: int = 4,
    **_,
) -> dict:
    p = _merge_params(params)
    return {
        "output":      hits_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "n_workers": n_workers},
    }


def _gpu(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    p = _merge_params(params)
    return {"output": hits_gpu(graph_csr, p), "extra_params": p}
