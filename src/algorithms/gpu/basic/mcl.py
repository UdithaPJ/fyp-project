"""
src/algorithms/gpu/basic/mcl.py
===============================

Markov Clustering — GPU baseline implementation.

Backend
-------
CuPy sparse matrix powers + element-wise inflation.  Raises
``ImportError`` immediately on module import if CuPy is not installed.

cuGraph has no native MCL routine in any released version, so CuPy is
the correct and only GPU baseline for MCL.  This module deliberately
does NOT import from ``_utils.py`` so that it remains independently
importable on a CuPy-only installation (``_utils.py`` requires
cuGraph).

This is a simple correct MCL: it stays on the GPU via CuPy's sparse
linear algebra without any custom kernels.  It is deliberately less
sophisticated than the optimized implementation in
``src/algorithms/gpu/cuda_optimized/mcl.py`` (which uses hash SpGEMM,
bitonic top-k, multi-stream pipelining, etc.).

Result keys (mirror src/algorithms/gpu/cuda_optimized/mcl.py)
-------------------------------------------------------------
    cluster_assignments, num_clusters, iterations, converged,
    note, overflow_warning
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

from src.algorithms.common.helpers import (
    _check_memory_or_raise,
    _estimate_mcl_peak_ram_bytes,
)

# Hard-fail: CuPy is required for MCL.  cuGraph has no MCL equivalent.
try:
    import cupy as cp                         # noqa: F401
    import cupyx.scipy.sparse as cpsp         # noqa: F401
except ImportError as _e:
    raise ImportError(
        "src/algorithms/gpu/basic/mcl.py requires CuPy "
        "(cuGraph has no MCL equivalent).  Install via:\n"
        "  pip install cupy-cuda12x   # or cupy-cuda11x\n"
        f"Original error: {_e}"
    ) from _e


# ---------------------------------------------------------------------------
# Mode identifier (does not import from _utils to avoid cuGraph dependency)
# ---------------------------------------------------------------------------

_BASELINE_MODE_CUPY: str = "gpu_baseline_cupy"

# CuPy sparse SpGEMM on mid-range GPUs (4–8 GB VRAM) OOMs above this
# input size before the VRAM-fraction guard can help — the intermediate
# is materialized in a single allocation with no incremental fallback.
_GPU_BASELINE_NNZ_HARD_CAP: int = 1_000_000


def _free_vram_bytes() -> int:
    """Live free-VRAM query via CuPy runtime; returns 0 on failure.

    Zero means "unknown — skip the guard" for the shared
    ``_check_memory_or_raise`` helper.
    """
    try:
        import cupy as _cp
        free, _total = _cp.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:                                       # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------
# Inlined helpers (copied from _utils to avoid cuGraph dependency)
# ---------------------------------------------------------------------------

def _build_envelope(
    *,
    algorithm: str,
    network_type: str,
    execution_time: float,
    graph_csr: sp.csr_matrix,
    inner: dict,
) -> dict:
    """Assemble the standard 7-key result dict for the MCL CuPy baseline."""
    return {
        "algorithm":      algorithm,
        "mode":           _BASELINE_MODE_CUPY,
        "network_type":   network_type,
        "execution_time": float(execution_time),
        "num_nodes":      int(graph_csr.shape[0]),
        "num_edges":      int(graph_csr.nnz),
        "result":         inner,
    }


def _symmetrize_for(graph_csr: sp.csr_matrix, network_type: str) -> sp.csr_matrix:
    """Network-type-aware undirected conversion (GRN/miRNA → binarise A+Aᵀ)."""
    nt = str(network_type).lower()
    if nt == "ppi":
        return graph_csr.astype(np.float32).tocsr()
    A = (graph_csr + graph_csr.T).astype(np.float32)
    if A.nnz > 0:
        A.data = np.ones_like(A.data, dtype=np.float32)
    return A.tocsr()


_DEFAULT_PARAMS: dict = {
    "expansion":        2,
    "inflation":        2.0,
    "prune_threshold":  1e-3,
    "top_k_per_column": 50,
    "max_iter":         100,
    "convergence_tol":  1e-4,
    "network_type":     "grn",
}


# ---------------------------------------------------------------------------
# Preprocessing — column-stochastic with self-loops
# ---------------------------------------------------------------------------

def _to_column_stochastic(csr: sp.csr_matrix) -> sp.csr_matrix:
    """Add self-loops and column-normalize so each column sums to 1."""
    n = csr.shape[0]
    eye = sp.eye(n, format="csr", dtype=np.float32)
    M = (csr + eye).tocsr().astype(np.float32)
    M.sum_duplicates()

    col_sums = np.asarray(M.sum(axis=0)).flatten()
    nonzero  = col_sums > 0
    inv      = np.zeros(n, dtype=np.float32)
    inv[nonzero] = 1.0 / col_sums[nonzero]
    D = sp.diags(inv, format="csr", dtype=np.float32)
    return (M @ D).tocsr().astype(np.float32)


# ---------------------------------------------------------------------------
# Cluster extraction (attractor method, CPU after convergence)
# ---------------------------------------------------------------------------

def _extract_clusters(M: sp.csr_matrix) -> np.ndarray:
    """Attractor-based cluster extraction; fallback to weakly-CCs."""
    n = M.shape[0]
    M_csr = M.tocsr()
    diag = np.asarray(M_csr.diagonal()).flatten()
    attractors = np.where(diag > 0)[0]

    if attractors.size == 0:
        _, labels = csgraph.connected_components(
            M_csr, directed=False, connection="weak",
        )
        return labels.astype(np.int64)

    sub = np.asarray(M_csr[attractors, :].todense())
    if sub.size == 0:
        return np.zeros(n, dtype=np.int64)
    best_local = np.argmax(sub, axis=0).flatten()
    return attractors[best_local].astype(np.int64)


# ---------------------------------------------------------------------------
# CuPy MCL — simple iterative SpGEMM + inflation
# ---------------------------------------------------------------------------

def _mcl_cupy(
    graph_csr: sp.csr_matrix, params: dict,
) -> tuple[np.ndarray, int, bool, str, str]:
    """Run a simple GPU MCL via CuPy sparse matrix powers.

    Returns ``(cluster_labels, iterations, converged, note, overflow_warning)``.
    Overflow is reported but never re-raised: the baseline favours
    correctness of the schema over raw speed.
    """
    import cupy as _cp
    import cupyx.scipy.sparse as _cpsp

    n = int(graph_csr.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0, True, "Empty graph.", ""

    expansion       = int(params["expansion"])
    inflation       = float(params["inflation"])
    prune_threshold = float(params["prune_threshold"])
    max_iter        = int(params["max_iter"])
    convergence_tol = float(params["convergence_tol"])
    nt              = str(params.get("network_type", "grn")).lower()

    # ---- Layer 1: hard nnz cap for the CuPy sparse baseline ----
    if int(graph_csr.nnz) > _GPU_BASELINE_NNZ_HARD_CAP:
        raise MemoryError(
            f"MCL gpu_baseline: refusing to run — input has "
            f"{graph_csr.nnz} edges, above the "
            f"{_GPU_BASELINE_NNZ_HARD_CAP} hard cap for the CuPy baseline. "
            f"Use mode=gpu (cuda_optimized) which fuses threshold pruning "
            f"into SpGEMM and scales to larger graphs."
        )

    # ---- Layer 2: VRAM-vs-estimate check with post-symmetrize sizing ----
    _check_memory_or_raise(
        _estimate_mcl_peak_ram_bytes(
            graph_csr, expansion=expansion, dtype_bytes=4, index_bytes=4,
        ),
        _free_vram_bytes(),
        backend="gpu_baseline",
        extra_hint=(
            "Use mode=gpu (cuda_optimized) — it fuses threshold pruning "
            "into SpGEMM and scales to larger graphs."
        ),
    )

    A_sym = _symmetrize_for(graph_csr, nt)
    M = _to_column_stochastic(A_sym)

    # Upload once; iterate entirely on the GPU.
    M_gpu = _cpsp.csr_matrix(
        (
            _cp.asarray(M.data,    dtype=_cp.float32),
            _cp.asarray(M.indices, dtype=_cp.int32),
            _cp.asarray(M.indptr,  dtype=_cp.int32),
        ),
        shape=M.shape,
    )

    overflow_warning = ""
    converged = False
    iterations = 0
    for it in range(max_iter):
        iterations = it + 1

        # ---- Expansion: M^expansion via repeated SpGEMM ----
        M_new = M_gpu
        try:
            for _ in range(expansion - 1):
                M_new = M_new @ M_gpu
        except _cp.cuda.memory.OutOfMemoryError as exc:
            overflow_warning = (
                f"CuPy SpGEMM out-of-memory at iteration {it}: {exc}.  "
                f"Returning current state without further expansion."
            )
            logging.warning(overflow_warning)
            break

        # ---- Prune: threshold ----
        if prune_threshold > 0.0:
            data = M_new.data
            mask = data >= _cp.float32(prune_threshold)
            if not bool(mask.all()):
                M_new.data = _cp.where(mask, data, _cp.float32(0.0))
                M_new.eliminate_zeros()

        # ---- Inflation: element-wise power then column-renormalize ----
        M_new.data = _cp.power(M_new.data, _cp.float32(inflation))

        M_csc = M_new.tocsc()
        col_sums = _cp.zeros((n,), dtype=_cp.float32)
        try:
            col_sums = _cp.add.reduceat(M_csc.data, M_csc.indptr[:-1].astype(_cp.int32))
            empty = (_cp.diff(M_csc.indptr) == 0)
            if bool(empty.any()):
                col_sums = _cp.where(empty, _cp.float32(0.0), col_sums)
        except Exception:                                    # noqa: BLE001
            indptr_host = M_csc.indptr.get()
            data_host   = M_csc.data.get()
            col_sums_h  = np.zeros(n, dtype=np.float32)
            for j in range(n):
                s, e = indptr_host[j], indptr_host[j + 1]
                if e > s:
                    col_sums_h[j] = float(data_host[s:e].sum())
            col_sums = _cp.asarray(col_sums_h)

        inv = _cp.where(col_sums > 0, _cp.float32(1.0) / col_sums, _cp.float32(0.0))
        D_inv = _cpsp.diags(inv, format="csc", dtype=_cp.float32)
        M_new = (M_csc @ D_inv).tocsr()

        # ---- Convergence check (Frobenius on matching pattern) ----
        same_pattern = (
            M_new.nnz == M_gpu.nnz
            and bool((M_new.indptr  == M_gpu.indptr ).all())
            and bool((M_new.indices == M_gpu.indices).all())
        )
        if same_pattern:
            diff = M_new.data - M_gpu.data
            frob = float(_cp.sqrt(_cp.sum(diff * diff)).get())
            M_gpu = M_new
            if frob < convergence_tol:
                converged = True
                break
        else:
            M_gpu = M_new

    # Pull result back to CPU for cluster extraction.
    M_host = sp.csr_matrix(
        (M_gpu.data.get(), M_gpu.indices.get(), M_gpu.indptr.get()),
        shape=M_gpu.shape, dtype=np.float32,
    )
    labels = _extract_clusters(M_host)

    note = (
        f"CuPy baseline MCL (expansion={expansion}, inflation={inflation}, "
        f"prune={prune_threshold:g}).  Network={nt}."
    )
    return labels, iterations, converged, note, overflow_warning


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def mcl_gpu_baseline(
    graph_csr: sp.csr_matrix, params: dict | None = None,
) -> dict:
    """Markov clustering — GPU baseline (CuPy; no cuGraph equivalent).

    Returns
    -------
    dict
        7-key standard result envelope; ``result["mode"]`` is
        ``"gpu_baseline_cupy"``.
    """
    p = {**_DEFAULT_PARAMS, **(params or {})}
    network_type = str(p.get("network_type", "grn")).lower()

    t0 = time.perf_counter()
    labels, iters, converged, note, overflow_warning = _mcl_cupy(graph_csr, p)
    elapsed = time.perf_counter() - t0

    # Compact 0..K-1 renumbering.
    _, compact = np.unique(labels, return_inverse=True)
    labels = compact.astype(np.int64)
    num_clusters = int(labels.max() + 1) if labels.size > 0 else 0

    inner = {
        "cluster_assignments": labels.tolist(),
        "num_clusters":        num_clusters,
        "iterations":          iters,
        "converged":           converged,
        "note":                f"{note}  Backend=cupy.",
        "overflow_warning":    overflow_warning,
        "backend":             "cupy",
    }

    return _build_envelope(
        algorithm="mcl",
        network_type=network_type,
        execution_time=elapsed,
        graph_csr=graph_csr,
        inner=inner,
    )
