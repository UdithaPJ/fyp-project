"""
src/algorithms/gpu/basic/mcl.py
===============================

Markov Clustering — GPU baseline implementation.

Backends
--------
cuGraph has no native MCL routine in any released version, so the
baseline is always CuPy-driven:

    1. CuPy sparse matrix powers + element-wise inflation  (fallback path)
    2. RuntimeError                                        (CuPy missing)

This is the simplest correct MCL: it stays on the GPU via CuPy's sparse
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

from ._utils import (
    CUPY_AVAILABLE,
    build_envelope, require_any_backend, symmetrize_for,
)


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
    import cupy as cp
    import cupyx.scipy.sparse as cpsp

    n = int(graph_csr.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0, True, "Empty graph.", ""

    expansion       = int(params["expansion"])
    inflation       = float(params["inflation"])
    prune_threshold = float(params["prune_threshold"])
    max_iter        = int(params["max_iter"])
    convergence_tol = float(params["convergence_tol"])
    nt              = str(params.get("network_type", "grn")).lower()

    A_sym = symmetrize_for(graph_csr, nt)
    M = _to_column_stochastic(A_sym)

    # Upload once; iterate entirely on the GPU.
    M_gpu = cpsp.csr_matrix(
        (
            cp.asarray(M.data,    dtype=cp.float32),
            cp.asarray(M.indices, dtype=cp.int32),
            cp.asarray(M.indptr,  dtype=cp.int32),
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
        except cp.cuda.memory.OutOfMemoryError as exc:
            overflow_warning = (
                f"CuPy SpGEMM out-of-memory at iteration {it}: {exc}.  "
                f"Returning current state without further expansion."
            )
            logging.warning(overflow_warning)
            break

        # ---- Prune: threshold ----
        # CuPy CSR has no in-place threshold; rebuild the values.
        if prune_threshold > 0.0:
            data = M_new.data
            mask = data >= cp.float32(prune_threshold)
            if not bool(mask.all()):
                M_new.data = cp.where(mask, data, cp.float32(0.0))
                M_new.eliminate_zeros()

        # ---- Inflation: element-wise power then column-renormalize ----
        M_new.data = cp.power(M_new.data, cp.float32(inflation))

        # Column sums via CSC view.
        M_csc = M_new.tocsc()
        col_sums = cp.zeros((n,), dtype=cp.float32)
        # Simple sum-of-segments via dense temporary — baseline simplicity over
        # micro-optimisation.  Falls back to one reduction per column.
        # CuPy supports cp.add.reduceat with sorted segment starts.
        try:
            col_sums = cp.add.reduceat(M_csc.data, M_csc.indptr[:-1].astype(cp.int32))
            # Fix segments that are empty (would otherwise read from the next).
            empty = (cp.diff(M_csc.indptr) == 0)
            if bool(empty.any()):
                col_sums = cp.where(empty, cp.float32(0.0), col_sums)
        except Exception:                                   # noqa: BLE001
            # Conservative fallback — host loop over columns.
            indptr_host = M_csc.indptr.get()
            data_host   = M_csc.data.get()
            col_sums_h  = np.zeros(n, dtype=np.float32)
            for j in range(n):
                s, e = indptr_host[j], indptr_host[j + 1]
                if e > s:
                    col_sums_h[j] = float(data_host[s:e].sum())
            col_sums = cp.asarray(col_sums_h)

        inv = cp.where(col_sums > 0, cp.float32(1.0) / col_sums, cp.float32(0.0))
        D_inv = cpsp.diags(inv, format="csc", dtype=cp.float32)
        M_new = (M_csc @ D_inv).tocsr()

        # ---- Convergence check (Frobenius on matching pattern) ----
        same_pattern = (
            M_new.nnz == M_gpu.nnz
            and bool((M_new.indptr  == M_gpu.indptr ).all())
            and bool((M_new.indices == M_gpu.indices).all())
        )
        if same_pattern:
            diff = M_new.data - M_gpu.data
            frob = float(cp.sqrt(cp.sum(diff * diff)).get())
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
    """Markov clustering — GPU baseline (CuPy)."""
    require_any_backend("mcl")
    if not CUPY_AVAILABLE:
        raise RuntimeError(
            "mcl_gpu_baseline requires CuPy — cuGraph does not provide MCL."
        )
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

    return build_envelope(
        algorithm="mcl",
        network_type=network_type,
        execution_time=elapsed,
        graph_csr=graph_csr,
        inner=inner,
    )
