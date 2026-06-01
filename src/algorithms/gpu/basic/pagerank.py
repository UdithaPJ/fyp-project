"""
src/algorithms/gpu/basic/pagerank.py
====================================

PageRank — GPU baseline implementation.

Backends (in priority order)
----------------------------
    1. cuGraph ``cugraph.pagerank``      (preferred, native RAPIDS)
    2. CuPy sparse power iteration       (fallback, no custom kernels)
    3. RuntimeError                      (neither backend available)

This is a *baseline* implementation: simple, readable, no custom
kernels, no graph reordering, no hybrid push/pull dispatch.  Used for
benchmarking against ``src/algorithms/gpu/cuda_optimized/pagerank.py``.

Result keys match the optimized implementation exactly so the two are
schema-comparable:

    grn   : scores, top_regulators, top_targets, iterations, converged
    ppi   : scores, top_nodes,                   iterations, converged
    mirna : scores, top_mirnas, top_target_genes, iterations, converged
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

from ._utils import (
    CUGRAPH_AVAILABLE, CUPY_AVAILABLE,
    TOP_NODES_PPI, TOP_REG, TOP_TGT,
    build_envelope, cugraph_build_graph, cugraph_extract_column,
    cugraph_function, out_in_degrees, require_any_backend,
    top_k_among, top_k_global,
    to_cupy_array, to_cupy_csr, cupy_get,
)


_DEFAULT_PARAMS: dict = {
    "damping":      0.85,
    "max_iter":     100,
    "tolerance":    1e-6,
    "network_type": "grn",
}


# ---------------------------------------------------------------------------
# Result packing — exactly mirrors src/algorithms/gpu/cuda_optimized/pagerank.py
# ---------------------------------------------------------------------------

def _pack_result(
    scores: np.ndarray,
    out_degrees: np.ndarray,
    iterations: int,
    converged: bool,
    network_type: str,
) -> dict:
    """Build the inner result dict in the network-type-specific shape."""
    nt = str(network_type).lower()
    scores_list = scores.astype(float).tolist()

    if nt == "ppi":
        return {
            "scores":     scores_list,
            "top_nodes":  top_k_global(scores, TOP_NODES_PPI),
            "iterations": iterations,
            "converged":  converged,
        }

    regulators = np.where(out_degrees > 0.0)[0]
    targets    = np.where(out_degrees == 0.0)[0]

    if nt == "mirna":
        return {
            "scores":           scores_list,
            "top_mirnas":       top_k_among(scores, regulators, TOP_REG),
            "top_target_genes": top_k_among(scores, targets,    TOP_TGT),
            "iterations":       iterations,
            "converged":        converged,
        }

    return {
        "scores":         scores_list,
        "top_regulators": top_k_among(scores, regulators, TOP_REG),
        "top_targets":    top_k_among(scores, targets,    TOP_TGT),
        "iterations":     iterations,
        "converged":      converged,
    }


# ---------------------------------------------------------------------------
# cuGraph path
# ---------------------------------------------------------------------------

def _pagerank_cugraph(
    graph_csr: sp.csr_matrix, params: dict,
) -> tuple[np.ndarray, int, bool]:
    """Run cuGraph PageRank.  Returns ``(scores, iterations, converged)``."""
    pagerank = cugraph_function("pagerank")
    if pagerank is None:
        raise RuntimeError("cugraph.pagerank is not available.")

    nt = str(params.get("network_type", "grn")).lower()
    directed = nt != "ppi"

    G = cugraph_build_graph(graph_csr, directed=directed, weighted=True)

    # Probe the installed signature; only forward kwargs that this version
    # accepts so we don't TypeError across RAPIDS releases.
    import inspect
    try:
        sig = inspect.signature(pagerank)
        accepted = set(sig.parameters.keys())
    except (TypeError, ValueError):
        accepted = set()

    kwargs: dict[str, Any] = {}
    if "alpha"          in accepted: kwargs["alpha"]          = float(params["damping"])
    if "damping_factor" in accepted: kwargs["damping_factor"] = float(params["damping"])
    if "max_iter"       in accepted: kwargs["max_iter"]       = int(params["max_iter"])
    if "tol"            in accepted: kwargs["tol"]            = float(params["tolerance"])
    if "tolerance"      in accepted: kwargs["tolerance"]      = float(params["tolerance"])

    df = pagerank(G, **kwargs)

    # Extract score column with flexible naming.
    score_col = cugraph_extract_column(df, ["pagerank", "score", "scores"])
    vertex_col = cugraph_extract_column(df, ["vertex", "node", "id"])

    n = int(graph_csr.shape[0])
    scores = np.zeros(n, dtype=np.float32)
    # cuGraph may return vertices in arbitrary order.
    np.put(scores, vertex_col.astype(np.int64), score_col.astype(np.float32))

    # cuGraph does not surface (iterations, converged) — report a best-effort
    # value: assume converged within max_iter if it returned without error.
    return scores, int(params["max_iter"]), True


# ---------------------------------------------------------------------------
# CuPy fallback — simple sparse power iteration, no custom kernels
# ---------------------------------------------------------------------------

def _pagerank_cupy(
    graph_csr: sp.csr_matrix, params: dict,
) -> tuple[np.ndarray, int, bool]:
    """Sparse power-iteration PageRank in CuPy.  Network-type-aware dangling."""
    import cupy as cp

    n = int(graph_csr.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float32), 0, True

    damping   = float(params["damping"])
    max_iter  = int(params["max_iter"])
    tolerance = float(params["tolerance"])
    nt        = str(params.get("network_type", "grn")).lower()

    # Build column-stochastic transition matrix M where M[v, u] = 1 / out_deg[u]
    # for each edge (u -> v).  We use the transpose so SpMV computes Sum_u of
    # contributions to v in one shot.
    out_deg, _ = out_in_degrees(graph_csr)

    # Eligible dangling-redistribution mask (network-type aware).
    if nt == "ppi":
        eligible = np.ones(n, dtype=np.bool_)
    else:  # grn or mirna — redistribute only to nodes with out_degree > 0
        eligible = out_deg > 0.0
        if not eligible.any():
            logging.warning(
                "pagerank_gpu_baseline (%s): no eligible regulators — "
                "falling back to uniform redistribution.", nt,
            )
            eligible = np.ones(n, dtype=np.bool_)

    # Build M = D^{-1} A on CPU once (simple — baseline), then upload.
    inv_out = np.zeros(n, dtype=np.float32)
    nonzero = out_deg > 0.0
    inv_out[nonzero] = 1.0 / out_deg[nonzero]
    D_inv = sp.diags(inv_out, format="csr", dtype=np.float32)
    M_T = (D_inv @ graph_csr.astype(np.float32)).T.tocsr()

    M_T_gpu     = to_cupy_csr(M_T)
    eligible_gp = cp.asarray(eligible.astype(np.bool_))
    n_eligible  = int(eligible.sum())

    p = cp.full((n,), 1.0 / n, dtype=cp.float32)
    teleport = cp.float32((1.0 - damping) / n)
    dangling_mask = cp.asarray((out_deg == 0.0).astype(np.bool_))

    converged = False
    iterations = 0
    for it in range(max_iter):
        iterations = it + 1
        # Dangling mass: sum over nodes with out_deg == 0.
        dangling_sum = cp.float32(damping) * cp.sum(p[dangling_mask])
        redist = dangling_sum / cp.float32(max(n_eligible, 1))

        # Vector form: p_new = teleport + (damping * M^T @ p)
        # plus dangling redistribution on eligible nodes.
        p_new = cp.full((n,), teleport, dtype=cp.float32)
        p_new = p_new + cp.float32(damping) * M_T_gpu.dot(p)
        p_new = cp.where(eligible_gp, p_new + redist, p_new)

        # Renormalize to guard against numerical drift.
        s = float(cp.sum(p_new).get())
        if s > 0:
            p_new = p_new / cp.float32(s)

        delta = float(cp.sum(cp.abs(p_new - p)).get())
        p = p_new
        if delta < tolerance:
            converged = True
            break

    scores = cupy_get(p).astype(np.float32)
    return scores, iterations, converged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def pagerank_gpu_baseline(
    graph_csr: sp.csr_matrix, params: dict | None = None,
) -> dict:
    """PageRank — GPU baseline (cuGraph preferred, CuPy fallback).

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
        Adjacency matrix (directed for GRN/miRNA, undirected for PPI).
    params : dict, optional
        ``damping``, ``max_iter``, ``tolerance``, ``network_type``.

    Returns
    -------
    dict
        7-key standard result envelope; inner keys mirror the optimized
        implementation per network type.
    """
    require_any_backend("pagerank")
    p = {**_DEFAULT_PARAMS, **(params or {})}
    network_type = str(p.get("network_type", "grn")).lower()

    out_deg, _ = out_in_degrees(graph_csr)

    t0 = time.perf_counter()
    scores: np.ndarray
    iterations: int
    converged: bool
    backend: str
    if CUGRAPH_AVAILABLE and cugraph_function("pagerank") is not None:
        try:
            scores, iterations, converged = _pagerank_cugraph(graph_csr, p)
            backend = "cugraph"
        except Exception as exc:                            # noqa: BLE001
            logging.warning(
                "pagerank_gpu_baseline: cuGraph path failed (%s) — "
                "falling back to CuPy.", exc,
            )
            if not CUPY_AVAILABLE:
                raise
            scores, iterations, converged = _pagerank_cupy(graph_csr, p)
            backend = "cupy"
    elif CUPY_AVAILABLE:
        scores, iterations, converged = _pagerank_cupy(graph_csr, p)
        backend = "cupy"
    else:
        raise RuntimeError("Unreachable — require_any_backend should have raised.")
    elapsed = time.perf_counter() - t0

    inner = _pack_result(
        scores=scores,
        out_degrees=out_deg,
        iterations=iterations,
        converged=converged,
        network_type=network_type,
    )
    inner["backend"] = backend  # additive metadata; doesn't break optimized parity

    return build_envelope(
        algorithm="pagerank",
        network_type=network_type,
        execution_time=elapsed,
        graph_csr=graph_csr,
        inner=inner,
    )
