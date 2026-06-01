"""
src/algorithms/gpu/basic/rwr.py
===============================

Random Walk with Restart — GPU baseline implementation.

Backends
--------
    1. cuGraph ``cugraph.personalized_pagerank``  (preferred)
    2. CuPy power iteration with restart vector   (fallback)
    3. RuntimeError                               (neither installed)

The "restart" formulation of RWR is mathematically equivalent to
personalized PageRank with the seed nodes as the personalization
distribution and ``alpha = 1 - restart_prob``.  We use that equivalence
in the cuGraph path.

Result keys (mirror src/algorithms/gpu/cuda_optimized/rwr.py)
-------------------------------------------------------------
    scores, top_nodes, top_seeds, iterations, converged, note
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

from ._utils import (
    CUGRAPH_AVAILABLE, CUDF_AVAILABLE, CUPY_AVAILABLE,
    TOP_RWR_NODES, TOP_RWR_SEEDS,
    build_envelope, cugraph_build_graph, cugraph_extract_column,
    cugraph_function, out_in_degrees, require_any_backend,
    symmetrize_for, top_k_among, top_k_global,
    to_cupy_csr, cupy_get,
)


_DEFAULT_PARAMS: dict = {
    "restart_prob": 0.3,
    "max_iter":     100,
    "tolerance":    1e-6,
    "seed_nodes":   [],
    "network_type": "grn",
}


# ---------------------------------------------------------------------------
# Seed normalisation (accepts ``[1,2,3]`` or ``[[1,2],[3,4]]``)
# ---------------------------------------------------------------------------

def _flatten_seeds(seed_nodes) -> list[int]:
    """Return a flat list of seed indices; supports single-set or multi-set."""
    if seed_nodes is None:
        return []
    if isinstance(seed_nodes, (list, tuple, np.ndarray)) and len(seed_nodes) > 0:
        first = seed_nodes[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            flat: list[int] = []
            for sub in seed_nodes:
                flat.extend(int(x) for x in sub)
            return flat
    return [int(x) for x in seed_nodes]


# ---------------------------------------------------------------------------
# cuGraph path — personalized PageRank
# ---------------------------------------------------------------------------

def _rwr_cugraph(
    graph_csr: sp.csr_matrix, params: dict, seeds: list[int],
) -> tuple[np.ndarray, int, bool, str]:
    ppr = cugraph_function("personalized_pagerank")
    if ppr is None:
        # Some RAPIDS releases route through a single ``pagerank`` with a
        # ``personalization`` argument instead.
        pr = cugraph_function("pagerank")
        if pr is None:
            raise RuntimeError(
                "Neither cugraph.personalized_pagerank nor cugraph.pagerank "
                "is available."
            )
        ppr = pr

    nt = str(params.get("network_type", "grn")).lower()
    directed = nt != "ppi"
    csr_in = symmetrize_for(graph_csr, nt) if nt == "ppi" else graph_csr
    G = cugraph_build_graph(csr_in, directed=directed, weighted=True)

    n = int(graph_csr.shape[0])
    restart_prob = float(params["restart_prob"])
    alpha = 1.0 - restart_prob   # standard equivalence

    if not CUDF_AVAILABLE:
        raise RuntimeError(
            "cuDF is required to pass the personalization vector to "
            "cuGraph personalized PageRank."
        )
    import cudf

    if not seeds:
        # No seeds => uniform; reduce to plain PageRank.
        personalization = None
    else:
        weight = 1.0 / float(len(seeds))
        personalization = cudf.DataFrame({
            "vertex": np.asarray(seeds, dtype=np.int32),
            "values": np.full(len(seeds), weight, dtype=np.float32),
        })

    import inspect
    try:
        accepted = set(inspect.signature(ppr).parameters.keys())
    except (TypeError, ValueError):
        accepted = set()

    kwargs: dict[str, Any] = {}
    if "alpha"          in accepted: kwargs["alpha"]          = alpha
    if "damping_factor" in accepted: kwargs["damping_factor"] = alpha
    if "max_iter"       in accepted: kwargs["max_iter"]       = int(params["max_iter"])
    if "tol"            in accepted: kwargs["tol"]            = float(params["tolerance"])
    if "tolerance"      in accepted: kwargs["tolerance"]      = float(params["tolerance"])
    if personalization is not None and "personalization" in accepted:
        kwargs["personalization"] = personalization

    df = ppr(G, **kwargs)
    vertex = cugraph_extract_column(df, ["vertex", "node", "id"]).astype(np.int64)
    score  = cugraph_extract_column(df, ["pagerank", "score", "scores"]).astype(np.float32)

    scores = np.zeros(n, dtype=np.float32)
    np.put(scores, vertex, score)

    note = (
        f"cuGraph personalized_pagerank used (alpha={alpha:.3f}, "
        f"seeds={len(seeds)})."
    )
    return scores, int(params["max_iter"]), True, note


# ---------------------------------------------------------------------------
# CuPy fallback — power iteration with restart
# ---------------------------------------------------------------------------

def _rwr_cupy(
    graph_csr: sp.csr_matrix, params: dict, seeds: list[int],
) -> tuple[np.ndarray, int, bool, str]:
    import cupy as cp

    n = int(graph_csr.shape[0])
    if n == 0:
        return np.zeros(0, np.float32), 0, True, "Empty graph."

    restart_prob = float(params["restart_prob"])
    max_iter     = int(params["max_iter"])
    tolerance    = float(params["tolerance"])
    nt           = str(params.get("network_type", "grn")).lower()

    A = symmetrize_for(graph_csr, nt) if nt == "ppi" else graph_csr.astype(np.float32)
    # Column-normalize so M^T @ p computes one walk step.
    out_deg, _ = out_in_degrees(A)
    inv_out = np.zeros(n, dtype=np.float32)
    nz = out_deg > 0.0
    inv_out[nz] = 1.0 / out_deg[nz]
    D_inv = sp.diags(inv_out, format="csr", dtype=np.float32)
    M_T = (D_inv @ A).T.tocsr().astype(np.float32)
    M_T_gpu = to_cupy_csr(M_T)

    if seeds:
        p0_np = np.zeros(n, dtype=np.float32)
        w = 1.0 / float(len(seeds))
        for s in seeds:
            if 0 <= int(s) < n:
                p0_np[int(s)] += w
    else:
        p0_np = np.full(n, 1.0 / n, dtype=np.float32)
    p0 = cp.asarray(p0_np)
    p  = p0.copy()

    converged = False
    iterations = 0
    for it in range(max_iter):
        iterations = it + 1
        p_new = cp.float32(1.0 - restart_prob) * M_T_gpu.dot(p) + cp.float32(restart_prob) * p0
        # Renormalize for numerical safety.
        s = float(cp.sum(p_new).get())
        if s > 0:
            p_new = p_new / cp.float32(s)
        delta = float(cp.sum(cp.abs(p_new - p)).get())
        p = p_new
        if delta < tolerance:
            converged = True
            break

    note = (
        f"CuPy power iteration with restart (restart_prob={restart_prob}, "
        f"seeds={len(seeds)})."
    )
    return cupy_get(p).astype(np.float32), iterations, converged, note


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def rwr_gpu_baseline(
    graph_csr: sp.csr_matrix, params: dict | None = None,
) -> dict:
    """Random Walk with Restart — GPU baseline."""
    require_any_backend("rwr")
    p = {**_DEFAULT_PARAMS, **(params or {})}
    network_type = str(p.get("network_type", "grn")).lower()
    seeds = _flatten_seeds(p.get("seed_nodes", []))

    t0 = time.perf_counter()
    backend: str
    if CUGRAPH_AVAILABLE and (
        cugraph_function("personalized_pagerank") is not None
        or cugraph_function("pagerank") is not None
    ):
        try:
            scores, iters, converged, note = _rwr_cugraph(graph_csr, p, seeds)
            backend = "cugraph"
        except Exception as exc:                            # noqa: BLE001
            logging.warning(
                "rwr_gpu_baseline: cuGraph path failed (%s) — falling back "
                "to CuPy.", exc,
            )
            if not CUPY_AVAILABLE:
                raise
            scores, iters, converged, note = _rwr_cupy(graph_csr, p, seeds)
            backend = "cupy"
    elif CUPY_AVAILABLE:
        scores, iters, converged, note = _rwr_cupy(graph_csr, p, seeds)
        backend = "cupy"
    else:
        raise RuntimeError("Unreachable — require_any_backend should have raised.")
    elapsed = time.perf_counter() - t0

    top_nodes = top_k_global(scores, TOP_RWR_NODES)
    if seeds:
        seed_arr = np.asarray(sorted(set(seeds)), dtype=np.int64)
        top_seeds = top_k_among(scores, seed_arr, TOP_RWR_SEEDS)
    else:
        top_seeds = top_nodes[:TOP_RWR_SEEDS]

    inner = {
        "scores":     scores.astype(float).tolist(),
        "top_nodes":  top_nodes,
        "top_seeds":  top_seeds,
        "iterations": iters,
        "converged":  converged,
        "note":       f"{note}  Backend={backend}.",
        "backend":    backend,
    }

    return build_envelope(
        algorithm="rwr",
        network_type=network_type,
        execution_time=elapsed,
        graph_csr=graph_csr,
        inner=inner,
    )
