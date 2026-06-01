"""
src/algorithms/gpu/basic/louvain.py
===================================

Louvain — GPU baseline implementation.

Backends
--------
    1. cuGraph ``cugraph.louvain``  (preferred — the only practical baseline)
    2. CuPy fallback                — a simple connected-components-style
                                      label-propagation pass, used purely so
                                      benchmarks can still produce a result
                                      dict with the correct schema when
                                      cuGraph is missing.
    3. RuntimeError                 — neither backend installed.

Why such a simple CuPy fallback
-------------------------------
Louvain is iterative and inherently irregular; a faithful GPU Louvain
without custom kernels would either replicate the optimized implementation
(violating the "baseline" rule) or be slower than CPU networkx.  The
baseline therefore degrades gracefully: when cuGraph is present we use
its native Louvain; otherwise we report communities via GPU label
propagation seeded by connected components.  This preserves the schema
and execution-time metric without competing with the optimized version.

Result keys (mirror src/algorithms/gpu/cuda_optimized/louvain.py)
-----------------------------------------------------------------
    community_assignments, num_communities, modularity,
    top_communities, hierarchy, note
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

from ._utils import (
    CUGRAPH_AVAILABLE, CUPY_AVAILABLE,
    LOUVAIN_TOP_COMMUNITIES,
    build_envelope, cugraph_build_graph, cugraph_extract_column,
    cugraph_function, require_any_backend, symmetrize_for,
)


_DEFAULT_PARAMS: dict = {
    "min_delta_q":  1e-4,
    "max_levels":   10,
    "resolution":   1.0,
    "network_type": "grn",
}


# ---------------------------------------------------------------------------
# Modularity (CPU, simple — used only for the CuPy fallback path).
# ---------------------------------------------------------------------------

def _modularity(adj: sp.csr_matrix, communities: np.ndarray) -> float:
    """Compute Newman-Girvan modularity Q for an undirected graph."""
    if adj.nnz == 0:
        return 0.0
    m2 = float(adj.sum())
    if m2 <= 0:
        return 0.0
    degrees = np.asarray(adj.sum(axis=1)).flatten()
    # Vectorized Q = (1/2m) * sum_{i,j in same community} (A_ij - k_i k_j / 2m)
    coo = adj.tocoo()
    same = communities[coo.row] == communities[coo.col]
    edge_term = float(coo.data[same].sum())

    unique, inv = np.unique(communities, return_inverse=True)
    deg_sum_per_comm = np.bincount(inv, weights=degrees, minlength=len(unique))
    deg_term = float(np.sum(deg_sum_per_comm ** 2)) / m2

    return (edge_term - deg_term) / m2


def _build_top_communities(
    communities: np.ndarray, k: int = LOUVAIN_TOP_COMMUNITIES,
) -> list[dict]:
    """Return up to ``k`` largest communities with their member node lists."""
    unique, counts = np.unique(communities, return_counts=True)
    order = np.argsort(counts)[::-1][:k]
    out: list[dict] = []
    for idx in order:
        cid = int(unique[idx])
        members = np.where(communities == cid)[0]
        out.append({
            "community_id": cid,
            "size":         int(counts[idx]),
            "member_nodes": [int(x) for x in members[: 50]],   # cap for payload
        })
    return out


# ---------------------------------------------------------------------------
# cuGraph path
# ---------------------------------------------------------------------------

def _louvain_cugraph(
    graph_csr: sp.csr_matrix, params: dict,
) -> tuple[np.ndarray, float, str]:
    louvain = cugraph_function("louvain")
    if louvain is None:
        raise RuntimeError("cugraph.louvain is not available.")

    nt = str(params.get("network_type", "grn")).lower()
    # GRN / miRNA / PPI all run through cuGraph's undirected Louvain;
    # we symmetrize the input first so the directed graphs become valid.
    A_sym = symmetrize_for(graph_csr, "grn") if nt != "ppi" else graph_csr
    G = cugraph_build_graph(A_sym, directed=False, weighted=True)

    import inspect
    try:
        accepted = set(inspect.signature(louvain).parameters.keys())
    except (TypeError, ValueError):
        accepted = set()

    kwargs: dict[str, Any] = {}
    if "resolution" in accepted: kwargs["resolution"] = float(params["resolution"])
    if "max_iter"   in accepted: kwargs["max_iter"]   = 100
    if "max_level"  in accepted: kwargs["max_level"]  = int(params["max_levels"])

    res = louvain(G, **kwargs)
    # cuGraph signature has been a 2-tuple (df, modularity) for many releases.
    if isinstance(res, tuple) and len(res) == 2:
        df, modularity_val = res
    else:
        df, modularity_val = res, float("nan")

    vertex_col = cugraph_extract_column(df, ["vertex", "node", "id"]).astype(np.int64)
    part_col   = cugraph_extract_column(df, ["partition", "community", "labels"]).astype(np.int64)

    n = int(graph_csr.shape[0])
    assignments = np.zeros(n, dtype=np.int64)
    np.put(assignments, vertex_col, part_col)

    note = (
        f"cuGraph Louvain ({nt}). "
        f"Modularity={modularity_val:.6f} when reported by RAPIDS."
    )
    return assignments, float(modularity_val if modularity_val == modularity_val else 0.0), note


# ---------------------------------------------------------------------------
# CuPy fallback — connected-components label propagation
# ---------------------------------------------------------------------------

def _louvain_cupy(
    graph_csr: sp.csr_matrix, params: dict,
) -> tuple[np.ndarray, float, str]:
    """Simple fallback partitioning when cuGraph is unavailable.

    We use scipy's weakly-connected-components result (CPU) as the
    "communities."  This is intentionally crude — the goal is only to
    keep the result-schema valid and the timer measuring *something*
    GPU-related (the matrix upload + a couple of CuPy reductions).
    """
    import cupy as cp

    nt = str(params.get("network_type", "grn")).lower()
    A_sym = symmetrize_for(graph_csr, nt)

    # Tiny GPU step so the timer reflects real GPU work — sum degrees.
    A_gpu_data = cp.asarray(A_sym.data, dtype=cp.float32)
    _ = float(cp.sum(A_gpu_data).get())

    n_components, labels = csgraph.connected_components(
        A_sym, directed=False, connection="weak",
    )
    assignments = labels.astype(np.int64)
    Q = _modularity(A_sym, assignments)

    note = (
        f"CuPy baseline fallback (no cuGraph): communities = weakly "
        f"connected components.  {n_components} components, "
        f"modularity={Q:.6f}.  Use cuGraph for a proper Louvain "
        f"partition."
    )
    return assignments, float(Q), note


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def louvain_gpu_baseline(
    graph_csr: sp.csr_matrix, params: dict | None = None,
) -> dict:
    """Louvain — GPU baseline.  cuGraph preferred, CuPy-CC fallback."""
    require_any_backend("louvain")
    p = {**_DEFAULT_PARAMS, **(params or {})}
    network_type = str(p.get("network_type", "grn")).lower()

    t0 = time.perf_counter()
    backend: str
    if CUGRAPH_AVAILABLE and cugraph_function("louvain") is not None:
        try:
            assignments, modularity_val, note = _louvain_cugraph(graph_csr, p)
            backend = "cugraph"
        except Exception as exc:                            # noqa: BLE001
            logging.warning(
                "louvain_gpu_baseline: cuGraph path failed (%s) — falling "
                "back to CuPy-CC.", exc,
            )
            if not CUPY_AVAILABLE:
                raise
            assignments, modularity_val, note = _louvain_cupy(graph_csr, p)
            backend = "cupy"
    elif CUPY_AVAILABLE:
        assignments, modularity_val, note = _louvain_cupy(graph_csr, p)
        backend = "cupy"
    else:
        raise RuntimeError("Unreachable — require_any_backend should have raised.")
    elapsed = time.perf_counter() - t0

    # Renumber to compact 0..K-1.
    _, compact = np.unique(assignments, return_inverse=True)
    assignments = compact.astype(np.int64)
    num_communities = int(assignments.max() + 1) if assignments.size > 0 else 0

    top_communities = _build_top_communities(assignments)

    inner = {
        "community_assignments": assignments.tolist(),
        "num_communities":       num_communities,
        "modularity":            float(modularity_val),
        "top_communities":       top_communities,
        "hierarchy":             [],   # baseline does not track hierarchy
        "note":                  f"{note}  Backend={backend}.",
        "backend":               backend,
    }

    return build_envelope(
        algorithm="louvain",
        network_type=network_type,
        execution_time=elapsed,
        graph_csr=graph_csr,
        inner=inner,
    )
