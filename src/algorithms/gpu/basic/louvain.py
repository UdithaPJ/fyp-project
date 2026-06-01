"""
src/algorithms/gpu/basic/louvain.py
===================================

Louvain — GPU baseline implementation.

Backend
-------
cuGraph ``cugraph.louvain`` only.  Raises ``ImportError`` immediately on
module import if cuGraph / cuDF (RAPIDS) are not installed.  There is
no CuPy fallback — the benchmarking baseline must use the cuGraph path.

Result keys (mirror src/algorithms/gpu/cuda_optimized/louvain.py)
-----------------------------------------------------------------
    community_assignments, num_communities, modularity,
    top_communities, hierarchy, note
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.sparse as sp

# Hard-fail: cuGraph + cuDF are required.
try:
    import cugraph   # noqa: F401
    import cudf      # noqa: F401
except ImportError as _e:
    raise ImportError(
        "src/algorithms/gpu/basic/louvain.py requires cuGraph and cuDF "
        "(RAPIDS).  Install via:\n"
        "  conda install -c rapidsai -c nvidia -c conda-forge "
        "rapids=24.02 python=3.10 cudatoolkit=11.8\n"
        f"Original error: {_e}"
    ) from _e

from ._utils import (
    BASELINE_MODE_CUGRAPH,
    LOUVAIN_TOP_COMMUNITIES,
    build_envelope, cugraph_build_graph, cugraph_extract_column,
    cugraph_function, symmetrize_for,
)


_DEFAULT_PARAMS: dict = {
    "min_delta_q":  1e-4,
    "max_levels":   10,
    "resolution":   1.0,
    "network_type": "grn",
}


# ---------------------------------------------------------------------------
# Community helpers
# ---------------------------------------------------------------------------

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
            "member_nodes": [int(x) for x in members[:50]],
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
        raise RuntimeError("cugraph.louvain is not available in this RAPIDS version.")

    nt = str(params.get("network_type", "grn")).lower()
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
# Public entry point
# ---------------------------------------------------------------------------

def louvain_gpu_baseline(
    graph_csr: sp.csr_matrix, params: dict | None = None,
) -> dict:
    """Louvain community detection — GPU baseline (cuGraph only).

    Returns
    -------
    dict
        7-key standard result envelope; ``result["mode"]`` is
        ``"gpu_baseline_cugraph"``.
    """
    p = {**_DEFAULT_PARAMS, **(params or {})}
    network_type = str(p.get("network_type", "grn")).lower()

    t0 = time.perf_counter()
    assignments, modularity_val, note = _louvain_cugraph(graph_csr, p)
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
        "hierarchy":             [],
        "note":                  f"{note}  Backend=cugraph.",
        "backend":               "cugraph",
    }

    return build_envelope(
        algorithm="louvain",
        network_type=network_type,
        execution_time=elapsed,
        graph_csr=graph_csr,
        inner=inner,
        mode=BASELINE_MODE_CUGRAPH,
    )
