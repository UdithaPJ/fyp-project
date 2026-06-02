"""
src/algorithms/gpu/basic/bfs.py
===============================

BFS — GPU baseline implementation.

Backend
-------
cuGraph ``cugraph.bfs`` only.  Raises ``ImportError`` immediately on
module import if cuGraph / cuDF (RAPIDS) are not installed.  There is
no CuPy fallback — the benchmarking baseline must use the cuGraph path.

Simple traversal — no warp-tiering, no direction switching, no bitmap
worklist.  Reports a single ``traversal_modes`` value of ``"push_only"``
per level so the result-schema matches the optimized implementation
(which records the per-level push/pull decision).

Result keys
-----------
    distances, visited_order, num_reachable, cascade_by_depth, traversal_modes
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
        "src/algorithms/gpu/basic/bfs.py requires cuGraph and cuDF "
        "(RAPIDS).  Install via:\n"
        "  conda install -c rapidsai -c nvidia -c conda-forge "
        "rapids=24.02 python=3.10 cudatoolkit=11.8\n"
        f"Original error: {_e}"
    ) from _e

from ._utils import (
    BASELINE_MODE_CUGRAPH,
    build_envelope, cugraph_build_graph, cugraph_extract_column,
    cugraph_function,
)


_DEFAULT_PARAMS: dict = {
    "source":       0,
    "max_depth":    5,
    "network_type": "grn",
}


# ---------------------------------------------------------------------------
# Result packing — mirrors src/algorithms/gpu/cuda_optimized/bfs.py
# ---------------------------------------------------------------------------

def _pack_result(
    distances: np.ndarray,
    visited_order: list[int],
    cascade_by_depth: dict[int, list[int]],
    traversal_modes: list[str],
) -> dict:
    return {
        "distances":        distances.tolist(),
        "visited_order":    visited_order,
        "num_reachable":    int((distances >= 0).sum()),
        "cascade_by_depth": {str(k): v for k, v in cascade_by_depth.items()},
        "traversal_modes":  traversal_modes,
    }


def _build_cascade(distances: np.ndarray, max_depth: int) -> dict[int, list[int]]:
    """Group reachable nodes by BFS level, capped at ``max_depth``."""
    cascade: dict[int, list[int]] = {}
    for d in range(max_depth + 1):
        idx = np.where(distances == d)[0]
        if idx.size > 0:
            cascade[d] = idx.tolist()
    return cascade


# ---------------------------------------------------------------------------
# cuGraph path
# ---------------------------------------------------------------------------

def _bfs_cugraph(
    G, graph_csr: sp.csr_matrix, params: dict,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Run cuGraph BFS.  Returns ``(distances, visited_order, traversal_modes)``.

    MEMORY_FIX (H-4): cuGraph.Graph is built by the caller before timing.
    """
    bfs = cugraph_function("bfs")
    if bfs is None:
        raise RuntimeError("cugraph.bfs is not available in this RAPIDS version.")

    import inspect
    try:
        accepted = set(inspect.signature(bfs).parameters.keys())
    except (TypeError, ValueError):
        accepted = set()

    kwargs: dict[str, Any] = {}
    src = int(params["source"])
    if "start"  in accepted:   kwargs["start"]  = src
    elif "source" in accepted: kwargs["source"] = src
    elif "src"  in accepted:   kwargs["src"]    = src
    else:
        df = bfs(G, src)
        return _parse_bfs_df(df, graph_csr.shape[0], params)

    df = bfs(G, **kwargs)
    return _parse_bfs_df(df, graph_csr.shape[0], params)


def _parse_bfs_df(
    df, n: int, params: dict,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Parse cuGraph BFS dataframe into our standard arrays."""
    vertex_col   = cugraph_extract_column(df, ["vertex", "node", "id"]).astype(np.int64)
    distance_col = cugraph_extract_column(df, ["distance", "distances"])

    distances = np.full(n, -1, dtype=np.int64)
    np.put(distances, vertex_col, distance_col.astype(np.int64))

    max_depth = int(params.get("max_depth", 5))
    cut_mask = (distances > max_depth) | (distances < 0)
    distances_capped = distances.copy()
    distances_capped[cut_mask] = np.where(distances[cut_mask] < 0, -1, -1)

    reachable = np.where((distances >= 0) & (distances <= max_depth))[0]
    order = np.argsort(distances[reachable], kind="stable")
    visited_order = [int(reachable[i]) for i in order]

    max_seen = int(distances_capped[distances_capped >= 0].max()) if reachable.size > 0 else 0
    traversal_modes = ["push_only"] * (max_seen + 1)
    return distances_capped, visited_order, traversal_modes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def bfs_gpu_baseline(
    graph_csr: sp.csr_matrix, params: dict | None = None,
) -> dict:
    """BFS — GPU baseline (cuGraph only).

    Returns
    -------
    dict
        7-key standard result envelope; ``result["mode"]`` is
        ``"gpu_baseline_cugraph"``.
    """
    p = {**_DEFAULT_PARAMS, **(params or {})}
    network_type = str(p.get("network_type", "grn")).lower()
    max_depth    = int(p["max_depth"])

    # MEMORY_FIX (H-4): build cugraph.Graph before the timed region.
    directed = network_type != "ppi"
    G = cugraph_build_graph(graph_csr, directed=directed, weighted=False)

    t0 = time.perf_counter()
    distances, visited_order, traversal_modes = _bfs_cugraph(G, graph_csr, p)
    elapsed = time.perf_counter() - t0

    cascade = _build_cascade(distances, max_depth)

    inner = _pack_result(
        distances=distances,
        visited_order=visited_order,
        cascade_by_depth=cascade,
        traversal_modes=traversal_modes,
    )
    inner["backend"] = "cugraph"

    return build_envelope(
        algorithm="bfs",
        network_type=network_type,
        execution_time=elapsed,
        graph_csr=graph_csr,
        inner=inner,
        mode=BASELINE_MODE_CUGRAPH,
    )
