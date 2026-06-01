"""
src/algorithms/gpu/basic/bfs.py
===============================

BFS — GPU baseline implementation.

Backends
--------
    1. cuGraph ``cugraph.bfs``  (preferred)
    2. CuPy frontier expansion via sparse SpMV bit-mask  (fallback)
    3. RuntimeError                                       (neither installed)

Simple traversal — no warp-tiering, no direction switching, no bitmap
worklist.  Reports a single ``traversal_modes`` value of ``"push_only"``
per level so the result-schema matches the optimized implementation
(which records the per-level push/pull decision).

Result keys
-----------
    distances, visited_order, num_reachable, cascade_by_depth, traversal_modes
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

from ._utils import (
    CUGRAPH_AVAILABLE, CUPY_AVAILABLE,
    build_envelope, cugraph_build_graph, cugraph_extract_column,
    cugraph_function, require_any_backend,
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
    graph_csr: sp.csr_matrix, params: dict,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Run cuGraph BFS.  Returns ``(distances, visited_order, traversal_modes)``."""
    bfs = cugraph_function("bfs")
    if bfs is None:
        raise RuntimeError("cugraph.bfs is not available.")

    nt = str(params.get("network_type", "grn")).lower()
    directed = nt != "ppi"
    G = cugraph_build_graph(graph_csr, directed=directed, weighted=False)

    import inspect
    try:
        accepted = set(inspect.signature(bfs).parameters.keys())
    except (TypeError, ValueError):
        accepted = set()

    kwargs: dict[str, Any] = {}
    src = int(params["source"])
    # Source argument name varies across releases: try the most common ones.
    if "start"  in accepted: kwargs["start"]  = src
    elif "source" in accepted: kwargs["source"] = src
    elif "src"  in accepted: kwargs["src"]  = src
    else:
        # Last resort: pass positionally.
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
    # Cut off at max_depth — anything farther is reported as -1 (unreachable
    # for our reporting purposes, matching the optimized BFS's max_depth cap).
    cut_mask = (distances > max_depth) | (distances < 0)
    distances_capped = distances.copy()
    distances_capped[cut_mask] = np.where(distances[cut_mask] < 0, -1, -1)

    reachable = np.where((distances >= 0) & (distances <= max_depth))[0]
    order = np.argsort(distances[reachable], kind="stable")
    visited_order = [int(reachable[i]) for i in order]

    # Levels of activity — baseline reports push for every level.
    max_seen = int(distances_capped[distances_capped >= 0].max()) if reachable.size > 0 else 0
    traversal_modes = ["push_only"] * (max_seen + 1)
    return distances_capped, visited_order, traversal_modes


# ---------------------------------------------------------------------------
# CuPy fallback — sparse-SpMV frontier expansion
# ---------------------------------------------------------------------------

def _bfs_cupy(
    graph_csr: sp.csr_matrix, params: dict,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Iterative BFS in CuPy using sparse SpMV on a frontier bitmask."""
    import cupy as cp
    import cupyx.scipy.sparse as cpsp

    n = int(graph_csr.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64), [], []

    source    = int(params["source"])
    max_depth = int(params["max_depth"])
    nt        = str(params.get("network_type", "grn")).lower()

    if source < 0 or source >= n:
        raise ValueError(f"BFS source {source} out of range [0, {n})")

    # Build adjacency.  PPI: symmetrize (treat as undirected).
    if nt == "ppi":
        A = (graph_csr + graph_csr.T).astype(np.float32)
        if A.nnz > 0:
            A.data = np.ones_like(A.data, dtype=np.float32)
        A = A.tocsr()
    else:
        A = graph_csr.astype(np.float32).tocsr()

    A_gpu = cpsp.csr_matrix(
        (
            cp.asarray(A.data,    dtype=cp.float32),
            cp.asarray(A.indices, dtype=cp.int32),
            cp.asarray(A.indptr,  dtype=cp.int32),
        ),
        shape=A.shape,
    )

    distances = cp.full((n,), -1, dtype=cp.int64)
    distances[source] = 0
    frontier = cp.zeros((n,), dtype=cp.float32)
    frontier[source] = 1.0

    visited_order: list[int] = [source]
    traversal_modes: list[str] = ["push_only"]   # source level

    for level in range(1, max_depth + 1):
        # SpMV: next-frontier candidates have at least one in-neighbour in
        # the current frontier (push semantics, A^T @ frontier).
        next_mask = A_gpu.T.dot(frontier)
        # Restrict to nodes not yet visited.
        next_mask = cp.where(distances < 0, next_mask, cp.float32(0.0))
        next_indices = cp.where(next_mask > 0)[0]
        if next_indices.size == 0:
            break

        distances[next_indices] = level
        new_frontier = cp.zeros((n,), dtype=cp.float32)
        new_frontier[next_indices] = 1.0
        frontier = new_frontier
        traversal_modes.append("push_only")

        # Append in ascending node-id order for determinism.
        order = cp.sort(next_indices)
        visited_order.extend(int(x) for x in order.get().tolist())

    return distances.get().astype(np.int64), visited_order, traversal_modes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def bfs_gpu_baseline(
    graph_csr: sp.csr_matrix, params: dict | None = None,
) -> dict:
    """BFS — GPU baseline.  cuGraph preferred, CuPy SpMV fallback."""
    require_any_backend("bfs")
    p = {**_DEFAULT_PARAMS, **(params or {})}
    network_type = str(p.get("network_type", "grn")).lower()
    max_depth    = int(p["max_depth"])

    t0 = time.perf_counter()
    backend: str
    distances: np.ndarray
    visited_order: list[int]
    traversal_modes: list[str]
    if CUGRAPH_AVAILABLE and cugraph_function("bfs") is not None:
        try:
            distances, visited_order, traversal_modes = _bfs_cugraph(graph_csr, p)
            backend = "cugraph"
        except Exception as exc:                            # noqa: BLE001
            logging.warning(
                "bfs_gpu_baseline: cuGraph path failed (%s) — falling back "
                "to CuPy SpMV.", exc,
            )
            if not CUPY_AVAILABLE:
                raise
            distances, visited_order, traversal_modes = _bfs_cupy(graph_csr, p)
            backend = "cupy"
    elif CUPY_AVAILABLE:
        distances, visited_order, traversal_modes = _bfs_cupy(graph_csr, p)
        backend = "cupy"
    else:
        raise RuntimeError("Unreachable — require_any_backend should have raised.")
    elapsed = time.perf_counter() - t0

    cascade = _build_cascade(distances, max_depth)

    inner = _pack_result(
        distances=distances,
        visited_order=visited_order,
        cascade_by_depth=cascade,
        traversal_modes=traversal_modes,
    )
    inner["backend"] = backend

    return build_envelope(
        algorithm="bfs",
        network_type=network_type,
        execution_time=elapsed,
        graph_csr=graph_csr,
        inner=inner,
    )
