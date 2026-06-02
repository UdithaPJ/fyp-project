"""
src/algorithms/cpu/multi_threaded/bfs.py
=========================================

BFS — level-synchronous parallel CPU implementation only.

At each depth level the entire frontier (nodes discovered at the previous
level) is split into chunks and dispatched to a multiprocessing.Pool.
Each worker returns the union of out-neighbours for its chunk of frontier
nodes.  The main process deduplicates the collected neighbours and removes
already-visited nodes before advancing to the next level.

Level-synchronous BFS is embarrassingly parallel across the frontier at
each level.  On wide GRN frontiers (hub TFs with many targets) this yields
a meaningful speedup; on narrow frontiers the overhead dominates, so a
single-threaded fallback is used when the frontier is smaller than n_workers.

Shared helpers imported from src.algorithms.common.helpers:
    bfs_pack_result  (imported as _pack_result)

For the single-threaded variant see:
    src.algorithms.cpu.single_threaded.bfs
"""

from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import (
    _worker_init_no_blas,
    bfs_pack_result as _pack_result,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "source":    0,
    "max_depth": 5,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# Module-level worker — must be at module scope for multiprocessing.Pool pickle
# ---------------------------------------------------------------------------

def _neighbors_chunk(args: tuple) -> list[int]:
    """
    Return the out-neighbours of a set of nodes from a CSR adjacency matrix.

    Receives (indices, indptr, node_list) — raw CSR arrays without the data
    array (connectivity only needed for BFS).  Deduplicates within the chunk.
    """
    csr_indices, csr_indptr, node_list = args
    neighbors: set[int] = set()
    for node in node_list:
        s = int(csr_indptr[node])
        e = int(csr_indptr[node + 1])
        if s < e:
            neighbors.update(csr_indices[s:e].tolist())
    return list(neighbors)


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def bfs_cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict,
    n_workers: int = 4,
) -> dict:
    """
    BFS — level-synchronous parallel CPU implementation.

    At each depth level the frontier is split across a multiprocessing.Pool.
    When the frontier is smaller than n_workers, the Pool overhead is avoided
    and neighbour expansion runs inline.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix — directed GRN adjacency.
    params    : dict
        source    (int, default 0)
        max_depth (int, default 5)
    n_workers : int

    Returns
    -------
    dict with keys: distances, visited_order, num_reachable, cascade_by_depth
    """
    p         = _merge_params(params)
    source    = int(p["source"])
    max_depth = int(p["max_depth"])
    N         = graph_csr.shape[0]

    distances     = np.full(N, -1, dtype=np.int32)
    visited_order: list[int] = []
    cascade: dict[int, list[int]] = {}
    visited: set[int] = {source}

    distances[source] = 0
    visited_order.append(source)
    cascade[0] = [source]

    frontier: list[int] = [source]
    csr_indices = graph_csr.indices
    csr_indptr  = graph_csr.indptr

    # MEMORY_FIX (H-1/H-3): create the pool once and reuse it across all
    # depth levels; old code created/destroyed it per level.  We only
    # actually enter the pooled branch when the frontier is wider than
    # n_workers, but spinning the pool up here is still cheaper than per-
    # level recreation because BFS often has multiple wide levels.
    pool: Pool | None = None
    try:
        for depth in range(1, max_depth + 1):
            if not frontier:
                break

            if len(frontier) < n_workers:
                new_neighbors: set[int] = set()
                for node in frontier:
                    s = int(csr_indptr[node])
                    e = int(csr_indptr[node + 1])
                    new_neighbors.update(int(x) for x in csr_indices[s:e])
            else:
                if pool is None:
                    pool = Pool(
                        processes=n_workers,
                        initializer=_worker_init_no_blas,
                    )
                chunk_size = max(1, (len(frontier) + n_workers - 1) // n_workers)
                chunks = [
                    frontier[i: i + chunk_size]
                    for i in range(0, len(frontier), chunk_size)
                ]
                args_list = [(csr_indices, csr_indptr, chunk) for chunk in chunks]
                partial = pool.map(_neighbors_chunk, args_list)
                new_neighbors = set().union(*partial)

            new_frontier: list[int] = []
            for nb in new_neighbors:
                if nb not in visited:
                    visited.add(nb)
                    distances[nb] = depth
                    visited_order.append(nb)
                    new_frontier.append(nb)

            if new_frontier:
                cascade[depth] = sorted(new_frontier)
            frontier = new_frontier
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return _pack_result(distances, visited_order, cascade)


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict | None = None,
    n_workers: int = 4,
    **_,
) -> dict:
    """Benchmark runner entry-point for cpu_multi mode."""
    p = _merge_params(params)
    return {
        "output":       bfs_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "n_workers": n_workers},
    }
