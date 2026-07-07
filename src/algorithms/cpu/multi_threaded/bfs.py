"""
src/algorithms/cpu/multi_threaded/bfs.py
=========================================

BFS — GraphBLAS-backed cpu_multi implementation.

The classic GraphBLAS BFS pattern: at each depth level, the frontier is
expanded via ``frontier.vxm(A, any_pair_bool)`` (single boolean SpMV)
and the already-visited mask is applied in numpy.  This typically beats
the ``collections.deque`` Python-loop CPU baseline by 5–20× on graphs
larger than a few hundred thousand nodes because the frontier expansion
runs inside SuiteSparse's parallel SpMV kernel.

For the deterministic single-thread variant see
``src.algorithms.cpu.single_threaded.bfs``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import bfs_pack_result as _pack_result
from src.algorithms.cpu.multi_threaded._graphblas_utils import (
    _configure_threads,
    _from_scipy,
    _require_graphblas,
    _vec_sparse_bool,
    _vec_to_np,
    gb,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "source":    0,
    "max_depth": 999,    # benchmark default: traverse full reachable component
}


def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def bfs_cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict,
    n_workers: int | None = None,
) -> dict:
    """BFS using the classic GraphBLAS ``vxm`` boolean expansion.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix — directed adjacency.
    params    : dict
        source    (int, default 0)
        max_depth (int, default 999)
    n_workers : int | None
        SuiteSparse OpenMP thread count.

    Returns
    -------
    dict — see ``bfs_pack_result``.
    """
    _require_graphblas()
    _configure_threads(n_workers)

    p         = _merge_params(params)
    source    = int(p["source"])
    max_depth = int(p["max_depth"])
    n         = int(graph_csr.shape[0])

    distances     = np.full(n, -1, dtype=np.int32)
    visited_order: list[int] = []
    cascade: dict[int, list[int]] = {}

    if n == 0 or source < 0 or source >= n:
        return _pack_result(distances, visited_order, cascade)

    distances[source] = 0
    visited_order.append(source)
    cascade[0] = [source]

    # Build boolean adjacency on the GraphBLAS side.
    A_gb = _from_scipy(graph_csr.astype(np.bool_))

    # Initial frontier: a sparse bool Vector with True at `source`.
    frontier = _vec_sparse_bool(np.array([source], dtype=np.int64), n)
    sr = gb.semiring.any_pair[gb.dtypes.BOOL]

    for depth in range(1, max_depth + 1):
        # ---- One BFS level: next_frontier = frontier @ A in (any, pair) -
        next_frontier_gb = frontier.vxm(A_gb, sr).new()

        # ---- Mask out visited (numpy intersection) ----------------------
        nf_np = _vec_to_np(next_frontier_gb, n, dtype=bool, fill=False)
        unvisited_new = nf_np & (distances < 0)
        new_idx = np.where(unvisited_new)[0]
        if new_idx.size == 0:
            break

        # ---- Record the new level ---------------------------------------
        distances[new_idx] = depth
        visited_order.extend(int(x) for x in new_idx)
        cascade[depth] = sorted(int(x) for x in new_idx)

        # ---- Build next frontier from newly-discovered nodes only -------
        frontier = _vec_sparse_bool(new_idx.astype(np.int64), n)

    return _pack_result(distances, visited_order, cascade)


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict | None = None,
    n_workers: int | None = None,
    **_,
) -> dict:
    """Benchmark runner entry-point for cpu_multi mode (now GraphBLAS-backed)."""
    p = _merge_params(params)
    return {
        "output":       bfs_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "backend": "graphblas"},
    }
