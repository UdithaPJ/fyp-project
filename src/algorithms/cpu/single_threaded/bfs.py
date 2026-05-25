"""
src/algorithms/cpu/single_threaded/bfs.py
==========================================

BFS — single-threaded CPU implementation only.

Traverses the directed GRN following out-edges via a standard FIFO queue.
The ``cascade_by_depth`` output groups genes by their regulatory distance
from the source TF, enabling biologists to identify direct targets (depth 1),
secondary targets (depth 2), etc.

Shared helpers imported from src.algorithms.common.helpers:
    bfs_pack_result  (imported as _pack_result)

For the level-synchronous parallel variant see:
    src.algorithms.cpu.multi_threaded.bfs
"""

from __future__ import annotations

from collections import deque

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import bfs_pack_result as _pack_result

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
# Public implementation
# ---------------------------------------------------------------------------

def bfs_cpu_single(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """
    BFS — single-threaded CPU using a FIFO queue (standard textbook BFS).

    Traverses the directed GRN following out-edges (row → column in CSR).
    Expansion of a node is skipped once the current depth exceeds *max_depth*,
    bounding the traversal to the biologically relevant regulatory cascade.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix — directed GRN adjacency.
    params    : dict
        source    (int, default 0) — index of the source TF node
        max_depth (int, default 5) — maximum BFS depth

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

    distances[source] = 0
    visited_order.append(source)
    cascade[0] = [source]

    queue: deque[tuple[int, int]] = deque([(source, 0)])

    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue

        row_start = int(graph_csr.indptr[node])
        row_end   = int(graph_csr.indptr[node + 1])
        for nb in graph_csr.indices[row_start:row_end]:
            nb = int(nb)
            if distances[nb] == -1:
                nd = depth + 1
                distances[nb] = nd
                visited_order.append(nb)
                cascade.setdefault(nd, []).append(nb)
                queue.append((nb, nd))

    return _pack_result(distances, visited_order, cascade)


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _cpu_single(
    graph_csr: sp.csr_matrix,
    params: dict | None = None,
    **_,
) -> dict:
    """Benchmark runner entry-point for cpu_single mode."""
    p = _merge_params(params)
    return {"output": bfs_cpu_single(graph_csr, p), "extra_params": p}
