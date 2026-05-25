"""
algorithms/bfs.py — Breadth-First Search for GRN Regulatory Cascade Tracing
=============================================================================

Biological Context
------------------
In a Gene Regulatory Network (GRN), a directed edge TF → gene encodes a
transcriptional regulatory event.  BFS from a *source TF* therefore traces its
full *regulatory cascade*: the set of genes reachable via successive regulatory
steps.

  Depth 1 — direct targets    (genes whose promoters the TF binds directly)
  Depth 2 — secondary targets (genes regulated by depth-1 targets)
  Depth k — k-th order effects

The ``cascade_by_depth`` output mirrors this directly, providing biologists with
a structured view of how regulatory influence propagates outward.  Limiting BFS
to ``max_depth`` avoids traversing distant, weakly connected network regions that
may not be biologically meaningful for the query TF.

BFS is also used as a preprocessing step in this framework:
  * Identify connected components before seeding RWR.
  * Confirm reachability when validating graph preprocessing.
  * Generate neighbourhood subgraphs for motif detection.

Parameter Guide
---------------
source    (int, default 0)   Index of the source TF node.
max_depth (int, default 5)   Maximum BFS depth.  Cascade beyond this depth is
                              truncated (nodes at depth > max_depth are left
                              unreachable, distance = −1).
"""

# ── GPU / CUDA-optimised implementation ──────────────────────────────────
# Source:    biological_network_framework/algorithms/bfs.py
# Requires:  cupy-cuda11x (or matching CUDA version), pycuda
# Used by:   webapp routes, GPU benchmarking (src.benchmarking.benchmark)
# Modes:     _cpu_single, _cpu_multi, _gpu  (all three)
# CPU-only counterparts (benchmarking only — never import in webapp):
#   src.algorithms.cpu.single_threaded.bfs
#   src.algorithms.cpu.multi_threaded.bfs
# ──────────────────────────────────────────────────────────────────────────

import warnings
from collections import deque
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Optional CuPy
# ---------------------------------------------------------------------------

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    cpsp = None
    _CUPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "source": 0,
    "max_depth": 5,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


def _pack_result(
    distances: np.ndarray,
    visited_order: list[int],
    cascade_by_depth: dict[int, list[int]],
) -> dict:
    return {
        "distances":       distances.tolist(),
        "visited_order":   visited_order,
        "num_reachable":   int((distances >= 0).sum()),
        "cascade_by_depth": {str(k): v for k, v in cascade_by_depth.items()},
    }


# ---------------------------------------------------------------------------
# Module-level worker for multiprocessing.Pool (must be picklable)
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
# Public implementations
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
    params    : dict — ``source`` and ``max_depth``.

    Returns
    -------
    dict with keys: distances, visited_order, num_reachable, cascade_by_depth
    """
    p         = _merge_params(params)
    source    = int(p["source"])
    max_depth = int(p["max_depth"])
    N         = graph_csr.shape[0]

    distances    = np.full(N, -1, dtype=np.int32)
    visited_order: list[int] = []
    cascade: dict[int, list[int]] = {}

    distances[source] = 0
    visited_order.append(source)
    cascade[0] = [source]

    # Queue entries: (node_index, depth)
    queue: deque[tuple[int, int]] = deque([(source, 0)])

    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue                         # do not expand beyond max_depth

        # Out-neighbours from CSR row
        row_start = int(graph_csr.indptr[node])
        row_end   = int(graph_csr.indptr[node + 1])
        for nb in graph_csr.indices[row_start:row_end]:
            nb = int(nb)
            if distances[nb] == -1:          # not yet visited
                nd = depth + 1
                distances[nb] = nd
                visited_order.append(nb)
                cascade.setdefault(nd, []).append(nb)
                queue.append((nb, nd))

    return _pack_result(distances, visited_order, cascade)


def bfs_cpu_multi(
    graph_csr: sp.csr_matrix,
    params: dict,
    n_workers: int = 4,
) -> dict:
    """
    BFS — level-synchronous parallel CPU implementation.

    At each depth level the entire frontier (set of nodes discovered at the
    previous level) is split into chunks and dispatched to a
    ``multiprocessing.Pool``.  Each worker returns the union of out-neighbours
    for its chunk of frontier nodes.  The main process deduplicates the
    collected neighbours and removes already-visited nodes before advancing
    to the next level.

    Level-synchronous BFS is embarrassingly parallel across the frontier at
    each level.  On wide GRN frontiers (hub TFs with many targets) this
    yields a meaningful speedup; on narrow frontiers the overhead dominates,
    so a single-threaded fallback is used when the frontier has fewer nodes
    than ``n_workers``.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
    params    : dict
    n_workers : int

    Returns
    -------
    Same structure as bfs_cpu_single.
    """
    p         = _merge_params(params)
    source    = int(p["source"])
    max_depth = int(p["max_depth"])
    N         = graph_csr.shape[0]

    distances    = np.full(N, -1, dtype=np.int32)
    visited_order: list[int] = []
    cascade: dict[int, list[int]] = {}
    visited: set[int] = {source}

    distances[source] = 0
    visited_order.append(source)
    cascade[0] = [source]

    frontier: list[int] = [source]
    # Pre-extract CSR arrays — passed to workers without pickling the matrix
    csr_indices = graph_csr.indices
    csr_indptr  = graph_csr.indptr

    for depth in range(1, max_depth + 1):
        if not frontier:
            break

        # For small frontiers, bypass multiprocessing overhead
        if len(frontier) < n_workers:
            new_neighbors: set[int] = set()
            for node in frontier:
                s = int(csr_indptr[node])
                e = int(csr_indptr[node + 1])
                new_neighbors.update(int(x) for x in csr_indices[s:e])
        else:
            chunk_size = max(1, (len(frontier) + n_workers - 1) // n_workers)
            chunks = [
                frontier[i: i + chunk_size]
                for i in range(0, len(frontier), chunk_size)
            ]
            args_list = [(csr_indices, csr_indptr, chunk) for chunk in chunks]
            with Pool(processes=n_workers) as pool:
                partial = pool.map(_neighbors_chunk, args_list)
            new_neighbors = set().union(*partial)

        # Deduplicate and filter already-visited
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

    return _pack_result(distances, visited_order, cascade)


def bfs_gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """
    BFS — level-synchronous GPU implementation using CuPy sparse SpMV.

    Represents the frontier as a float32 indicator vector ``f`` of length N
    (f[i] = 1.0 iff node i is in the current frontier).  At each level:

        raw = A_T_gpu @ f           # A_T[j, i] = edge i→j; raw[j] > 0 iff j
                                    # is reachable from some frontier node
        next_f = (raw > 0) & ~visited_gpu   # mask already-visited nodes

    This reduces one full BFS level to a single cuSPARSE SpMV plus two
    element-wise GPU operations — highly efficient for sparse GRN graphs.

    ``A_T_gpu`` (the CSR transpose of the adjacency matrix) is computed once
    before the loop and reused across all levels.

    Falls back to ``bfs_cpu_single`` with a warning if CuPy is unavailable.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
    params    : dict

    Returns
    -------
    Same structure as bfs_cpu_single.
    """
    try:
        from src.benchmarking.benchmark import _ensure_cuda_context
        if not _ensure_cuda_context():
            raise RuntimeError("no CUDA context")
    except Exception:
        pass

    if not _CUPY_AVAILABLE:
        warnings.warn(
            "CuPy unavailable — falling back to bfs_cpu_single.",
            RuntimeWarning,
            stacklevel=2,
        )
        return bfs_cpu_single(graph_csr, params)

    p         = _merge_params(params)
    source    = int(p["source"])
    max_depth = int(p["max_depth"])
    N         = graph_csr.shape[0]

    # Pre-compute A^T on GPU once (A_T[j, i] = A[i, j], i.e. edge i→j)
    # A^T @ frontier gives, for each node j, the summed frontier weights of
    # all nodes i that have an edge to j — positive iff j is reachable.
    A_T_cpu = graph_csr.T.tocsr().astype(np.float32)
    A_T_coo = A_T_cpu.tocoo()
    A_T_gpu = cpsp.csr_matrix(
        (
            cp.asarray(A_T_coo.data),
            (cp.asarray(A_T_coo.row), cp.asarray(A_T_coo.col)),
        ),
        shape=A_T_coo.shape,
    )

    # GPU state
    distances_gpu = cp.full(N, -1, dtype=cp.int32)
    visited_gpu   = cp.zeros(N, dtype=cp.bool_)
    frontier_vec  = cp.zeros(N, dtype=cp.float32)

    distances_gpu[source] = 0
    visited_gpu[source]   = True
    frontier_vec[source]  = 1.0

    # CPU bookkeeping (small; no overhead)
    visited_order: list[int] = [source]
    cascade: dict[int, list[int]] = {0: [source]}

    for depth in range(1, max_depth + 1):
        if not bool(frontier_vec.any()):
            break

        # Kernel: find all nodes reachable from current frontier in one hop
        raw = A_T_gpu @ frontier_vec                  # (N,) float32
        reachable = raw > 0.0                         # (N,) bool
        next_frontier_mask = reachable & ~visited_gpu # (N,) bool — unvisited only

        if not bool(next_frontier_mask.any()):
            break

        # Update distances and visited on GPU
        distances_gpu[next_frontier_mask] = depth
        visited_gpu[next_frontier_mask]   = True

        # Transfer new frontier node indices to CPU for bookkeeping only
        new_nodes: list[int] = cp.asnumpy(cp.where(next_frontier_mask)[0]).tolist()
        visited_order.extend(new_nodes)
        cascade[depth] = sorted(new_nodes)

        # Rebuild frontier indicator vector for next level
        frontier_vec = next_frontier_mask.astype(cp.float32)

    distances_cpu = cp.asnumpy(distances_gpu)

    del A_T_gpu, frontier_vec, visited_gpu, distances_gpu, raw
    cp.get_default_memory_pool().free_all_blocks()

    return _pack_result(
        distances_cpu.astype(np.int32),
        visited_order,
        cascade,
    )


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _cpu_single(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    p = _merge_params(params)
    return {"output": bfs_cpu_single(graph_csr, p), "extra_params": p}


def _cpu_multi(
    graph_csr: sp.csr_matrix,
    params: dict | None = None,
    n_workers: int = 4,
    **_,
) -> dict:
    p = _merge_params(params)
    return {
        "output":      bfs_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "n_workers": n_workers},
    }


def _gpu(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    p = _merge_params(params)
    return {"output": bfs_gpu(graph_csr, p), "extra_params": p}
