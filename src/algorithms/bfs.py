"""
src/algorithms/bfs.py — BFS for regulatory cascade tracing in GRNs
====================================================================

Biological context
------------------
BFS from a source TF traces the regulatory cascade downstream — at depth
1 the direct targets, at depth 2 the targets of those targets, etc.  In
GRN analysis this answers "what genes does this master TF eventually
influence?"  Output ``cascade_by_depth`` groups reachable genes by hop
distance so the frontend can visualise the cascade as concentric layers.
"""

from __future__ import annotations

import time
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

from .base import AlgorithmBase

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    cpsp = None
    _CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Worker for ProcessPoolExecutor — module-level (pickle-safe)
# ---------------------------------------------------------------------------

def _neighbors_chunk(args: tuple) -> set:
    """Return union of out-neighbours for a list of source nodes."""
    csr_indices, csr_indptr, node_list = args
    out = set()
    for u in node_list:
        s = int(csr_indptr[u])
        e = int(csr_indptr[u + 1])
        for j in csr_indices[s:e]:
            out.add(int(j))
    return out


# ---------------------------------------------------------------------------
# Cascade builder (shared)
# ---------------------------------------------------------------------------

def _build_cascade(
    distances: np.ndarray,
    max_depth: int,
) -> dict[str, list[int]]:
    """Group node indices by BFS depth.  Keys are stringified depths."""
    cascade: dict[str, list[int]] = {}
    finite = (distances >= 0) & (distances <= max_depth)
    for d in range(int(max_depth) + 1):
        layer = np.where(distances == d)[0]
        if layer.size > 0:
            cascade[str(d)] = layer.tolist()
        # Always include depth 0 (source) even if no further nodes reachable
        elif d == 0 and bool(finite.any()):
            cascade[str(d)] = []
    return cascade


# ---------------------------------------------------------------------------
# Public algorithm class
# ---------------------------------------------------------------------------

class BFS(AlgorithmBase):
    """Breadth-First Search — directed traversal from a single source TF."""

    NAME = "bfs"
    PARAM_SCHEMA = {
        "source":    0,
        "max_depth": 5,
    }

    @staticmethod
    def cpu_single(graph_csr: sp.csr_matrix, params: dict) -> dict:
        t0 = time.perf_counter()
        N = graph_csr.shape[0]
        source    = int(params.get("source",    0))
        max_depth = int(params.get("max_depth", 5))

        if not (0 <= source < N):
            raise ValueError(f"BFS source {source} out of bounds for N={N}")

        distances = np.full(N, -1, dtype=np.int32)
        distances[source] = 0
        visited_order = [source]
        queue = deque([source])

        while queue:
            u = queue.popleft()
            depth_u = distances[u]
            if depth_u >= max_depth:
                continue
            s = int(graph_csr.indptr[u])
            e = int(graph_csr.indptr[u + 1])
            for v in graph_csr.indices[s:e]:
                v = int(v)
                if distances[v] == -1:
                    distances[v] = depth_u + 1
                    visited_order.append(v)
                    queue.append(v)

        elapsed = time.perf_counter() - t0
        num_reachable = int((distances >= 0).sum())
        cascade = _build_cascade(distances, max_depth)

        return BFS.build_result(
            mode="cpu_single",
            execution_time=elapsed,
            graph_csr=graph_csr,
            result_data={
                "distances":        distances.tolist(),
                "visited_order":    visited_order,
                "num_reachable":    num_reachable,
                "cascade_by_depth": cascade,
            },
        )

    @staticmethod
    def cpu_multi(graph_csr: sp.csr_matrix, params: dict) -> dict:
        t0 = time.perf_counter()
        N = graph_csr.shape[0]
        source    = int(params.get("source",    0))
        max_depth = int(params.get("max_depth", 5))
        n_workers = int(params.get("n_workers", 4))

        if not (0 <= source < N):
            raise ValueError(f"BFS source {source} out of bounds for N={N}")

        distances = np.full(N, -1, dtype=np.int32)
        distances[source] = 0
        visited_order = [source]
        frontier = [source]

        # Level-synchronous: bypass pool for tiny frontiers (overhead exceeds gain)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for depth in range(max_depth):
                if not frontier:
                    break
                if len(frontier) < n_workers * 4:
                    new_nodes = _neighbors_chunk(
                        (graph_csr.indices, graph_csr.indptr, frontier)
                    )
                else:
                    chunk_size = max(1, (len(frontier) + n_workers - 1) // n_workers)
                    args_list = [
                        (graph_csr.indices, graph_csr.indptr,
                         frontier[i:i + chunk_size])
                        for i in range(0, len(frontier), chunk_size)
                    ]
                    new_nodes = set()
                    for s in ex.map(_neighbors_chunk, args_list):
                        new_nodes |= s

                next_frontier = []
                for v in new_nodes:
                    if distances[v] == -1:
                        distances[v] = depth + 1
                        visited_order.append(int(v))
                        next_frontier.append(int(v))
                frontier = next_frontier

        elapsed = time.perf_counter() - t0
        num_reachable = int((distances >= 0).sum())
        cascade = _build_cascade(distances, max_depth)

        return BFS.build_result(
            mode="cpu_multi",
            execution_time=elapsed,
            graph_csr=graph_csr,
            result_data={
                "distances":        distances.tolist(),
                "visited_order":    visited_order,
                "num_reachable":    num_reachable,
                "cascade_by_depth": cascade,
            },
        )

    @staticmethod
    def gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
        if not _CUPY_AVAILABLE:
            warnings.warn(
                "CuPy unavailable — falling back to bfs cpu_single.",
                RuntimeWarning, stacklevel=2,
            )
            r = BFS.cpu_single(graph_csr, params)
            r["mode"] = "gpu"
            return r

        t0 = time.perf_counter()
        N = graph_csr.shape[0]
        source    = int(params.get("source",    0))
        max_depth = int(params.get("max_depth", 5))

        if not (0 <= source < N):
            raise ValueError(f"BFS source {source} out of bounds for N={N}")

        # Use A.T so SpMV gives us out-neighbour discovery per level:
        # frontier_next = (A.T @ frontier_curr) marks rows reachable in 1 hop.
        # Actually for forward BFS along directed edges, we want A.T @ frontier
        # because frontier is a column vector indexed by "source", and we need
        # to find columns that source rows point to.  Equivalent: A @ frontier
        # when A[i,j] = edge i→j gives, for source vector e_u, the columns of
        # row u — i.e. u's out-neighbours.  We use the row form here.
        coo = graph_csr.tocoo()
        A_gpu = cpsp.csr_matrix(
            (cp.asarray(coo.data, dtype=cp.float32),
             (cp.asarray(coo.row), cp.asarray(coo.col))),
            shape=coo.shape,
        )
        A_T_gpu = A_gpu.T.tocsr()

        distances = cp.full(N, -1, dtype=cp.int32)
        distances[source] = 0
        visited = cp.zeros(N, dtype=cp.bool_)
        visited[source] = True

        frontier = cp.zeros(N, dtype=cp.float32)
        frontier[source] = 1.0
        visited_order_chunks: list[np.ndarray] = [np.array([source], dtype=np.int32)]

        for depth in range(max_depth):
            # Out-neighbours of frontier rows: SpMV with A^T on the frontier
            # (column vector); non-zero rows of the result are reachable.
            raw = A_T_gpu @ frontier
            new_mask = (raw > 0) & (~visited)
            if not bool(new_mask.any()):
                break
            new_indices = cp.where(new_mask)[0]
            distances[new_indices] = depth + 1
            visited |= new_mask
            visited_order_chunks.append(cp.asnumpy(new_indices).astype(np.int32))
            frontier = new_mask.astype(cp.float32)

        distances_cpu = cp.asnumpy(distances)
        visited_order = np.concatenate(visited_order_chunks).tolist()
        num_reachable = int((distances_cpu >= 0).sum())
        cascade = _build_cascade(distances_cpu, max_depth)

        del A_gpu, A_T_gpu, distances, visited, frontier
        cp.get_default_memory_pool().free_all_blocks()

        elapsed = time.perf_counter() - t0
        return BFS.build_result(
            mode="gpu",
            execution_time=elapsed,
            graph_csr=graph_csr,
            result_data={
                "distances":        distances_cpu.tolist(),
                "visited_order":    visited_order,
                "num_reachable":    num_reachable,
                "cascade_by_depth": cascade,
            },
        )
