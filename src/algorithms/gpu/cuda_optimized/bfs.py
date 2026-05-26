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

# ── GPU / CUDA-optimised implementation (PyCUDA) ─────────────────────────
# Source:    biological_network_framework/algorithms/bfs.py
# Requires:  pycuda  (with a working NVCC toolchain)
# Used by:   webapp routes, GPU benchmarking (src.benchmarking.benchmark)
# Modes:     _gpu only — this module is GPU-exclusive.
# CPU-only counterparts (benchmarking only — never import in webapp):
#   src.algorithms.cpu.single_threaded.bfs
#   src.algorithms.cpu.multi_threaded.bfs
#
# Implementation
# --------------
# A level-synchronous, frontier-based BFS executed entirely on the device.
# Each BFS level is one launch of the ``bfs_expand`` CUDA kernel:
#
#   for each node u in the current frontier (one thread per frontier vertex):
#       for each out-neighbour v of u (CSR row scan):
#           if atomicCAS(&distances[v], -1, depth) == -1:
#               pos = atomicAdd(&next_size, 1)
#               next_frontier[pos] = v
#
# This is the classic GPU-friendly traversal described in the algorithm
# analysis report:
#   * Frontier expansion (edge traversal) is parallel across the worklist.
#   * Visited test-and-set uses atomicCAS to guarantee a vertex is enqueued
#     exactly once even when many threads race to claim it.
#   * Frontier compaction is implicit — winning threads atomically reserve
#     a slot in the next-frontier worklist via atomicAdd, producing a
#     densely-packed list with no separate prefix-sum pass.
#
# Two device worklists are allocated up front (size N) and ping-ponged
# between levels; the device distances array doubles as the visited mask
# (-1 ⇔ unvisited).  Only the next-frontier indices are copied back to
# the host per level, for visited_order / cascade bookkeeping.
# ──────────────────────────────────────────────────────────────────────────

import warnings

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Optional PyCUDA
# ---------------------------------------------------------------------------

try:
    import pycuda.autoinit  # noqa: F401  (initialises a default context)
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    _PYCUDA_AVAILABLE = True
except Exception:
    cuda = None
    SourceModule = None
    _PYCUDA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "source": 0,
    "max_depth": 5,
}

_BLOCK_SIZE: int = 256

# ---------------------------------------------------------------------------
# CUDA kernel source
# ---------------------------------------------------------------------------

_BFS_CU_SRC = r"""
extern "C" {

__global__ void bfs_expand(
    const int* __restrict__ row_offsets,   // CSR indptr, size N+1
    const int* __restrict__ col_indices,   // CSR indices, size NNZ
    const int* __restrict__ frontier,      // current frontier worklist
    const int                frontier_size,
    int*       __restrict__ next_frontier, // output worklist (size <= N)
    int*       __restrict__ next_size,     // atomic counter for next_frontier
    int*       __restrict__ distances,     // size N, -1 if unvisited
    const int                current_depth)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    const int u         = frontier[tid];
    const int row_start = row_offsets[u];
    const int row_end   = row_offsets[u + 1];

    // Per-vertex parallelism: one thread scans the out-neighbour list of u.
    // For very high-degree (hub) vertices this is suboptimal — a more
    // sophisticated implementation would dispatch hubs to a whole warp/block.
    for (int e = row_start; e < row_end; ++e) {
        const int v = col_indices[e];

        // Test-and-set: claim v atomically iff it is still unvisited (-1).
        // The thread that wins the CAS is the unique discoverer of v and
        // is responsible for appending it to the next frontier.
        const int prev = atomicCAS(&distances[v], -1, current_depth);
        if (prev == -1) {
            const int pos = atomicAdd(next_size, 1);
            next_frontier[pos] = v;
        }
    }
}

}  // extern "C"
"""

_KERNEL_CACHE: dict = {}


def _get_kernel():
    """Compile (or fetch from cache) the bfs_expand kernel."""
    key = "bfs_expand"
    if key not in _KERNEL_CACHE:
        mod = SourceModule(_BFS_CU_SRC, no_extern_c=True)
        _KERNEL_CACHE[key] = mod.get_function(key)
    return _KERNEL_CACHE[key]


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
# GPU BFS — PyCUDA frontier traversal
# ---------------------------------------------------------------------------

def bfs_gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """
    BFS — level-synchronous frontier traversal on the GPU via a custom
    PyCUDA kernel.

    Each level launches ``bfs_expand`` with one thread per frontier vertex.
    Threads scan their vertex's out-neighbour list from the CSR adjacency
    and attempt an ``atomicCAS`` on the per-vertex distance slot to claim
    unvisited neighbours.  Winners append their newly-claimed vertex to a
    densely-packed next-frontier worklist via ``atomicAdd``.

    The host loop drives one launch per BFS level until either the frontier
    is empty or ``max_depth`` is reached, ping-ponging two device worklists
    of size N.  Only the indices appended at each level are copied back to
    the host (for ``visited_order`` and ``cascade_by_depth`` bookkeeping);
    the distances array is copied back a single time at the end.

    Raises
    ------
    RuntimeError
        If PyCUDA is unavailable or no CUDA device can be initialised.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix — directed adjacency, rows = sources.
    params    : dict — ``source`` and ``max_depth``.

    Returns
    -------
    dict with keys: distances, visited_order, num_reachable, cascade_by_depth.
    """
    if not _PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA is not available — cannot run GPU BFS.  "
            "Install pycuda and ensure a working CUDA toolchain is on PATH."
        )

    p         = _merge_params(params)
    source    = int(p["source"])
    max_depth = int(p["max_depth"])
    N         = int(graph_csr.shape[0])

    if not (0 <= source < N):
        raise ValueError(f"BFS source {source} out of bounds for N={N}")

    # CSR arrays expected by the kernel — both int32 on the device.
    row_offsets_host = np.asarray(graph_csr.indptr, dtype=np.int32)
    col_indices_host = np.asarray(graph_csr.indices, dtype=np.int32)

    kernel = _get_kernel()

    # ---- Device allocation ----
    d_row_offsets = cuda.mem_alloc(row_offsets_host.nbytes)
    d_col_indices = cuda.mem_alloc(col_indices_host.nbytes)
    d_distances   = cuda.mem_alloc(N * np.int32().nbytes)
    d_frontier_a  = cuda.mem_alloc(N * np.int32().nbytes)
    d_frontier_b  = cuda.mem_alloc(N * np.int32().nbytes)
    d_next_size   = cuda.mem_alloc(np.int32().nbytes)

    try:
        # ---- H2D copies (graph + initial state) ----
        cuda.memcpy_htod(d_row_offsets, row_offsets_host)
        cuda.memcpy_htod(d_col_indices, col_indices_host)

        distances_host = np.full(N, -1, dtype=np.int32)
        distances_host[source] = 0
        cuda.memcpy_htod(d_distances, distances_host)

        # Seed frontier A with the single source vertex.
        source_arr = np.array([source], dtype=np.int32)
        cuda.memcpy_htod(d_frontier_a, source_arr)
        frontier_size = 1

        visited_order: list[int] = [source]
        cascade: dict[int, list[int]] = {0: [source]}

        d_cur, d_nxt = d_frontier_a, d_frontier_b
        zero_i32 = np.int32(0)

        for depth in range(1, max_depth + 1):
            if frontier_size == 0:
                break

            # Reset next-frontier counter.
            cuda.memcpy_htod(d_next_size, zero_i32)

            grid_x = (frontier_size + _BLOCK_SIZE - 1) // _BLOCK_SIZE
            kernel(
                d_row_offsets,
                d_col_indices,
                d_cur,
                np.int32(frontier_size),
                d_nxt,
                d_next_size,
                d_distances,
                np.int32(depth),
                block=(_BLOCK_SIZE, 1, 1),
                grid=(grid_x, 1, 1),
            )

            # Read back the size of the next frontier.
            next_size_host = np.zeros(1, dtype=np.int32)
            cuda.memcpy_dtoh(next_size_host, d_next_size)
            next_size = int(next_size_host[0])

            if next_size == 0:
                break

            # Copy the new frontier indices back for host-side bookkeeping.
            new_nodes_host = np.empty(next_size, dtype=np.int32)
            cuda.memcpy_dtoh(new_nodes_host, d_nxt)
            new_nodes_list = new_nodes_host.tolist()

            visited_order.extend(new_nodes_list)
            cascade[depth] = sorted(new_nodes_list)

            # Ping-pong the worklists for the next level.
            d_cur, d_nxt = d_nxt, d_cur
            frontier_size = next_size

        # ---- Final D2H of the distance array ----
        cuda.memcpy_dtoh(distances_host, d_distances)

    finally:
        # Explicit frees — PyCUDA's deallocation can race with context teardown
        # if we leave these to GC, especially across repeated benchmark runs.
        for buf in (
            d_row_offsets, d_col_indices, d_distances,
            d_frontier_a, d_frontier_b, d_next_size,
        ):
            try:
                buf.free()
            except Exception:
                pass

    return _pack_result(distances_host, visited_order, cascade)


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _gpu(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    p = _merge_params(params)
    return {"output": bfs_gpu(graph_csr, p), "extra_params": p}
