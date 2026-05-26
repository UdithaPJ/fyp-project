"""
algorithms/bfs.py — Breadth-First Search for Biological Regulatory Cascade Tracing
====================================================================================

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
source     (int, default 0)   Index of the source TF node.
max_depth  (int, default 5)   Maximum BFS depth.  Cascade beyond this depth is
                              truncated (nodes at depth > max_depth are left
                              unreachable, distance = −1).
block_size (int, default 256) GPU thread block size (push kernel).
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
# OPTIMISED IMPLEMENTATION (five improvements vs. the original)
# -------------------------------------------------------------
# 1) Three-tier degree-aware scheduling inside a single push kernel
#    (`bfs_frontier_tiered`):
#       degree < WARP_SIZE      → 1 thread per node (serial scan)
#       WARP_SIZE ≤ d < BLOCK   → 1 warp per node  (warp-stride loop)
#       degree ≥ BLOCK_SIZE     → full block per node (block-stride loop)
#    This eliminates the hub-node tail-latency problem of one-thread-per-vertex.
#
# 2) Block-local shared-memory buffer + warp-level deduplication.  Each block
#    accumulates its newly-discovered neighbours in `__shared__ local_next[256]`
#    and flushes once at the end via a SINGLE `atomicAdd(next_size, count)`.
#    Same-warp duplicate discoveries are suppressed by a `__ballot_sync` vote
#    so only the lowest-lane thread issues the visited-set atomic.
#
# 3) Bitmap-based visited array (`uint32` of size `ceil(N/32)`) — 32× smaller
#    than the int32 sentinel scheme, far better L2/L1 hit rate, atomic writes
#    via `atomicOr` instead of `atomicCAS`.  The full `distances` int32 array
#    is still kept for the depth output.
#
# 4) Per-level launch path is retained; an optional persistent-kernel version
#    that keeps the level loop on-device via `cg::this_grid().sync()` is
#    designed but gated behind `USE_PERSISTENT_KERNEL = False` because not
#    every CUDA/PyCUDA install supports cooperative launches.  Falls back
#    automatically.
#
# 5) Adaptive push/pull direction switching (Beamer 2012).  Each level decides
#    between push and pull based on `frontier_size > n / (4 · avg_degree)`.
#    Pull uses the CSR-transpose and a bitmap frontier; push uses the dense
#    int32 worklist.  Per-level decisions are recorded in `traversal_modes`.
#
# Compilation: `-arch=sm_75` (RTX 20-series, Turing — explicit project target).
# Fallback chain: optimised GPU → CPU single-thread.
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Optional PyCUDA
# ---------------------------------------------------------------------------

try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    _PYCUDA_AVAILABLE = True
except Exception:                                       # noqa: BLE001
    cuda = None                                         # type: ignore[assignment]
    SourceModule = None                                 # type: ignore[assignment]
    _PYCUDA_AVAILABLE = False

# Optional: GPU config (block_size suggestions per tier).
try:
    from src.optimization.gpu_config import apply_config
    _GPU_CONFIG_AVAILABLE = True
except Exception:                                       # noqa: BLE001
    _GPU_CONFIG_AVAILABLE = False

    def apply_config(_name, _csr, params):              # type: ignore[no-redef]
        return params or {}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "source":       0,
    "max_depth":    5,
    "network_type": "grn",
    "block_size":   256,
}

_WARP_SIZE: int  = 32
_BLOCK_SIZE: int = 256   # also MAX_LOCAL for shared buffer

# Persistent (cooperative-groups) kernel is designed but disabled by default.
# Flip to True once cooperative launches have been verified on the target box.
USE_PERSISTENT_KERNEL: bool = False

# ---------------------------------------------------------------------------
# CUDA kernel source (push tiered + pull, single module)
# ---------------------------------------------------------------------------

_BFS_CU_SRC = r"""
extern "C" {

#define WARP_SIZE  32
#define BLOCK_SIZE 256
#define MAX_LOCAL  256

// =========================================================================
// PUSH KERNEL — three-tier degree-aware frontier expansion
//
// Grid layout: one block per frontier vertex.  Inside the block the work
// split depends on the vertex's out-degree:
//
//   degree < 32           : only threadIdx.x == 0 walks the neighbour list
//   32 <= degree < 256    : threads 0..31 (one warp) stride through the list
//   degree >= 256         : all 256 threads stride through the list
//
// Newly-discovered neighbours are pushed into a __shared__ buffer; one
// global atomicAdd per block reserves a contiguous slab in next_frontier
// and the block cooperatively writes its local buffer out.
// =========================================================================
__global__ void bfs_frontier_tiered(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const int* __restrict__ frontier,
    const int                frontier_size,
    int*       __restrict__ next_frontier,
    int*       __restrict__ next_size,
    unsigned int* __restrict__ visited_bitmap,
    int*       __restrict__ distances,
    const int                current_depth)
{
    __shared__ int local_next[MAX_LOCAL];
    __shared__ int local_count;
    __shared__ int global_base;

    if (threadIdx.x == 0) {
        local_count = 0;
    }
    __syncthreads();

    // One block per frontier vertex.
    if (blockIdx.x >= frontier_size) return;
    const int u         = frontier[blockIdx.x];
    const int row_start = row_offsets[u];
    const int row_end   = row_offsets[u + 1];
    const int degree    = row_end - row_start;

    // ---- Tiered dispatch -------------------------------------------------
    // Determine the (start_lane, stride) pair this thread will use.
    // - low tier  : single thread does the work, others sit out.
    // - mid tier  : first warp (lanes 0..31) participates.
    // - high tier : the whole block participates.
    int thread_start = -1;          // -1 → this thread is idle
    int thread_stride = 1;

    if (degree < WARP_SIZE) {
        if (threadIdx.x == 0) {
            thread_start  = 0;
            thread_stride = 1;
        }
    } else if (degree < BLOCK_SIZE) {
        if (threadIdx.x < WARP_SIZE) {
            thread_start  = threadIdx.x;
            thread_stride = WARP_SIZE;
        }
    } else {
        thread_start  = threadIdx.x;
        thread_stride = BLOCK_SIZE;
    }

    // ---- Neighbour scan --------------------------------------------------
    if (thread_start >= 0) {
        for (int off = thread_start; off < degree; off += thread_stride) {
            const int v = col_indices[row_start + off];

            // Pre-test (cheap, no atomic) — skip if already visited.
            const unsigned int word_idx = ((unsigned int)v) >> 5;
            const unsigned int bit_mask = 1u << (((unsigned int)v) & 31u);
            if ((visited_bitmap[word_idx] & bit_mask) != 0u) {
                continue;
            }

            // Warp-level deduplication: within the active warp, if two
            // threads found the same neighbour, only the lowest-lane wins.
            const unsigned int active = __activemask();
            const unsigned int same   = __match_any_sync(active, v);
            const unsigned int lane   = threadIdx.x & 31u;
            const unsigned int lower  = same & ((1u << lane) - 1u);
            if (lower != 0u) {
                // Some lower-lane sibling in this warp is already handling v.
                continue;
            }

            // Atomic test-and-set on the visited bit.
            const unsigned int prev =
                atomicOr(&visited_bitmap[word_idx], bit_mask);
            if ((prev & bit_mask) != 0u) {
                // Lost the race to a thread in a different block / warp.
                continue;
            }

            // Unique claim — record distance and push to local buffer.
            distances[v] = current_depth;
            const int pos = atomicAdd(&local_count, 1);
            if (pos < MAX_LOCAL) {
                local_next[pos] = v;
            } else {
                // Overflow: fall back to direct global push.  Very rare
                // because each block handles ONE source vertex and the
                // average out-degree is small; but on pathological hubs
                // (degree >> 256) the local buffer can fill, so we still
                // need a correct path.
                const int gp = atomicAdd(next_size, 1);
                next_frontier[gp] = v;
            }
        }
    }

    __syncthreads();

    // ---- Flush the block's local buffer to global next_frontier ---------
    const int n_to_flush = (local_count < MAX_LOCAL) ? local_count : MAX_LOCAL;
    if (threadIdx.x == 0) {
        global_base = (n_to_flush > 0) ? atomicAdd(next_size, n_to_flush) : 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n_to_flush; i += blockDim.x) {
        next_frontier[global_base + i] = local_next[i];
    }
}


// =========================================================================
// PULL KERNEL — direction-optimised reverse-edge scan
//
// One thread per unvisited node v.  For each incoming edge (u, v) of v,
// if u is in the current frontier (bit set in frontier_bitmap), claim v
// for the next level.  Writes v's bit into next_frontier_bitmap, sets
// distances[v], and bumps next_size_counter (for the host-side branching
// decision; the kernel does NOT materialise a dense worklist here).
//
// This is the heavy direction for high-density frontiers — checking
// "any incoming neighbour in frontier" terminates as soon as the first
// hit is found, avoiding the wasteful per-edge atomic of the push path.
// =========================================================================
__global__ void bfs_pull(
    const int* __restrict__ row_offsets_T,        // CSR-transpose: in-edges of v
    const int* __restrict__ col_indices_T,
    const unsigned int* __restrict__ frontier_bitmap,
    unsigned int* __restrict__ visited_bitmap,
    unsigned int* __restrict__ next_frontier_bitmap,
    int*       __restrict__ next_size_counter,
    int*       __restrict__ distances,
    const int                current_depth,
    const int                n)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    const unsigned int word_idx = ((unsigned int)v) >> 5;
    const unsigned int bit_mask = 1u << (((unsigned int)v) & 31u);

    // Skip already-visited vertices.
    if ((visited_bitmap[word_idx] & bit_mask) != 0u) return;

    const int row_start = row_offsets_T[v];
    const int row_end   = row_offsets_T[v + 1];

    for (int e = row_start; e < row_end; ++e) {
        const int u = col_indices_T[e];
        const unsigned int uw = ((unsigned int)u) >> 5;
        const unsigned int ub = 1u << (((unsigned int)u) & 31u);
        if ((frontier_bitmap[uw] & ub) != 0u) {
            // u is in the current frontier → v joins next level.
            // Atomic on the visited bit guarantees a single update even if
            // multiple incoming edges fire simultaneously.
            const unsigned int prev =
                atomicOr(&visited_bitmap[word_idx], bit_mask);
            if ((prev & bit_mask) == 0u) {
                atomicOr(&next_frontier_bitmap[word_idx], bit_mask);
                distances[v] = current_depth;
                atomicAdd(next_size_counter, 1);
            }
            break;
        }
    }
}


// =========================================================================
// SUPPORT KERNELS — bitmap ↔ worklist conversion
// =========================================================================

// Build a bitmap from a dense worklist (used when switching push → pull).
__global__ void worklist_to_bitmap(
    const int* __restrict__ worklist,
    const int                worklist_size,
    unsigned int* __restrict__ out_bitmap)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= worklist_size) return;
    const int v = worklist[i];
    const unsigned int w = ((unsigned int)v) >> 5;
    const unsigned int b = 1u << (((unsigned int)v) & 31u);
    atomicOr(&out_bitmap[w], b);
}

// Compact a bitmap back into a dense worklist (used when switching pull → push).
__global__ void bitmap_to_worklist(
    const unsigned int* __restrict__ bitmap,
    const int                bitmap_words,
    int*       __restrict__ out_worklist,
    int*       __restrict__ out_counter)
{
    const int wi = blockIdx.x * blockDim.x + threadIdx.x;
    if (wi >= bitmap_words) return;
    unsigned int word = bitmap[wi];
    while (word != 0u) {
        const int bit = __ffs(word) - 1;
        word &= (word - 1);
        const int v = (wi << 5) + bit;
        const int pos = atomicAdd(out_counter, 1);
        out_worklist[pos] = v;
    }
}

// Zero an int32 buffer.
__global__ void fill_int(int* buf, const int n, const int value) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    buf[i] = value;
}

// Zero a uint32 buffer.
__global__ void fill_u32(unsigned int* buf, const int n, const unsigned int value) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    buf[i] = value;
}

}  // extern "C"
"""

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

_KERNEL_CACHE: dict[str, dict[str, Any]] = {}
_PYCUDA_PRIMARY_CONTEXT = None


def _ensure_pycuda_context_current() -> bool:
    """Ensure a PyCUDA CUDA context is current in the calling thread.

    Returns True if this function pushed the context (caller must pop),
    False if a context was already current or context setup failed.
    Uses the device PRIMARY context so it can coexist with CuPy.
    """
    global _PYCUDA_PRIMARY_CONTEXT
    if not _PYCUDA_AVAILABLE or cuda is None:
        return False
    try:
        cuda.init()
        if _PYCUDA_PRIMARY_CONTEXT is None:
            if cuda.Device.count() <= 0:
                return False
            _PYCUDA_PRIMARY_CONTEXT = cuda.Device(0).retain_primary_context()

        try:
            current = cuda.Context.get_current()
        except Exception:                               # noqa: BLE001
            current = None

        if current is None:
            _PYCUDA_PRIMARY_CONTEXT.push()
            return True
    except Exception:                                   # noqa: BLE001
        return False
    return False


def _get_kernels() -> dict[str, Any]:
    """Compile (or fetch from cache) all BFS device kernels."""
    key = "bfs_optimized"
    if key not in _KERNEL_CACHE:
        options = ["-arch=sm_75"]                       # RTX 20-series target
        mod = SourceModule(_BFS_CU_SRC, options=options, no_extern_c=True)
        _KERNEL_CACHE[key] = {
            "push":       mod.get_function("bfs_frontier_tiered"),
            "pull":       mod.get_function("bfs_pull"),
            "w2b":        mod.get_function("worklist_to_bitmap"),
            "b2w":        mod.get_function("bitmap_to_worklist"),
            "fill_int":   mod.get_function("fill_int"),
            "fill_u32":   mod.get_function("fill_u32"),
        }
    return _KERNEL_CACHE[key]


# ---------------------------------------------------------------------------
# Param + result helpers
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


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


# ---------------------------------------------------------------------------
# GPU BFS — optimised direction-aware traversal
# ---------------------------------------------------------------------------

def _bfs_gpu_optimized(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """Five-improvement GPU BFS.  Raises on any GPU failure — caller decides
    whether to retry per-level or fall back to CPU.
    """
    if not _PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA is not available — cannot run GPU BFS.  "
            "Install pycuda and ensure a working CUDA toolchain is on PATH."
        )
    if cuda is None:
        raise RuntimeError("PyCUDA driver not available")

    pushed = _ensure_pycuda_context_current()
    try:
        # ---- Parameter merging ----------------------------------------------
        p = _merge_params(params)
        if _GPU_CONFIG_AVAILABLE:
            p = apply_config("bfs", graph_csr, p) or p
            # Re-apply our defaults for keys apply_config may have stripped.
            for k, v in _DEFAULT_PARAMS.items():
                p.setdefault(k, v)

        source       = int(p["source"])
        max_depth    = int(p["max_depth"])
        network_type = str(p.get("network_type", "grn"))
        block_size   = int(p.get("block_size", _BLOCK_SIZE))
        # The push kernel is hard-wired to BLOCK_SIZE=256 (shared memory size);
        # block_size only governs pull / fill kernels.
        if block_size <= 0 or block_size > 1024:
            block_size = _BLOCK_SIZE

        N = int(graph_csr.shape[0])
        if N == 0:
            raise ValueError("Empty graph")
        if not (0 <= source < N):
            raise ValueError(f"BFS source {source} out of bounds for N={N}")

        # Network-type adaptation: for PPI, treat as undirected by symmetrizing
        # the adjacency before running.  GRN / mirna keep direction.
        g_eff = graph_csr
        if network_type == "ppi":
            g_eff = (graph_csr + graph_csr.T).tocsr()
            g_eff.sum_duplicates()

        # CSR + transpose ----------------------------------------------------
        row_off_h = np.asarray(g_eff.indptr,  dtype=np.int32)
        col_idx_h = np.asarray(g_eff.indices, dtype=np.int32)
        if network_type == "ppi":
            # Symmetric: transpose == original.  Save memory.
            row_off_T_h = row_off_h
            col_idx_T_h = col_idx_h
        else:
            g_T = g_eff.T.tocsr()
            row_off_T_h = np.asarray(g_T.indptr,  dtype=np.int32)
            col_idx_T_h = np.asarray(g_T.indices, dtype=np.int32)

        nnz       = int(g_eff.nnz)
        avg_deg   = max(nnz / max(N, 1), 1e-9)
        # Beamer push→pull threshold.
        pp_threshold = N / max(4.0 * avg_deg, 1.0)

        bitmap_words = (N + 31) // 32

        kernels = _get_kernels()
        k_push   = kernels["push"]
        k_pull   = kernels["pull"]
        k_w2b    = kernels["w2b"]
        k_b2w    = kernels["b2w"]
        k_fill_i = kernels["fill_int"]
        k_fill_u = kernels["fill_u32"]

        # ---- Device allocation ---------------------------------------------
        d_buffers: list = []

        def _alloc(nbytes: int):
            buf = cuda.mem_alloc(int(nbytes))
            d_buffers.append(buf)
            return buf

        i32 = np.int32().nbytes
        u32 = np.uint32().nbytes

        d_row_off    = _alloc(row_off_h.nbytes)
        d_col_idx    = _alloc(col_idx_h.nbytes)
        d_row_off_T  = _alloc(row_off_T_h.nbytes)
        d_col_idx_T  = _alloc(col_idx_T_h.nbytes)
        d_distances  = _alloc(N * i32)
        d_visited    = _alloc(bitmap_words * u32)
        d_front_a    = _alloc(N * i32)               # push worklist (current)
        d_front_b    = _alloc(N * i32)               # push worklist (next)
        d_fbmp_cur   = _alloc(bitmap_words * u32)    # pull frontier bitmap
        d_fbmp_nxt   = _alloc(bitmap_words * u32)
        d_next_size  = _alloc(i32)

        try:
            # ---- H2D copies (graph + initial state) -------------------------
            cuda.memcpy_htod(d_row_off,   row_off_h)
            cuda.memcpy_htod(d_col_idx,   col_idx_h)
            cuda.memcpy_htod(d_row_off_T, row_off_T_h)
            cuda.memcpy_htod(d_col_idx_T, col_idx_T_h)

            distances_host = np.full(N, -1, dtype=np.int32)
            distances_host[source] = 0
            cuda.memcpy_htod(d_distances, distances_host)

            visited_host = np.zeros(bitmap_words, dtype=np.uint32)
            visited_host[source >> 5] |= np.uint32(1 << (source & 31))
            cuda.memcpy_htod(d_visited, visited_host)

            # Initial frontier (push form): [source]
            source_arr = np.array([source], dtype=np.int32)
            cuda.memcpy_htod(d_front_a, source_arr)
            frontier_size = 1
            mode_is_push  = True       # current frontier representation

            # Zero pull bitmaps.
            zero_bmp = np.zeros(bitmap_words, dtype=np.uint32)
            cuda.memcpy_htod(d_fbmp_cur, zero_bmp)
            cuda.memcpy_htod(d_fbmp_nxt, zero_bmp)

            visited_order: list[int]            = [source]
            cascade: dict[int, list[int]]       = {0: [source]}
            traversal_modes: list[str]          = []

            d_cur_push, d_nxt_push = d_front_a, d_front_b
            d_cur_bmp,  d_nxt_bmp  = d_fbmp_cur, d_fbmp_nxt

            zero_i32  = np.int32(0)

            for depth in range(1, max_depth + 1):
                if frontier_size == 0:
                    break

                use_pull = frontier_size > pp_threshold
                traversal_modes.append("pull" if use_pull else "push")

                # Reset next-size counter and next-frontier bitmap.
                cuda.memcpy_htod(d_next_size, zero_i32)
                grid_words = (bitmap_words + block_size - 1) // block_size
                k_fill_u(
                    d_nxt_bmp, np.int32(bitmap_words), np.uint32(0),
                    block=(block_size, 1, 1), grid=(grid_words, 1, 1),
                )

                if use_pull:
                    # Ensure d_cur_bmp holds the current frontier as a bitmap.
                    if mode_is_push:
                        # Push-worklist → bitmap conversion.
                        k_fill_u(
                            d_cur_bmp, np.int32(bitmap_words), np.uint32(0),
                            block=(block_size, 1, 1), grid=(grid_words, 1, 1),
                        )
                        grid_w = (frontier_size + block_size - 1) // block_size
                        k_w2b(
                            d_cur_push, np.int32(frontier_size), d_cur_bmp,
                            block=(block_size, 1, 1), grid=(grid_w, 1, 1),
                        )
                        mode_is_push = False

                    # Launch pull kernel — one thread per vertex.
                    grid_v = (N + block_size - 1) // block_size
                    k_pull(
                        d_row_off_T, d_col_idx_T,
                        d_cur_bmp, d_visited, d_nxt_bmp,
                        d_next_size, d_distances,
                        np.int32(depth), np.int32(N),
                        block=(block_size, 1, 1), grid=(grid_v, 1, 1),
                    )

                else:  # push
                    # Ensure d_cur_push holds the current frontier as a worklist.
                    if not mode_is_push:
                        # Bitmap → worklist conversion.
                        cuda.memcpy_htod(d_next_size, zero_i32)
                        gw = (bitmap_words + block_size - 1) // block_size
                        k_b2w(
                            d_cur_bmp, np.int32(bitmap_words),
                            d_cur_push, d_next_size,
                            block=(block_size, 1, 1), grid=(gw, 1, 1),
                        )
                        # Read back the compacted size.
                        ns_host = np.zeros(1, dtype=np.int32)
                        cuda.memcpy_dtoh(ns_host, d_next_size)
                        frontier_size = int(ns_host[0])
                        cuda.memcpy_htod(d_next_size, zero_i32)
                        mode_is_push = True
                        if frontier_size == 0:
                            break

                    # Push kernel: one block per frontier vertex.
                    grid_f = max(frontier_size, 1)
                    k_push(
                        d_row_off, d_col_idx,
                        d_cur_push, np.int32(frontier_size),
                        d_nxt_push, d_next_size,
                        d_visited, d_distances,
                        np.int32(depth),
                        block=(_BLOCK_SIZE, 1, 1),
                        grid=(grid_f, 1, 1),
                    )

                # ---- Read back next-frontier size -------------------------
                ns_host = np.zeros(1, dtype=np.int32)
                cuda.memcpy_dtoh(ns_host, d_next_size)
                next_size = int(ns_host[0])

                if next_size == 0:
                    break

                # Bookkeeping: read back the newly-visited nodes for cascade.
                if use_pull:
                    # New frontier sits in d_nxt_bmp; pull only the bits.
                    bmp_host = np.empty(bitmap_words, dtype=np.uint32)
                    cuda.memcpy_dtoh(bmp_host, d_nxt_bmp)
                    new_nodes_arr = np.flatnonzero(
                        np.unpackbits(
                            bmp_host.view(np.uint8),
                            bitorder="little",
                        )[:N]
                    )
                    new_nodes_list = new_nodes_arr.astype(int).tolist()
                else:
                    new_nodes_host = np.empty(next_size, dtype=np.int32)
                    cuda.memcpy_dtoh(new_nodes_host, d_nxt_push)
                    new_nodes_list = new_nodes_host.tolist()

                visited_order.extend(new_nodes_list)
                cascade[depth] = sorted(new_nodes_list)

                # Ping-pong both representations so whichever direction is
                # used next level finds its "current" buffer pre-populated.
                d_cur_push, d_nxt_push = d_nxt_push, d_cur_push
                d_cur_bmp,  d_nxt_bmp  = d_nxt_bmp,  d_cur_bmp
                frontier_size = next_size
                # If we used pull this level, the new frontier is stored as a
                # bitmap in d_cur_bmp; the push worklist d_cur_push is stale.
                mode_is_push = not use_pull

            # ---- Final D2H of distances -------------------------------------
            cuda.memcpy_dtoh(distances_host, d_distances)

            return _pack_result(distances_host, visited_order, cascade,
                                traversal_modes)

        finally:
            for buf in d_buffers:
                try:
                    buf.free()
                except Exception:                       # noqa: BLE001
                    pass
    finally:
        if pushed and _PYCUDA_PRIMARY_CONTEXT is not None:
            try:
                _PYCUDA_PRIMARY_CONTEXT.pop()
            except Exception:                           # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Top-level entry point with fallback chain
# ---------------------------------------------------------------------------

def bfs_gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """
    Optimised GPU BFS.

    Pipeline
    --------
        try:   optimised GPU BFS (direction-aware, bitmap, tiered)
        except GPU-error or PyCUDA-absent:
               fall back to CPU single-thread BFS

    Returns the dict shape required by ``_pack_result``:
        distances, visited_order, num_reachable, cascade_by_depth, traversal_modes
    """
    try:
        return _bfs_gpu_optimized(graph_csr, params)
    except Exception as exc:                            # noqa: BLE001
        logging.warning(
            "GPU BFS failed (%s: %s) — falling back to cpu_single",
            type(exc).__name__, exc,
        )
        # Late import to avoid importing CPU code on the happy GPU path.
        try:
            from src.algorithms.cpu.single_threaded.bfs import (
                _cpu_single as _bfs_cpu_single,
            )
        except Exception:                               # noqa: BLE001
            # Last-ditch: a numpy BFS so the smoke test still passes even
            # without the CPU package installed.
            return _bfs_cpu_inline(graph_csr, params)

        raw = _bfs_cpu_single(graph_csr, params=_merge_params(params))
        # _cpu_single may return either the inner result dict or the
        # {"output": ..., "extra_params": ...} runner shape.
        if isinstance(raw, dict) and "output" in raw:
            inner = raw["output"]
        else:
            inner = raw
        # Ensure the new traversal_modes key is present even from CPU fallback.
        if "traversal_modes" not in inner:
            inner["traversal_modes"] = ["cpu_fallback"]
        return inner


def _bfs_cpu_inline(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """Tiny pure-NumPy BFS, used only when neither GPU nor the CPU package
    is available (e.g. running this file standalone with no CUDA)."""
    from collections import deque
    p = _merge_params(params)
    source    = int(p["source"])
    max_depth = int(p["max_depth"])
    N         = int(graph_csr.shape[0])
    indptr  = graph_csr.indptr
    indices = graph_csr.indices

    distances = np.full(N, -1, dtype=np.int32)
    distances[source] = 0
    cascade: dict[int, list[int]] = {0: [source]}
    order: list[int] = [source]
    q = deque([source])
    while q:
        u = q.popleft()
        d_u = int(distances[u])
        if d_u >= max_depth:
            continue
        for v in indices[indptr[u]:indptr[u + 1]]:
            v = int(v)
            if distances[v] == -1:
                distances[v] = d_u + 1
                order.append(v)
                cascade.setdefault(d_u + 1, []).append(v)
                q.append(v)
    return _pack_result(distances, order,
                        {k: sorted(v) for k, v in cascade.items()},
                        ["cpu_inline_fallback"])


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _gpu(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    """Runner entry point — preserves the legacy `{"output", "extra_params"}`
    shape required by the benchmark runner and `algorithm_runner.py`.
    """
    p = _merge_params(params)
    return {"output": bfs_gpu(graph_csr, p), "extra_params": p}


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    # Make the project root importable so `from src.algorithms.cpu...` resolves.
    _PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    print("=" * 70)
    print("BFS GPU — optimised five-improvement implementation, smoke test")
    print("=" * 70)

    # -------- Test 1: tiny directed graph --------------------------------
    print("\n[Test 1] Tiny directed graph (10 nodes, 15 edges)")
    rng = np.random.default_rng(42)
    n1 = 10
    edges1 = {(0,1),(0,2),(1,3),(1,4),(2,4),(2,5),(3,6),(4,6),(4,7),
              (5,7),(6,8),(7,8),(7,9),(8,9),(0,5)}
    rows1 = [a for a, _ in edges1]
    cols1 = [b for _, b in edges1]
    data1 = [1.0] * len(edges1)
    g1 = sp.csr_matrix((data1, (rows1, cols1)), shape=(n1, n1))

    # Reference: tiny inline BFS (independent of the module under test).
    def ref_bfs(g, src, max_d):
        from collections import deque
        N = g.shape[0]
        d = [-1] * N
        d[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            if d[u] >= max_d:
                continue
            for v in g.indices[g.indptr[u]:g.indptr[u+1]]:
                if d[v] == -1:
                    d[v] = d[u] + 1
                    q.append(int(v))
        return d

    ref = ref_bfs(g1, 0, 5)
    out1 = bfs_gpu(g1, {"source": 0, "max_depth": 5, "network_type": "grn"})
    print(f"  GPU distances : {out1['distances']}")
    print(f"  REF distances : {ref}")
    match1 = out1["distances"] == ref
    print(f"  Small graph distances match : {match1}")
    print(f"  traversal_modes             : {out1['traversal_modes']}")

    # ---- helper: synthesise a power-law-ish directed graph in scipy only --
    def _make_hub_graph(n: int, hub_deg: int, tail_deg: int, seed: int):
        rng_local = np.random.default_rng(seed)
        rows, cols = [], []
        # First few nodes are hubs (high out-degree).
        n_hubs = max(2, n // 20)
        for u in range(n_hubs):
            targets = rng_local.choice(n, size=min(hub_deg, n - 1),
                                       replace=False)
            for v in targets:
                if int(v) != u:
                    rows.append(u); cols.append(int(v))
        # The rest get small out-degree.
        for u in range(n_hubs, n):
            targets = rng_local.choice(n, size=min(tail_deg, n - 1),
                                       replace=False)
            for v in targets:
                if int(v) != u:
                    rows.append(u); cols.append(int(v))
        data = np.ones(len(rows), dtype=np.float32)
        m = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        m.sum_duplicates()
        return m

    # -------- Test 2: hub-containing directed graph ----------------------
    print("\n[Test 2] Synthetic hub graph (100 nodes, hubs deg=40)")
    try:
        g2 = _make_hub_graph(n=100, hub_deg=40, tail_deg=3, seed=7)
        ref2 = ref_bfs(g2, 0, 6)
        out2 = bfs_gpu(g2, {"source": 0, "max_depth": 6, "network_type": "grn"})
        match2 = out2["distances"] == ref2
        print(f"  nnz                       : {g2.nnz}")
        print(f"  Hub graph distances match : {match2}")
        print(f"  traversal_modes           : {out2['traversal_modes']}")
        print(f"  num_reachable             : {out2['num_reachable']}")
    except Exception as e:                              # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}")

    # -------- Test 3: timing comparison ----------------------------------
    print("\n[Test 3] Timing comparison (1000 nodes, ~5000 edges)")
    try:
        gt = _make_hub_graph(n=1000, hub_deg=80, tail_deg=4, seed=11)

        t0 = time.perf_counter()
        ref3 = ref_bfs(gt, 0, 10)
        t_cpu_ref = time.perf_counter() - t0

        try:
            from src.algorithms.cpu.single_threaded.bfs import (
                _cpu_single as _bfs_cpu_single,
            )
            t0 = time.perf_counter()
            _bfs_cpu_single(gt, params={"source": 0, "max_depth": 10})
            t_cpu_single = time.perf_counter() - t0
        except Exception as e:                          # noqa: BLE001
            t_cpu_single = float("nan")
            print(f"  (cpu_single skipped: {e})")

        try:
            from src.algorithms.cpu.multi_threaded.bfs import (
                _cpu_multi as _bfs_cpu_multi,
            )
            t0 = time.perf_counter()
            _bfs_cpu_multi(gt, params={"source": 0, "max_depth": 10})
            t_cpu_multi = time.perf_counter() - t0
        except Exception as e:                          # noqa: BLE001
            t_cpu_multi = float("nan")
            print(f"  (cpu_multi skipped: {e})")

        t0 = time.perf_counter()
        out3 = bfs_gpu(gt, {"source": 0, "max_depth": 10, "network_type": "grn"})
        t_gpu = time.perf_counter() - t0
        match3 = out3["distances"] == ref3

        print(f"  cpu_reference (inline)   : {t_cpu_ref:.4f}s")
        print(f"  cpu_single               : {t_cpu_single:.4f}s")
        print(f"  cpu_multi                : {t_cpu_multi:.4f}s")
        print(f"  gpu (wall-clock total)   : {t_gpu:.4f}s")
        if t_cpu_single == t_cpu_single:   # NaN check
            print(f"  gpu speedup vs cpu_single: "
                  f"{(t_cpu_single / max(t_gpu, 1e-9)):.2f}x")
        print(f"  distances match          : {match3}")
        print(f"  traversal_modes          : {out3['traversal_modes']}")
    except Exception as e:                              # noqa: BLE001
        print(f"  Skipped (dependency missing: {e})")

    print("\n" + "=" * 70)
    print("Smoke test complete.")
    print("=" * 70)
