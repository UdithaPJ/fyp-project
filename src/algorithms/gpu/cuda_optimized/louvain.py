"""
algorithms/louvain.py — Louvain Community Detection (GPU / PyCUDA)
==================================================================

Biological context
------------------
Louvain on a biological network identifies modules of nodes that are more
densely connected to each other than to the rest of the graph:

  GRN   — co-regulated gene modules (shared TF inputs, feedback arcs).
  PPI   — protein complexes / functional modules.
  miRNA — miRNA–gene regulons (a miRNA and its co-targeted genes).

The ``top_communities`` output is the natural input to GO-term enrichment
tools (Enrichr, g:Profiler) — each module can be tested for functional
coherence.

Why the graph must be undirected for Louvain
--------------------------------------------
Modularity Q is only defined for undirected graphs:

    Q = (1 / 2m) · Σ_{i,j} ( A_{ij} − γ · k_i · k_j / 2m ) · δ(c_i, c_j)

The null model k_i·k_j / 2m assumes undirected degree.  For GRN / miRNA
networks this module applies A ← A + Aᵀ internally (mutual edges keep
their summed weight = 2 × original, encoding tighter coupling).  PPI
networks are already undirected and are used as-is.

The symmetrisation step happens *inside* this module rather than during
preprocessing because every other algorithm (HITS, RWR, BFS, PageRank,
MCL) needs the original directed graph.

Parallel non-determinism
------------------------
Phase 1 is bulk-synchronous on the GPU: every node evaluates its best
move against a *snapshot* of the community state, then all moves are
applied in a single batch.  Two nodes that beneficially swap into each
other's communities will BOTH apply simultaneously — the resulting
partition may differ between runs.  This is a documented property of
parallel Louvain (Traag et al. 2019, "From Louvain to Leiden").
Modularity remains non-decreasing in practice; the partition is still
valid.

Algorithm outline
-----------------
Phase 1 — node-level moves (GPU):
    For each node u, compute ΔQ for moving u to each neighbouring
    community.  Propose the best move.  Apply all proposals as a batch.
    Repeat until no node moves, or ``max_phase1_passes`` reached.

Phase 2 — graph coarsening (mixed GPU + CPU):
    Each community becomes a super-node; inter-community edges are
    summed.  Self-loops on super-nodes carry intra-community weight.
    Recurse Phase 1 on the coarsened graph.

Parameter guide
---------------
min_delta_q  (float, default 1e-4)  Minimum ΔQ to accept a move.
max_levels   (int,   default 10)    Maximum Phase 1+2 recursion depth.
resolution   (float, default 1.0)   γ in the modularity formula.
                                    > 1.0 → more, smaller communities.
                                    < 1.0 → fewer, larger communities.
network_type (str,   default "grn") One of "grn", "ppi", "mirna".
block_size   (int,   default 256)   CUDA block dimension.
"""

# ── GPU / CUDA-optimised implementation (PyCUDA custom kernels) ──────────
# Source:    biological_network_framework/algorithms/louvain.py
# Requires:  pycuda (with a working NVCC toolchain)
# Used by:   webapp routes, GPU benchmarking (src.benchmarking.benchmark)
# Modes:     _gpu only — this module is GPU-exclusive.
# CPU-only counterparts (benchmarking only — never import in webapp):
#   src.algorithms.cpu.single_threaded.louvain
#   src.algorithms.cpu.multi_threaded.louvain
#
# Four PyCUDA kernels (one SourceModule, compiled once, cached):
#   compute_proposed_moves     — Phase 1: three-tier degree-aware ΔQ scan
#                                + block-wide best-move reduction.
#                                Writes proposals only; never touches the
#                                community array directly.
#   apply_moves                — Phase 1: batch-apply all proposals.
#                                Updates community + comm_degree_sum via
#                                atomicAdd.
#   count_community_edges      — Phase 2: edge-parallel mapping
#                                (u, v, w) → (comm[u], comm[v], w).
#   compute_modularity_partial — Final Q: per-block partial sums of
#                                A_{ij} − γ·k_i·k_j/(2m) over same-
#                                community edges.
#
# Sort + reduce + scan for Phase 2 coarsening run CPU-side
# (np.lexsort + np.add.reduceat) — correctness-first, no thrust/CUB
# dependency.  GPU radix sort is a future optimisation (see TODO).
#
# Compilation: -arch=sm_75 (RTX 20-series, Turing — explicit target).
# Does NOT silently fall back to CPU; raises RuntimeError / MemoryError /
# cuda.LogicError so the runner can surface the failure.
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Optional PyCUDA
# ---------------------------------------------------------------------------

try:
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
    PYCUDA_AVAILABLE = True
except Exception:                                       # noqa: BLE001
    cuda = None                                         # type: ignore[assignment]
    gpuarray = None                                     # type: ignore[assignment]
    SourceModule = None                                 # type: ignore[assignment]
    PYCUDA_AVAILABLE = False
    logging.warning("PyCUDA not available — louvain_gpu() will raise.")

# Optional: GPU config (block_size + chunking suggestions per tier).
try:
    from src.optimization.gpu_config import apply_config, get_gpu_config
    _GPU_CONFIG_AVAILABLE = True
except Exception:                                       # noqa: BLE001
    _GPU_CONFIG_AVAILABLE = False

    def apply_config(_name, _csr, params=None):         # type: ignore[no-redef]
        return params or {}

    def get_gpu_config():                               # type: ignore[no-redef]
        return {"free_vram_mb": 0}

try:
    from src.benchmarking.benchmark import _ensure_cuda_context
except Exception:                                       # noqa: BLE001
    def _ensure_cuda_context() -> bool:                 # type: ignore[no-redef]
        return True

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "min_delta_q":        1e-4,
    "max_levels":         10,
    "resolution":         1.0,
    # Maximum Phase 1 passes per Louvain level.  Bulk-synchronous parallel
    # Phase 1 can oscillate (two adjacent nodes swapping communities in
    # alternating passes), so the cap is the primary termination guarantee.
    "max_phase1_passes":  100,
    "network_type":       "grn",
    "block_size":         256,
}

BLOCK_SIZE: int    = 256
WARP_SIZE: int     = 32
VRAM_SAFETY: float = 0.80    # warn if working set > 80 % of free VRAM
_TOP_K: int        = 5


# ---------------------------------------------------------------------------
# CUDA kernel source (all four kernels, single SourceModule)
# ---------------------------------------------------------------------------

KERNEL_SOURCE = r"""
extern "C" {

#define BLOCK_SIZE 256
#define WARP_SIZE  32

// =========================================================================
// KERNEL 1: compute_proposed_moves
//
// Phase 1 core.  One block per source node u; threads inside the block
// cooperatively scan u's edges with three-tier degree-aware scheduling:
//
//   LOW   (degree < 32)         : thread 0 only, serial scan
//   MED   (32 <= degree < 256)  : first warp, stride-32 + warp shuffles
//   HIGH  (degree >= 256)       : whole block, stride-256 + shared-mem
//
// Two passes per block:
//
//   Pass 1 — accumulate k_self = sum w_uv over edges where comm[v] == comm[u].
//            Reduce across active threads; broadcast via shared memory.
//   Pass 2 — each active thread evaluates per-edge gain for moving u to
//            comm[v], tracks its local (best_gain, best_comm).  Block-wide
//            reduction finds the block's winning pair.  Thread 0 writes
//            proposed_comm[u] (or c_old if best gain <= min_delta_q).
//
// Per-edge gain (the standard parallel-Louvain approximation):
//
//   join_gain(e=(u,v,w)) = w/m  -  gamma * k_u * Sigma_tot[c_v] / (2 m^2)
//   leave_gain(u)        = k_self/m  -  gamma * (Sigma_tot[c_u] - k_u) * k_u / (2 m^2)
//   delta_Q(e)           = join_gain(e) - leave_gain(u)
//
// The kernel writes to proposed_comm only — community[] is never modified
// here.  apply_moves performs the batch update separately.
// =========================================================================
__global__ void compute_proposed_moves(
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ edge_wt,
    const int*   __restrict__ community,
    const float* __restrict__ comm_degree_sum,
    const float* __restrict__ node_degree,
    int*         __restrict__ proposed_comm,
    const float                min_delta_q,
    const float                inv_2m,        // = 1 / (2m)
    const float                resolution,
    const int                  n)
{
    __shared__ float smem_f[BLOCK_SIZE];
    __shared__ int   smem_i[BLOCK_SIZE];
    __shared__ float s_leave;
    __shared__ float s_k_self;
    __shared__ int   s_proposed;

    const int u = blockIdx.x;
    if (u >= n) return;

    const int row_start = row_ptr[u];
    const int row_end   = row_ptr[u + 1];
    const int degree    = row_end - row_start;
    const int c_old     = community[u];
    const float k_u     = node_degree[u];

    if (degree == 0) {
        if (threadIdx.x == 0) proposed_comm[u] = c_old;
        return;
    }

    // ---- Tier decision (uniform across the block; degree is shared) ----
    int t_start  = -1;
    int t_stride = 1;
    if (degree < WARP_SIZE) {
        if (threadIdx.x == 0) { t_start = 0;            t_stride = 1; }
    } else if (degree < BLOCK_SIZE) {
        if (threadIdx.x < WARP_SIZE) {
            t_start = threadIdx.x;                       t_stride = WARP_SIZE;
        }
    } else {
        t_start = threadIdx.x;                           t_stride = BLOCK_SIZE;
    }

    // =================================================================
    // PASS 1 — accumulate k_self = sum w_uv where comm[v] == c_old.
    // =================================================================
    float local_k_self = 0.0f;
    if (t_start >= 0) {
        for (int off = t_start; off < degree; off += t_stride) {
            const int v = col_idx[row_start + off];
            if (community[v] == c_old) {
                local_k_self += edge_wt[row_start + off];
            }
        }
    }

    // Reduce local_k_self across active threads, write to s_k_self.
    if (degree < WARP_SIZE) {
        if (threadIdx.x == 0) s_k_self = local_k_self;
    } else if (degree < BLOCK_SIZE) {
        if (threadIdx.x < WARP_SIZE) {
            for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
                local_k_self += __shfl_down_sync(0xffffffffu, local_k_self, off);
            }
            if (threadIdx.x == 0) s_k_self = local_k_self;
        }
    } else {
        smem_f[threadIdx.x] = local_k_self;
        __syncthreads();
        for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) smem_f[threadIdx.x] += smem_f[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) s_k_self = smem_f[0];
    }
    __syncthreads();

    // Compute leave_gain on thread 0; broadcast via shared memory.
    if (threadIdx.x == 0) {
        const float sigma_old = comm_degree_sum[c_old];
        // 1/m = 2 * inv_2m;  1/(2 m^2) = 2 * inv_2m * inv_2m
        s_leave = 2.0f * inv_2m * s_k_self
                  - 2.0f * resolution * (sigma_old - k_u) * k_u
                    * inv_2m * inv_2m;
    }
    __syncthreads();

    // =================================================================
    // PASS 2 — per-edge gain; track best (gain, comm) per active thread.
    // =================================================================
    float best_gain = 0.0f;       // require strict improvement
    int   best_comm = c_old;

    if (t_start >= 0) {
        const float leave = s_leave;
        for (int off = t_start; off < degree; off += t_stride) {
            const int v   = col_idx[row_start + off];
            const int c_v = community[v];
            if (c_v == c_old) continue;          // moving to current = no-op
            const float w         = edge_wt[row_start + off];
            const float sigma_new = comm_degree_sum[c_v];
            const float join      = 2.0f * inv_2m * w
                                    - 2.0f * resolution * k_u * sigma_new
                                      * inv_2m * inv_2m;
            const float gain      = join - leave;
            if (gain > best_gain) {
                best_gain = gain;
                best_comm = c_v;
            }
        }
    }

    // -------------------------------------------------------------------
    // Reduce (best_gain, best_comm) pair across active threads.
    // -------------------------------------------------------------------
    if (degree < WARP_SIZE) {
        if (threadIdx.x == 0) {
            s_proposed = (best_gain > min_delta_q) ? best_comm : c_old;
        }
    } else if (degree < BLOCK_SIZE) {
        if (threadIdx.x < WARP_SIZE) {
            float my_gain = best_gain;
            int   my_comm = best_comm;
            for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
                const float ogain = __shfl_down_sync(0xffffffffu, my_gain, off);
                const int   ocomm = __shfl_down_sync(0xffffffffu, my_comm, off);
                if (ogain > my_gain) {
                    my_gain = ogain;
                    my_comm = ocomm;
                }
            }
            if (threadIdx.x == 0) {
                s_proposed = (my_gain > min_delta_q) ? my_comm : c_old;
            }
        }
    } else {
        smem_f[threadIdx.x] = best_gain;
        smem_i[threadIdx.x] = best_comm;
        __syncthreads();
        for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                if (smem_f[threadIdx.x + s] > smem_f[threadIdx.x]) {
                    smem_f[threadIdx.x] = smem_f[threadIdx.x + s];
                    smem_i[threadIdx.x] = smem_i[threadIdx.x + s];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            s_proposed = (smem_f[0] > min_delta_q) ? smem_i[0] : c_old;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) proposed_comm[u] = s_proposed;
}


// =========================================================================
// KERNEL 2: apply_moves
//
// One thread per node.  Atomically applies proposed_comm[u] to community[u]
// and rebalances comm_degree_sum.  Increments improvement_flag whenever a
// real move happens; the host loops Phase 1 until this counter stays zero.
//
// Conflict resolution: if A proposes comm[B] and B proposes comm[A], both
// moves apply (A swaps to comm_B, B swaps to comm_A; the comm_degree_sum
// updates remain balanced).  This is the documented parallel Louvain
// non-determinism.
// =========================================================================
__global__ void apply_moves(
    int*       __restrict__ community,
    const int* __restrict__ proposed_comm,
    int*       __restrict__ improvement_flag,
    const float* __restrict__ node_degree,
    float*     __restrict__ comm_degree_sum,
    const int                n)
{
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;

    const int old_c = community[u];
    const int new_c = proposed_comm[u];
    if (new_c == old_c) return;

    community[u] = new_c;
    atomicAdd(improvement_flag, 1);

    const float k_u = node_degree[u];
    atomicAdd(&comm_degree_sum[old_c], -k_u);
    atomicAdd(&comm_degree_sum[new_c],  k_u);
}


// =========================================================================
// KERNEL 3: count_community_edges
//
// Phase 2 step 1.  One thread per edge: emit a (comm_src, comm_dst, weight)
// triple.  edge_src[e] is the source-node mapping, precomputed CPU-side as
// np.repeat(arange(n), diff(indptr)).
// =========================================================================
__global__ void count_community_edges(
    const int*   __restrict__ edge_src,
    const int*   __restrict__ col_idx,
    const float* __restrict__ edge_wt,
    const int*   __restrict__ community,
    int*         __restrict__ out_csrc,
    int*         __restrict__ out_cdst,
    float*       __restrict__ out_cwt,
    const int                  nnz)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nnz) return;
    const int u = edge_src[e];
    const int v = col_idx[e];
    out_csrc[e] = community[u];
    out_cdst[e] = community[v];
    out_cwt[e]  = edge_wt[e];
}


// =========================================================================
// KERNEL 4: compute_modularity_partial
//
// One thread per edge.  Accumulates ( w − gamma · k_i · k_j · inv_2m ) for
// each edge whose endpoints share a community.  Block-reduces in shared
// memory; host sums the per-block partials and multiplies by inv_2m for
// the final Q.
// =========================================================================
__global__ void compute_modularity_partial(
    const int*   __restrict__ edge_src,
    const int*   __restrict__ col_idx,
    const float* __restrict__ edge_wt,
    const int*   __restrict__ community,
    const float* __restrict__ node_degree,
    float*       __restrict__ partial_Q,
    const float                inv_2m,
    const float                resolution,
    const int                  nnz)
{
    __shared__ float smem[BLOCK_SIZE];
    const int e = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (e < nnz) {
        const int u = edge_src[e];
        const int v = col_idx[e];
        if (community[u] == community[v]) {
            const float w  = edge_wt[e];
            const float ki = node_degree[u];
            const float kj = node_degree[v];
            val = w - resolution * ki * kj * inv_2m;
        }
    }
    smem[threadIdx.x] = val;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial_Q[blockIdx.x] = smem[0];
}

}  // extern "C"
"""


# ---------------------------------------------------------------------------
# Module-level kernel cache
# ---------------------------------------------------------------------------

_kernel_cache: dict[str, dict[str, Any]] = {}


def _get_kernels() -> dict[str, Any]:
    """Compile (or fetch from cache) the four Louvain device kernels."""
    if "louvain" not in _kernel_cache:
        if not PYCUDA_AVAILABLE:
            raise RuntimeError(
                "PyCUDA is required to compile Louvain kernels — "
                "install pycuda and ensure NVCC is on PATH."
            )
        mod = SourceModule(
            KERNEL_SOURCE,
            options=["-arch=sm_75"],        # RTX 20-series Turing
            no_extern_c=True,
        )
        _kernel_cache["louvain"] = {
            "proposed_moves": mod.get_function("compute_proposed_moves"),
            "apply_moves":    mod.get_function("apply_moves"),
            "count_edges":    mod.get_function("count_community_edges"),
            "modularity":     mod.get_function("compute_modularity_partial"),
        }
    return _kernel_cache["louvain"]


# ---------------------------------------------------------------------------
# Parameter merging
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# CPU preprocessing helpers
# ---------------------------------------------------------------------------

def _symmetrize(
    graph_csr: sp.csr_matrix,
    network_type: str,
) -> tuple[sp.csr_matrix, str]:
    """Convert the graph to an undirected weighted CSR for Louvain.

    Louvain requires an undirected graph — modularity Q is undefined for
    directed edges (the null model k_i·k_j / 2m assumes undirected degree).

    Behaviour by network type
    -------------------------
    GRN    : A_sym = A + Aᵀ.  Mutual TF↔gene edges get weight 2 (stronger
             co-regulation).  One-way TF→gene edges keep their original
             weight.
    PPI    : graph used as-is (already undirected; symmetrising would
             double every edge weight).
    miRNA  : A_sym = A + Aᵀ (bipartite → mutual reads as weight 2).

    Returns
    -------
    (csr_float32, note_string)
    """
    nt = str(network_type).lower()
    if nt == "ppi":
        A = graph_csr.astype(np.float32).tocsr()
        A.sum_duplicates()
        return A, "PPI: graph used as-is (already undirected)"
    # grn / mirna / anything else → symmetrise
    A = (graph_csr + graph_csr.T).astype(np.float32).tocsr()
    A.sum_duplicates()
    return A, f"{nt.upper()}: A + Aᵀ applied (mutual edges weight 2)"


def _remove_self_loops(csr: sp.csr_matrix) -> sp.csr_matrix:
    """Zero the diagonal and drop the now-explicit zeros.

    Self-loops on the *original* graph are not biologically meaningful for
    Louvain at level 0.  They will be regenerated correctly at higher
    levels as collapsed intra-community weight by :func:`_build_coarsened_graph`.
    """
    csr = csr.copy()
    csr.setdiag(0)
    csr.eliminate_zeros()
    return csr


def _handle_isolated_nodes(
    csr: sp.csr_matrix,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Identify zero-degree nodes.

    The CSR is returned unchanged; isolated nodes naturally remain in
    singleton communities (``compute_proposed_moves`` returns
    ``proposed = current`` when ``degree == 0``).  The list is returned
    purely for downstream reporting / re-integration logic.
    """
    degrees = np.asarray(csr.sum(axis=1), dtype=np.float64).flatten()
    isolated = np.where(degrees == 0)[0].astype(np.int32)
    return csr, isolated


def _normalize_weights(csr: sp.csr_matrix) -> sp.csr_matrix:
    """Rescale edge weights so max(w) <= 1.

    Modularity Q is invariant under uniform weight scaling, so this is
    safe and protects against float32 overflow in dot-product sums for
    dense or high-weight graphs (e.g. miRNA expression-correlation
    networks).
    """
    if csr.nnz == 0:
        return csr
    max_w = float(csr.data.max())
    if max_w <= 0.0 or math.isclose(max_w, 1.0):
        return csr
    csr = csr.copy()
    csr.data = (csr.data / max_w).astype(np.float32)
    return csr


# ---------------------------------------------------------------------------
# Phase 1 GPU helper — run one Louvain level + modularity
# ---------------------------------------------------------------------------

def _louvain_level(
    csr: sp.csr_matrix,
    kernels: dict[str, Any],
    min_delta_q: float,
    resolution: float,
    max_phase1_passes: int,
    block_size: int,
    stream_compute,
    stream_transfer,
) -> tuple[np.ndarray, float]:
    """Run Phase 1 on the GPU until convergence (or pass cap) and compute Q.

    Returns
    -------
    (community_assignment, modularity_Q)

    The community assignment is NOT renumbered to 0..K-1 here — that is
    done in :func:`louvain_gpu` before passing to :func:`_build_coarsened_graph`.
    """
    n   = int(csr.shape[0])
    nnz = int(csr.nnz)
    if n == 0:
        return np.zeros(0, dtype=np.int32), 0.0

    # ---- Host arrays --------------------------------------------------
    row_ptr_h  = np.ascontiguousarray(csr.indptr,  dtype=np.int32)
    col_idx_h  = np.ascontiguousarray(csr.indices, dtype=np.int32)
    edge_wt_h  = np.ascontiguousarray(csr.data,    dtype=np.float32)
    degree_h   = np.asarray(csr.sum(axis=1), dtype=np.float32).flatten()
    edge_src_h = np.repeat(
        np.arange(n, dtype=np.int32), np.diff(row_ptr_h)
    ).astype(np.int32)

    total_weight = float(degree_h.sum())   # = 2m for symmetric CSR
    if total_weight <= 0.0:
        return np.arange(n, dtype=np.int32), 0.0
    inv_2m = 1.0 / total_weight            # = 1 / (2m)

    community_h    = np.arange(n, dtype=np.int32)
    comm_degsum_h  = degree_h.copy()       # each node alone → degsum = degree

    # ---- Device allocation -------------------------------------------
    d_buffers: list = []

    def _to_gpu(arr: np.ndarray):
        ga = gpuarray.to_gpu_async(arr, stream=stream_transfer)
        d_buffers.append(ga)
        return ga

    def _empty(shape, dtype):
        ga = gpuarray.empty(shape, dtype=dtype)
        d_buffers.append(ga)
        return ga

    d_row_ptr    = _to_gpu(row_ptr_h)
    d_col_idx    = _to_gpu(col_idx_h)
    d_edge_wt    = _to_gpu(edge_wt_h)
    d_edge_src   = _to_gpu(edge_src_h)
    d_degree     = _to_gpu(degree_h)
    d_community  = _to_gpu(community_h)
    d_comm_degs  = _to_gpu(comm_degsum_h)
    d_proposed   = _empty((n,), np.int32)
    d_improve    = _empty((1,), np.int32)

    mod_blocks   = max(1, (nnz + BLOCK_SIZE - 1) // BLOCK_SIZE)
    d_partial_Q  = _empty((mod_blocks,), np.float32)

    stream_transfer.synchronize()

    try:
        k_prop  = kernels["proposed_moves"]
        k_apply = kernels["apply_moves"]
        k_mod   = kernels["modularity"]

        bs_apply = int(block_size)
        if bs_apply <= 0 or bs_apply > 1024:
            bs_apply = BLOCK_SIZE
        grid_apply = ((n + bs_apply - 1) // bs_apply, 1, 1)
        grid_n     = (n, 1, 1)            # one block per node for Phase 1
        block_256  = (BLOCK_SIZE, 1, 1)

        # ---- Phase 1 iterative loop -----------------------------------
        for _pass in range(max_phase1_passes):
            cuda.memset_d32(d_improve.gpudata, 0, 1)

            # Compute proposals (block per node, three-tier dispatch)
            k_prop(
                d_row_ptr, d_col_idx, d_edge_wt,
                d_community, d_comm_degs, d_degree,
                d_proposed,
                np.float32(min_delta_q),
                np.float32(inv_2m),
                np.float32(resolution),
                np.int32(n),
                block=block_256, grid=grid_n, stream=stream_compute,
            )

            # Apply proposals in batch (one thread per node)
            k_apply(
                d_community, d_proposed, d_improve,
                d_degree, d_comm_degs, np.int32(n),
                block=(bs_apply, 1, 1), grid=grid_apply,
                stream=stream_compute,
            )

            stream_compute.synchronize()
            if int(d_improve.get()[0]) == 0:
                break

        # ---- Modularity (one thread per edge) -------------------------
        mod_grid = (mod_blocks, 1, 1)
        k_mod(
            d_edge_src, d_col_idx, d_edge_wt,
            d_community, d_degree, d_partial_Q,
            np.float32(inv_2m), np.float32(resolution),
            np.int32(nnz),
            block=block_256, grid=mod_grid, stream=stream_compute,
        )
        stream_compute.synchronize()

        modularity = float(inv_2m * float(np.sum(d_partial_Q.get())))
        community_out = d_community.get().astype(np.int32)
        return community_out, modularity

    finally:
        for arr in d_buffers:
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Phase 2 — build the coarsened graph for the next level
# ---------------------------------------------------------------------------

def _build_coarsened_graph(
    csr: sp.csr_matrix,
    community: np.ndarray,        # already renumbered to 0..K-1
    K: int,
    kernels: dict[str, Any],
    stream_compute,
    block_size: int,
) -> sp.csr_matrix:
    """Collapse the graph into a (K × K) weighted super-node CSR.

    Implementation (mixed GPU + CPU)
    --------------------------------
      1. GPU: count_community_edges  → (csrc, cdst, cwt) triples
      2. D2H copy
      3. CPU: np.lexsort by (csrc, cdst)         (TODO: GPU radix sort)
      4. CPU: segment-head detect + np.add.reduceat to sum weights
      5. CPU: assemble scipy.sparse.csr_matrix on (K × K) shape

    Self-loops are KEPT — they encode intra-community edge weight, which
    contributes to the modularity null model at higher levels.
    """
    n   = int(csr.shape[0])
    nnz = int(csr.nnz)
    if nnz == 0:
        return sp.csr_matrix((K, K), dtype=np.float32)

    row_ptr_h  = np.ascontiguousarray(csr.indptr,  dtype=np.int32)
    col_idx_h  = np.ascontiguousarray(csr.indices, dtype=np.int32)
    edge_wt_h  = np.ascontiguousarray(csr.data,    dtype=np.float32)
    edge_src_h = np.repeat(
        np.arange(n, dtype=np.int32), np.diff(row_ptr_h)
    ).astype(np.int32)

    d_local: list = []

    def _to_gpu(arr: np.ndarray):
        ga = gpuarray.to_gpu(arr)
        d_local.append(ga)
        return ga

    def _empty(shape, dtype):
        ga = gpuarray.empty(shape, dtype=dtype)
        d_local.append(ga)
        return ga

    try:
        d_edge_src  = _to_gpu(edge_src_h)
        d_col_idx   = _to_gpu(col_idx_h)
        d_edge_wt   = _to_gpu(edge_wt_h)
        d_community = _to_gpu(community.astype(np.int32))
        d_csrc      = _empty((nnz,), np.int32)
        d_cdst      = _empty((nnz,), np.int32)
        d_cwt       = _empty((nnz,), np.float32)

        bs = int(block_size)
        if bs <= 0 or bs > 1024:
            bs = BLOCK_SIZE
        grid = ((nnz + bs - 1) // bs, 1, 1)

        k_count = kernels["count_edges"]
        k_count(
            d_edge_src, d_col_idx, d_edge_wt, d_community,
            d_csrc, d_cdst, d_cwt, np.int32(nnz),
            block=(bs, 1, 1), grid=grid, stream=stream_compute,
        )
        stream_compute.synchronize()

        csrc_h = d_csrc.get()
        cdst_h = d_cdst.get()
        cwt_h  = d_cwt.get()
    finally:
        for arr in d_local:
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass

    # ---- CPU sort + reduce-by-key + assemble new CSR --------------------
    # TODO(optimisation): replace lexsort with a GPU radix sort (CUB
    # DeviceRadixSort via cupy / pycuda-cub bindings) once the dependency
    # cost is justified.  Current correctness-first path: numpy lexsort.
    order  = np.lexsort((cdst_h, csrc_h))
    csrc_h = csrc_h[order]
    cdst_h = cdst_h[order]
    cwt_h  = cwt_h[order]

    # Segment heads: positions where (csrc, cdst) changes vs. previous.
    if nnz == 1:
        seg_starts = np.array([0], dtype=np.int64)
    else:
        change = (csrc_h[1:] != csrc_h[:-1]) | (cdst_h[1:] != cdst_h[:-1])
        seg_starts = np.concatenate(([0], np.flatnonzero(change) + 1)).astype(np.int64)

    new_csrc = csrc_h[seg_starts]
    new_cdst = cdst_h[seg_starts]
    new_cwt  = np.add.reduceat(cwt_h, seg_starts).astype(np.float32)

    new_csr = sp.csr_matrix(
        (new_cwt, (new_csrc, new_cdst)),
        shape=(K, K),
        dtype=np.float32,
    )
    # sum_duplicates is a no-op here (reduce_by-segment already did it) but
    # it canonicalises the CSR layout for downstream calls.
    new_csr.sum_duplicates()
    return new_csr


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def louvain_gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """Louvain — GPU-accelerated via custom PyCUDA kernels.

    No CuPy dependency.  All four kernels (compiled once, cached) are
    launched from a single PyCUDA driver-API context retained via
    :py:meth:`Device.retain_primary_context` so the implementation
    coexists cleanly with any CuPy code running in the same process.

    Pipeline
    --------
    CPU preprocessing (NOT timed):
      _symmetrize          — network-type-aware undirected conversion
      _remove_self_loops   — drop diagonal
      _handle_isolated_nodes
      _normalize_weights   — divide by max (Q invariant under scaling)

    GPU loop (timed):
      For each level (up to max_levels):
        _louvain_level         — Phase 1 + modularity
        _build_coarsened_graph — Phase 2 (mixed GPU + CPU)
        break when no merging occurs

    Returns
    -------
    dict — see CLAUDE.md "Louvain" result spec (outer envelope +
    ``result`` sub-dict with community_assignments, num_communities,
    modularity, top_communities, hierarchy, note).

    Raises
    ------
    RuntimeError
        If PyCUDA is unavailable or no CUDA device can be initialised.
    MemoryError
        If GPU allocation fails.
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA is required for louvain_gpu(). "
            "Install it or use louvain_cpu_single() from "
            "src/algorithms/cpu/single_threaded/louvain.py"
        )

    # Push the device PRIMARY context unconditionally — same rationale as
    # hits.py: `_ensure_cuda_context()` activates the CuPy runtime-API
    # context, which PyCUDA's `cuModuleLoadDataEx` does not recognise as
    # current on the driver-API stack.  retain_primary_context() is
    # ref-counted and shared with CuPy under the hood.
    pushed_ctx = None
    try:
        cuda.init()
        if cuda.Device.count() <= 0:
            raise RuntimeError("No CUDA device available")
        pushed_ctx = cuda.Device(0).retain_primary_context()
        pushed_ctx.push()
    except cuda.LogicError as e:
        raise RuntimeError(f"CUDA initialisation failed: {e}") from e

    try:
        # ---- Parameter merging ----------------------------------------
        p = _merge_params(params)
        if _GPU_CONFIG_AVAILABLE:
            p = apply_config("louvain", graph_csr, p) or p
            for k, v in _DEFAULT_PARAMS.items():
                p.setdefault(k, v)

        min_delta_q       = float(p["min_delta_q"])
        max_levels        = int(p["max_levels"])
        resolution        = float(p["resolution"])
        max_phase1_passes = int(p["max_phase1_passes"])
        network_type      = str(p.get("network_type", "grn"))
        block_size        = int(p.get("block_size", BLOCK_SIZE))
        if block_size <= 0 or block_size > 1024:
            block_size = BLOCK_SIZE

        n_original = int(graph_csr.shape[0])
        if n_original == 0:
            raise ValueError("Empty graph")

        # ---- CPU preprocessing ----------------------------------------
        A_sym, sym_note = _symmetrize(graph_csr, network_type)
        A_sym           = _remove_self_loops(A_sym)
        A_sym, _isolated = _handle_isolated_nodes(A_sym)
        A_sym           = _normalize_weights(A_sym)

        # Ensure float32 throughout the GPU pipeline.
        if A_sym.dtype != np.float32:
            A_sym = A_sym.astype(np.float32)

        kernels = _get_kernels()

        # ---- Streams + timing ------------------------------------------
        stream_compute  = cuda.Stream()
        stream_transfer = cuda.Stream()
        start_event     = cuda.Event()
        end_event       = cuda.Event()
        start_event.record(stream_compute)

        # ---- Hierarchy loop -------------------------------------------
        current          = A_sym
        global_community = np.arange(n_original, dtype=np.int32)
        hierarchy: list[list[int]] = []
        final_modularity = 0.0

        for _level in range(max_levels):
            n_cur = int(current.shape[0])

            # Phase 1 + modularity for this level
            level_community, level_modularity = _louvain_level(
                current, kernels,
                min_delta_q=min_delta_q,
                resolution=resolution,
                max_phase1_passes=max_phase1_passes,
                block_size=block_size,
                stream_compute=stream_compute,
                stream_transfer=stream_transfer,
            )
            final_modularity = level_modularity

            # Renumber communities to 0..K-1 (compact label space)
            _, renum = np.unique(level_community, return_inverse=True)
            level_renum = renum.astype(np.int32)
            K = int(level_renum.max()) + 1 if level_renum.size else 0

            # Update global mapping: each original node now points to its
            # super-node at this level.
            global_community = level_renum[global_community]
            hierarchy.append(global_community.astype(np.int32).tolist())

            # Convergence: no merging occurred at this level.
            if K >= n_cur:
                break

            # Phase 2: build coarsened CSR for the next level.
            current = _build_coarsened_graph(
                current, level_renum, K, kernels,
                stream_compute=stream_compute,
                block_size=block_size,
            )

            # Defensive: if coarsening produced nothing useful, stop.
            if current.shape[0] == 0 or current.shape[0] == n_cur:
                break

        end_event.record(stream_compute)
        end_event.synchronize()
        elapsed = start_event.time_till(end_event) / 1000.0   # ms → s

        # ---- Result construction (NOT timed) --------------------------
        # Compact label space for the final assignment.
        _, final_labels = np.unique(global_community, return_inverse=True)
        final_labels    = final_labels.astype(np.int32)
        K_final         = int(final_labels.max()) + 1 if final_labels.size else 0

        sizes      = np.bincount(final_labels, minlength=K_final)
        top_idx    = np.argsort(sizes)[::-1][:_TOP_K]
        top_communities = [
            {
                "community_id": int(c),
                "size":         int(sizes[c]),
                "member_nodes": np.where(final_labels == c)[0].tolist(),
            }
            for c in top_idx if sizes[c] > 0
        ]

        note = (
            f"Graph symmetrised for Louvain ({sym_note}). "
            "Parallel Phase 1 updates may produce a different partition "
            "than serial Louvain — expected behaviour, not an error."
        )

        return {
            "algorithm":      "louvain",
            "mode":           "gpu",
            "network_type":   network_type,
            "execution_time": elapsed,
            "num_nodes":      n_original,
            "num_edges":      int(graph_csr.nnz),
            "result": {
                "community_assignments": final_labels.tolist(),
                "num_communities":       int(K_final),
                "modularity":            float(final_modularity),
                "top_communities":       top_communities,
                "hierarchy":             hierarchy,
                "note":                  note,
            },
        }

    except cuda.LogicError as e:
        logging.error("CUDA error in louvain_gpu: %s", e)
        raise
    except MemoryError:
        logging.warning(
            "VRAM exhausted in louvain_gpu. "
            "Try a smaller graph or a higher-VRAM device."
        )
        raise
    finally:
        if pushed_ctx is not None:
            try:
                pushed_ctx.pop()
            except Exception:                           # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _gpu(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    """Runner entry point — preserves the legacy ``{output, extra_params}``
    shape required by ``src.benchmarking.benchmark`` and
    ``src.runner.algorithm_runner``.
    """
    p = _merge_params(params)
    full = louvain_gpu(graph_csr, p)
    # `full` already has the outer envelope; the benchmark runner only
    # consumes "output" and "extra_params" — keep both layers available.
    return {"output": full, "extra_params": p}
