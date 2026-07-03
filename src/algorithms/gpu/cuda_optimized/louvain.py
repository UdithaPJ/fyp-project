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

    Q = (1 / 2m) * sum_{i,j} ( A_ij - gamma * k_i * k_j / (2m) ) * delta(c_i, c_j)

The null model k_i*k_j / (2m) assumes undirected degree.  For GRN / miRNA
networks this module applies A <- A + A^T internally (mutual edges keep
their summed weight = 2 * original, encoding tighter coupling).  PPI
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
    For each node u, compute dQ for moving u to each neighbouring
    community.  Propose the best move.  Apply all proposals as a batch.
    Repeat until no node moves, or ``max_phase1_passes`` reached.

Phase 2 — graph coarsening (mixed GPU + CPU):
    Each community becomes a super-node; inter-community edges are
    summed.  Self-loops on super-nodes carry intra-community weight.
    Recurse Phase 1 on the coarsened graph.

Parameter guide
---------------
min_delta_q  (float, default 1e-4)  Minimum dQ to accept a move.
max_levels   (int,   default 10)    Maximum Phase 1+2 recursion depth.
resolution   (float, default 1.0)   gamma in the modularity formula.
                                    > 1.0 -> more, smaller communities.
                                    < 1.0 -> fewer, larger communities.
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
#   compute_proposed_moves     — Phase 1: three-tier degree-aware dQ scan
#                                + block-wide best-move reduction.
#                                Writes proposals only; never touches the
#                                community array directly.
#   apply_moves                — Phase 1: batch-apply all proposals.
#                                Updates community + comm_degree_sum via
#                                atomicAdd.
#   count_community_edges      — Phase 2: edge-parallel mapping
#                                (u, v, w) -> (comm[u], comm[v], w).
#   compute_modularity_partial — Final Q: per-block partial sums of
#                                A_ij - gamma*k_i*k_j/(2m) over same-
#                                community edges.
#
# Phase 2 sort + reduce now runs on the GPU.  CuPy path (preferred):
# cp.argsort + cp.add.reduceat fully on the device.  Fallback path
# (no CuPy): GPU-side compound sort keys + CPU argsort of the keys +
# GPU gather_by_index + GPU segmented_weight_reduce.  Both paths
# eliminate the bulk PCIe transfer of edge triples.
#
# Compilation: adaptive (_detect_arch_flag).  Queries
# cuda.Device(0).compute_capability() at runtime; falls back to
# -arch=sm_75 on probe failure.  Does NOT silently fall back to CPU;
# raises RuntimeError / MemoryError / cuda.LogicError so the runner
# can surface the failure.
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

# Optional CuPy for Phase 2 GPU-side sort/reduce.  When available, Phase 2
# coarsening uses ``cp.argsort`` + ``cp.add.reduceat`` directly on the
# device — no PCIe round trip during sort.  Falls back to the
# CPU-numpy hybrid path if CuPy is missing.
try:
    import cupy as _cp                                  # type: ignore
    CUPY_SORT_AVAILABLE = True
except Exception:                                       # noqa: BLE001
    _cp = None                                          # type: ignore[assignment]
    CUPY_SORT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "min_delta_q":          1e-4,
    "max_levels":           10,
    "resolution":           1.0,
    # Maximum Phase 1 passes per Louvain level.  Bulk-synchronous parallel
    # Phase 1 can oscillate (two adjacent nodes swapping communities in
    # alternating passes), so the cap is the primary termination guarantee.
    "max_phase1_passes":    100,
    "network_type":         "grn",
    "block_size":           256,
    # Community freezing: a node unchanged for ``freeze_threshold`` consecutive
    # passes is skipped in subsequent passes.
    "freeze_threshold":     3,
    # Early termination: stop Phase 1 when fewer than this fraction of nodes
    # moved in the previous pass.
    "early_stop_fraction":  0.01,
    # Force chunked Phase 1 even if VRAM is sufficient (mainly for testing).
    "use_chunking":         False,
}

BLOCK_SIZE: int       = 256
WARP_SIZE: int        = 32
VRAM_SAFETY: float    = 0.80    # warn if working set > 80 % of free VRAM
_TOP_K: int           = 5
SMEM_HASH_SIZE: int   = 64       # buckets per block for community accumulator


# ---------------------------------------------------------------------------
# CUDA kernel source (all four kernels, single SourceModule)
# ---------------------------------------------------------------------------

KERNEL_SOURCE = r"""
extern "C" {

#define BLOCK_SIZE      256
#define WARP_SIZE       32
#define SMEM_HASH_SIZE  64    // per-block community-weight hash table

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
//   Pass 1 -- accumulate k_self = sum w_uv over edges where comm[v] == comm[u].
//            Reduce across active threads; broadcast via shared memory.
//   Pass 2 -- each active thread evaluates per-edge gain for moving u to
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
// The kernel writes to proposed_comm only -- community[] is never modified
// here.  apply_moves performs the batch update separately.
// =========================================================================
__global__ void compute_proposed_moves(
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ edge_wt,
    const int*   __restrict__ community,
    const float* __restrict__ comm_degree_sum,
    const float* __restrict__ node_degree,
    const int*   __restrict__ frozen,         // length n; 1=skip, 0=process. May be NULL.
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
    // Per-block community-weight hash table.  Pass 2 accumulates weights
    // into this table keyed by neighbour community; thread 0 then walks
    // the table to pick the winning community.  Collisions saturate the
    // probe chain (drop), matching documented non-determinism.
    __shared__ int   smem_comm_keys[SMEM_HASH_SIZE];
    __shared__ float smem_comm_wts[SMEM_HASH_SIZE];

    const int u = blockIdx.x;
    if (u >= n) return;

    // Frozen-node skip: keep current community unchanged.
    if (frozen != 0 && frozen[u] != 0) {
        if (threadIdx.x == 0) proposed_comm[u] = community[u];
        return;
    }

    // ---- Initialise SMEM hash table -------------------------------------
    for (int b = threadIdx.x; b < SMEM_HASH_SIZE; b += BLOCK_SIZE) {
        smem_comm_keys[b] = -1;
        smem_comm_wts[b]  = 0.0f;
    }
    __syncthreads();

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
    // PASS 1 -- accumulate k_self = sum w_uv where comm[v] == c_old.
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
    // PASS 2 -- accumulate per-community edge weights into the SMEM hash
    // table.  Each thread visits its assigned subset of u's neighbours
    // and atomically merges (community -> weight) entries.  Linear probe
    // with bounded chain length (= SMEM_HASH_SIZE); saturation drops the
    // entry (documented non-determinism).
    // =================================================================
    if (t_start >= 0) {
        for (int off = t_start; off < degree; off += t_stride) {
            const int v   = col_idx[row_start + off];
            const int c_v = community[v];
            if (c_v == c_old) continue;          // moving to current = no-op
            const float w = edge_wt[row_start + off];

            int bucket = c_v & (SMEM_HASH_SIZE - 1);
            for (int probe = 0; probe < SMEM_HASH_SIZE; ++probe) {
                const int old_key = atomicCAS(&smem_comm_keys[bucket], -1, c_v);
                if (old_key == -1 || old_key == c_v) {
                    atomicAdd(&smem_comm_wts[bucket], w);
                    break;
                }
                bucket = (bucket + 1) & (SMEM_HASH_SIZE - 1);
            }
        }
    }
    __syncthreads();

    // -------------------------------------------------------------------
    // Hash walk -- thread 0 evaluates deltaQ once per unique neighbour
    // community (<= SMEM_HASH_SIZE entries).  Eliminates redundant gain
    // recomputation for hub nodes with many same-community neighbours.
    // -------------------------------------------------------------------
    if (threadIdx.x == 0) {
        float best_gain = 0.0f;
        int   best_comm = c_old;
        const float leave = s_leave;
        for (int b = 0; b < SMEM_HASH_SIZE; ++b) {
            const int nc = smem_comm_keys[b];
            if (nc < 0 || nc == c_old) continue;
            const float k_u_in    = smem_comm_wts[b];
            const float sigma_new = comm_degree_sum[nc];
            const float join      = 2.0f * inv_2m * k_u_in
                                    - 2.0f * resolution * k_u * sigma_new
                                      * inv_2m * inv_2m;
            const float gain = join - leave;
            if (gain > best_gain) {
                best_gain = gain;
                best_comm = nc;
            }
        }
        s_proposed = (best_gain > min_delta_q) ? best_comm : c_old;
        proposed_comm[u] = s_proposed;
    }
}


// =========================================================================
// KERNEL 2: apply_moves
//
// One thread per node.  Atomically applies proposed_comm[u] to community[u]
// and rebalances comm_degree_sum.  Increments move_counter for every node
// that actually moves; the host compares the counter against an early-stop
// threshold (fraction of n) to decide whether Phase 1 has converged.
// =========================================================================
__global__ void apply_moves(
    int*         __restrict__ community,
    const int*   __restrict__ proposed_comm,
    int*         __restrict__ move_counter,
    const float* __restrict__ node_degree,
    float*       __restrict__ comm_degree_sum,
    const int                  n)
{
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;

    const int old_c = community[u];
    const int new_c = proposed_comm[u];
    if (new_c == old_c) return;

    community[u] = new_c;
    atomicAdd(move_counter, 1);

    const float k_u = node_degree[u];
    atomicAdd(&comm_degree_sum[old_c], -k_u);
    atomicAdd(&comm_degree_sum[new_c],  k_u);
}


// =========================================================================
// KERNEL 2b: update_freeze_status  (Improvement 3 -- community freezing)
//
// One thread per node.  If a node remained in the same community as the
// previous pass, increment its freeze counter; once the counter reaches
// freeze_threshold the node is marked frozen and will be skipped by
// compute_proposed_moves on subsequent passes.  Any move resets the
// counter and clears the frozen flag.
// =========================================================================
__global__ void update_freeze_status(
    const int* __restrict__ community,
    const int* __restrict__ prev_community,
    int*       __restrict__ freeze_counter,
    int*       __restrict__ frozen,
    const int                freeze_threshold,
    const int                n)
{
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;

    if (community[u] == prev_community[u]) {
        const int c = freeze_counter[u] + 1;
        freeze_counter[u] = c;
        if (c >= freeze_threshold) {
            frozen[u] = 1;
        }
    } else {
        freeze_counter[u] = 0;
        frozen[u] = 0;
    }
}


// =========================================================================
// KERNEL 2c: gather_by_index  (Improvement 1 -- Phase 2 GPU reorder)
//
// Stream-gather of (src, dst, wt) edge triples through an indirection
// array.  Used as the GPU half of the fallback Phase-2 coarsening path
// when CuPy is not available (host computes indices via argsort,
// device applies the permutation).
// =========================================================================
__global__ void gather_by_index(
    const int*   __restrict__ src_in,
    const int*   __restrict__ dst_in,
    const float* __restrict__ wt_in,
    int*         __restrict__ src_out,
    int*         __restrict__ dst_out,
    float*       __restrict__ wt_out,
    const int*   __restrict__ indices,
    const int                  num_edges)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    const int idx = indices[e];
    src_out[e] = src_in[idx];
    dst_out[e] = dst_in[idx];
    wt_out[e]  = wt_in[idx];
}


// =========================================================================
// KERNEL 2d: segmented_weight_reduce  (Improvement 1 -- Phase 2 GPU reduce)
//
// One thread per sorted (src, dst, wt) edge.  Segment heads -- edges whose
// (src, dst) differs from the previous one -- claim a slot in the output
// arrays via an atomic counter and forward-scan the segment to accumulate
// the weight sum.  Self-loops (src == dst) are skipped because they do
// not contribute to higher-level modularity beyond the intra-community
// weight already encoded by collapsed edges.
//
// Correctness: forward scan has variable work per thread (skewed segments
// dominate one thread).  A production version should use a parallel
// segmented scan (e.g. cub::DeviceSegmentedReduce); the current path is
// correctness-first and still avoids the PCIe round trip of CPU reduce.
// =========================================================================
__global__ void segmented_weight_reduce(
    const int*   __restrict__ src_sorted,
    const int*   __restrict__ dst_sorted,
    const float* __restrict__ wt_sorted,
    int*         __restrict__ out_src,
    int*         __restrict__ out_dst,
    float*       __restrict__ out_wt,
    int*         __restrict__ out_count,
    const int                  num_edges,
    const int                  drop_self_loops)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_edges) return;

    const int s = src_sorted[tid];
    const int d = dst_sorted[tid];

    const bool is_head =
        (tid == 0) ||
        (src_sorted[tid] != src_sorted[tid - 1]) ||
        (dst_sorted[tid] != dst_sorted[tid - 1]);
    if (!is_head) return;

    if (drop_self_loops && s == d) return;

    // Forward scan the segment.
    float sum = 0.0f;
    int j = tid;
    while (j < num_edges
           && src_sorted[j] == s
           && dst_sorted[j] == d) {
        sum += wt_sorted[j];
        ++j;
    }

    const int pos = atomicAdd(out_count, 1);
    out_src[pos] = s;
    out_dst[pos] = d;
    out_wt[pos]  = sum;
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
// One thread per edge.  Accumulates ( w - gamma * k_i * k_j * inv_2m ) for
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


def _detect_arch_flag() -> tuple[str, tuple[int, int]]:
    """Return (``-arch=sm_XY``, (cc_major, cc_minor)) for the current device.

    Falls back to ``sm_75`` (RTX 20-series) if PyCUDA cannot probe the
    device — that matches the project's target hardware.
    """
    try:
        cuda.init()
        cc_major, cc_minor = cuda.Device(0).compute_capability()
        return f"-arch=sm_{cc_major}{cc_minor}", (int(cc_major), int(cc_minor))
    except Exception:                                   # noqa: BLE001
        return "-arch=sm_75", (7, 5)


def _get_kernels() -> dict[str, Any]:
    """Compile (or fetch from cache) all Louvain device kernels.

    Adaptive arch detection: queries the device's compute capability at
    runtime and builds with ``-arch=sm_<major><minor>``.  Enables
    ``-use_fast_math`` on Ampere+ and disables cooperative-groups
    features on pre-Volta devices.
    """
    if "louvain" not in _kernel_cache:
        if not PYCUDA_AVAILABLE:
            raise RuntimeError(
                "PyCUDA is required to compile Louvain kernels — "
                "install pycuda and ensure NVCC is on PATH."
            )
        arch_flag, (cc_major, cc_minor) = _detect_arch_flag()
        options = [arch_flag, "-O3"]
        if cc_major >= 8:
            options.append("-use_fast_math")
        if cc_major < 7:
            options.append("-DDISABLE_COOPERATIVE_GROUPS")

        mod = SourceModule(
            KERNEL_SOURCE,
            options=options,
            no_extern_c=True,
        )
        _kernel_cache["louvain"] = {
            "proposed_moves": mod.get_function("compute_proposed_moves"),
            "apply_moves":    mod.get_function("apply_moves"),
            "count_edges":    mod.get_function("count_community_edges"),
            "modularity":     mod.get_function("compute_modularity_partial"),
            "freeze":         mod.get_function("update_freeze_status"),
            "gather":         mod.get_function("gather_by_index"),
            "seg_reduce":     mod.get_function("segmented_weight_reduce"),
            "_arch_flag":     arch_flag,
            "_compute_capability": (cc_major, cc_minor),
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

    Louvain requires an undirected graph - modularity Q is undefined for
    directed edges (the null model k_i*k_j / (2m) assumes undirected degree).

    Behaviour by network type
    -------------------------
    GRN    : A_sym = A + A^T.  Mutual TF<->gene edges get weight 2 (stronger
             co-regulation).  One-way TF->gene edges keep their original
             weight.
    PPI    : graph used as-is (already undirected; symmetrising would
             double every edge weight).
    miRNA  : A_sym = A + A^T (bipartite -> mutual reads as weight 2).

    Returns
    -------
    (csr_float32, note_string)
    """
    nt = str(network_type).lower()
    if nt == "ppi":
        A = graph_csr.astype(np.float32).tocsr()
        A.sum_duplicates()
        return A, "PPI: graph used as-is (already undirected)"
    # grn / mirna / anything else -> symmetrise
    A = (graph_csr + graph_csr.T).astype(np.float32).tocsr()
    A.sum_duplicates()
    return A, f"{nt.upper()}: A + A^T applied (mutual edges weight 2)"


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
    freeze_threshold: int = 3,
    early_stop_fraction: float = 0.01,
) -> tuple[np.ndarray, float]:
    """Run Phase 1 on the GPU until convergence (or pass cap) and compute Q.

    Includes the three iteration-level optimisations:
      - SMEM hash table in ``compute_proposed_moves`` (kernel-side).
      - Community freezing via ``update_freeze_status``.
      - Early termination via ``move_counter`` and ``early_stop_fraction``.

    The community assignment is NOT renumbered to 0..K-1 here — that is
    done in :func:`louvain_gpu` before passing to :func:`_build_coarsened_graph`.
    """
    n   = int(csr.shape[0])
    nnz = int(csr.nnz)
    if n == 0:
        return np.zeros(0, dtype=np.int32), 0.0

    row_ptr_h  = np.ascontiguousarray(csr.indptr,  dtype=np.int32)
    col_idx_h  = np.ascontiguousarray(csr.indices, dtype=np.int32)
    edge_wt_h  = np.ascontiguousarray(csr.data,    dtype=np.float32)
    degree_h   = np.asarray(csr.sum(axis=1), dtype=np.float32).flatten()
    edge_src_h = np.repeat(
        np.arange(n, dtype=np.int32), np.diff(row_ptr_h)
    ).astype(np.int32)

    total_weight = float(degree_h.sum())
    if total_weight <= 0.0:
        return np.arange(n, dtype=np.int32), 0.0
    inv_2m = 1.0 / total_weight

    community_h    = np.arange(n, dtype=np.int32)
    comm_degsum_h  = degree_h.copy()

    d_buffers: list = []

    def _to_gpu(arr: np.ndarray):
        ga = gpuarray.to_gpu_async(arr, stream=stream_transfer)
        d_buffers.append(ga)
        return ga

    def _empty(shape, dtype):
        ga = gpuarray.empty(shape, dtype=dtype)
        d_buffers.append(ga)
        return ga

    d_row_ptr   = _to_gpu(row_ptr_h)
    d_col_idx   = _to_gpu(col_idx_h)
    d_edge_wt   = _to_gpu(edge_wt_h)
    d_edge_src  = _to_gpu(edge_src_h)
    d_degree    = _to_gpu(degree_h)
    d_community = _to_gpu(community_h)
    d_prev_comm = _to_gpu(community_h.copy())
    d_comm_degs = _to_gpu(comm_degsum_h)
    d_proposed  = _empty((n,), np.int32)
    d_best_comm = _empty((n,), np.int32)   # best partition seen (Q-tracked)
    d_move_cnt  = _empty((1,), np.int32)
    d_freeze_ct = gpuarray.zeros((n,), np.int32)
    d_frozen    = gpuarray.zeros((n,), np.int32)
    d_buffers.extend([d_freeze_ct, d_frozen])

    mod_blocks  = max(1, (nnz + BLOCK_SIZE - 1) // BLOCK_SIZE)
    d_partial_Q = _empty((mod_blocks,), np.float32)

    stream_transfer.synchronize()

    try:
        k_prop   = kernels["proposed_moves"]
        k_apply  = kernels["apply_moves"]
        k_freeze = kernels["freeze"]
        k_mod    = kernels["modularity"]

        bs_apply = int(block_size)
        if bs_apply <= 0 or bs_apply > 1024:
            bs_apply = BLOCK_SIZE
        grid_apply = ((n + bs_apply - 1) // bs_apply, 1, 1)
        grid_n     = (n, 1, 1)
        block_256  = (BLOCK_SIZE, 1, 1)
        early_stop_threshold = max(1, int(n * float(early_stop_fraction)))
        mod_grid = (mod_blocks, 1, 1)

        # Objective-based convergence.  Bulk-synchronous parallel Phase 1
        # OSCILLATES on scale-free / random graphs (adjacent nodes swap
        # communities every pass), so ``move_count`` stays high and the loop
        # burns all max_phase1_passes without the modularity improving — slow
        # AND leaving a poor partition.  We therefore compute Q after every
        # pass, KEEP THE BEST partition seen (parallel Phase 1 can overshoot a
        # peak then degrade, so the last pass is not necessarily the best), and
        # stop once Q fails to improve by min_delta_q for MOD_PATIENCE passes.
        # This finally gives min_delta_q its proper role as a level-convergence
        # tolerance (not a per-move gate — that is handled above via inv_2m).
        MOD_PATIENCE = 3
        best_Q      = -1.0e30
        best_valid  = False
        no_improve  = 0

        for _pass in range(max_phase1_passes):
            cuda.memset_d32(d_move_cnt.gpudata, 0, 1)

            # Save current → prev_community before computing proposals
            # (used by update_freeze_status after apply_moves).
            cuda.memcpy_dtod_async(
                d_prev_comm.gpudata, d_community.gpudata, n * 4,
                stream_compute,
            )

            k_prop(
                d_row_ptr, d_col_idx, d_edge_wt,
                d_community, d_comm_degs, d_degree,
                d_frozen,                                # NEW: frozen mask
                d_proposed,
                # Per-move acceptance threshold.  A single node's modularity
                # gain scales as ~1/(2m), so a fixed absolute threshold (the
                # old default min_delta_q=1e-4) rejects EVERY move on any graph
                # with more than a few thousand edges — leaving every node in
                # its own community (num_communities == n, modularity == 0).
                # Scaling by inv_2m makes the threshold track the natural gain
                # magnitude, so min_delta_q stays a meaningful tuning knob while
                # actually accepting improving moves at any graph size.
                np.float32(min_delta_q * inv_2m),
                np.float32(inv_2m),
                np.float32(resolution),
                np.int32(n),
                block=block_256, grid=grid_n, stream=stream_compute,
            )

            k_apply(
                d_community, d_proposed, d_move_cnt,
                d_degree, d_comm_degs, np.int32(n),
                block=(bs_apply, 1, 1), grid=grid_apply,
                stream=stream_compute,
            )

            # Update freeze counters (consults community vs prev_community).
            k_freeze(
                d_community, d_prev_comm,
                d_freeze_ct, d_frozen,
                np.int32(int(freeze_threshold)), np.int32(n),
                block=(bs_apply, 1, 1), grid=grid_apply,
                stream=stream_compute,
            )

            # Modularity of the partition AFTER this pass.
            k_mod(
                d_edge_src, d_col_idx, d_edge_wt,
                d_community, d_degree, d_partial_Q,
                np.float32(inv_2m), np.float32(resolution),
                np.int32(nnz),
                block=block_256, grid=mod_grid, stream=stream_compute,
            )

            stream_compute.synchronize()
            move_count = int(d_move_cnt.get()[0])
            cur_Q = float(inv_2m * float(np.sum(d_partial_Q.get())))

            gained = cur_Q - best_Q
            if cur_Q > best_Q:
                best_Q = cur_Q
                cuda.memcpy_dtod_async(
                    d_best_comm.gpudata, d_community.gpudata, n * 4,
                    stream_compute,
                )
                best_valid = True
            # Patience: a pass that improves Q by less than min_delta_q counts
            # as "no meaningful progress".
            no_improve = 0 if gained >= min_delta_q else no_improve + 1

            if move_count < early_stop_threshold or no_improve >= MOD_PATIENCE:
                break

        stream_compute.synchronize()
        if best_valid:
            community_out = d_best_comm.get().astype(np.int32)
            modularity    = best_Q
        else:
            community_out = d_community.get().astype(np.int32)
            modularity    = 0.0
        return community_out, modularity

    finally:
        for arr in d_buffers:
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Chunked Phase 1 for graphs that exceed VRAM
# ---------------------------------------------------------------------------

def _louvain_level_chunked(
    csr: sp.csr_matrix,
    kernels: dict[str, Any],
    min_delta_q: float,
    resolution: float,
    max_phase1_passes: int,
    block_size: int,
    stream_compute,
    stream_transfer,
    freeze_threshold: int = 3,
    early_stop_fraction: float = 0.01,
) -> tuple[np.ndarray, float]:
    """Chunked Phase 1 for graphs that exceed VRAM.

    Strategy
    --------
    The community / comm_degree_sum / freeze / proposed arrays stay
    full-size on the device (each O(n)).  Only the per-chunk CSR rows
    are streamed in.  For each pass:

      for chunk in row_chunks(csr, chunk_size):
          upload chunk CSR (row_ptr_local, col_idx_chunk, wt_chunk)
          launch compute_proposed_moves with chunk_node_ids → proposed
      apply_moves over the full proposed array
      update_freeze_status over the full community array
      sync; read move_counter; early-stop if below threshold

    Chunk size budget: 30 % of free VRAM, divided by the average
    bytes-per-row (= avg_degree × 8).
    """
    n   = int(csr.shape[0])
    nnz = int(csr.nnz)
    if n == 0:
        return np.zeros(0, dtype=np.int32), 0.0

    try:
        free_bytes, _total = cuda.mem_get_info()
    except Exception:                                   # noqa: BLE001
        free_bytes = 1 << 30
    avg_deg = max(1.0, nnz / max(n, 1))
    bytes_per_row = max(1, int(avg_deg * 8))
    chunk_size = max(1, min(n, int(free_bytes * 0.3) // bytes_per_row))

    degree_h = np.asarray(csr.sum(axis=1), dtype=np.float32).flatten()
    total_weight = float(degree_h.sum())
    if total_weight <= 0.0:
        return np.arange(n, dtype=np.int32), 0.0
    inv_2m = 1.0 / total_weight

    # Persistent O(n) device arrays
    d_degree    = gpuarray.to_gpu(degree_h)
    d_community = gpuarray.to_gpu(np.arange(n, dtype=np.int32))
    d_prev_comm = gpuarray.to_gpu(np.arange(n, dtype=np.int32))
    d_comm_degs = gpuarray.to_gpu(degree_h.copy())
    d_proposed  = gpuarray.zeros((n,), np.int32)
    d_freeze_ct = gpuarray.zeros((n,), np.int32)
    d_frozen    = gpuarray.zeros((n,), np.int32)
    d_move_cnt  = gpuarray.zeros((1,), np.int32)

    persistent_buffers = [
        d_degree, d_community, d_prev_comm, d_comm_degs, d_proposed,
        d_freeze_ct, d_frozen, d_move_cnt,
    ]

    try:
        k_prop   = kernels["proposed_moves"]
        k_apply  = kernels["apply_moves"]
        k_freeze = kernels["freeze"]
        k_mod    = kernels["modularity"]

        bs_apply = int(block_size)
        if bs_apply <= 0 or bs_apply > 1024:
            bs_apply = BLOCK_SIZE
        grid_apply = ((n + bs_apply - 1) // bs_apply, 1, 1)
        block_256  = (BLOCK_SIZE, 1, 1)
        early_stop_threshold = max(1, int(n * float(early_stop_fraction)))

        # Pre-split the CSR into row-chunks on the host (zero-copy slices).
        chunks: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            rs = int(csr.indptr[start])
            re = int(csr.indptr[end])
            local_row_ptr = (csr.indptr[start:end + 1] - rs).astype(np.int32)
            local_col_idx = csr.indices[rs:re].astype(np.int32, copy=False)
            local_values  = csr.data[rs:re].astype(np.float32, copy=False)
            chunks.append((start, end, local_row_ptr, local_col_idx, local_values))

        for _pass in range(max_phase1_passes):
            cuda.memset_d32(d_move_cnt.gpudata, 0, 1)
            cuda.memcpy_dtod_async(
                d_prev_comm.gpudata, d_community.gpudata, n * 4,
                stream_compute,
            )

            for (start, end, local_rp, local_ci, local_wt) in chunks:
                cs = end - start
                d_row_ptr_c = gpuarray.to_gpu_async(local_rp, stream=stream_transfer)
                d_col_idx_c = gpuarray.to_gpu_async(local_ci, stream=stream_transfer)
                d_wt_c      = gpuarray.to_gpu_async(local_wt, stream=stream_transfer)
                stream_transfer.synchronize()

                # We re-use compute_proposed_moves but it expects row_ptr
                # indexed by GLOBAL node id.  To make this work for a
                # chunk, we copy the chunk's local proposals back into the
                # global ``d_proposed`` array via a host-side gather.
                # For simplicity in this initial implementation we launch
                # the kernel with chunk_size blocks AND a temporary local
                # proposed buffer, then memcpy the slice into d_proposed.
                d_proposed_local = gpuarray.zeros((cs,), np.int32)

                # Build a temporary "community_local" view: we pass the
                # global community array since col_idx still references
                # global node ids — that is correct.  Only the row dimension
                # is chunked.
                # Note: compute_proposed_moves writes proposed_comm[u]
                # where u = blockIdx.x.  Here blockIdx.x ranges 0..cs-1,
                # so the writes land in d_proposed_local.  We then copy
                # d_proposed_local → d_proposed[start:end].

                # For the frozen mask we pass a chunk-local view as well.
                # The simplest correct approach: pass d_frozen offset by
                # start.  PyCUDA supports pointer arithmetic via .ptr +
                # offset.  We avoid that complication by NOT skipping
                # frozen nodes in the chunked path (chunked is already a
                # low-VRAM fallback; correctness > optimisation here).

                k_prop(
                    d_row_ptr_c, d_col_idx_c, d_wt_c,
                    d_community, d_comm_degs, d_degree,
                    np.intp(0),                          # NULL frozen ptr
                    d_proposed_local,
                    # Scale-invariant per-move threshold — see _louvain_level.
                    np.float32(min_delta_q * inv_2m),
                    np.float32(inv_2m),
                    np.float32(resolution),
                    np.int32(cs),
                    block=block_256, grid=(cs, 1, 1), stream=stream_compute,
                )

                # Copy local proposals back into the global slice.
                cuda.memcpy_dtod_async(
                    int(d_proposed.gpudata) + start * 4,
                    d_proposed_local.gpudata,
                    cs * 4,
                    stream_compute,
                )

                stream_compute.synchronize()
                for arr in (d_row_ptr_c, d_col_idx_c, d_wt_c,
                            d_proposed_local):
                    try:
                        arr.gpudata.free()
                    except Exception:                   # noqa: BLE001
                        pass

            k_apply(
                d_community, d_proposed, d_move_cnt,
                d_degree, d_comm_degs, np.int32(n),
                block=(bs_apply, 1, 1), grid=grid_apply,
                stream=stream_compute,
            )
            k_freeze(
                d_community, d_prev_comm, d_freeze_ct, d_frozen,
                np.int32(int(freeze_threshold)), np.int32(n),
                block=(bs_apply, 1, 1), grid=grid_apply,
                stream=stream_compute,
            )
            stream_compute.synchronize()
            if int(d_move_cnt.get()[0]) < early_stop_threshold:
                break

        # Modularity computed on full graph (single pass over edges).
        row_ptr_h  = np.ascontiguousarray(csr.indptr,  dtype=np.int32)
        edge_src_h = np.repeat(
            np.arange(n, dtype=np.int32), np.diff(row_ptr_h)
        ).astype(np.int32)
        col_idx_h  = np.ascontiguousarray(csr.indices, dtype=np.int32)
        edge_wt_h  = np.ascontiguousarray(csr.data,    dtype=np.float32)

        d_edge_src = gpuarray.to_gpu(edge_src_h)
        d_col_idx  = gpuarray.to_gpu(col_idx_h)
        d_edge_wt  = gpuarray.to_gpu(edge_wt_h)
        mod_blocks = max(1, (nnz + BLOCK_SIZE - 1) // BLOCK_SIZE)
        d_partial_Q = gpuarray.zeros((mod_blocks,), np.float32)

        k_mod(
            d_edge_src, d_col_idx, d_edge_wt,
            d_community, d_degree, d_partial_Q,
            np.float32(inv_2m), np.float32(resolution),
            np.int32(nnz),
            block=block_256, grid=(mod_blocks, 1, 1),
            stream=stream_compute,
        )
        stream_compute.synchronize()

        modularity = float(inv_2m * float(np.sum(d_partial_Q.get())))
        community_out = d_community.get().astype(np.int32)

        for arr in (d_edge_src, d_col_idx, d_edge_wt, d_partial_Q):
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass

        return community_out, modularity

    finally:
        for arr in persistent_buffers:
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Phase 2 — build the coarsened graph for the next level
# ---------------------------------------------------------------------------

def _coarsen_gpu_cupy(
    d_csrc, d_cdst, d_cwt, nnz: int, K: int,
) -> sp.csr_matrix:
    """Fully GPU-side Phase 2 coarsening using CuPy sort + segmented reduce.

    Inputs are PyCUDA gpuarrays containing the per-edge (csrc, cdst, cwt)
    triples emitted by ``count_community_edges``.  We bridge to CuPy via
    ``cp.asarray`` on the same device pointer (zero-copy when memory
    pools are compatible; otherwise a single contiguous copy on the
    device).  No host transfers occur during sort/reduce.

    Self-loops (csrc == cdst) are kept — they carry intra-community
    weight that contributes to the next level's modularity.
    """
    assert CUPY_SORT_AVAILABLE and _cp is not None

    # Wrap PyCUDA gpuarrays as CuPy arrays via the int pointer.  The
    # arrays remain owned by PyCUDA — we just create a non-owning view.
    src_cp = _cp.ndarray((nnz,), dtype=_cp.int32,
                         memptr=_cp.cuda.MemoryPointer(
                             _cp.cuda.UnownedMemory(
                                 int(d_csrc.gpudata), nnz * 4, owner=None,
                             ),
                             0,
                         ))
    dst_cp = _cp.ndarray((nnz,), dtype=_cp.int32,
                         memptr=_cp.cuda.MemoryPointer(
                             _cp.cuda.UnownedMemory(
                                 int(d_cdst.gpudata), nnz * 4, owner=None,
                             ),
                             0,
                         ))
    wt_cp  = _cp.ndarray((nnz,), dtype=_cp.float32,
                         memptr=_cp.cuda.MemoryPointer(
                             _cp.cuda.UnownedMemory(
                                 int(d_cwt.gpudata), nnz * 4, owner=None,
                             ),
                             0,
                         ))

    # Compound 64-bit sort key: src * K + dst.  K may be smaller than 2^31
    # so int64 is plenty.
    sort_keys = src_cp.astype(_cp.int64) * _cp.int64(K) + dst_cp.astype(_cp.int64)
    order = _cp.argsort(sort_keys)
    src_sorted = src_cp[order]
    dst_sorted = dst_cp[order]
    wt_sorted  = wt_cp[order]

    # Segment heads: positions where (src, dst) changes.
    if nnz <= 1:
        seg_starts_cp = _cp.zeros(1, dtype=_cp.int64)
    else:
        change = (src_sorted[1:] != src_sorted[:-1]) | \
                 (dst_sorted[1:] != dst_sorted[:-1])
        seg_starts_cp = _cp.concatenate((
            _cp.zeros(1, dtype=_cp.int64),
            (_cp.where(change)[0] + 1).astype(_cp.int64),
        ))

    new_csrc_cp = src_sorted[seg_starts_cp]
    new_cdst_cp = dst_sorted[seg_starts_cp]
    new_cwt_cp  = _cp.add.reduceat(wt_sorted, seg_starts_cp)

    # Single H2D copy of the reduced triples (much smaller than the raw
    # edge stream).  scipy CSR assembly stays on CPU.
    new_csrc = _cp.asnumpy(new_csrc_cp).astype(np.int32, copy=False)
    new_cdst = _cp.asnumpy(new_cdst_cp).astype(np.int32, copy=False)
    new_cwt  = _cp.asnumpy(new_cwt_cp).astype(np.float32, copy=False)

    new_csr = sp.csr_matrix(
        (new_cwt, (new_csrc, new_cdst)),
        shape=(K, K), dtype=np.float32,
    )
    new_csr.sum_duplicates()
    return new_csr


def _coarsen_gpu_fallback(
    d_csrc, d_cdst, d_cwt, nnz: int, K: int,
    kernels: dict[str, Any], stream_compute,
) -> sp.csr_matrix:
    """Hybrid coarsen: GPU compound key + CPU argsort + GPU gather + GPU reduce.

    Used when CuPy is not available.  PCIe traffic is roughly halved
    vs. the pure-CPU path because the large CSR arrays never leave the
    device — only the sort_keys (int64, 8 B per edge) and the resulting
    permutation indices cross the bus.
    """
    # Build compound sort key on the device.  We do this with a small
    # PyCUDA-side helper via gpuarray arithmetic (vectorised), which
    # avoids writing a dedicated kernel for this one operation.
    src_i64 = d_csrc.astype(np.int64)
    dst_i64 = d_cdst.astype(np.int64)
    sort_keys = src_i64 * np.int64(K) + dst_i64

    # Single small D2H of just the keys.
    keys_host = sort_keys.get()
    order = np.argsort(keys_host, kind="stable").astype(np.int32)

    # Free temporary GPU buffers for the keys.
    for arr in (src_i64, dst_i64, sort_keys):
        try:
            arr.gpudata.free()
        except Exception:                               # noqa: BLE001
            pass

    # Upload permutation back to the device for the gather.
    d_indices = gpuarray.to_gpu(order)
    d_src_sorted = gpuarray.empty((nnz,), np.int32)
    d_dst_sorted = gpuarray.empty((nnz,), np.int32)
    d_wt_sorted  = gpuarray.empty((nnz,), np.float32)

    k_gather = kernels["gather"]
    bs = BLOCK_SIZE
    grid = ((nnz + bs - 1) // bs, 1, 1)
    k_gather(
        d_csrc, d_cdst, d_cwt,
        d_src_sorted, d_dst_sorted, d_wt_sorted,
        d_indices, np.int32(nnz),
        block=(bs, 1, 1), grid=grid, stream=stream_compute,
    )

    # GPU segmented reduction.  Output buffers are bounded by nnz.
    d_out_src = gpuarray.empty((nnz,), np.int32)
    d_out_dst = gpuarray.empty((nnz,), np.int32)
    d_out_wt  = gpuarray.empty((nnz,), np.float32)
    d_out_cnt = gpuarray.zeros((1,), np.int32)

    k_seg = kernels["seg_reduce"]
    k_seg(
        d_src_sorted, d_dst_sorted, d_wt_sorted,
        d_out_src, d_out_dst, d_out_wt, d_out_cnt,
        np.int32(nnz),
        np.int32(0),                                    # keep self-loops
        block=(bs, 1, 1), grid=grid, stream=stream_compute,
    )
    stream_compute.synchronize()
    out_count = int(d_out_cnt.get()[0])

    if out_count == 0:
        result = sp.csr_matrix((K, K), dtype=np.float32)
    else:
        new_csrc = d_out_src.get()[:out_count]
        new_cdst = d_out_dst.get()[:out_count]
        new_cwt  = d_out_wt.get()[:out_count]
        result = sp.csr_matrix(
            (new_cwt, (new_csrc, new_cdst)),
            shape=(K, K), dtype=np.float32,
        )
        result.sum_duplicates()

    for arr in (d_indices, d_src_sorted, d_dst_sorted, d_wt_sorted,
                d_out_src, d_out_dst, d_out_wt, d_out_cnt):
        try:
            arr.gpudata.free()
        except Exception:                               # noqa: BLE001
            pass
    return result


def _build_coarsened_graph(
    csr: sp.csr_matrix,
    community: np.ndarray,        # already renumbered to 0..K-1
    K: int,
    kernels: dict[str, Any],
    stream_compute,
    block_size: int,
) -> sp.csr_matrix:
    """Collapse the graph into a (K × K) weighted super-node CSR.

    Dispatches between:
      1. CuPy fully GPU sort+reduce (``_coarsen_gpu_cupy``) when CuPy
         is available — zero CPU sort, only one small D2H of the
         reduced triples.
      2. GPU-gather + CPU-argsort + GPU segmented reduce
         (``_coarsen_gpu_fallback``) when CuPy is missing — ~50 % less
         PCIe traffic than the original pure-CPU coarsening path.

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

    d_edge_src  = gpuarray.to_gpu(edge_src_h)
    d_col_idx   = gpuarray.to_gpu(col_idx_h)
    d_edge_wt   = gpuarray.to_gpu(edge_wt_h)
    d_community = gpuarray.to_gpu(community.astype(np.int32))
    d_csrc      = gpuarray.empty((nnz,), np.int32)
    d_cdst      = gpuarray.empty((nnz,), np.int32)
    d_cwt       = gpuarray.empty((nnz,), np.float32)

    bs = int(block_size)
    if bs <= 0 or bs > 1024:
        bs = BLOCK_SIZE
    grid = ((nnz + bs - 1) // bs, 1, 1)

    try:
        k_count = kernels["count_edges"]
        k_count(
            d_edge_src, d_col_idx, d_edge_wt, d_community,
            d_csrc, d_cdst, d_cwt, np.int32(nnz),
            block=(bs, 1, 1), grid=grid, stream=stream_compute,
        )
        stream_compute.synchronize()

        if CUPY_SORT_AVAILABLE:
            try:
                return _coarsen_gpu_cupy(d_csrc, d_cdst, d_cwt, nnz, K)
            except Exception as exc:                    # noqa: BLE001
                logging.warning(
                    "CuPy Phase-2 coarsening failed (%s); "
                    "falling back to GPU-gather hybrid.", exc,
                )

        return _coarsen_gpu_fallback(
            d_csrc, d_cdst, d_cwt, nnz, K, kernels, stream_compute,
        )

    finally:
        for arr in (d_edge_src, d_col_idx, d_edge_wt, d_community,
                    d_csrc, d_cdst, d_cwt):
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass


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

        min_delta_q         = float(p["min_delta_q"])
        max_levels          = int(p["max_levels"])
        resolution          = float(p["resolution"])
        max_phase1_passes   = int(p["max_phase1_passes"])
        network_type        = str(p.get("network_type", "grn"))
        block_size          = int(p.get("block_size", BLOCK_SIZE))
        if block_size <= 0 or block_size > 1024:
            block_size = BLOCK_SIZE
        freeze_threshold    = int(p.get("freeze_threshold", 3))
        early_stop_fraction = float(p.get("early_stop_fraction", 0.01))
        use_chunking_req    = bool(p.get("use_chunking", False))

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

            # Use the chunked Phase-1 path ONLY when this level's CSR working
            # set genuinely does not fit in VRAM.  The chunked path streams CSR
            # rows and (per its docstring) drops the freeze optimisation, so it
            # is strictly slower when the graph fits.  A use_chunking=True flag
            # from params is deliberately NOT honoured: apply_config's
            # MemoryManager injects it from a coarse estimate and, since the
            # runner calls apply_config before this function, that injected
            # flag is indistinguishable from a user-supplied one.  est_bytes
            # (this level's precise estimate) is the sole authority.
            try:
                free_bytes, _total = cuda.mem_get_info()
            except Exception:                           # noqa: BLE001
                free_bytes = 1 << 30
            est_bytes = int(current.nnz) * 8 + n_cur * 8
            use_chunked = est_bytes > VRAM_SAFETY * free_bytes
            if use_chunking_req and not use_chunked and _level == 0:
                logging.info(
                    "louvain_gpu: use_chunking ignored — level-0 working set "
                    "%.1f MB fits in %.1f MB free.",
                    est_bytes / 1e6, free_bytes / 1e6,
                )

            level_fn = _louvain_level_chunked if use_chunked else _louvain_level
            level_community, level_modularity = level_fn(
                current, kernels,
                min_delta_q=min_delta_q,
                resolution=resolution,
                max_phase1_passes=max_phase1_passes,
                block_size=block_size,
                stream_compute=stream_compute,
                stream_transfer=stream_transfer,
                freeze_threshold=freeze_threshold,
                early_stop_fraction=early_stop_fraction,
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
            f"GPU pipeline: SMEM hash table (size {SMEM_HASH_SIZE}), "
            f"community freezing (threshold {freeze_threshold}), "
            f"early termination at {early_stop_fraction:.1%}, "
            f"Phase-2 backend "
            f"{'cupy' if CUPY_SORT_AVAILABLE else 'gpu_gather_hybrid'}, "
            f"arch={kernels.get('_arch_flag', '?')}. "
            "Parallel Phase 1 updates combined with frozen communities "
            "and early termination may produce different partitions than "
            "serial Louvain. This is expected behaviour, not an error."
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
