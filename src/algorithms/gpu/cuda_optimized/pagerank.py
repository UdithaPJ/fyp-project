"""
algorithms/pagerank.py — PageRank for Biological Network Hub Identification
============================================================================

Biological context
------------------
PageRank models the propagation of "influence" through a directed network.
At each step a fraction d of the mass at each node follows an outgoing
edge; the remaining fraction (1 − d) teleports.  The fixed-point ranking
identifies the most central nodes — what those mean depends on the network:

  GRN   — top_regulators are master TFs (receive influence from other
          important TFs AND drive many targets); top_targets are
          convergence points for regulatory signals.
  PPI   — top_nodes are hub proteins central to the interaction network.
  miRNA — top_mirnas are master post-transcriptional regulators;
          top_target_genes are heavily co-targeted effector genes.

Dangling-node handling (network-type aware)
-------------------------------------------
Nodes with no outgoing edges (out_degree == 0) cannot propagate their
mass via the standard scatter; the mass must be redistributed somewhere.
The choice of destination depends on the biology of the network type:

  GRN   : redistribute ONLY to nodes with out_degree > 0 (regulators).
          A pure target gene receiving dangling mass artificially
          elevates its score; restricting to regulators preserves the
          TF↔target asymmetry that is the whole point of the GRN view.
  PPI   : redistribute UNIFORMLY across all nodes.  The graph is
          undirected; every node is a valid "starting point" for a
          random walk reset.
  miRNA : redistribute ONLY to miRNA nodes (out_degree > 0 in the
          directed bipartite graph).  Gene targets have zero outgoing
          edges by construction; sending dangling mass to them inverts
          the regulator/target direction.

If the eligible set turns out empty (degenerate graph), the kernel
falls back to uniform redistribution with a logged warning.

Algorithm (power iteration, scatter form)
-----------------------------------------
Initialise PR[i] = 1 / N for all i.
Repeat until ‖PR_new − PR_old‖₁ < tolerance, or max_iter:
  1.  PR_new[v] = (1 − d) / N                    (teleport baseline)
  2.  For each source u with deg_u > 0:
        for each (u → v) edge with weight w_uv:
          PR_new[v] += d * PR_old[u] / deg_u * w_uv
  3.  For each eligible v:
        PR_new[v] += d * Σ PR_old[dangling] / |eligible|
  4.  Swap PR_old ↔ PR_new (no data copy on GPU — pointer swap).

The scatter form avoids transposing the CSR (no Mᵀ build) and is the
natural mapping to one-thread-per-edge GPU work.

Parameter guide
---------------
damping      (float, default 0.85)  Probability of following an edge.
max_iter     (int,   default 100)   Hard iteration cap.
tolerance    (float, default 1e-6)  L1-norm early-stop threshold.
network_type (str,   default "grn") One of "grn", "ppi", "mirna".
block_size   (int,   default 256)   CUDA block dimension.
"""

# ── GPU / CUDA-optimised implementation (PyCUDA custom kernels) ──────────
# Source:    biological_network_framework/algorithms/pagerank.py
# Requires:  pycuda (with a working NVCC toolchain)
# Used by:   webapp routes, GPU benchmarking (src.benchmarking.benchmark)
# Modes:     _gpu only — this module is GPU-exclusive.
# CPU-only counterparts (benchmarking only — never import in webapp):
#   src.algorithms.cpu.single_threaded.pagerank
#   src.algorithms.cpu.multi_threaded.pagerank
#
# Six PyCUDA kernels (one SourceModule, compiled once, cached):
#   initialize_pr               — PR_new[i] = (1 − d) / N (teleport base)
#   scatter_contributions_csr   — low/medium-degree source nodes;
#                                  three-tier (thread / warp / block);
#                                  SMEM hash aggregation in HIGH tier
#                                  reduces global atomicAdd pressure.
#   scatter_contributions_ellpack — hub source nodes via padded ELLPACK
#                                  arrays for predictable stride access.
#   sum_dangling_pr             — block-partial Σ PR_old[i] over dangling
#                                  nodes (host sums the partials).
#   distribute_dangling_mass    — adds d · Σ_dangling / |eligible| to
#                                  every eligible node (network-type
#                                  decides "eligible" CPU-side).
#   compute_l1_convergence      — warp-shuffle + final warp reduction of
#                                  Σ |PR_new[i] − PR_old[i]| per block.
#
# Hybrid storage:  CSR + ELLPACK split at HUB_THRESHOLD = 32.
#   Nodes with 0 < deg < 32     → CSR scatter kernel (original CSR arrays)
#   Nodes with deg >= 32        → ELLPACK scatter kernel (built CPU-side)
#   Nodes with deg == 0         → dangling, handled separately.
#
# Compilation: -arch=sm_75 (RTX 20-series Turing).
# Context: retain_primary_context().push() pattern (matches hits/louvain/mcl).
# Does NOT silently fall back to CPU — raises RuntimeError / MemoryError /
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
    logging.warning("PyCUDA not available — pagerank_gpu() will raise.")

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
    "damping":      0.85,
    "max_iter":     100,
    "tolerance":    1e-6,
    "network_type": "grn",
    "block_size":   256,
}

BLOCK_SIZE: int     = 256
WARP_SIZE: int      = 32
SMEM_BUCKETS: int   = 256        # = BLOCK_SIZE so each thread flushes one bucket
HUB_THRESHOLD: int  = 32         # deg >= this → ELLPACK kernel
_TOP_REG: int       = 15         # top regulators returned (GRN / miRNA)
_TOP_TGT: int       = 15         # top targets returned   (GRN / miRNA)
_TOP_NODES: int     = 20         # top nodes returned     (PPI)


# ---------------------------------------------------------------------------
# CUDA kernel source (all six kernels, single SourceModule)
# ---------------------------------------------------------------------------

KERNEL_SOURCE = r"""
extern "C" {

#define BLOCK_SIZE   256
#define WARP_SIZE    32
#define SMEM_BUCKETS 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

// =========================================================================
// KERNEL 1: initialize_pr
//
// PR_new[i] = (1 − d) / N for every i.  Runs separately from the scatter
// kernels because the scatters atomicAdd INTO PR_new and assume it
// starts at the teleport value.
// =========================================================================
__global__ void initialize_pr(
    float*     __restrict__ PR_new,
    const float              teleport_val,
    const int                n)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) PR_new[tid] = teleport_val;
}


// =========================================================================
// KERNEL 2: scatter_contributions_csr
//
// One block per low/medium-degree source node u.  Threads in the block
// scatter contributions to u's out-neighbours via atomicAdd on PR_new.
//
// Three-tier degree-aware scheduling:
//   LOW  (deg_u < 32)         : thread 0 only, serial scatter
//   MED  (32 <= deg_u < 256)  : first warp, stride-32 scatter
//   HIGH (deg_u >= 256)       : full block + SMEM hash aggregation
//
// HIGH-tier SMEM hash: per-block 256-bucket open-addressing table.
// Each thread inserts (target_node, contribution) via atomicCAS +
// atomicAdd with linear probing; falls back to direct global atomicAdd
// if the table saturates.  After __syncthreads, every thread flushes
// its assigned bucket (SMEM_BUCKETS == BLOCK_SIZE).  This reduces the
// number of global atomicAdds per block from up to `deg_u` (thousands
// for a super-hub) to at most SMEM_BUCKETS + saturation-fallback count.
// =========================================================================
__global__ void scatter_contributions_csr(
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ edge_weights,
    const float* __restrict__ PR_old,
    float*       __restrict__ PR_new,
    const float* __restrict__ out_degree,    // FP32: sum of out-edge weights
    const int*   __restrict__ node_ids,
    const int                  num_nodes,
    const float                damping,
    const int                  n)
{
    __shared__ int   smem_keys[SMEM_BUCKETS];
    __shared__ float smem_vals[SMEM_BUCKETS];

    const int idx = blockIdx.x;
    if (idx >= num_nodes) return;

    const int   u     = node_ids[idx];
    const float deg_u = out_degree[u];
    if (deg_u <= 0.0f) return;          // dangling — handled separately

    const int   row_start = row_ptr[u];
    const int   row_end   = row_ptr[u + 1];
    const int   deg_int   = row_end - row_start;
    const float contribution = damping * PR_old[u] / deg_u;

    // ---- LOW tier --------------------------------------------------------
    if (deg_int < WARP_SIZE) {
        if (threadIdx.x == 0) {
            for (int j = 0; j < deg_int; ++j) {
                const int   v = col_idx[row_start + j];
                const float w = edge_weights[row_start + j];
                atomicAdd(&PR_new[v], contribution * w);
            }
        }
        return;
    }

    // ---- MED tier --------------------------------------------------------
    if (deg_int < BLOCK_SIZE) {
        if (threadIdx.x < WARP_SIZE) {
            for (int j = threadIdx.x; j < deg_int; j += WARP_SIZE) {
                const int   v = col_idx[row_start + j];
                const float w = edge_weights[row_start + j];
                atomicAdd(&PR_new[v], contribution * w);
            }
        }
        return;
    }

    // ---- HIGH tier: SMEM hash aggregation --------------------------------
    // Initialise hash table.
    smem_keys[threadIdx.x] = -1;
    smem_vals[threadIdx.x] = 0.0f;
    __syncthreads();

    // Each thread inserts its share of edges.
    for (int j = threadIdx.x; j < deg_int; j += BLOCK_SIZE) {
        const int   v = col_idx[row_start + j];
        const float w = edge_weights[row_start + j] * contribution;

        int  bucket = v & (SMEM_BUCKETS - 1);
        bool placed = false;
        // Cap probes at SMEM_BUCKETS to guarantee termination.
        for (int probe = 0; probe < SMEM_BUCKETS; ++probe) {
            const int prev = atomicCAS(&smem_keys[bucket], -1, v);
            if (prev == -1 || prev == v) {
                atomicAdd(&smem_vals[bucket], w);
                placed = true;
                break;
            }
            bucket = (bucket + 1) & (SMEM_BUCKETS - 1);
        }
        if (!placed) {
            // Hash saturated for this v's probe chain — direct global write.
            atomicAdd(&PR_new[v], w);
        }
    }
    __syncthreads();

    // Flush hash table.  SMEM_BUCKETS == BLOCK_SIZE so each thread owns
    // exactly one bucket.
    const int k = smem_keys[threadIdx.x];
    if (k != -1) {
        atomicAdd(&PR_new[k], smem_vals[threadIdx.x]);
    }
}


// =========================================================================
// KERNEL 3: scatter_contributions_ellpack
//
// One block per hub.  Threads stride through max_row_len padded columns
// in the ELLPACK arrays.  Padding entries have ellpack_cols[...] = -1
// and are skipped.  Coalesced reads within a hub's row; cross-hub
// access is by block, so no cross-hub coalescing — the win versus CSR
// is the absence of row_ptr indirection and predictable stride.
// =========================================================================
__global__ void scatter_contributions_ellpack(
    const int*   __restrict__ ellpack_cols,
    const float* __restrict__ ellpack_vals,
    const int*   __restrict__ hub_node_ids,
    const float* __restrict__ PR_old,
    float*       __restrict__ PR_new,
    const float* __restrict__ out_degree,
    const int                  num_hubs,
    const int                  max_row_len,
    const float                damping)
{
    const int hub_idx = blockIdx.x;
    if (hub_idx >= num_hubs) return;

    const int   u     = hub_node_ids[hub_idx];
    const float deg_u = out_degree[u];
    if (deg_u <= 0.0f) return;

    const float contribution = damping * PR_old[u] / deg_u;
    const int   base         = hub_idx * max_row_len;

    for (int j = threadIdx.x; j < max_row_len; j += BLOCK_SIZE) {
        const int v = ellpack_cols[base + j];
        if (v >= 0) {                  // -1 = padding sentinel
            const float w = ellpack_vals[base + j];
            atomicAdd(&PR_new[v], contribution * w);
        }
    }
}


// =========================================================================
// KERNEL 4: distribute_dangling_mass
//
// Adds redistribution_val = d · dangling_sum / |eligible| to every node
// in the eligible_nodes array via atomicAdd (so prior scatter writes are
// preserved).  Network-type decides which nodes are eligible (CPU-side);
// the kernel is structure-agnostic.
// =========================================================================
__global__ void distribute_dangling_mass(
    float*       __restrict__ PR_new,
    const int*   __restrict__ eligible_nodes,
    const int                  num_eligible,
    const float                redistribution_val)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_eligible) return;
    const int v = eligible_nodes[tid];
    atomicAdd(&PR_new[v], redistribution_val);
}


// =========================================================================
// KERNEL 5: sum_dangling_pr
//
// Per-block partial sum of PR_old[i] over dangling nodes (flag == 1).
// Host sums the per-block partials.
// =========================================================================
__global__ void sum_dangling_pr(
    const float* __restrict__ PR_old,
    const int*   __restrict__ dangling_flags,
    float*       __restrict__ partial_sums,
    const int                  n)
{
    __shared__ float smem[BLOCK_SIZE];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (tid < n && dangling_flags[tid] != 0) {
        val = PR_old[tid];
    }
    smem[threadIdx.x] = val;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial_sums[blockIdx.x] = smem[0];
}


// =========================================================================
// KERNEL 6: compute_l1_convergence
//
// Σ |PR_new[i] − PR_old[i]| per block via warp-shuffle reduction +
// final warp aggregation in shared memory.  Host sums the per-block
// partials.
// =========================================================================
__global__ void compute_l1_convergence(
    const float* __restrict__ PR_new,
    const float* __restrict__ PR_old,
    float*       __restrict__ partial_sums,
    const int                  n)
{
    __shared__ float smem[WARPS_PER_BLOCK];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < n) ? fabsf(PR_new[tid] - PR_old[tid]) : 0.0f;

    // Intra-warp reduction via shuffles.
    for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, off);
    }

    const int lane    = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    // Final reduction in the first warp (WARPS_PER_BLOCK entries).
    if (threadIdx.x < WARP_SIZE) {
        val = (threadIdx.x < WARPS_PER_BLOCK) ? smem[threadIdx.x] : 0.0f;
        // WARPS_PER_BLOCK = 8 for BLOCK_SIZE=256 → reduce over 8 lanes.
        for (int off = WARPS_PER_BLOCK >> 1; off > 0; off >>= 1) {
            val += __shfl_down_sync(0xffffffffu, val, off);
        }
        if (threadIdx.x == 0) partial_sums[blockIdx.x] = val;
    }
}

}  // extern "C"
"""


# ---------------------------------------------------------------------------
# Module-level kernel cache
# ---------------------------------------------------------------------------

_kernel_cache: dict[str, dict[str, Any]] = {}


def _get_kernels() -> dict[str, Any]:
    """Compile (or fetch from cache) the six PageRank device kernels."""
    if "pagerank" not in _kernel_cache:
        if not PYCUDA_AVAILABLE:
            raise RuntimeError(
                "PyCUDA is required to compile PageRank kernels — "
                "install pycuda and ensure NVCC is on PATH."
            )
        mod = SourceModule(
            KERNEL_SOURCE,
            options=["-arch=sm_75"],        # RTX 20-series Turing
            no_extern_c=True,
        )
        _kernel_cache["pagerank"] = {
            "init":         mod.get_function("initialize_pr"),
            "scatter_csr":  mod.get_function("scatter_contributions_csr"),
            "scatter_ell":  mod.get_function("scatter_contributions_ellpack"),
            "dangling":     mod.get_function("distribute_dangling_mass"),
            "sum_dangling": mod.get_function("sum_dangling_pr"),
            "l1_conv":      mod.get_function("compute_l1_convergence"),
        }
    return _kernel_cache["pagerank"]


# ---------------------------------------------------------------------------
# Parameter merging
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# CPU preprocessing helpers
# ---------------------------------------------------------------------------

def _compute_out_degrees(graph_csr: sp.csr_matrix) -> np.ndarray:
    """Sum of outgoing edge weights per node (FP32).

    For weighted graphs (e.g. PPI confidence scores) the weighted sum is
    the correct denominator in ``contribution = d · PR[u] / deg_u``.  For
    unweighted (binary) graphs this collapses to the integer out-degree.
    """
    return np.asarray(
        graph_csr.sum(axis=1), dtype=np.float32
    ).flatten()


def _identify_dangling_nodes(out_degrees: np.ndarray) -> np.ndarray:
    """Boolean mask: True where out_degree == 0."""
    return out_degrees == 0.0


def _identify_regulator_nodes(out_degrees: np.ndarray) -> np.ndarray:
    """Indices of nodes with out_degree > 0 (GRN regulators)."""
    return np.where(out_degrees > 0.0)[0].astype(np.int32)


def _identify_mirna_nodes(
    graph_csr: sp.csr_matrix,
    node_index_map: dict | None,
) -> np.ndarray:
    """Indices of miRNA nodes in a bipartite miRNA→gene graph.

    If ``node_index_map`` is None (the framework currently does not pass
    node-type labels through to algorithms), miRNA identity is inferred
    from the bipartite structure: only miRNA nodes have outgoing edges
    (genes are sinks with out_degree == 0).
    """
    out_degrees = _compute_out_degrees(graph_csr)
    return np.where(out_degrees > 0.0)[0].astype(np.int32)


def _build_ellpack(
    graph_csr: sp.csr_matrix,
    out_degrees: np.ndarray,
    hub_threshold: int = HUB_THRESHOLD,
) -> tuple[dict, dict]:
    """Split source nodes into CSR (low-degree) and ELLPACK (hub) groups.

    Returns
    -------
    (ellpack_data, csr_remainder) — two dicts.

    ellpack_data
    ------------
        hub_ids      : int32 (num_hubs,)               — global node indices
        cols         : int32 (num_hubs, max_row_len)   — padded with -1
        vals         : float32 (num_hubs, max_row_len) — padded with 0.0
        max_row_len  : int

    csr_remainder
    -------------
        node_ids     : int32 (num_low,) — global indices of non-hub
                       nodes with out_degree > 0.  Uses the ORIGINAL CSR
                       arrays (no row copying).
    """
    n        = int(graph_csr.shape[0])
    indptr   = np.ascontiguousarray(graph_csr.indptr,  dtype=np.int32)
    indices  = np.ascontiguousarray(graph_csr.indices, dtype=np.int32)
    data     = np.ascontiguousarray(graph_csr.data,    dtype=np.float32)
    deg_int  = np.diff(indptr).astype(np.int32)           # integer count

    hub_mask  = deg_int >= hub_threshold
    low_mask  = (deg_int > 0) & (~hub_mask)

    hub_ids = np.where(hub_mask)[0].astype(np.int32)
    low_ids = np.where(low_mask)[0].astype(np.int32)

    if hub_ids.size > 0:
        max_row_len = int(deg_int[hub_ids].max())
        cols = np.full((hub_ids.size, max_row_len), -1, dtype=np.int32)
        vals = np.zeros((hub_ids.size, max_row_len),    dtype=np.float32)
        for h, u in enumerate(hub_ids):
            s = int(indptr[u]); e = int(indptr[u + 1])
            L = e - s
            cols[h, :L] = indices[s:e]
            vals[h, :L] = data[s:e]
    else:
        max_row_len = 0
        cols = np.zeros((0, 0), dtype=np.int32)
        vals = np.zeros((0, 0), dtype=np.float32)

    ellpack_data = {
        "hub_ids":     hub_ids,
        "cols":        cols.reshape(-1),
        "vals":        vals.reshape(-1),
        "max_row_len": max_row_len,
        "num_hubs":    int(hub_ids.size),
    }
    csr_remainder = {
        "node_ids":  low_ids,
        "num_low":   int(low_ids.size),
        "row_ptr":   indptr,                # full CSR — kernel reads via node_ids
        "col_idx":   indices,
        "values":    data,
    }
    return ellpack_data, csr_remainder


def _estimate_pagerank_vram(
    n: int,
    nnz: int,
    num_hubs: int,
    max_row_len: int,
) -> int:
    """Rough VRAM estimate for the full PageRank working set, in bytes.

    Components:
        CSR arrays         : (n+1 + nnz + nnz) * 4
        ELLPACK arrays     : num_hubs * max_row_len * (4 + 4)
        PR vectors x 2     : n * 4 * 2
        partial_sums       : ceil(n / BLOCK_SIZE) * 4
        misc int32 arrays  : n * 4 * 4   (degree, dangling flags, eligible, low_ids)
    """
    csr_b      = (n + 1 + 2 * nnz) * 4
    ellpack_b  = num_hubs * max_row_len * 8
    pr_b       = 2 * n * 4
    partial_b  = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE) * 4
    misc_b     = n * 4 * 4
    return int(csr_b + ellpack_b + pr_b + partial_b + misc_b)


# ---------------------------------------------------------------------------
# Network-type-aware result packing
# ---------------------------------------------------------------------------

def _top_k_among(
    scores: np.ndarray,
    candidate_indices: np.ndarray,
    k: int,
) -> list[int]:
    """Return the top-k indices (by score, descending) drawn from candidates."""
    if candidate_indices.size == 0:
        return []
    sub_scores = scores[candidate_indices]
    order = np.argsort(sub_scores)[::-1][:k]
    return candidate_indices[order].astype(int).tolist()


def _pack_result(
    scores: np.ndarray,
    out_degrees: np.ndarray,
    iterations: int,
    converged: bool,
    network_type: str,
) -> dict:
    """Build the inner result dict in the network-type-specific shape."""
    n = int(scores.size)
    nt = str(network_type).lower()
    scores_list = scores.astype(float).tolist()

    if nt == "ppi":
        order = np.argsort(scores)[::-1][:_TOP_NODES]
        return {
            "scores":     scores_list,
            "top_nodes":  order.astype(int).tolist(),
            "iterations": iterations,
            "converged":  converged,
        }

    regulators = np.where(out_degrees > 0.0)[0]
    targets    = np.where(out_degrees == 0.0)[0]

    if nt == "mirna":
        return {
            "scores":            scores_list,
            "top_mirnas":        _top_k_among(scores, regulators, _TOP_REG),
            "top_target_genes":  _top_k_among(scores, targets,    _TOP_TGT),
            "iterations":        iterations,
            "converged":         converged,
        }

    # Default: GRN
    return {
        "scores":          scores_list,
        "top_regulators":  _top_k_among(scores, regulators, _TOP_REG),
        "top_targets":     _top_k_among(scores, targets,    _TOP_TGT),
        "iterations":      iterations,
        "converged":       converged,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def pagerank_gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """PageRank — GPU-accelerated via custom PyCUDA kernels.

    No CuPy dependency.  Scatter-form power iteration over the original
    CSR (no transpose), with a hybrid CSR + ELLPACK split for hub vs.
    non-hub source nodes (HUB_THRESHOLD = 32).

    Network-type handling
    ---------------------
    GRN    : dangling mass redistributed to regulator nodes only.
    PPI    : dangling mass redistributed uniformly across all nodes.
    miRNA  : dangling mass redistributed to miRNA nodes only.

    Returns
    -------
    dict — see CLAUDE.md "PageRank" result spec (outer envelope +
    ``result`` sub-dict; result keys differ per network_type).

    Raises
    ------
    RuntimeError
        If PyCUDA is unavailable or no CUDA device can be initialised.
    MemoryError
        If allocation fails (graph too large for current VRAM).
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA is required for pagerank_gpu(). "
            "Install it or use pagerank_cpu_single() from "
            "src/algorithms/cpu/single_threaded/pagerank.py"
        )

    # Push the device PRIMARY context unconditionally — same rationale as
    # hits.py / louvain.py / mcl.py.  retain_primary_context() is
    # ref-counted and coexists with any CuPy code that may be active.
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
            p = apply_config("pagerank", graph_csr, p) or p
            for k, v in _DEFAULT_PARAMS.items():
                p.setdefault(k, v)

        damping        = float(p["damping"])
        max_iter       = int(p["max_iter"])
        tolerance      = float(p["tolerance"])
        network_type   = str(p.get("network_type", "grn"))
        block_size     = int(p.get("block_size", BLOCK_SIZE))
        if block_size <= 0 or block_size > 1024:
            block_size = BLOCK_SIZE
        use_chunking   = bool(p.get("use_chunking", False))
        node_index_map = p.get("node_index_map")

        n = int(graph_csr.shape[0])
        if n == 0:
            raise ValueError("Empty graph")

        teleport_val = np.float32((1.0 - damping) / n)
        init_val     = np.float32(1.0 / n)

        # ---- CPU preprocessing ----------------------------------------
        # FP32 weighted out-degree (supports weighted PPI / GRN edges).
        out_degrees    = _compute_out_degrees(graph_csr)
        dangling_mask  = _identify_dangling_nodes(out_degrees)
        dangling_flags = dangling_mask.astype(np.int32)

        # Network-type-aware "eligible" nodes for dangling redistribution.
        nt = network_type.lower()
        if nt == "grn":
            eligible = _identify_regulator_nodes(out_degrees)
            eligible_note = "dangling mass → regulator nodes only (out_degree > 0)"
        elif nt == "mirna":
            eligible = _identify_mirna_nodes(graph_csr, node_index_map)
            eligible_note = "dangling mass → miRNA nodes only (out_degree > 0)"
        else:  # ppi (and default)
            eligible = np.arange(n, dtype=np.int32)
            eligible_note = "dangling mass → all nodes (uniform)"

        # Degenerate fallback: no eligible nodes → uniform.
        if eligible.size == 0:
            logging.warning(
                "pagerank_gpu: no eligible nodes for dangling redistribution "
                "(network_type=%s) — falling back to uniform.", network_type,
            )
            eligible = np.arange(n, dtype=np.int32)
            eligible_note += " (fallback to uniform — no eligible nodes found)"

        # Hybrid CSR + ELLPACK split.
        ellpack_data, csr_remainder = _build_ellpack(
            graph_csr, out_degrees, HUB_THRESHOLD,
        )

        # ---- VRAM check / chunked path notice -------------------------
        est_bytes = _estimate_pagerank_vram(
            n=n, nnz=int(graph_csr.nnz),
            num_hubs=ellpack_data["num_hubs"],
            max_row_len=ellpack_data["max_row_len"],
        )
        try:
            free_bytes, _total = cuda.mem_get_info()
        except Exception:                                   # noqa: BLE001
            free_bytes = 1 << 30   # 1 GB fallback when probe fails
        if est_bytes > free_bytes:
            raise MemoryError(
                f"PageRank GPU needs ~{est_bytes/1e6:.1f} MB but only "
                f"{free_bytes/1e6:.1f} MB free.  Use a higher-VRAM "
                f"device or reduce graph size."
            )
        if use_chunking:
            # The chunked path is a planned optimisation; for now we run
            # the regular path with a warning so user code is not silently
            # broken when apply_config sets use_chunking=True on mid-tier.
            logging.info(
                "pagerank_gpu: use_chunking=True requested but chunked "
                "path is a TODO — running regular path."
            )

        kernels = _get_kernels()

        # ---- Streams + timing -----------------------------------------
        stream_compute  = cuda.Stream()
        stream_transfer = cuda.Stream()
        start_event     = cuda.Event()
        end_event       = cuda.Event()
        start_event.record(stream_compute)

        # ---- Device allocation + async H2D ----------------------------
        d_buffers: list = []

        def _to_gpu(arr: np.ndarray):
            ga = gpuarray.to_gpu_async(arr, stream=stream_transfer)
            d_buffers.append(ga)
            return ga

        def _empty(shape, dtype):
            ga = gpuarray.empty(shape, dtype=dtype)
            d_buffers.append(ga)
            return ga

        # Original full CSR (used by scatter_csr via low_node_ids).
        d_csr_row_ptr = _to_gpu(csr_remainder["row_ptr"])
        d_csr_col_idx = _to_gpu(csr_remainder["col_idx"])
        d_csr_values  = _to_gpu(csr_remainder["values"])
        d_low_ids     = (
            _to_gpu(csr_remainder["node_ids"])
            if csr_remainder["num_low"] > 0
            else None
        )

        # ELLPACK arrays for hubs (flattened to 1-D for PyCUDA).
        num_hubs    = ellpack_data["num_hubs"]
        max_row_len = ellpack_data["max_row_len"]
        if num_hubs > 0:
            d_ellpack_cols = _to_gpu(ellpack_data["cols"])
            d_ellpack_vals = _to_gpu(ellpack_data["vals"])
            d_hub_ids      = _to_gpu(ellpack_data["hub_ids"])
        else:
            d_ellpack_cols = None
            d_ellpack_vals = None
            d_hub_ids      = None

        d_out_degree     = _to_gpu(out_degrees)
        d_dangling_flags = _to_gpu(dangling_flags)
        d_eligible       = _to_gpu(eligible)

        # PR vectors (initialised after transfer sync).
        d_PR_old = _empty((n,), np.float32)
        d_PR_new = _empty((n,), np.float32)

        n_partial_blocks = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE)
        d_partial        = _empty((n_partial_blocks,), np.float32)

        # Initialise PR_old = 1/N via H2D (small, on transfer stream).
        init_host = np.full(n, init_val, dtype=np.float32)
        cuda.memcpy_htod_async(d_PR_old.gpudata, init_host, stream_transfer)
        stream_transfer.synchronize()

        # ---- Iteration loop -------------------------------------------
        converged  = False
        iterations = 0

        n_init_grid  = ((n + block_size - 1) // block_size, 1, 1)
        n_block_dim  = (block_size, 1, 1)
        partial_grid = (n_partial_blocks, 1, 1)
        partial_block = (BLOCK_SIZE, 1, 1)

        k_init     = kernels["init"]
        k_sc_csr   = kernels["scatter_csr"]
        k_sc_ell   = kernels["scatter_ell"]
        k_dang     = kernels["dangling"]
        k_sum_dang = kernels["sum_dangling"]
        k_l1       = kernels["l1_conv"]

        for it in range(max_iter):
            iterations = it + 1

            # 1. Initialise PR_new with teleport baseline.
            k_init(
                d_PR_new, teleport_val, np.int32(n),
                block=n_block_dim, grid=n_init_grid,
                stream=stream_compute,
            )

            # 2. Partial sum of dangling PR_old.
            k_sum_dang(
                d_PR_old, d_dangling_flags, d_partial, np.int32(n),
                block=partial_block, grid=partial_grid,
                stream=stream_compute,
            )
            stream_compute.synchronize()
            dangling_sum = float(np.sum(d_partial.get()))

            # 3. Scatter from CSR (low/medium-degree) sources.
            if csr_remainder["num_low"] > 0 and d_low_ids is not None:
                k_sc_csr(
                    d_csr_row_ptr, d_csr_col_idx, d_csr_values,
                    d_PR_old, d_PR_new, d_out_degree, d_low_ids,
                    np.int32(csr_remainder["num_low"]),
                    np.float32(damping), np.int32(n),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(csr_remainder["num_low"], 1, 1),
                    stream=stream_compute,
                )

            # 4. Scatter from ELLPACK (hub) sources.
            if num_hubs > 0 and d_ellpack_cols is not None:
                k_sc_ell(
                    d_ellpack_cols, d_ellpack_vals, d_hub_ids,
                    d_PR_old, d_PR_new, d_out_degree,
                    np.int32(num_hubs), np.int32(max_row_len),
                    np.float32(damping),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(num_hubs, 1, 1),
                    stream=stream_compute,
                )

            # 5. Distribute dangling mass to eligible nodes.
            if dangling_sum > 0.0 and eligible.size > 0:
                redistribution_val = np.float32(
                    damping * dangling_sum / float(eligible.size)
                )
                eligible_grid = (
                    (int(eligible.size) + block_size - 1) // block_size,
                    1, 1,
                )
                k_dang(
                    d_PR_new, d_eligible,
                    np.int32(int(eligible.size)),
                    redistribution_val,
                    block=n_block_dim, grid=eligible_grid,
                    stream=stream_compute,
                )

            # 6. L1 convergence reduction.
            k_l1(
                d_PR_new, d_PR_old, d_partial, np.int32(n),
                block=partial_block, grid=partial_grid,
                stream=stream_compute,
            )
            stream_compute.synchronize()
            l1_norm = float(np.sum(d_partial.get()))

            # 7. Pointer swap (no data copy).
            d_PR_old, d_PR_new = d_PR_new, d_PR_old

            if l1_norm < tolerance:
                converged = True
                break

        end_event.record(stream_compute)
        end_event.synchronize()
        elapsed = start_event.time_till(end_event) / 1000.0   # ms → s

        # ---- Result extraction (NOT timed) ----------------------------
        scores_host = d_PR_old.get()

        inner = _pack_result(
            scores_host, out_degrees, iterations, converged, network_type,
        )
        inner["note"] = eligible_note

        return {
            "algorithm":      "pagerank",
            "mode":           "gpu",
            "network_type":   network_type,
            "execution_time": elapsed,
            "num_nodes":      n,
            "num_edges":      int(graph_csr.nnz),
            "result":         inner,
        }

    except cuda.LogicError as e:
        logging.error("CUDA error in pagerank_gpu: %s", e)
        raise
    except MemoryError:
        logging.warning(
            "VRAM exhausted in pagerank_gpu.  Retry with a higher-VRAM "
            "device, or reduce graph size."
        )
        raise
    finally:
        # Explicit buffer release before context pop.
        try:
            for arr in d_buffers:                       # noqa: F821
                try:
                    arr.gpudata.free()
                except Exception:                       # noqa: BLE001
                    pass
        except NameError:
            pass
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
    full = pagerank_gpu(graph_csr, p)
    # `full` already has the outer envelope; the benchmark runner only
    # consumes "output" and "extra_params" — keep both layers available.
    return {"output": full, "extra_params": p}
