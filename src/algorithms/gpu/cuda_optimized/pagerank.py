"""
algorithms/pagerank.py - PageRank for Biological Network Hub Identification
============================================================================

Biological context
------------------
PageRank models the propagation of "influence" through a directed network.
At each step a fraction d of the mass at each node follows an outgoing
edge; the remaining fraction (1 - d) teleports.  The fixed-point ranking
identifies the most central nodes - what those mean depends on the network:

  GRN   - top_regulators are master TFs (receive influence from other
          important TFs AND drive many targets); top_targets are
          convergence points for regulatory signals.
  PPI   - top_nodes are hub proteins central to the interaction network.
  miRNA - top_mirnas are master post-transcriptional regulators;
          top_target_genes are heavily co-targeted effector genes.

Dangling-node handling (network-type aware)
-------------------------------------------
Nodes with no outgoing edges (out_degree == 0) cannot propagate their
mass via the standard scatter; the mass must be redistributed somewhere.
The choice of destination depends on the biology of the network type:

  GRN   : redistribute ONLY to nodes with out_degree > 0 (regulators).
  PPI   : redistribute UNIFORMLY across all nodes.
  miRNA : redistribute ONLY to miRNA nodes (out_degree > 0).

If the eligible set turns out empty (degenerate graph), the kernel
falls back to uniform redistribution with a logged warning.

Algorithm (power iteration, scatter form, fused init + dangling)
-----------------------------------------------------------------
Initialise PR[i] = 1 / N for all i.
Repeat until ||PR_new - PR_old||_1 < tolerance, or max_iter:
  1.  sum_dangling_sum = sum_i PR_old[i] for i in dangling set  (GPU)
  2.  redist           = damping * sum_dangling / |eligible|    (GPU scalar)
  3.  PR_new[i] = (1 - d)/N + (redist if i in eligible else 0)  (GPU, fused)
  4.  For each source u with deg_u > 0:
        for each (u -> v) edge with weight w_uv:
          PR_new[v] += d * PR_old[u] / deg_u * w_uv
  5.  Optional: gather_contributions_pull writes PR_new[v]
        directly (no atomic) for nodes with in_degree >= pull_threshold.
        Push kernels skip edges to those targets to avoid double-counting.
  6.  L1 norm + GPU-side scalar reduction.
  7.  Swap PR_old <-> PR_new (pointer swap, no data copy).

Parameter guide
---------------
damping              (float, default 0.85)  Probability of following an edge.
max_iter             (int,   default 100)   Hard iteration cap.
tolerance            (float, default 1e-6)  L1-norm early-stop threshold.
network_type         (str,   default "grn") One of "grn", "ppi", "mirna".
block_size           (int,   default 256)   CUDA block dimension.
ellpack_fraction     (float, default 0.05)  Fraction of nodes routed to ELLPACK.
pull_threshold       (int,   default 0)     0 disables pull; >0 enables it
                                              for nodes with in_degree >= this.
use_chunking         (bool,  default False) Force the chunked path.
"""

# -- GPU / CUDA-optimised implementation (PyCUDA custom kernels) ----------
# Source:    src/algorithms/gpu/cuda_optimized/pagerank.py
# Requires:  pycuda (with a working NVCC toolchain)
# Used by:   webapp routes, GPU benchmarking (src.benchmarking.benchmark)
# Modes:     _gpu only - this module is GPU-exclusive.
# -------------------------------------------------------------------------

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
    logging.warning("PyCUDA not available - pagerank_gpu() will raise.")

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
    "damping":          0.85,
    "max_iter":         100,
    "tolerance":        1e-6,
    "network_type":     "grn",
    "block_size":       256,
    "ellpack_fraction": 0.05,
    "pull_threshold":   0,        # 0 disables pull-mode
    "use_chunking":     False,
}

BLOCK_SIZE: int     = 256
WARP_SIZE: int      = 32          # hardware-fixed (Turing+); not a tunable
SMEM_BUCKETS: int   = 256         # = BLOCK_SIZE so each thread flushes one bucket
_TOP_REG: int       = 15          # top regulators returned (GRN / miRNA)
_TOP_TGT: int       = 15          # top targets returned   (GRN / miRNA)
_TOP_NODES: int     = 20          # top nodes returned     (PPI)


# ---------------------------------------------------------------------------
# CUDA kernel source
# ---------------------------------------------------------------------------

KERNEL_SOURCE = r"""
extern "C" {

#define BLOCK_SIZE      256
#define WARP_SIZE       32
#define SMEM_BUCKETS    256
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

// =========================================================================
// KERNEL: initialize_pr (BACKUP - kept for chunked path, not on hot loop)
//
// PR_new[i] = teleport_val for every i.
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
// KERNEL: initialize_pr_with_dangling  (NEW - Improvement 3, FUSION A)
//
// Fuses teleportation baseline + dangling redistribution into one pass.
//
// PR_new[i] = teleport_val
//            + (d * (*dangling_sum_ptr) / num_eligible) if eligible_mask[i]
//
// Reads dangling_sum from a GPU pointer (length-1 array) - no CPU sync
// required between sum_dangling_pr -> reduce_to_scalar -> this kernel.
// eligible_mask is a boolean (uint8) array of length n; replaces the
// previous index-array + atomicAdd pattern, avoiding per-thread search.
// =========================================================================
__global__ void initialize_pr_with_dangling(
    float*       __restrict__ PR_new,
    const float                teleport_val,
    const float* __restrict__ dangling_sum_ptr,
    const unsigned char* __restrict__ eligible_mask,
    const int                  num_eligible,
    const float                damping,
    const int                  n)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float v = teleport_val;
    if (num_eligible > 0 && eligible_mask[tid]) {
        const float dangling = *dangling_sum_ptr;
        v += damping * dangling / (float)num_eligible;
    }
    PR_new[tid] = v;
}


// =========================================================================
// KERNEL: scatter_contributions_csr  (UPDATED - skip pull-target edges)
//
// One block per low/medium-degree source node u.  Threads in the block
// scatter contributions to u's out-neighbours via atomicAdd on PR_new.
//
// pull_target_mask (NULLABLE):
//   If non-null, edges to v with pull_target_mask[v] != 0 are SKIPPED
//   - those contributions are handled by gather_contributions_pull
//   instead.  When the mask is null, this kernel behaves identically
//   to the original scatter.
//
// Three-tier degree-aware scheduling:
//   LOW  (deg_u < 32)         : thread 0 only, serial scatter
//   MED  (32 <= deg_u < 256)  : first warp, stride-32 scatter
//   HIGH (deg_u >= 256)       : full block + SMEM hash aggregation
// =========================================================================
__global__ void scatter_contributions_csr(
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ edge_weights,
    const float* __restrict__ PR_old,
    float*       __restrict__ PR_new,
    const float* __restrict__ out_degree,
    const int*   __restrict__ node_ids,
    const unsigned char* __restrict__ pull_target_mask,
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
    if (deg_u <= 0.0f) return;

    const int   row_start = row_ptr[u];
    const int   row_end   = row_ptr[u + 1];
    const int   deg_int   = row_end - row_start;
    const float contribution = damping * PR_old[u] / deg_u;

    // ---- LOW tier --------------------------------------------------------
    if (deg_int < WARP_SIZE) {
        if (threadIdx.x == 0) {
            for (int j = 0; j < deg_int; ++j) {
                const int   v = col_idx[row_start + j];
                if (pull_target_mask != 0 && pull_target_mask[v]) continue;
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
                if (pull_target_mask != 0 && pull_target_mask[v]) continue;
                const float w = edge_weights[row_start + j];
                atomicAdd(&PR_new[v], contribution * w);
            }
        }
        return;
    }

    // ---- HIGH tier: SMEM hash aggregation --------------------------------
    smem_keys[threadIdx.x] = -1;
    smem_vals[threadIdx.x] = 0.0f;
    __syncthreads();

    for (int j = threadIdx.x; j < deg_int; j += BLOCK_SIZE) {
        const int v = col_idx[row_start + j];
        if (pull_target_mask != 0 && pull_target_mask[v]) continue;
        const float w = edge_weights[row_start + j] * contribution;

        int  bucket = v & (SMEM_BUCKETS - 1);
        bool placed = false;
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
            atomicAdd(&PR_new[v], w);
        }
    }
    __syncthreads();

    const int k = smem_keys[threadIdx.x];
    if (k != -1) {
        atomicAdd(&PR_new[k], smem_vals[threadIdx.x]);
    }
}


// =========================================================================
// KERNEL: scatter_contributions_ellpack  (UPDATED - skip pull targets)
// =========================================================================
__global__ void scatter_contributions_ellpack(
    const int*   __restrict__ ellpack_cols,
    const float* __restrict__ ellpack_vals,
    const int*   __restrict__ hub_node_ids,
    const float* __restrict__ PR_old,
    float*       __restrict__ PR_new,
    const float* __restrict__ out_degree,
    const unsigned char* __restrict__ pull_target_mask,
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
        if (v >= 0) {
            if (pull_target_mask != 0 && pull_target_mask[v]) continue;
            const float w = ellpack_vals[base + j];
            atomicAdd(&PR_new[v], contribution * w);
        }
    }
}


// =========================================================================
// KERNEL: distribute_dangling_mass  (BACKUP - kept for compatibility)
// Now superseded by initialize_pr_with_dangling.
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
// KERNEL: sum_dangling_pr  (UNCHANGED - feeds reduce_to_scalar_f32)
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
// KERNEL: compute_l1_convergence  (UNCHANGED - feeds reduce_to_scalar_f32)
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

    for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, off);
    }

    const int lane    = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    if (threadIdx.x < WARP_SIZE) {
        val = (threadIdx.x < WARPS_PER_BLOCK) ? smem[threadIdx.x] : 0.0f;
        for (int off = WARPS_PER_BLOCK >> 1; off > 0; off >>= 1) {
            val += __shfl_down_sync(0xffffffffu, val, off);
        }
        if (threadIdx.x == 0) partial_sums[blockIdx.x] = val;
    }
}


// =========================================================================
// KERNEL: reduce_to_scalar_f32  (NEW - Improvement 1)
//
// Single-block kernel.  Reads a per-block partial-sums array (length
// `input_len`, typically ceil(n / BLOCK_SIZE) entries) and writes the
// total sum to scalar_output[0] (a length-1 GPU array).  Used by both
// sum_dangling_pr and compute_l1_convergence to keep their final
// reductions fully on GPU.
//
// Launch contract: grid=(1, 1, 1), block=(BLOCK_SIZE, 1, 1).
// =========================================================================
__global__ void reduce_to_scalar_f32(
    const float* __restrict__ partial_input,
    float*       __restrict__ scalar_output,
    const int                  input_len)
{
    __shared__ float smem[BLOCK_SIZE];
    const int tid = threadIdx.x;

    float val = 0.0f;
    for (int i = tid; i < input_len; i += BLOCK_SIZE) {
        val += partial_input[i];
    }
    smem[tid] = val;
    __syncthreads();

    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) scalar_output[0] = smem[0];
}


// =========================================================================
// KERNEL: gather_contributions_pull  (NEW - Improvement 6)
//
// Pull-based contribution accumulation for high-in-degree target nodes.
// One block per pull node u.  Threads in the block cooperatively scan
// u's incoming edges (rows of A^T) and accumulate
//     sum_v  values_T[v,u] * PR_old[v] / out_degree[v]
// via warp/block reductions.  Thread 0 writes the final result to
// PR_new[u] using a plain (non-atomic) += : because the scatter kernels
// have already skipped pull-target edges (via pull_target_mask) and
// only one block ever writes to PR_new[u] in this kernel, the write is
// race-free.
//
// IMPORTANT: PR_new[u] must already contain the teleport + dangling
// baseline (set by initialize_pr_with_dangling) before this kernel runs.
// =========================================================================
__global__ void gather_contributions_pull(
    const int*   __restrict__ row_ptr_T,
    const int*   __restrict__ col_idx_T,
    const float* __restrict__ values_T,
    const float* __restrict__ PR_old,
    float*       __restrict__ PR_new,
    const float* __restrict__ out_degree,
    const int*   __restrict__ pull_node_ids,
    const int                  num_pull_nodes,
    const float                damping)
{
    __shared__ float smem[BLOCK_SIZE];

    const int pid = blockIdx.x;
    if (pid >= num_pull_nodes) return;
    const int u = pull_node_ids[pid];

    const int row_start = row_ptr_T[u];
    const int row_end   = row_ptr_T[u + 1];

    float partial = 0.0f;
    for (int j = row_start + threadIdx.x; j < row_end; j += BLOCK_SIZE) {
        const int   v     = col_idx_T[j];
        const float deg_v = out_degree[v];
        if (deg_v > 0.0f) {
            partial += values_T[j] * PR_old[v] / deg_v;
        }
    }

    smem[threadIdx.x] = partial;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        PR_new[u] += damping * smem[0];
    }
}

}  // extern "C"
"""


# ---------------------------------------------------------------------------
# Module-level kernel cache + adaptive compilation
# ---------------------------------------------------------------------------

_kernel_cache: dict[str, dict[str, Any]] = {}


def _detect_arch_flag() -> str:
    """Return ``-arch=sm_XY`` for the current device, fallback to sm_75."""
    try:
        cuda.init()
        cc_major, cc_minor = cuda.Device(0).compute_capability()
        return f"-arch=sm_{cc_major}{cc_minor}"
    except Exception:                                   # noqa: BLE001
        return "-arch=sm_75"


def _get_kernels() -> dict[str, Any]:
    """Compile (or fetch from cache) the PageRank device kernels."""
    if "pagerank" not in _kernel_cache:
        if not PYCUDA_AVAILABLE:
            raise RuntimeError(
                "PyCUDA is required to compile PageRank kernels - "
                "install pycuda and ensure NVCC is on PATH."
            )
        arch_flag = _detect_arch_flag()
        mod = SourceModule(
            KERNEL_SOURCE,
            options=[arch_flag, "-O3"],
            no_extern_c=True,
        )
        _kernel_cache["pagerank"] = {
            "init":            mod.get_function("initialize_pr"),
            "init_dangling":   mod.get_function("initialize_pr_with_dangling"),
            "scatter_csr":     mod.get_function("scatter_contributions_csr"),
            "scatter_ell":     mod.get_function("scatter_contributions_ellpack"),
            "dangling":        mod.get_function("distribute_dangling_mass"),
            "sum_dangling":    mod.get_function("sum_dangling_pr"),
            "l1_conv":         mod.get_function("compute_l1_convergence"),
            "reduce_scalar":   mod.get_function("reduce_to_scalar_f32"),
            "pull_gather":     mod.get_function("gather_contributions_pull"),
            "_arch_flag":      arch_flag,
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
    """Sum of outgoing edge weights per node (FP32)."""
    return np.asarray(
        graph_csr.sum(axis=1), dtype=np.float32
    ).flatten()


def _compute_in_degrees(graph_csr: sp.csr_matrix) -> np.ndarray:
    """Sum of incoming edge weights per node (FP32)."""
    return np.asarray(
        graph_csr.sum(axis=0), dtype=np.float32
    ).flatten()


def _identify_dangling_nodes(out_degrees: np.ndarray) -> np.ndarray:
    """Boolean mask: True where out_degree == 0."""
    return out_degrees == 0.0


def _eligible_mask(
    graph_csr: sp.csr_matrix,
    out_degrees: np.ndarray,
    network_type: str,
    node_index_map: dict | None,
) -> tuple[np.ndarray, str]:
    """Return a uint8 boolean mask of "eligible" nodes for dangling redistribution.

    Replaces the previous index-array convention with a length-n mask so
    that ``initialize_pr_with_dangling`` can decide per-thread without a
    binary search.

    Returns (mask, note) where mask[i] == 1 iff node i receives dangling mass.
    """
    n = int(graph_csr.shape[0])
    nt = str(network_type).lower()
    mask = np.zeros(n, dtype=np.uint8)
    if nt == "grn":
        mask[out_degrees > 0.0] = 1
        note = "dangling mass -> regulator nodes only (out_degree > 0)"
    elif nt == "mirna":
        # Without type labels in node_index_map, miRNA identity is
        # inferred from the bipartite structure (only miRNAs have outgoing
        # edges).  This matches the previous _identify_mirna_nodes path.
        mask[out_degrees > 0.0] = 1
        note = "dangling mass -> miRNA nodes only (out_degree > 0)"
    else:  # ppi (and default)
        mask[:] = 1
        note = "dangling mass -> all nodes (uniform)"
    return mask, note


def _compute_adaptive_hub_threshold(
    out_degrees: np.ndarray,
    target_ellpack_fraction: float = 0.05,
) -> int:
    """Compute a hub threshold so ~target_ellpack_fraction of nodes go to ELLPACK.

    For biological power-law networks the threshold lands near the
    (1 - target) quantile of the degree distribution.  The result is
    clamped to >= WARP_SIZE (so ELLPACK actually benefits from warp-
    level processing) and snapped to the nearest power of two for
    efficient warp/block math.
    """
    if out_degrees.size == 0:
        return WARP_SIZE

    # Use integer out-degree counts for the percentile (weight-free).
    deg_int = out_degrees.astype(np.int64)
    pct = (1.0 - max(0.0, min(target_ellpack_fraction, 1.0))) * 100.0
    threshold = int(np.percentile(deg_int, pct))

    threshold = max(threshold, WARP_SIZE)
    max_deg = int(deg_int.max()) if deg_int.size > 0 else WARP_SIZE
    threshold = min(threshold, max_deg)
    threshold = max(threshold, 1)

    # Snap to nearest power of two.
    snapped = int(2 ** round(math.log2(max(threshold, 1))))
    snapped = max(snapped, WARP_SIZE)
    return snapped


def _build_ellpack(
    graph_csr: sp.csr_matrix,
    out_degrees: np.ndarray,
    hub_threshold: int,
) -> tuple[dict, dict]:
    """Split source nodes into CSR (low-degree) and ELLPACK (hub) groups."""
    n        = int(graph_csr.shape[0])
    indptr   = np.ascontiguousarray(graph_csr.indptr,  dtype=np.int32)
    indices  = np.ascontiguousarray(graph_csr.indices, dtype=np.int32)
    data     = np.ascontiguousarray(graph_csr.data,    dtype=np.float32)
    deg_int  = np.diff(indptr).astype(np.int32)

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
        "row_ptr":   indptr,
        "col_idx":   indices,
        "values":    data,
    }
    return ellpack_data, csr_remainder


def _estimate_pagerank_vram(
    n: int,
    nnz: int,
    num_hubs: int,
    max_row_len: int,
    has_pull: bool = False,
) -> int:
    """Rough VRAM estimate for the full PageRank working set, in bytes."""
    csr_b      = (n + 1 + 2 * nnz) * 4
    ellpack_b  = num_hubs * max_row_len * 8
    pr_b       = 2 * n * 4
    partial_b  = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE) * 4
    misc_b     = n * 4 * 5
    pull_b     = ((n + 1 + 2 * nnz) * 4) if has_pull else 0
    return int(csr_b + ellpack_b + pr_b + partial_b + misc_b + pull_b)


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
# Chunked-path helpers (Improvement 5)
# ---------------------------------------------------------------------------

class _ChunkBuffer:
    """Pre-allocated per-chunk GPU buffer set (double-buffer slot)."""
    __slots__ = ("d_row_ptr", "d_col_idx", "d_values", "d_node_ids",
                 "max_nnz", "max_nodes")

    def __init__(self, max_nnz: int, max_nodes: int) -> None:
        self.max_nnz   = int(max_nnz)
        self.max_nodes = int(max_nodes)
        self.d_row_ptr  = gpuarray.empty((self.max_nodes + 1,), np.int32)
        self.d_col_idx  = gpuarray.empty((self.max_nnz,),       np.int32)
        self.d_values   = gpuarray.empty((self.max_nnz,),       np.float32)
        self.d_node_ids = gpuarray.empty((self.max_nodes,),     np.int32)

    def free(self) -> None:
        for arr in (self.d_row_ptr, self.d_col_idx,
                    self.d_values,  self.d_node_ids):
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass


def _allocate_chunk_buffer(max_nnz: int, max_nodes: int) -> _ChunkBuffer:
    return _ChunkBuffer(max_nnz, max_nodes)


def _transfer_chunk_async(
    chunk_indptr: np.ndarray,
    chunk_indices: np.ndarray,
    chunk_values: np.ndarray,
    chunk_node_ids: np.ndarray,
    buf: _ChunkBuffer,
    stream,
) -> None:
    """Async H2D of a chunk's CSR triple + node-id list into ``buf``."""
    cuda.memcpy_htod_async(buf.d_row_ptr.gpudata,
                            np.ascontiguousarray(chunk_indptr,  np.int32),
                            stream)
    cuda.memcpy_htod_async(buf.d_col_idx.gpudata,
                            np.ascontiguousarray(chunk_indices, np.int32),
                            stream)
    cuda.memcpy_htod_async(buf.d_values.gpudata,
                            np.ascontiguousarray(chunk_values,  np.float32),
                            stream)
    cuda.memcpy_htod_async(buf.d_node_ids.gpudata,
                            np.ascontiguousarray(chunk_node_ids, np.int32),
                            stream)


def _build_csr_chunks(
    graph_csr: sp.csr_matrix,
    out_degrees: np.ndarray,
    chunk_size: int,
) -> list[dict]:
    """Split the CSR row-wise into chunks of at most ``chunk_size`` rows.

    Each chunk dict carries:
        node_ids : global row indices included in this chunk
        indptr   : LOCAL CSR row_ptr (re-based to start at 0)
        indices  : column indices (full graph - global node IDs preserved)
        values   : edge weights
        nnz      : len(indices)
    """
    n = int(graph_csr.shape[0])
    indptr  = graph_csr.indptr
    indices = graph_csr.indices
    values  = graph_csr.data.astype(np.float32, copy=False)

    chunks: list[dict] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # Restrict to rows that actually contribute (out_degree > 0).
        node_ids = np.arange(start, end, dtype=np.int32)
        mask     = out_degrees[start:end] > 0.0
        node_ids = node_ids[mask]
        if node_ids.size == 0:
            continue

        # Slice the CSR by the kept rows.
        row_starts = indptr[node_ids]
        row_ends   = indptr[node_ids + 1]
        row_lens   = (row_ends - row_starts).astype(np.int32)
        # Concatenate the kept rows' slices.
        nnz = int(row_lens.sum())
        if nnz == 0:
            continue
        local_cols = np.empty((nnz,), dtype=np.int32)
        local_vals = np.empty((nnz,), dtype=np.float32)
        offset = 0
        for u, s, e in zip(node_ids, row_starts, row_ends):
            L = int(e - s)
            local_cols[offset:offset + L] = indices[s:e]
            local_vals[offset:offset + L] = values[s:e]
            offset += L
        local_indptr = np.zeros((node_ids.size + 1,), dtype=np.int32)
        np.cumsum(row_lens, out=local_indptr[1:])

        chunks.append({
            "node_ids": node_ids,
            "indptr":   local_indptr,
            "indices":  local_cols,
            "values":   local_vals,
            "nnz":      nnz,
        })
    return chunks


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def pagerank_gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """PageRank - GPU-accelerated via custom PyCUDA kernels.

    See module docstring for the full algorithm; this routine adds:
      - GPU-side scalar reductions (one sync per iteration);
      - fused teleport + dangling redistribution;
      - adaptive hub threshold for the CSR / ELLPACK split;
      - optional pull-based contribution gather for extreme-in-degree hubs;
      - chunked execution path for graphs that exceed VRAM.
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA is required for pagerank_gpu(). "
            "Install it or use pagerank_cpu_single() from "
            "src/algorithms/cpu/single_threaded/pagerank.py"
        )

    pushed_ctx = None
    try:
        cuda.init()
        if cuda.Device.count() <= 0:
            raise RuntimeError("No CUDA device available")
        pushed_ctx = cuda.Device(0).retain_primary_context()
        pushed_ctx.push()
    except cuda.LogicError as e:
        raise RuntimeError(f"CUDA initialisation failed: {e}") from e

    d_buffers: list = []

    try:
        # ---- Parameter merging ----------------------------------------
        p = _merge_params(params)
        if _GPU_CONFIG_AVAILABLE:
            p = apply_config("pagerank", graph_csr, p) or p
            for k, v in _DEFAULT_PARAMS.items():
                p.setdefault(k, v)

        damping           = float(p["damping"])
        max_iter          = int(p["max_iter"])
        tolerance         = float(p["tolerance"])
        network_type      = str(p.get("network_type", "grn"))
        block_size        = int(p.get("block_size", BLOCK_SIZE))
        if block_size <= 0 or block_size > 1024:
            block_size = BLOCK_SIZE
        use_chunking      = bool(p.get("use_chunking", False))
        ellpack_fraction  = float(p.get("ellpack_fraction", 0.05))
        pull_threshold    = int(p.get("pull_threshold", 0))
        node_index_map    = p.get("node_index_map")

        n = int(graph_csr.shape[0])
        if n == 0:
            raise ValueError("Empty graph")

        teleport_val = np.float32((1.0 - damping) / n)
        init_val     = np.float32(1.0 / n)

        # ---- CPU preprocessing ----------------------------------------
        out_degrees    = _compute_out_degrees(graph_csr)
        dangling_mask  = _identify_dangling_nodes(out_degrees)
        dangling_flags = dangling_mask.astype(np.int32)

        eligible_mask_host, eligible_note = _eligible_mask(
            graph_csr, out_degrees, network_type, node_index_map,
        )
        num_eligible = int(eligible_mask_host.sum())
        if num_eligible == 0:
            logging.warning(
                "pagerank_gpu: no eligible nodes for dangling redistribution "
                "(network_type=%s) - falling back to uniform.", network_type,
            )
            eligible_mask_host[:] = 1
            num_eligible = n
            eligible_note += " (fallback to uniform - no eligible nodes found)"

        # ---- Adaptive hub threshold + ELLPACK split -------------------
        hub_threshold = _compute_adaptive_hub_threshold(
            out_degrees, target_ellpack_fraction=ellpack_fraction,
        )
        ellpack_data, csr_remainder = _build_ellpack(
            graph_csr, out_degrees, hub_threshold,
        )
        logging.info(
            "PageRank adaptive hub threshold: %d "
            "(%d hubs, %d CSR nodes; ellpack_fraction target=%.3f)",
            hub_threshold,
            ellpack_data["num_hubs"],
            csr_remainder["num_low"],
            ellpack_fraction,
        )

        # ---- Pull-mode setup (optional) --------------------------------
        in_degrees   = _compute_in_degrees(graph_csr).astype(np.int64)
        pull_enabled = pull_threshold > 0
        if pull_enabled:
            pull_ids = np.where(in_degrees >= pull_threshold)[0].astype(np.int32)
            pull_enabled = pull_ids.size > 0
        else:
            pull_ids = np.zeros((0,), dtype=np.int32)

        pull_target_mask_host = np.zeros((n,), dtype=np.uint8)
        if pull_enabled:
            pull_target_mask_host[pull_ids] = 1
            logging.info(
                "PageRank pull-mode: %d pull-target nodes (in_degree >= %d)",
                int(pull_ids.size), int(pull_threshold),
            )

        # ---- VRAM check / chunked path decision -----------------------
        est_bytes = _estimate_pagerank_vram(
            n=n, nnz=int(graph_csr.nnz),
            num_hubs=ellpack_data["num_hubs"],
            max_row_len=ellpack_data["max_row_len"],
            has_pull=pull_enabled,
        )
        try:
            free_bytes, _total = cuda.mem_get_info()
        except Exception:                               # noqa: BLE001
            free_bytes = 1 << 30
        if est_bytes > free_bytes:
            raise MemoryError(
                f"PageRank GPU needs ~{est_bytes/1e6:.1f} MB but only "
                f"{free_bytes/1e6:.1f} MB free.  Use a higher-VRAM "
                f"device or enable use_chunking=True."
            )
        auto_chunk = est_bytes > 0.8 * free_bytes
        if use_chunking or auto_chunk:
            return _pagerank_gpu_chunked(
                graph_csr, p, out_degrees, dangling_flags,
                eligible_mask_host, eligible_note, num_eligible,
                pull_enabled, pull_ids, pull_target_mask_host,
                in_degrees, ellpack_data, csr_remainder,
                damping, max_iter, tolerance, network_type, block_size,
                teleport_val, init_val, n,
            )

        kernels = _get_kernels()

        # ---- Streams + timing -----------------------------------------
        stream_compute  = cuda.Stream()
        stream_transfer = cuda.Stream()
        start_event     = cuda.Event()
        end_event       = cuda.Event()
        start_event.record(stream_compute)

        def _to_gpu(arr: np.ndarray):
            ga = gpuarray.to_gpu_async(arr, stream=stream_transfer)
            d_buffers.append(ga)
            return ga

        def _empty(shape, dtype):
            ga = gpuarray.empty(shape, dtype=dtype)
            d_buffers.append(ga)
            return ga

        # ---- Device allocation + async H2D ----------------------------
        d_csr_row_ptr = _to_gpu(csr_remainder["row_ptr"])
        d_csr_col_idx = _to_gpu(csr_remainder["col_idx"])
        d_csr_values  = _to_gpu(csr_remainder["values"])
        d_low_ids     = (
            _to_gpu(csr_remainder["node_ids"])
            if csr_remainder["num_low"] > 0
            else None
        )

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
        d_eligible_mask  = _to_gpu(eligible_mask_host)
        d_pull_target_mask = (_to_gpu(pull_target_mask_host)
                              if pull_enabled else None)

        # Pull mode arrays (transposed CSR + pull-target node IDs).
        d_row_ptr_T = d_col_idx_T = d_values_T = d_pull_ids = None
        if pull_enabled:
            graph_csr_T = graph_csr.T.tocsr().astype(np.float32)
            d_row_ptr_T = _to_gpu(
                np.ascontiguousarray(graph_csr_T.indptr,  np.int32))
            d_col_idx_T = _to_gpu(
                np.ascontiguousarray(graph_csr_T.indices, np.int32))
            d_values_T  = _to_gpu(
                np.ascontiguousarray(graph_csr_T.data,    np.float32))
            d_pull_ids  = _to_gpu(pull_ids)

        d_PR_old = _empty((n,), np.float32)
        d_PR_new = _empty((n,), np.float32)

        n_partial_blocks = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE)
        d_partial         = _empty((n_partial_blocks,), np.float32)
        d_dangling_scalar = _empty((1,), np.float32)
        d_l1_scalar       = _empty((1,), np.float32)

        # PR_old = 1/N (initial uniform distribution).
        init_host = np.full(n, init_val, dtype=np.float32)
        cuda.memcpy_htod_async(d_PR_old.gpudata, init_host, stream_transfer)
        stream_transfer.synchronize()

        # ---- Iteration loop -------------------------------------------
        converged   = False
        iterations  = 0
        n_init_grid = ((n + block_size - 1) // block_size, 1, 1)
        n_block_dim = (block_size, 1, 1)
        partial_grid  = (n_partial_blocks, 1, 1)
        partial_block = (BLOCK_SIZE, 1, 1)

        k_init_dang = kernels["init_dangling"]
        k_sc_csr    = kernels["scatter_csr"]
        k_sc_ell    = kernels["scatter_ell"]
        k_sum_dang  = kernels["sum_dangling"]
        k_l1        = kernels["l1_conv"]
        k_reduce    = kernels["reduce_scalar"]
        k_pull      = kernels["pull_gather"]

        # Null mask for scatter kernels when pull mode is disabled - PyCUDA
        # cannot pass a Python None as a pointer arg; we wrap it in a
        # device-side 0-byte trick by passing the address of a single byte
        # zero buffer.  When pull_target_mask is the null pointer, the
        # kernel branch `if (pull_target_mask != 0 && ...)` short-circuits
        # to false and behaviour matches the original scatter.
        if d_pull_target_mask is None:
            # Use a single-element 0-byte to keep the address valid; the
            # kernel only dereferences when pull_target_mask != 0 - but
            # we pass np.intp(0) as the address via a zero-length array
            # is not portable.  Simpler: always allocate a 1-byte zero
            # mask and pass it; the kernel's check `pull_target_mask[v]`
            # with v < n must read 0 for every v.  Allocate a length-n
            # zero mask only when pull is disabled.
            d_pull_target_mask = _to_gpu(
                np.zeros((n,), dtype=np.uint8))

        for it in range(max_iter):
            iterations = it + 1

            # 1. Partial sum of dangling PR_old.
            k_sum_dang(
                d_PR_old, d_dangling_flags, d_partial, np.int32(n),
                block=partial_block, grid=partial_grid,
                stream=stream_compute,
            )

            # 2. GPU-side reduction -> scalar (no CPU sync).
            k_reduce(
                d_partial, d_dangling_scalar,
                np.int32(n_partial_blocks),
                block=(BLOCK_SIZE, 1, 1), grid=(1, 1, 1),
                stream=stream_compute,
            )

            # 3. Fused teleport init + dangling redistribution.
            k_init_dang(
                d_PR_new, teleport_val, d_dangling_scalar,
                d_eligible_mask, np.int32(num_eligible),
                np.float32(damping), np.int32(n),
                block=n_block_dim, grid=n_init_grid,
                stream=stream_compute,
            )

            # 4. Scatter from CSR (low/medium-degree) sources.
            if csr_remainder["num_low"] > 0 and d_low_ids is not None:
                k_sc_csr(
                    d_csr_row_ptr, d_csr_col_idx, d_csr_values,
                    d_PR_old, d_PR_new, d_out_degree, d_low_ids,
                    d_pull_target_mask,
                    np.int32(csr_remainder["num_low"]),
                    np.float32(damping), np.int32(n),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(csr_remainder["num_low"], 1, 1),
                    stream=stream_compute,
                )

            # 5. Scatter from ELLPACK (hub) sources.
            if num_hubs > 0 and d_ellpack_cols is not None:
                k_sc_ell(
                    d_ellpack_cols, d_ellpack_vals, d_hub_ids,
                    d_PR_old, d_PR_new, d_out_degree,
                    d_pull_target_mask,
                    np.int32(num_hubs), np.int32(max_row_len),
                    np.float32(damping),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(num_hubs, 1, 1),
                    stream=stream_compute,
                )

            # 6. Pull-mode gather for extreme-in-degree targets.
            if pull_enabled and d_pull_ids is not None:
                k_pull(
                    d_row_ptr_T, d_col_idx_T, d_values_T,
                    d_PR_old, d_PR_new, d_out_degree,
                    d_pull_ids, np.int32(int(pull_ids.size)),
                    np.float32(damping),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(int(pull_ids.size), 1, 1),
                    stream=stream_compute,
                )

            # 7. L1 convergence reduction.
            k_l1(
                d_PR_new, d_PR_old, d_partial, np.int32(n),
                block=partial_block, grid=partial_grid,
                stream=stream_compute,
            )
            k_reduce(
                d_partial, d_l1_scalar,
                np.int32(n_partial_blocks),
                block=(BLOCK_SIZE, 1, 1), grid=(1, 1, 1),
                stream=stream_compute,
            )

            # 8. ONE sync per iteration.
            stream_compute.synchronize()
            l1_norm = float(d_l1_scalar.get()[0])

            # 9. Pointer swap (no data copy).
            d_PR_old, d_PR_new = d_PR_new, d_PR_old

            if l1_norm < tolerance:
                converged = True
                break

        end_event.record(stream_compute)
        end_event.synchronize()
        elapsed = start_event.time_till(end_event) / 1000.0

        # ---- Result extraction (NOT timed) ----------------------------
        scores_host = d_PR_old.get()

        inner = _pack_result(
            scores_host, out_degrees, iterations, converged, network_type,
        )
        inner["note"] = (
            f"{eligible_note}; "
            f"hub_threshold={hub_threshold}, "
            f"num_hubs={ellpack_data['num_hubs']}, "
            f"pull_targets={int(pull_ids.size) if pull_enabled else 0}, "
            f"arch={kernels.get('_arch_flag', '?')}, "
            f"syncs_per_iter=1"
        )

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
            "device, or set use_chunking=True / reduce graph size."
        )
        raise
    finally:
        for arr in d_buffers:
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass
        if pushed_ctx is not None:
            try:
                pushed_ctx.pop()
            except Exception:                           # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Chunked execution path (Improvement 5)
# ---------------------------------------------------------------------------

def _pagerank_gpu_chunked(
    graph_csr: sp.csr_matrix,
    p: dict,
    out_degrees: np.ndarray,
    dangling_flags: np.ndarray,
    eligible_mask_host: np.ndarray,
    eligible_note: str,
    num_eligible: int,
    pull_enabled: bool,
    pull_ids: np.ndarray,
    pull_target_mask_host: np.ndarray,
    in_degrees: np.ndarray,
    ellpack_data: dict,
    csr_remainder: dict,
    damping: float,
    max_iter: int,
    tolerance: float,
    network_type: str,
    block_size: int,
    teleport_val: np.float32,
    init_val: np.float32,
    n: int,
) -> dict:
    """PageRank with chunked CSR uploads + double-buffer transfer pipeline.

    PR_old / PR_new stay fully resident on the device (n floats each);
    only the CSR row data is streamed in per chunk via a ping-pong pair
    of pre-allocated transfer buffers.  Compute on chunk i overlaps with
    transfer of chunk i+1 via ``stream_transfer`` + ``cuda.Event``.

    The ELLPACK hub kernel is NOT chunked here (hubs are typically a tiny
    fraction of nodes and already fit in VRAM); it runs once per iteration
    after the chunked CSR scatter completes.  Pull-mode is similarly
    treated as a single full-graph launch.
    """
    kernels = _get_kernels()

    # ---- Chunk size from free VRAM --------------------------------------
    try:
        free_bytes, _total = cuda.mem_get_info()
    except Exception:                                   # noqa: BLE001
        free_bytes = 1 << 30
    avg_nnz = max(1.0, graph_csr.nnz / max(1, n))
    bytes_per_node = (1 + 2 * avg_nnz) * 4
    available = int(free_bytes * 0.60)        # 60% budget for chunks
    chunk_size = max(1, min(n, int(available / max(1.0, bytes_per_node))))

    chunks = _build_csr_chunks(graph_csr, out_degrees, chunk_size)
    if not chunks:
        # No non-dangling rows - degenerate; return uniform scores.
        scores_host = np.full(n, init_val, dtype=np.float32)
        inner = _pack_result(scores_host, out_degrees, 0, False, network_type)
        inner["note"] = eligible_note + "; chunked path (no rows to scatter)"
        return {
            "algorithm":      "pagerank",
            "mode":           "gpu",
            "network_type":   network_type,
            "execution_time": 0.0,
            "num_nodes":      n,
            "num_edges":      int(graph_csr.nnz),
            "result":         inner,
        }

    # ---- Streams + events + double buffers ------------------------------
    stream_compute  = cuda.Stream()
    stream_transfer = cuda.Stream()
    start_event     = cuda.Event()
    end_event       = cuda.Event()

    max_chunk_nnz   = max(c["nnz"] for c in chunks)
    max_chunk_nodes = max(c["node_ids"].size for c in chunks)
    buffers = [
        _allocate_chunk_buffer(max_chunk_nnz, max_chunk_nodes),
        _allocate_chunk_buffer(max_chunk_nnz, max_chunk_nodes),
    ]
    events = [cuda.Event(), cuda.Event()]

    d_buffers: list = []

    def _to_gpu(arr: np.ndarray):
        ga = gpuarray.to_gpu_async(arr, stream=stream_transfer)
        d_buffers.append(ga)
        return ga

    def _empty(shape, dtype):
        ga = gpuarray.empty(shape, dtype=dtype)
        d_buffers.append(ga)
        return ga

    try:
        # ELLPACK hubs (small enough to fully resident).
        num_hubs    = ellpack_data["num_hubs"]
        max_row_len = ellpack_data["max_row_len"]
        if num_hubs > 0:
            d_ellpack_cols = _to_gpu(ellpack_data["cols"])
            d_ellpack_vals = _to_gpu(ellpack_data["vals"])
            d_hub_ids      = _to_gpu(ellpack_data["hub_ids"])
        else:
            d_ellpack_cols = d_ellpack_vals = d_hub_ids = None

        d_out_degree     = _to_gpu(out_degrees)
        d_dangling_flags = _to_gpu(dangling_flags)
        d_eligible_mask  = _to_gpu(eligible_mask_host)
        d_pull_target_mask = _to_gpu(
            pull_target_mask_host
            if pull_enabled else np.zeros((n,), dtype=np.uint8)
        )

        # Pull-mode arrays (full-graph transposed CSR).
        d_row_ptr_T = d_col_idx_T = d_values_T = d_pull_ids = None
        if pull_enabled:
            graph_csr_T = graph_csr.T.tocsr().astype(np.float32)
            d_row_ptr_T = _to_gpu(
                np.ascontiguousarray(graph_csr_T.indptr,  np.int32))
            d_col_idx_T = _to_gpu(
                np.ascontiguousarray(graph_csr_T.indices, np.int32))
            d_values_T  = _to_gpu(
                np.ascontiguousarray(graph_csr_T.data,    np.float32))
            d_pull_ids  = _to_gpu(pull_ids)

        d_PR_old = _empty((n,), np.float32)
        d_PR_new = _empty((n,), np.float32)
        n_partial_blocks = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE)
        d_partial         = _empty((n_partial_blocks,), np.float32)
        d_dangling_scalar = _empty((1,), np.float32)
        d_l1_scalar       = _empty((1,), np.float32)

        init_host = np.full(n, init_val, dtype=np.float32)
        cuda.memcpy_htod_async(d_PR_old.gpudata, init_host, stream_transfer)
        stream_transfer.synchronize()

        start_event.record(stream_compute)

        n_init_grid   = ((n + block_size - 1) // block_size, 1, 1)
        n_block_dim   = (block_size, 1, 1)
        partial_grid  = (n_partial_blocks, 1, 1)
        partial_block = (BLOCK_SIZE, 1, 1)

        k_init_dang = kernels["init_dangling"]
        k_sc_csr    = kernels["scatter_csr"]
        k_sc_ell    = kernels["scatter_ell"]
        k_sum_dang  = kernels["sum_dangling"]
        k_l1        = kernels["l1_conv"]
        k_reduce    = kernels["reduce_scalar"]
        k_pull      = kernels["pull_gather"]

        converged   = False
        iterations  = 0

        for it in range(max_iter):
            iterations = it + 1

            # 1. Dangling sum + GPU reduction.
            k_sum_dang(
                d_PR_old, d_dangling_flags, d_partial, np.int32(n),
                block=partial_block, grid=partial_grid,
                stream=stream_compute,
            )
            k_reduce(
                d_partial, d_dangling_scalar,
                np.int32(n_partial_blocks),
                block=(BLOCK_SIZE, 1, 1), grid=(1, 1, 1),
                stream=stream_compute,
            )

            # 2. Fused init + dangling redistribution.
            k_init_dang(
                d_PR_new, teleport_val, d_dangling_scalar,
                d_eligible_mask, np.int32(num_eligible),
                np.float32(damping), np.int32(n),
                block=n_block_dim, grid=n_init_grid,
                stream=stream_compute,
            )

            # 3. Chunked scatter with double-buffer pipeline.
            _transfer_chunk_async(
                chunks[0]["indptr"], chunks[0]["indices"],
                chunks[0]["values"], chunks[0]["node_ids"],
                buffers[0], stream_transfer,
            )
            events[0].record(stream_transfer)

            for i, chunk in enumerate(chunks):
                slot = i % 2
                buf  = buffers[slot]
                stream_compute.wait_for_event(events[slot])

                k_sc_csr(
                    buf.d_row_ptr, buf.d_col_idx, buf.d_values,
                    d_PR_old, d_PR_new, d_out_degree, buf.d_node_ids,
                    d_pull_target_mask,
                    np.int32(int(chunk["node_ids"].size)),
                    np.float32(damping), np.int32(n),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(int(chunk["node_ids"].size), 1, 1),
                    stream=stream_compute,
                )

                if i + 1 < len(chunks):
                    next_slot = (i + 1) % 2
                    nc = chunks[i + 1]
                    _transfer_chunk_async(
                        nc["indptr"], nc["indices"],
                        nc["values"], nc["node_ids"],
                        buffers[next_slot], stream_transfer,
                    )
                    events[next_slot].record(stream_transfer)

            # 4. ELLPACK hub scatter (not chunked).
            if num_hubs > 0 and d_ellpack_cols is not None:
                k_sc_ell(
                    d_ellpack_cols, d_ellpack_vals, d_hub_ids,
                    d_PR_old, d_PR_new, d_out_degree,
                    d_pull_target_mask,
                    np.int32(num_hubs), np.int32(max_row_len),
                    np.float32(damping),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(num_hubs, 1, 1),
                    stream=stream_compute,
                )

            # 5. Pull-mode gather (not chunked).
            if pull_enabled and d_pull_ids is not None:
                k_pull(
                    d_row_ptr_T, d_col_idx_T, d_values_T,
                    d_PR_old, d_PR_new, d_out_degree,
                    d_pull_ids, np.int32(int(pull_ids.size)),
                    np.float32(damping),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(int(pull_ids.size), 1, 1),
                    stream=stream_compute,
                )

            # 6. L1 convergence + GPU reduction.
            k_l1(
                d_PR_new, d_PR_old, d_partial, np.int32(n),
                block=partial_block, grid=partial_grid,
                stream=stream_compute,
            )
            k_reduce(
                d_partial, d_l1_scalar,
                np.int32(n_partial_blocks),
                block=(BLOCK_SIZE, 1, 1), grid=(1, 1, 1),
                stream=stream_compute,
            )

            stream_compute.synchronize()
            l1_norm = float(d_l1_scalar.get()[0])

            d_PR_old, d_PR_new = d_PR_new, d_PR_old
            if l1_norm < tolerance:
                converged = True
                break

        end_event.record(stream_compute)
        end_event.synchronize()
        elapsed = start_event.time_till(end_event) / 1000.0

        scores_host = d_PR_old.get()
        inner = _pack_result(
            scores_host, out_degrees, iterations, converged, network_type,
        )
        inner["note"] = (
            f"{eligible_note}; chunked path "
            f"({len(chunks)} chunks of ~{chunk_size} rows each); "
            f"hub_threshold={ellpack_data['num_hubs']}, "
            f"arch={kernels.get('_arch_flag', '?')}, "
            f"syncs_per_iter=1"
        )
        return {
            "algorithm":      "pagerank",
            "mode":           "gpu",
            "network_type":   network_type,
            "execution_time": elapsed,
            "num_nodes":      n,
            "num_edges":      int(graph_csr.nnz),
            "result":         inner,
        }

    finally:
        for buf in buffers:
            buf.free()
        for arr in d_buffers:
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _gpu(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    """Runner entry point - preserves the legacy ``{output, extra_params}`` shape."""
    p = _merge_params(params)
    full = pagerank_gpu(graph_csr, p)
    return {"output": full, "extra_params": p}
