"""
algorithms/rwr.py — Random Walk with Restart for Biological Network Diffusion
==============================================================================

Biological context
------------------
Random Walk with Restart propagates probability mass from a seed set
across a network.  At each step the walker follows an out-edge with
probability (1 - r) or teleports back to a seed with probability r.
The steady-state vector ranks every node by its diffusion-influence
relative to the seeds:

  GRN    — seed with disease-associated TFs; top-ranked genes are candidate
            downstream regulatory targets.
  PPI    — seed with known disease proteins; top-ranked proteins are candidate
            disease modifiers.
  miRNA  — seed with miRNAs of interest; top-ranked genes are candidate
            co-targeted effectors.

Steady-state equation:
    p* = (1 - r) · W · p* + r · p₀

W is the column-stochastic transition matrix and p₀ is the seed
distribution (uniform over seeds; uniform over all nodes when no seeds).

Network-type-aware transition matrix
-------------------------------------
GRN   : directed CSR, column-normalised by out-degree.
        Dangling columns (out_degree == 0) get a self-loop.
PPI   : A_sym = A + Aᵀ, binarised, then column-normalised.
miRNA : directed bipartite CSR (miRNA → gene), column-normalised.
        Gene-target nodes (out_degree == 0) get self-loops.

Optimised GPU pipeline
-----------------------
1.  Warp-per-node SpMV (rwr_spmv_warp_per_node / _fp16w / _batched):
    One 256-thread block processes NODES_PER_BLOCK = 8 nodes at a time.
    Each warp (32 threads) handles exactly one node — warp-shuffle
    reduction, no shared-memory overhead, no per-warp __syncthreads__.
    Compared to the previous one-block-per-node design:
      • 8× fewer blocks → 8× lower scheduler pressure
      • All warps in a block do useful work simultaneously
      • Superior occupancy for the low-degree-majority of biological nets

2.  GPU-side convergence reduction:
    l1_convergence_rwr writes per-block L1 partials; reduce_to_scalar_f32
    collapses them to a single scalar on the GPU.  Only that one float
    crosses PCIe per convergence check.

3.  Convergence checked every N iterations (default 10):
    SpMV kernels queue continuously; sync + scalar read happen only every N
    iterations.  Eliminates ~90 % of CPU-GPU synchronisation stalls vs
    checking every iteration.  The final iteration always checks.

4.  Transition matrix W cached CPU-side:
    The scipy symmetrisation + column-normalisation is recomputed only
    when the graph object or network_type changes.

5.  GPU CSR arrays cached device-side:
    row_ptr / col_idx / w_values are re-uploaded only when the graph or
    precision_mode changes.  Subsequent runs skip H2D entirely.

6.  Persistent GPU working buffers (_WorkingBufferCache):
    d_p / d_pn / d_p0 / d_partial / d_l1_scalar are allocated once
    and reused across repeated calls on the same n.  Reallocation
    only occurs when the graph size (n) changes.

7.  Optional mixed precision (FP16 weights, FP32 accumulation):
    rwr_spmv_warp_per_node_fp16w halves the w_values bandwidth.

8.  Batched execution (rwr_spmv_warp_per_node_batched) for B ≤ MAX_BATCH.

9.  Chunked execution (_rwr_gpu_chunked) for graphs exceeding VRAM.

10. CUDA-event profiling (enable_profiling=True):
    Per-run breakdown of SpMV time, convergence time, H2D transfer time,
    total time, and derived overhead.  Zero overhead when disabled.

11. benchmark_precision_modes() utility:
    Automates FP32 vs FP16 comparison on any graph: warm-up, timed runs,
    speedup ratio, top-node overlap, and a plain-English recommendation.

Removed optimisations (introduced more overhead than benefit):
  - Hub-row ELLPACK: preprocessing cost + two kernel launches per iter.
  - SMEM-cached p vector: reduced SM occupancy; L1/L2 + __ldg suffices.
  - Node reordering: O(n log n) sort + CSR reconstruction cost.

Does NOT silently fall back to CPU.  Target: adaptive (sm_75 fallback).

Multi-seed-set support
------------------------
seed_nodes accepts a flat list (single RWR) or list-of-lists (one RWR per
inner list, scores averaged to form a consensus influence vector).
Batched kernel handles 1 < B <= MAX_BATCH = 4; larger batches loop the
single-seed kernel.

Parameter guide
---------------
restart_prob        (float, default 0.3)    Teleport probability per step.
max_iter            (int,   default 100)    Hard iteration cap.
tolerance           (float, default 1e-6)   L1-norm early-stop threshold.
seed_nodes          (list)                  Seeds for p₀.
network_type        (str,   default "grn")  One of "grn", "ppi", "mirna".
block_size          (int,   default 256)    CUDA block dimension.
precision_mode      (str,   default "fp32") "fp32" or "mixed" (FP16 W).
use_chunking        (bool,  default False)  Force chunked path on.
conv_check_interval (int,   default 10)     Check convergence every N iters.
enable_profiling    (bool,  default False)  Attach CUDA-event timing data.
"""

# ── GPU / CUDA-optimised implementation (PyCUDA custom kernels) ──────────
# Source:    src/algorithms/gpu/cuda_optimized/rwr.py
# Requires:  pycuda (with a working NVCC toolchain)
# Used by:   webapp routes, GPU benchmarking (src.benchmarking.benchmark)
# Modes:     _gpu only — this module is GPU-exclusive.
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import weakref
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
    logging.warning("PyCUDA not available — rwr_gpu() will raise.")

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
    "restart_prob":         0.3,
    "max_iter":             100,
    "tolerance":            1e-6,
    "seed_nodes":           [],
    "network_type":         "grn",
    "block_size":           256,
    "precision_mode":       "fp32",   # "fp32" or "mixed"
    "use_chunking":         False,
    "use_zero_copy":        False,
    "conv_check_interval":  10,       # changed from 5 → 10 (improvement 1)
    "enable_profiling":     False,    # CUDA-event profiling (improvement 5)
}

BLOCK_SIZE: int             = 256
WARP_SIZE: int              = 32
NODES_PER_BLOCK: int        = BLOCK_SIZE // WARP_SIZE   # = 8 (improvement 4)
MAX_BATCH: int              = 4
_TOP_NODES: int             = 20
_TOP_SEEDS: int             = 10
VRAM_BUDGET_FRACTION: float = 0.80

# ---------------------------------------------------------------------------
# CPU-side transition matrix cache
# ---------------------------------------------------------------------------

_W_CACHE: dict      = {}   # key -> (W: csr_matrix, note: str)
_W_CACHE_REFS: dict = {}   # key -> weakref to graph_csr
_W_CACHE_MAX: int   = 8

# ---------------------------------------------------------------------------
# GPU CSR cache — most recently used graph's device arrays
# ---------------------------------------------------------------------------

class _GPUCSRCache:
    """Single-entry device-array cache for the transition matrix CSR.

    Thread safety: not thread-safe; intended for single-threaded use.
    All invalidate() calls must occur while the CUDA primary context is
    active (i.e. from inside rwr_gpu after the context push).
    """

    __slots__ = ("key", "d_row_ptr", "d_col_idx", "d_w_values")

    def __init__(self) -> None:
        self.key: Any  = None
        self.d_row_ptr = None
        self.d_col_idx = None
        self.d_w_values = None

    def matches(self, key: Any) -> bool:
        return self.key == key and self.d_row_ptr is not None

    def store(self, key: Any, d_row_ptr, d_col_idx, d_w_values) -> None:
        self.key        = key
        self.d_row_ptr  = d_row_ptr
        self.d_col_idx  = d_col_idx
        self.d_w_values = d_w_values

    def invalidate(self) -> None:
        for attr in ("d_row_ptr", "d_col_idx", "d_w_values"):
            arr = getattr(self, attr, None)
            if arr is not None:
                try:
                    arr.gpudata.free()
                except Exception:                       # noqa: BLE001
                    pass
                setattr(self, attr, None)
        self.key = None


_GPU_CSR_CACHE = _GPUCSRCache()


# ---------------------------------------------------------------------------
# Persistent GPU working buffers — improvement 3
# ---------------------------------------------------------------------------

class _WorkingBufferCache:
    """Reusable GPU working buffers for the single-seed serial path.

    Avoids repeated gpuarray.empty() / gpuarray.free() on every call.
    Buffers are reallocated only when n (number of nodes) changes.

    Usage (inside rwr_gpu after context push):
        _GPU_WORKING_BUFFERS.ensure(n)
        buf = _GPU_WORKING_BUFFERS
        cuda.memcpy_htod_async(buf.d_p.gpudata, h_p0, stream)
        cuda.memcpy_htod_async(buf.d_p0.gpudata, h_p0, stream)
        d_p, d_pn = buf.d_p, buf.d_pn   # local aliases for pointer swap

    Thread safety: not thread-safe (single-threaded benchmarking use).
    free() must be called while the CUDA primary context is active.
    """

    __slots__ = (
        "n", "n_partial_blocks",
        "d_p", "d_pn", "d_p0", "d_partial", "d_l1_scalar",
    )

    def __init__(self) -> None:
        self.n               = 0
        self.n_partial_blocks = 0
        self.d_p             = None
        self.d_pn            = None
        self.d_p0            = None
        self.d_partial       = None
        self.d_l1_scalar     = None

    # ------------------------------------------------------------------
    def ensure(self, n: int) -> None:
        """Allocate buffers for n nodes, or no-op if already allocated."""
        if self.n == n and self.d_p is not None:
            return
        self.free()
        npb = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE)
        self.n                = n
        self.n_partial_blocks = npb
        self.d_p          = gpuarray.empty((n,),   np.float32)
        self.d_pn         = gpuarray.empty((n,),   np.float32)
        self.d_p0         = gpuarray.empty((n,),   np.float32)
        self.d_partial    = gpuarray.empty((npb,), np.float32)
        self.d_l1_scalar  = gpuarray.empty((1,),   np.float32)

    # ------------------------------------------------------------------
    def free(self) -> None:
        """Free all cached device buffers (call with active CUDA context)."""
        for attr in ("d_p", "d_pn", "d_p0", "d_partial", "d_l1_scalar"):
            arr = getattr(self, attr, None)
            if arr is not None:
                try:
                    arr.gpudata.free()
                except Exception:                   # noqa: BLE001
                    pass
                setattr(self, attr, None)
        self.n                = 0
        self.n_partial_blocks = 0


_GPU_WORKING_BUFFERS = _WorkingBufferCache()


# ---------------------------------------------------------------------------
# CUDA kernel source — improvement 4: warp-per-node scheduling
# ---------------------------------------------------------------------------
#
# Kernel inventory:
#   reduce_to_scalar_f32              — unchanged; GPU partial-sum reducer.
#   rwr_spmv_warp_per_node            — FP32  SpMV + restart, warp-per-node.
#   rwr_spmv_warp_per_node_fp16w      — FP16w SpMV + restart, warp-per-node.
#   l1_convergence_rwr                — unchanged; flat L1 partial kernel.
#   rwr_spmv_warp_per_node_batched    — batched SpMV, warp-per-node.
#
# Design rationale for warp-per-node vs previous one-block-per-node:
#   Biological networks are power-law: the vast majority of nodes have
#   degree << BLOCK_SIZE.  Under the old scheme a 256-thread block was
#   launched for each node, but only 1 or 32 threads did useful work.
#   With NODES_PER_BLOCK = 8, each block runs 8 warps simultaneously,
#   all doing useful SpMV work.  This raises SM occupancy ~8× for the
#   common low-degree case and removes shared-memory pressure entirely
#   (warp shuffles replace the SMEM tree).
#
# Grid sizes (Python side):
#   SpMV single : grid = (ceil(n / NODES_PER_BLOCK), 1, 1)
#   SpMV batched: grid = (ceil(n / NODES_PER_BLOCK), B, 1)
#   L1 conv     : grid = (ceil(n / BLOCK_SIZE), 1, 1)   [unchanged]
#   Reduce scalar: grid = (1, 1, 1)                      [unchanged]

KERNEL_SOURCE = r"""
#include <cuda_fp16.h>

extern "C" {

#define BLOCK_SIZE       256
#define WARP_SIZE        32
#define WARPS_PER_BLOCK  (BLOCK_SIZE / WARP_SIZE)    // 8
#define NODES_PER_BLOCK  (BLOCK_SIZE / WARP_SIZE)    // 8


// =========================================================================
// KERNEL: reduce_to_scalar_f32  (unchanged)
//
// Strided load + shared-memory tree reduction.  Launched grid=(1,1,1).
// Writes a SINGLE float to scalar_output[0].
// =========================================================================
__global__ void reduce_to_scalar_f32(
    const float* __restrict__ partial_input,
    float*       __restrict__ scalar_output,
    const int                  input_len)
{
    __shared__ float smem[BLOCK_SIZE];
    const int tid = threadIdx.x;
    float val = 0.0f;
    for (int i = tid; i < input_len; i += BLOCK_SIZE)
        val += partial_input[i];
    smem[tid] = val;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) scalar_output[0] = smem[0];
}


// =========================================================================
// KERNEL: rwr_spmv_warp_per_node  (FP32, warp-per-node)
//
// One 256-thread block processes NODES_PER_BLOCK = 8 nodes simultaneously.
// Each warp (32 threads) is responsible for exactly one node:
//   - warp_id  = threadIdx.x / WARP_SIZE  →  which node in this block
//   - lane     = threadIdx.x % WARP_SIZE  →  which edge within that node
//
// Threads stride across the row with step WARP_SIZE, accumulating into a
// warp-private float.  Five __shfl_down_sync calls reduce the warp to
// lane-0 in ~5 cycles (no __syncthreads needed).
//
// __ldg(&p[col_idx[j]]) routes p reads through the read-only cache.
// For degree == 0 (dangling node, should not occur after W construction
// adds self-loops but kept as a safety guard), lane 0 writes r*p0[node_i].
// =========================================================================
__global__ void rwr_spmv_warp_per_node(
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ w_values,
    const float* __restrict__ p,
    const float* __restrict__ p0,
    float*       __restrict__ p_new,
    const float                one_minus_r,
    const float                r,
    const int                  n)
{
    const int warp_id = threadIdx.x >> 5;               // threadIdx.x / 32
    const int lane    = threadIdx.x & 31;               // threadIdx.x % 32
    const int node_i  = (int)blockIdx.x * NODES_PER_BLOCK + warp_id;

    if (node_i >= n) return;

    const int row_start = row_ptr[node_i];
    const int row_end   = row_ptr[node_i + 1];

    float partial = 0.0f;
    for (int j = row_start + lane; j < row_end; j += WARP_SIZE)
        partial += w_values[j] * __ldg(&p[col_idx[j]]);

    // Warp reduction (unrolled — compiler-friendly, avoids loop overhead)
    partial += __shfl_down_sync(0xffffffffu, partial, 16);
    partial += __shfl_down_sync(0xffffffffu, partial, 8);
    partial += __shfl_down_sync(0xffffffffu, partial, 4);
    partial += __shfl_down_sync(0xffffffffu, partial, 2);
    partial += __shfl_down_sync(0xffffffffu, partial, 1);

    if (lane == 0)
        p_new[node_i] = one_minus_r * partial + r * p0[node_i];
}


// =========================================================================
// KERNEL: rwr_spmv_tprN  (FP32, adaptive vector width = N threads per row)
//
// Sub-warp "CSR-vector" SpMV: N contiguous lanes cooperate on one row, so
// BLOCK_SIZE / N rows are processed per 256-thread block.  For the low-
// average-degree biological / random graphs (avg degree ~6) a full 32-lane
// warp per row leaves ~26 lanes idle every iteration; matching the vector
// width to the degree (N ~ next_pow2(avg_degree)) packs several rows into
// each warp and lifts useful-lane occupancy several-fold.  Only wired up
// for network_type == "mirna" (grn / ppi keep the warp-per-node kernel).
//
// Boundary safety: threads whose row is out of range do NOT early-return;
// they participate in the shuffle with partial = 0 so the full-warp mask
// 0xffffffff stays valid on Volta+ (no inactive-lane shuffle UB).
// Reduction correctness: N divides 32 and the N lanes of a group are
// contiguous, so __shfl_down_sync offsets < N never cross a group boundary
// for lane 0 of the group, which alone writes the result.
// =========================================================================
#define RWR_SPMV_TPR_KERNEL(NAME, TPR)                                        \
__global__ void NAME(                                                         \
    const int*   __restrict__ row_ptr,                                        \
    const int*   __restrict__ col_idx,                                        \
    const float* __restrict__ w_values,                                       \
    const float* __restrict__ p,                                              \
    const float* __restrict__ p0,                                             \
    float*       __restrict__ p_new,                                          \
    const float                one_minus_r,                                   \
    const float                r,                                             \
    const int                  n)                                             \
{                                                                             \
    const int rows_per_block = BLOCK_SIZE / (TPR);                            \
    const int local_row = threadIdx.x / (TPR);                                \
    const int sub_lane  = threadIdx.x % (TPR);                                \
    const int node_i    = (int)blockIdx.x * rows_per_block + local_row;       \
    const bool active   = (node_i < n);                                       \
    const int row_start = active ? row_ptr[node_i]     : 0;                   \
    const int row_end   = active ? row_ptr[node_i + 1] : 0;                   \
    float partial = 0.0f;                                                     \
    for (int j = row_start + sub_lane; j < row_end; j += (TPR))               \
        partial += w_values[j] * __ldg(&p[col_idx[j]]);                       \
    for (int off = (TPR) >> 1; off > 0; off >>= 1)                            \
        partial += __shfl_down_sync(0xffffffffu, partial, off);               \
    if (active && sub_lane == 0)                                              \
        p_new[node_i] = one_minus_r * partial + r * p0[node_i];               \
}

RWR_SPMV_TPR_KERNEL(rwr_spmv_tpr2,  2)
RWR_SPMV_TPR_KERNEL(rwr_spmv_tpr4,  4)
RWR_SPMV_TPR_KERNEL(rwr_spmv_tpr8,  8)
RWR_SPMV_TPR_KERNEL(rwr_spmv_tpr16, 16)


// =========================================================================
// KERNEL: rwr_spmv_warp_per_node_fp16w  (FP16 weights, FP32 accum)
//
// Identical warp-per-node structure; reads w_values as FP16 bit patterns
// and converts to float at use time.  Halves w_values bandwidth.
// p / p_new / p0 remain FP32; accumulation is FP32.
// =========================================================================
__global__ void rwr_spmv_warp_per_node_fp16w(
    const int*           __restrict__ row_ptr,
    const int*           __restrict__ col_idx,
    const unsigned short* __restrict__ w_values_h,
    const float*         __restrict__ p,
    const float*         __restrict__ p0,
    float*               __restrict__ p_new,
    const float                        one_minus_r,
    const float                        r,
    const int                          n)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int node_i  = (int)blockIdx.x * NODES_PER_BLOCK + warp_id;

    if (node_i >= n) return;

    const int row_start = row_ptr[node_i];
    const int row_end   = row_ptr[node_i + 1];

    float partial = 0.0f;
    for (int j = row_start + lane; j < row_end; j += WARP_SIZE)
        partial += __half2float(*reinterpret_cast<const __half*>(&w_values_h[j]))
                   * __ldg(&p[col_idx[j]]);

    partial += __shfl_down_sync(0xffffffffu, partial, 16);
    partial += __shfl_down_sync(0xffffffffu, partial, 8);
    partial += __shfl_down_sync(0xffffffffu, partial, 4);
    partial += __shfl_down_sync(0xffffffffu, partial, 2);
    partial += __shfl_down_sync(0xffffffffu, partial, 1);

    if (lane == 0)
        p_new[node_i] = one_minus_r * partial + r * p0[node_i];
}


// =========================================================================
// KERNEL: l1_convergence_rwr  (unchanged — already optimal flat kernel)
//
// One thread per element: partial |p_new[i] - p[i]| accumulated into a
// warp-level sum via __shfl_down_sync, then one warp in SMEM reduction.
// Grid = (ceil(n / BLOCK_SIZE), 1, 1); partials fed to reduce_to_scalar.
// =========================================================================
__global__ void l1_convergence_rwr(
    const float* __restrict__ p_new,
    const float* __restrict__ p,
    float*       __restrict__ partial_sums,
    const int                  n)
{
    __shared__ float smem[WARPS_PER_BLOCK];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < n) ? fabsf(p_new[tid] - p[tid]) : 0.0f;

    for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, off);

    const int lane    = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    if (threadIdx.x < WARP_SIZE) {
        val = (threadIdx.x < WARPS_PER_BLOCK) ? smem[threadIdx.x] : 0.0f;
        for (int off = WARPS_PER_BLOCK >> 1; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffffu, val, off);
        if (threadIdx.x == 0) partial_sums[blockIdx.x] = val;
    }
}


// =========================================================================
// KERNEL: rwr_spmv_warp_per_node_batched  (multi-seed, B <= MAX_BATCH)
//
// Grid: (ceil(n / NODES_PER_BLOCK), B, 1).
// Each warp handles node (blockIdx.x * NPB + warp_id) for seed blockIdx.y.
// p / p0 / p_new stored as [n × B] row-major flat arrays.
// =========================================================================
__global__ void rwr_spmv_warp_per_node_batched(
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ w_values,
    const float* __restrict__ p,
    const float* __restrict__ p0,
    float*       __restrict__ p_new,
    const float                one_minus_r,
    const float                r,
    const int                  n,
    const int                  B)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int node_i  = (int)blockIdx.x * NODES_PER_BLOCK + warp_id;
    const int seed_b  = (int)blockIdx.y;

    if (node_i >= n || seed_b >= B) return;

    const int row_start = row_ptr[node_i];
    const int row_end   = row_ptr[node_i + 1];
    const int out_idx   = node_i * B + seed_b;

    float partial = 0.0f;
    for (int j = row_start + lane; j < row_end; j += WARP_SIZE)
        partial += w_values[j] * __ldg(&p[col_idx[j] * B + seed_b]);

    partial += __shfl_down_sync(0xffffffffu, partial, 16);
    partial += __shfl_down_sync(0xffffffffu, partial, 8);
    partial += __shfl_down_sync(0xffffffffu, partial, 4);
    partial += __shfl_down_sync(0xffffffffu, partial, 2);
    partial += __shfl_down_sync(0xffffffffu, partial, 1);

    if (lane == 0)
        p_new[out_idx] = one_minus_r * partial + r * p0[out_idx];
}

}  // extern "C"
"""


# ---------------------------------------------------------------------------
# Module-level kernel cache + adaptive compilation
# ---------------------------------------------------------------------------

_kernel_cache: dict[str, dict[str, Any]] = {}


def _detect_arch_flag() -> str:
    """Return ``-arch=sm_XY`` for the current device (sm_75 fallback)."""
    try:
        cuda.init()
        cc_major, cc_minor = cuda.Device(0).compute_capability()
        return f"-arch=sm_{cc_major}{cc_minor}"
    except Exception:                                   # noqa: BLE001
        return "-arch=sm_75"


def _get_kernels() -> dict[str, Any]:
    """Compile (or fetch from cache) all RWR device kernels."""
    if "rwr" not in _kernel_cache:
        if not PYCUDA_AVAILABLE:
            raise RuntimeError(
                "PyCUDA is required to compile RWR kernels — "
                "install pycuda and ensure NVCC is on PATH."
            )
        arch_flag = _detect_arch_flag()
        mod = SourceModule(
            KERNEL_SOURCE,
            options=[arch_flag, "-O3"],
            no_extern_c=True,
        )
        # Python-facing keys are unchanged so callers don't need updating.
        # CUDA function names reflect the new warp-per-node design.
        _kernel_cache["rwr"] = {
            "reduce_scalar":        mod.get_function("reduce_to_scalar_f32"),
            "spmv_restart":         mod.get_function("rwr_spmv_warp_per_node"),
            "spmv_restart_fp16w":   mod.get_function("rwr_spmv_warp_per_node_fp16w"),
            "l1_conv":              mod.get_function("l1_convergence_rwr"),
            "spmv_restart_batched": mod.get_function("rwr_spmv_warp_per_node_batched"),
            # Adaptive vector-width SpMV (mirna only — see dispatch below).
            "spmv_tpr2":            mod.get_function("rwr_spmv_tpr2"),
            "spmv_tpr4":            mod.get_function("rwr_spmv_tpr4"),
            "spmv_tpr8":            mod.get_function("rwr_spmv_tpr8"),
            "spmv_tpr16":           mod.get_function("rwr_spmv_tpr16"),
            "_arch_flag":           arch_flag,
        }
    return _kernel_cache["rwr"]


# ---------------------------------------------------------------------------
# Parameter merging
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# CPU preprocessing helpers
# ---------------------------------------------------------------------------

def _build_transition_matrix(
    graph_csr: sp.csr_matrix,
    network_type: str,
) -> tuple[sp.csr_matrix, str]:
    """Build the column-stochastic transition matrix W for RWR."""
    nt = str(network_type).lower()
    if nt == "ppi":
        A = (graph_csr + graph_csr.T).astype(np.float32)
        A.data = np.ones_like(A.data, dtype=np.float32)
        note = "PPI: A + Aᵀ binarised, then column-normalised"
    else:
        A = graph_csr.astype(np.float32)
        note = f"{nt.upper()}: directed CSR, column-normalised by out-degree"

    A = A.tocsr()
    A.sum_duplicates()
    out_degrees = np.asarray(A.sum(axis=1), dtype=np.float64).flatten()
    safe_degs   = np.where(out_degrees == 0.0, 1.0, out_degrees)
    D_inv       = sp.diags(1.0 / safe_degs, format="csr")
    W           = (D_inv @ A).T.tocsr().astype(np.float32)

    col_sums  = np.asarray(W.sum(axis=0), dtype=np.float64).flatten()
    zero_cols = np.where(col_sums == 0.0)[0]
    if zero_cols.size > 0:
        W_lil = W.tolil()
        for j in zero_cols:
            W_lil[int(j), int(j)] = 1.0
        W = W_lil.tocsr().astype(np.float32)
    return W, note


def _get_cached_W(
    graph_csr: sp.csr_matrix,
    network_type: str,
) -> tuple[sp.csr_matrix, str]:
    """Return cached (W, note); recompute if graph or network_type changed.

    Key: (object id, shape, nnz, network_type).  Weakref guards against
    Python reusing the same id for a different object after GC.
    """
    key = (id(graph_csr), graph_csr.shape, int(graph_csr.nnz), network_type)
    if key in _W_CACHE:
        ref = _W_CACHE_REFS.get(key)
        if ref is None or ref() is graph_csr:
            return _W_CACHE[key]
        del _W_CACHE[key]
        _W_CACHE_REFS.pop(key, None)

    W, note = _build_transition_matrix(graph_csr, network_type)
    if len(_W_CACHE) >= _W_CACHE_MAX:
        evict = next(iter(_W_CACHE))
        del _W_CACHE[evict]
        _W_CACHE_REFS.pop(evict, None)
    _W_CACHE[key] = (W, note)
    try:
        _W_CACHE_REFS[key] = weakref.ref(graph_csr)
    except TypeError:
        _W_CACHE_REFS[key] = None
    return W, note


def _build_p0(seed_nodes, n: int, network_type: str) -> np.ndarray:
    """Build the restart vector p₀ (length n, sums to 1.0)."""
    p0 = np.zeros(n, dtype=np.float32)
    valid = [int(s) for s in (seed_nodes or []) if isinstance(s, (int, np.integer))
             and 0 <= int(s) < n]
    if len(valid) == 0:
        p0[:] = np.float32(1.0 / n)
    else:
        p0[valid] = np.float32(1.0 / len(valid))
    return p0


def _fp32_to_fp16_bits(values: np.ndarray) -> np.ndarray:
    """Convert FP32 array to FP16 bit pattern stored as uint16."""
    return values.astype(np.float16).view(np.uint16).astype(np.uint16)


def _choose_tpr(avg_degree: float) -> int:
    """Pick threads-per-row (power of 2 in [2, 32]) ~ next_pow2(avg_degree).

    A full 32-lane warp per row wastes lanes when rows are short; matching
    the vector width to the average degree packs BLOCK_SIZE / tpr rows into
    each block and raises useful-lane occupancy.  ``32`` means "use the
    original warp-per-node kernel" (no change).
    """
    if avg_degree <= 2.0:
        return 2
    if avg_degree <= 4.0:
        return 4
    if avg_degree <= 8.0:
        return 8
    if avg_degree <= 16.0:
        return 16
    return 32


def _estimate_rwr_vram(n: int, nnz: int, batch_size: int = 1,
                       fp16_weights: bool = False) -> int:
    """Conservative VRAM estimate (bytes) for RWR working set."""
    w_val_bytes = 2 if fp16_weights else 4
    csr_b     = (n + 1) * 4 + nnz * 4 + nnz * w_val_bytes
    pr_b      = n * 4 * (2 + batch_size)
    partial_b = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE) * 4
    scalar_b  = 4 * 4
    return int(csr_b + pr_b + partial_b + scalar_b)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _top_k(scores: np.ndarray, k: int = _TOP_NODES) -> list[int]:
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    return np.argsort(scores)[::-1][:k].astype(int).tolist()


def _top_k_among(
    scores: np.ndarray,
    candidate_indices: np.ndarray,
    k: int,
) -> list[int]:
    if candidate_indices.size == 0:
        return []
    sub   = scores[candidate_indices]
    order = np.argsort(sub)[::-1][:k]
    return candidate_indices[order].astype(int).tolist()


# ---------------------------------------------------------------------------
# Chunked execution helpers
# ---------------------------------------------------------------------------

class _ChunkBuffer:
    """Pre-allocated GPU buffer pair for chunked CSR streaming."""

    __slots__ = ("d_row_ptr", "d_col_idx", "d_values",
                 "d_node_ids", "max_nnz", "max_nodes")

    def __init__(self, max_nnz: int, max_nodes: int):
        self.d_row_ptr  = gpuarray.empty((max_nodes + 1,), np.int32)
        self.d_col_idx  = gpuarray.empty((max_nnz,),       np.int32)
        self.d_values   = gpuarray.empty((max_nnz,),       np.float32)
        self.d_node_ids = gpuarray.empty((max_nodes,),     np.int32)
        self.max_nnz    = max_nnz
        self.max_nodes  = max_nodes

    def free(self) -> None:
        for arr in (self.d_row_ptr, self.d_col_idx,
                    self.d_values, self.d_node_ids):
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass


def _build_csr_chunks(W: sp.csr_matrix, chunk_size: int) -> list[dict]:
    """Split W into row-chunks for chunked SpMV."""
    n = int(W.shape[0])
    chunks: list[dict] = []
    for start in range(0, n, chunk_size):
        end      = min(start + chunk_size, n)
        node_ids = np.arange(start, end, dtype=np.int32)
        rs       = int(W.indptr[start])
        re       = int(W.indptr[end])
        chunks.append({
            "node_ids": node_ids,
            "indptr":   (W.indptr[start:end + 1] - rs).astype(np.int32),
            "indices":  W.indices[rs:re].astype(np.int32, copy=False),
            "values":   W.data[rs:re].astype(np.float32, copy=False),
        })
    return chunks


# ---------------------------------------------------------------------------
# Chunk-aware SpMV kernel — warp-per-node layout
# ---------------------------------------------------------------------------

_CHUNK_KERNEL_SOURCE = r"""
extern "C" {

#define BLOCK_SIZE      256
#define WARP_SIZE       32
#define NODES_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)    // 8

// Warp-per-node chunk kernel.
//   local_i = blockIdx.x * NODES_PER_BLOCK + warp_id   (chunk-local row)
//   node_i  = chunk_node_ids[local_i]                   (global node id)
// row_ptr / col_idx / values use chunk-local indices (rebased indptr).
__global__ void rwr_spmv_warp_per_node_chunk(
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ w_values,
    const int*   __restrict__ chunk_node_ids,
    const float* __restrict__ p,
    const float* __restrict__ p0,
    float*       __restrict__ p_new,
    const float                one_minus_r,
    const float                r,
    const int                  chunk_size)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int local_i = (int)blockIdx.x * NODES_PER_BLOCK + warp_id;

    if (local_i >= chunk_size) return;
    const int node_i = chunk_node_ids[local_i];

    const int row_start = row_ptr[local_i];
    const int row_end   = row_ptr[local_i + 1];

    float partial = 0.0f;
    for (int j = row_start + lane; j < row_end; j += WARP_SIZE)
        partial += w_values[j] * __ldg(&p[col_idx[j]]);

    partial += __shfl_down_sync(0xffffffffu, partial, 16);
    partial += __shfl_down_sync(0xffffffffu, partial, 8);
    partial += __shfl_down_sync(0xffffffffu, partial, 4);
    partial += __shfl_down_sync(0xffffffffu, partial, 2);
    partial += __shfl_down_sync(0xffffffffu, partial, 1);

    if (lane == 0)
        p_new[node_i] = one_minus_r * partial + r * p0[node_i];
}

}  // extern "C"
"""


def _get_chunk_kernel() -> Any:
    """Compile (or fetch) the warp-per-node chunk SpMV kernel."""
    if "rwr_chunk" not in _kernel_cache:
        if not PYCUDA_AVAILABLE:
            raise RuntimeError("PyCUDA required.")
        arch_flag = _detect_arch_flag()
        mod = SourceModule(
            _CHUNK_KERNEL_SOURCE,
            options=[arch_flag, "-O3"],
            no_extern_c=True,
        )
        _kernel_cache["rwr_chunk"] = {
            "spmv_chunk": mod.get_function("rwr_spmv_warp_per_node_chunk"),
        }
    return _kernel_cache["rwr_chunk"]


# ---------------------------------------------------------------------------
# Chunked GPU iteration
# ---------------------------------------------------------------------------

def _rwr_gpu_chunked(
    W: sp.csr_matrix,
    p0_np: np.ndarray,
    one_minus_r: float,
    r_val: float,
    max_iter: int,
    tolerance: float,
    conv_check_interval: int,
    kernels: dict,
    stream_compute,
    stream_transfer,
    chunk_size: int,
) -> tuple[np.ndarray, int, bool]:
    """Chunked warp-per-node SpMV for graphs that exceed VRAM.

    p / p_new / p0 remain full-size on the device.  CSR rows stream in
    chunks via a double-buffer pipeline.  Convergence is checked every
    conv_check_interval iterations; the final iteration always checks.
    """
    chunk_kernels = _get_chunk_kernel()
    k_chunk  = chunk_kernels["spmv_chunk"]
    k_l1     = kernels["l1_conv"]
    k_reduce = kernels["reduce_scalar"]

    n      = int(W.shape[0])
    chunks = _build_csr_chunks(W, chunk_size=chunk_size)
    max_chunk_nnz   = max((c["values"].size for c in chunks), default=1)
    max_chunk_nodes = max((c["node_ids"].size for c in chunks), default=1)

    h_p0 = np.ascontiguousarray(p0_np, np.float32)
    d_p0 = gpuarray.to_gpu(h_p0)
    d_p  = gpuarray.to_gpu(h_p0.copy())
    d_pn = gpuarray.zeros((n,), np.float32)

    n_partial_blocks = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE)
    d_partial   = gpuarray.zeros((n_partial_blocks,), np.float32)
    d_l1_scalar = gpuarray.zeros((1,), np.float32)

    buffers = [
        _ChunkBuffer(max_chunk_nnz, max_chunk_nodes),
        _ChunkBuffer(max_chunk_nnz, max_chunk_nodes),
    ]
    events = [cuda.Event(), cuda.Event()]

    def _upload_chunk(chunk: dict, buf: _ChunkBuffer) -> None:
        cuda.memcpy_htod_async(buf.d_row_ptr.gpudata,  chunk["indptr"],   stream_transfer)
        cuda.memcpy_htod_async(buf.d_col_idx.gpudata,  chunk["indices"],  stream_transfer)
        cuda.memcpy_htod_async(buf.d_values.gpudata,   chunk["values"],   stream_transfer)
        cuda.memcpy_htod_async(buf.d_node_ids.gpudata, chunk["node_ids"], stream_transfer)

    _upload_chunk(chunks[0], buffers[0])
    events[0].record(stream_transfer)

    converged  = False
    iterations = 0
    try:
        for it in range(max_iter):
            iterations = it + 1

            for ci, chunk in enumerate(chunks):
                buf = buffers[ci % 2]
                stream_compute.wait_for_event(events[ci % 2])
                chunk_sz = int(chunk["node_ids"].size)
                chunk_blocks = (chunk_sz + NODES_PER_BLOCK - 1) // NODES_PER_BLOCK
                k_chunk(
                    buf.d_row_ptr, buf.d_col_idx, buf.d_values,
                    buf.d_node_ids, d_p, d_p0, d_pn,
                    np.float32(one_minus_r), np.float32(r_val),
                    np.int32(chunk_sz),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(chunk_blocks, 1, 1),
                    stream=stream_compute,
                )
                if ci + 1 < len(chunks):
                    nxt_buf = buffers[(ci + 1) % 2]
                    _upload_chunk(chunks[ci + 1], nxt_buf)
                    events[(ci + 1) % 2].record(stream_transfer)
                elif it + 1 < max_iter:
                    _upload_chunk(chunks[0], buffers[0])
                    events[0].record(stream_transfer)

            # Convergence check every N iterations (improvement 1 + 3)
            do_check = (iterations % conv_check_interval == 0) or (iterations == max_iter)
            if do_check:
                k_l1(
                    d_pn, d_p, d_partial, np.int32(n),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(n_partial_blocks, 1, 1),
                    stream=stream_compute,
                )
                k_reduce(
                    d_partial, d_l1_scalar, np.int32(n_partial_blocks),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(1, 1, 1),
                    stream=stream_compute,
                )

            d_p, d_pn = d_pn, d_p

            if do_check:
                stream_compute.synchronize()
                if float(d_l1_scalar.get()[0]) < tolerance:
                    converged = True
                    break

        scores_np = d_p.get()
        return scores_np, iterations, converged

    finally:
        for buf in buffers:
            buf.free()
        for arr in (d_p0, d_p, d_pn, d_partial, d_l1_scalar):
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def rwr_gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """RWR — GPU-accelerated via custom PyCUDA kernels.

    Changes in this version:
      1.  conv_check_interval default raised to 10 (~90 % fewer syncs).
      2.  Warp-per-node kernel: 8 nodes per 256-thread block.
      3.  Persistent working buffers: d_p/d_pn/d_p0/d_partial/d_l1_scalar
          allocated once per graph size, reused across repeated calls.
      4.  CUDA-event profiling returned in result["result"]["profiling"]
          when enable_profiling=True.

    Raises
    ------
    RuntimeError   PyCUDA unavailable or no CUDA device found.
    MemoryError    GPU allocation fails even after chunked path.
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA is required for rwr_gpu(). "
            "Install pycuda or use the CPU implementation."
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

    # Temporary GPU allocations freed in the finally block.
    # Cached arrays (GPU CSR, working buffers) are NOT added here.
    d_buffers: list = []

    try:
        # ---- Parameter merging -------------------------------------------
        p = _merge_params(params)

        # apply_config() is ALREADY run by algorithm_runner.run_algorithm()
        # before this function is called, and every key it injects for RWR
        # (spmv_mode, reorder_nodes, execution_mode, chunk_size, …) is ignored
        # below — rwr_gpu re-derives its own VRAM / chunking decision and
        # honours user-supplied max_iter / tolerance regardless.  For the
        # mirna network type we therefore skip this SECOND, redundant
        # apply_config() so its CPU graph-profiling and live mem_get_info()
        # VRAM queries are not charged to the CUDA-event-timed region (the
        # runner times the whole _gpu() call, and its start event fires before
        # this line).  grn / ppi keep the original path so their tuning and
        # results stay byte-identical.
        _nt_raw = str(p.get("network_type", "grn")).lower()
        if _GPU_CONFIG_AVAILABLE and _nt_raw != "mirna":
            p = apply_config("rwr", graph_csr, p) or p
            for k, v in _DEFAULT_PARAMS.items():
                p.setdefault(k, v)

        restart_prob        = float(p["restart_prob"])
        max_iter            = int(p["max_iter"])
        tolerance           = float(p["tolerance"])
        seed_nodes          = p.get("seed_nodes") or []
        network_type        = str(p.get("network_type", "grn"))
        block_size          = int(p.get("block_size", BLOCK_SIZE))
        if block_size <= 0 or block_size > 1024:
            block_size = BLOCK_SIZE
        precision_mode      = str(p.get("precision_mode", "fp32")).lower()
        use_chunking_req    = bool(p.get("use_chunking", False))
        conv_check_interval = max(1, int(p.get("conv_check_interval", 10)))
        enable_profiling    = bool(p.get("enable_profiling", False))

        n = int(graph_csr.shape[0])
        if n == 0:
            raise ValueError("Empty graph")

        one_minus_r = np.float32(1.0 - restart_prob)
        r_val       = np.float32(restart_prob)

        # ---- Detect batched input ----------------------------------------
        if (len(seed_nodes) > 0
                and isinstance(seed_nodes[0], (list, tuple, np.ndarray))):
            seed_sets = [list(s) for s in seed_nodes]
        else:
            seed_sets = [list(seed_nodes)]
        batch_size = len(seed_sets)

        # ---- Transition matrix (CPU cache) -------------------------------
        W, transition_note = _get_cached_W(graph_csr, network_type)

        # ---- p₀ vectors --------------------------------------------------
        p0_list = [_build_p0(sset, n, network_type) for sset in seed_sets]

        fp16_weights = (precision_mode == "mixed")

        # ---- VRAM check + chunking decision ------------------------------
        est_bytes = _estimate_rwr_vram(
            n=n, nnz=int(W.nnz), batch_size=batch_size,
            fp16_weights=fp16_weights,
        )
        try:
            free_bytes, _ = cuda.mem_get_info()
        except Exception:                               # noqa: BLE001
            free_bytes = 1 << 30

        # Chunk ONLY when the graph genuinely does not fit in VRAM.  The
        # chunked path re-streams the ENTIRE W CSR host→device every iteration
        # (see _upload_chunk inside the iteration loop) — catastrophic (e.g.
        # 100x re-upload of the full matrix) for a graph that fits.  A
        # use_chunking=True flag from params is deliberately NOT honoured:
        # apply_config's MemoryManager injects it from a coarse estimate, and
        # by the time this function runs (the runner calls apply_config first)
        # that injected flag is indistinguishable from a user-supplied one.
        # est_bytes (RWR's own precise estimate) is the sole authority.
        use_chunking = est_bytes > VRAM_BUDGET_FRACTION * free_bytes
        if use_chunking_req and not use_chunking:
            logging.debug(
                "rwr_gpu: use_chunking ignored — working set %.1f MB fits in "
                "%.1f MB free (chunking would re-stream W every iteration).",
                est_bytes / 1e6, free_bytes / 1e6,
            )
        if batch_size > 1 and use_chunking:
            logging.info(
                "rwr_gpu: chunked + batched not implemented; "
                "running batched path on full graph (may OOM)."
            )
            use_chunking = False

        if est_bytes > free_bytes and not use_chunking:
            raise MemoryError(
                f"RWR GPU needs ~{est_bytes/1e6:.1f} MB but only "
                f"{free_bytes/1e6:.1f} MB free.  Try use_chunking=True "
                f"or precision_mode='mixed'."
            )

        kernels         = _get_kernels()
        stream_compute  = cuda.Stream()
        stream_transfer = cuda.Stream()
        start_event     = cuda.Event()
        end_event       = cuda.Event()

        # Warp-per-node grid dimension
        n_spmv_blocks = (n + NODES_PER_BLOCK - 1) // NODES_PER_BLOCK
        n_partial_blocks = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE)

        # ---- Chunked path ------------------------------------------------
        if use_chunking and batch_size == 1:
            avg_nnz        = max(1.0, W.nnz / n)
            bytes_per_node = int((1 + 2 * avg_nnz) * 4)
            chunk_size     = max(1, min(n, int(free_bytes * 0.6) // max(bytes_per_node, 1)))

            start_event.record(stream_compute)
            scores_np, iters, converged = _rwr_gpu_chunked(
                W=W, p0_np=p0_list[0],
                one_minus_r=float(one_minus_r), r_val=float(r_val),
                max_iter=max_iter, tolerance=tolerance,
                conv_check_interval=conv_check_interval,
                kernels=kernels,
                stream_compute=stream_compute,
                stream_transfer=stream_transfer,
                chunk_size=chunk_size,
            )
            end_event.record(stream_compute)
            end_event.synchronize()
            elapsed = start_event.time_till(end_event) / 1000.0

            return _finalize_result(
                all_scores=[{"seed_set": seed_sets[0], "scores": scores_np,
                             "iterations": iters, "converged": converged}],
                n=n, seed_sets=seed_sets, network_type=network_type,
                graph_csr=graph_csr, elapsed=elapsed,
                transition_note=transition_note,
                arch_flag=kernels.get("_arch_flag", "?"),
                precision_mode=precision_mode, chunked=True,
                profiling_data=None,
            )

        # ---- Prepare GPU CSR arrays (use device cache if available) ------
        gpu_cache_key = (id(graph_csr), graph_csr.shape,
                         int(graph_csr.nnz), network_type, precision_mode)

        if _GPU_CSR_CACHE.matches(gpu_cache_key):
            d_row_ptr  = _GPU_CSR_CACHE.d_row_ptr
            d_col_idx  = _GPU_CSR_CACHE.d_col_idx
            d_w_values = _GPU_CSR_CACHE.d_w_values
        else:
            _GPU_CSR_CACHE.invalidate()          # free old (context active)
            h_row_ptr = np.ascontiguousarray(W.indptr,  dtype=np.int32)
            h_col_idx = np.ascontiguousarray(W.indices, dtype=np.int32)
            h_w_values = (
                np.ascontiguousarray(
                    _fp32_to_fp16_bits(W.data.astype(np.float32)), dtype=np.uint16)
                if fp16_weights
                else np.ascontiguousarray(W.data, dtype=np.float32)
            )
            d_row_ptr  = gpuarray.to_gpu_async(h_row_ptr,  stream=stream_transfer)
            d_col_idx  = gpuarray.to_gpu_async(h_col_idx,  stream=stream_transfer)
            d_w_values = gpuarray.to_gpu_async(h_w_values, stream=stream_transfer)
            _GPU_CSR_CACHE.store(gpu_cache_key, d_row_ptr, d_col_idx, d_w_values)

        # ---- Select SpMV kernel + launch grid ----------------------------
        # Default (grn / ppi, or FP16 weights): warp-per-node, 8 rows/block.
        #
        # mirna (FP32, non-batched): degree-adaptive sub-warp SpMV.  Match the
        # threads-per-row to the average degree so short rows do not each
        # occupy a full 32-lane warp — packs BLOCK_SIZE / tpr rows per block.
        # tpr == 32 falls back to the identical warp-per-node kernel/grid, so
        # this is a no-op for high-degree mirna graphs.
        spmv_grid = (n_spmv_blocks, 1, 1)
        if fp16_weights:
            k_spmv = kernels["spmv_restart_fp16w"]
        elif network_type.lower() == "mirna":
            avg_deg = float(W.nnz) / max(1, n)
            tpr = _choose_tpr(avg_deg)
            if tpr >= 32:
                k_spmv = kernels["spmv_restart"]
            else:
                k_spmv = kernels[f"spmv_tpr{tpr}"]
                rows_per_block = BLOCK_SIZE // tpr
                spmv_grid = ((n + rows_per_block - 1) // rows_per_block, 1, 1)
        else:
            k_spmv = kernels["spmv_restart"]
        k_l1     = kernels["l1_conv"]
        k_reduce = kernels["reduce_scalar"]

        use_batched = (1 < batch_size <= MAX_BATCH)
        all_scores: list[dict] = []

        # ==================================================================
        # BATCHED PATH  (B <= MAX_BATCH, no working buffer cache)
        # ==================================================================
        if use_batched:
            p0_matrix = np.column_stack(p0_list).astype(np.float32)
            h_p0_flat = np.ascontiguousarray(p0_matrix.reshape(-1), np.float32)

            def _tmp(arr=None, shape=None, dtype=np.float32):
                ga = (gpuarray.to_gpu_async(arr, stream=stream_transfer)
                      if arr is not None else gpuarray.empty(shape, dtype))
                d_buffers.append(ga)
                return ga

            d_p0 = _tmp(arr=h_p0_flat)
            d_p  = _tmp(arr=h_p0_flat.copy())
            d_pn = _tmp(shape=(n * batch_size,))

            nB               = n * batch_size
            n_partial_batched = max(1, (nB + BLOCK_SIZE - 1) // BLOCK_SIZE)
            d_partial_b   = _tmp(shape=(n_partial_batched,))
            d_l1_scalar_b = _tmp(shape=(1,))
            k_batched     = kernels["spmv_restart_batched"]
            n_batched_blocks = (n + NODES_PER_BLOCK - 1) // NODES_PER_BLOCK

            # Profiling setup for batched path
            b_spmv_evts: list = []
            b_conv_evts: list = []

            stream_transfer.synchronize()
            start_event.record(stream_compute)

            converged  = False
            iterations = 0
            for it in range(max_iter):
                iterations = it + 1

                if enable_profiling:
                    _es = cuda.Event(); _es.record(stream_compute)

                k_batched(
                    d_row_ptr, d_col_idx, d_w_values,
                    d_p, d_p0, d_pn,
                    one_minus_r, r_val,
                    np.int32(n), np.int32(batch_size),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(n_batched_blocks, batch_size, 1),
                    stream=stream_compute,
                )

                if enable_profiling:
                    _ee = cuda.Event(); _ee.record(stream_compute)
                    b_spmv_evts.append((_es, _ee))

                do_check = (iterations % conv_check_interval == 0) or (iterations == max_iter)
                if do_check:
                    if enable_profiling:
                        _cs = cuda.Event(); _cs.record(stream_compute)

                    k_l1(
                        d_pn, d_p, d_partial_b, np.int32(nB),
                        block=(BLOCK_SIZE, 1, 1),
                        grid=(n_partial_batched, 1, 1),
                        stream=stream_compute,
                    )
                    k_reduce(
                        d_partial_b, d_l1_scalar_b, np.int32(n_partial_batched),
                        block=(BLOCK_SIZE, 1, 1),
                        grid=(1, 1, 1),
                        stream=stream_compute,
                    )

                    if enable_profiling:
                        _ce = cuda.Event(); _ce.record(stream_compute)
                        b_conv_evts.append((_cs, _ce))

                d_p, d_pn = d_pn, d_p

                if do_check:
                    stream_compute.synchronize()
                    if float(d_l1_scalar_b.get()[0]) < tolerance * batch_size:
                        converged = True
                        break

            p_final = d_p.get().reshape(n, batch_size)
            for b in range(batch_size):
                all_scores.append({
                    "seed_set":   seed_sets[b],
                    "scores":     p_final[:, b].copy(),
                    "iterations": iterations,
                    "converged":  converged,
                })

            end_event.record(stream_compute)
            end_event.synchronize()
            elapsed = start_event.time_till(end_event) / 1000.0

            prof = None
            if enable_profiling:
                spmv_ms = sum(s.time_till(e) for s, e in b_spmv_evts)
                conv_ms = sum(s.time_till(e) for s, e in b_conv_evts)
                total_ms = elapsed * 1000.0
                prof = {
                    "spmv_ms":              round(spmv_ms, 4),
                    "convergence_ms":       round(conv_ms, 4),
                    "transfer_ms":          0.0,  # CSR upload excluded from timed region
                    "total_ms":             round(total_ms, 4),
                    "overhead_ms":          round(total_ms - spmv_ms - conv_ms, 4),
                    "iterations":           iterations,
                    "convergence_checks":   len(b_conv_evts),
                    "avg_spmv_ms_per_iter": round(spmv_ms / max(1, iterations), 4),
                    "avg_conv_ms_per_check": round(conv_ms / max(1, len(b_conv_evts)), 4),
                }

            return _finalize_result(
                all_scores=all_scores, n=n, seed_sets=seed_sets,
                network_type=network_type, graph_csr=graph_csr,
                elapsed=elapsed, transition_note=transition_note,
                arch_flag=kernels.get("_arch_flag", "?"),
                precision_mode=precision_mode, chunked=False,
                profiling_data=prof,
            )

        # ==================================================================
        # SERIAL PER-SEED-SET PATH  (uses persistent working buffer cache)
        # ==================================================================

        # Ensure working buffers are allocated for this n (no-op if already done)
        _GPU_WORKING_BUFFERS.ensure(n)
        wbuf = _GPU_WORKING_BUFFERS

        # Accumulated profiling across all seed sets
        prof_spmv_ms  = 0.0
        prof_conv_ms  = 0.0
        prof_xfer_ms  = 0.0
        prof_conv_checks = 0

        stream_transfer.synchronize()   # ensure CSR upload is done before timing
        start_event.record(stream_compute)

        for b_idx, (sset, p0_np) in enumerate(zip(seed_sets, p0_list)):
            h_p0 = np.ascontiguousarray(p0_np, np.float32)

            # ---- H2D transfer timing (improvement 5) --------------------
            if enable_profiling:
                ev_xfer_s = cuda.Event()
                ev_xfer_e = cuda.Event()
                ev_xfer_s.record(stream_transfer)

            # Re-initialise persistent buffers with this seed set's p0.
            # d_pn is a scratch buffer; it will be fully overwritten by the
            # first SpMV so no zeroing is needed.
            cuda.memcpy_htod_async(wbuf.d_p.gpudata,  h_p0, stream_transfer)
            cuda.memcpy_htod_async(wbuf.d_p0.gpudata, h_p0, stream_transfer)

            if enable_profiling:
                ev_xfer_e.record(stream_transfer)

            stream_transfer.synchronize()

            if enable_profiling:
                prof_xfer_ms += ev_xfer_s.time_till(ev_xfer_e)

            # Local aliases for pointer swap (wbuf attrs are stable)
            d_p  = wbuf.d_p
            d_pn = wbuf.d_pn
            d_p0 = wbuf.d_p0

            converged  = False
            iterations = 0
            it_spmv_evts: list = []
            it_conv_evts: list = []

            # ---- Iteration loop -----------------------------------------
            for it in range(max_iter):
                iterations = it + 1

                if enable_profiling:
                    _es = cuda.Event(); _es.record(stream_compute)

                # SpMV: warp-per-node (grn/ppi/fp16) or adaptive tpr (mirna)
                k_spmv(
                    d_row_ptr, d_col_idx, d_w_values,
                    d_p, d_p0, d_pn,
                    one_minus_r, r_val, np.int32(n),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=spmv_grid,
                    stream=stream_compute,
                )

                if enable_profiling:
                    _ee = cuda.Event(); _ee.record(stream_compute)
                    it_spmv_evts.append((_es, _ee))

                # Convergence check every N iters (improvement 1)
                do_check = (iterations % conv_check_interval == 0) or (iterations == max_iter)
                if do_check:
                    if enable_profiling:
                        _cs = cuda.Event(); _cs.record(stream_compute)

                    k_l1(
                        d_pn, d_p, wbuf.d_partial, np.int32(n),
                        block=(BLOCK_SIZE, 1, 1),
                        grid=(n_partial_blocks, 1, 1),
                        stream=stream_compute,
                    )
                    k_reduce(
                        wbuf.d_partial, wbuf.d_l1_scalar, np.int32(n_partial_blocks),
                        block=(BLOCK_SIZE, 1, 1),
                        grid=(1, 1, 1),
                        stream=stream_compute,
                    )

                    if enable_profiling:
                        _ce = cuda.Event(); _ce.record(stream_compute)
                        it_conv_evts.append((_cs, _ce))

                # Pointer swap (no GPU work, just Python references)
                d_p, d_pn = d_pn, d_p

                if do_check:
                    stream_compute.synchronize()          # ONE sync per check
                    if float(wbuf.d_l1_scalar.get()[0]) < tolerance:
                        converged = True
                        break

            # ---- Collect per-seed-set profiling -------------------------
            if enable_profiling:
                prof_spmv_ms   += sum(s.time_till(e) for s, e in it_spmv_evts)
                prof_conv_ms   += sum(s.time_till(e) for s, e in it_conv_evts)
                prof_conv_checks += len(it_conv_evts)

            scores_b = d_p.get()
            all_scores.append({
                "seed_set":   sset,
                "scores":     scores_b,
                "iterations": iterations,
                "converged":  converged,
            })

        end_event.record(stream_compute)
        end_event.synchronize()
        elapsed = start_event.time_till(end_event) / 1000.0

        prof = None
        if enable_profiling:
            total_ms = elapsed * 1000.0
            prof = {
                "spmv_ms":               round(prof_spmv_ms, 4),
                "convergence_ms":        round(prof_conv_ms, 4),
                "transfer_ms":           round(prof_xfer_ms, 4),
                "total_ms":              round(total_ms, 4),
                "overhead_ms":           round(total_ms - prof_spmv_ms
                                               - prof_conv_ms - prof_xfer_ms, 4),
                "iterations":            sum(s["iterations"] for s in all_scores),
                "convergence_checks":    prof_conv_checks,
                "avg_spmv_ms_per_iter":  round(
                    prof_spmv_ms / max(1, sum(s["iterations"] for s in all_scores)), 4),
                "avg_conv_ms_per_check": round(
                    prof_conv_ms / max(1, prof_conv_checks), 4),
            }

        return _finalize_result(
            all_scores=all_scores, n=n, seed_sets=seed_sets,
            network_type=network_type, graph_csr=graph_csr,
            elapsed=elapsed, transition_note=transition_note,
            arch_flag=kernels.get("_arch_flag", "?"),
            precision_mode=precision_mode, chunked=False,
            profiling_data=prof,
        )

    except cuda.LogicError as e:
        logging.error("CUDA error in rwr_gpu: %s", e)
        raise
    except MemoryError:
        logging.warning(
            "VRAM exhausted in rwr_gpu.  Retry with use_chunking=True "
            "or precision_mode='mixed'."
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
# Result aggregation
# ---------------------------------------------------------------------------

def _finalize_result(
    *,
    all_scores: list[dict],
    n: int,
    seed_sets: list[list[int]],
    network_type: str,
    graph_csr: sp.csr_matrix,
    elapsed: float,
    transition_note: str,
    arch_flag: str,
    precision_mode: str,
    chunked: bool,
    profiling_data: dict | None = None,
) -> dict:
    """Build the final result dict (matches the CLAUDE.md RWR spec)."""
    if len(all_scores) == 1:
        primary_scores   = all_scores[0]["scores"]
        total_iterations = int(all_scores[0]["iterations"])
        any_converged    = bool(all_scores[0]["converged"])
        batch_results    = None
    else:
        score_matrix     = np.array([s["scores"] for s in all_scores], dtype=np.float64)
        primary_scores   = score_matrix.mean(axis=0).astype(np.float32)
        total_iterations = int(max(s["iterations"] for s in all_scores))
        any_converged    = bool(all(s["converged"] for s in all_scores))
        batch_results    = [
            {"seed_set": list(s["seed_set"]),
             "scores":   [float(x) for x in s["scores"]],
             "iterations": int(s["iterations"]),
             "converged":  bool(s["converged"])}
            for s in all_scores
        ]

    top_nodes = _top_k(primary_scores, _TOP_NODES)
    all_seed_indices = np.array(
        sorted({int(idx) for sset in seed_sets for idx in sset
                if 0 <= int(idx) < n}),
        dtype=np.int32,
    )
    if all_seed_indices.size > 0:
        top_seeds = _top_k_among(primary_scores, all_seed_indices, _TOP_SEEDS)
    else:
        top_seeds = top_nodes[:_TOP_SEEDS]

    note = (
        f"{transition_note}. "
        f"GPU pipeline: warp_per_node, precision={precision_mode}, "
        f"chunked={chunked}, arch={arch_flag}, "
        f"syncs_per_check=1 (check every N iters)."
    )

    inner: dict = {
        "scores":     [float(x) for x in primary_scores],
        "top_nodes":  top_nodes,
        "top_seeds":  top_seeds,
        "iterations": total_iterations,
        "converged":  any_converged,
        "note":       note,
    }
    if batch_results is not None:
        inner["batch_results"] = batch_results
    if profiling_data is not None:
        inner["profiling"] = profiling_data

    return {
        "algorithm":      "rwr",
        "mode":           "gpu",
        "network_type":   network_type,
        "execution_time": elapsed,
        "num_nodes":      n,
        "num_edges":      int(graph_csr.nnz),
        "result":         inner,
    }


# ---------------------------------------------------------------------------
# FP32 vs Mixed-Precision benchmark utility — improvement 2
# ---------------------------------------------------------------------------

def benchmark_precision_modes(
    graph_csr: sp.csr_matrix,
    params: dict | None = None,
    num_warmup: int = 2,
    num_runs: int = 5,
) -> dict:
    """Compare FP32 and mixed-precision (FP16 weights) on the given graph.

    Methodology
    -----------
    1.  Run each mode `num_warmup` times to populate all caches (W, GPU CSR,
        working buffers, kernel compilation).
    2.  Run each mode `num_runs` times; use execution_time from each result
        (CUDA-event timing already excludes context setup and CSR upload).
    3.  Report mean / std / min / max times, speedup ratio, top-node overlap,
        and a plain-English recommendation.

    Important: the function does NOT automatically switch the active
    precision mode.  It only benchmarks and reports.

    Parameters
    ----------
    graph_csr   : scipy CSR matrix to benchmark on.
    params      : optional base parameter dict (network_type, seed_nodes, …).
                  precision_mode is overridden internally for each mode.
    num_warmup  : warm-up runs per mode (populate all caches).
    num_runs    : timed runs per mode.

    Returns
    -------
    dict with keys:
        fp32, mixed           — per-mode timing and convergence statistics.
        speedup_mixed_over_fp32 — fp32_mean / mixed_mean (>1 means mixed faster).
        recommendation        — plain-English string.
        numerically_equivalent — True if top-20 nodes overlap >= 90 %.
        top_nodes_overlap_fraction — exact overlap fraction.
        num_warmup, num_runs  — benchmark configuration for reproducibility.
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError("PyCUDA is required for benchmark_precision_modes().")

    base = _merge_params(params)
    # Disable nested profiling during the benchmark (would skew timings)
    base["enable_profiling"] = False

    results: dict[str, dict] = {}

    for mode in ("fp32", "mixed"):
        mode_params = {**base, "precision_mode": mode}

        # ---- Warm-up (populate all caches) ------------------------------
        last_result: dict | None = None
        for _ in range(max(1, num_warmup)):
            try:
                last_result = rwr_gpu(graph_csr, mode_params)
            except Exception as exc:                    # noqa: BLE001
                results[mode] = {"error": str(exc)}
                break
        else:
            # ---- Timed runs ---------------------------------------------
            times: list[float] = []
            iters_list: list[int] = []
            conv_list: list[bool] = []
            for _ in range(max(1, num_runs)):
                try:
                    r = rwr_gpu(graph_csr, mode_params)
                    times.append(float(r["execution_time"]) * 1000.0)  # → ms
                    iters_list.append(int(r["result"]["iterations"]))
                    conv_list.append(bool(r["result"]["converged"]))
                    last_result = r
                except Exception as exc:                # noqa: BLE001
                    logging.warning("benchmark_precision_modes [%s] run failed: %s",
                                    mode, exc)

            if times:
                arr = np.asarray(times, dtype=np.float64)
                results[mode] = {
                    "mean_ms":   float(np.mean(arr)),
                    "std_ms":    float(np.std(arr)),
                    "min_ms":    float(np.min(arr)),
                    "max_ms":    float(np.max(arr)),
                    "iterations": int(round(float(np.median(iters_list)))),
                    "converged":  bool(all(conv_list)),
                    "top_nodes":  (last_result["result"]["top_nodes"]
                                   if last_result is not None else []),
                    "num_valid_runs": len(times),
                }
            else:
                results[mode] = {"error": "all timed runs failed", "top_nodes": []}

    # ---- Comparison metrics ---------------------------------------------
    fp32_ok  = "error" not in results.get("fp32",  {"error": True})
    mixed_ok = "error" not in results.get("mixed", {"error": True})

    if fp32_ok and mixed_ok:
        fp32_mean  = results["fp32"]["mean_ms"]
        mixed_mean = results["mixed"]["mean_ms"]
        speedup    = fp32_mean / mixed_mean if mixed_mean > 0 else 0.0

        fp32_top  = set(results["fp32"].get("top_nodes",  []))
        mixed_top = set(results["mixed"].get("top_nodes", []))
        union_sz  = max(1, len(fp32_top | mixed_top))
        overlap   = len(fp32_top & mixed_top)
        overlap_frac = overlap / union_sz
        num_equiv = overlap_frac >= 0.90

        if speedup > 1.10:
            rec = (f"Use precision_mode='mixed': {speedup:.2f}× faster than FP32 "
                   f"({mixed_mean:.2f} ms vs {fp32_mean:.2f} ms).")
        elif speedup < 0.90:
            rec = (f"Use precision_mode='fp32': mixed-precision is {1/speedup:.2f}× "
                   f"slower on this graph ({mixed_mean:.2f} ms vs {fp32_mean:.2f} ms).  "
                   f"FP32 benefits from better DRAM burst alignment at this sparsity.")
        else:
            rec = (f"Both modes perform similarly ({fp32_mean:.2f} ms vs "
                   f"{mixed_mean:.2f} ms, speedup={speedup:.2f}×).  "
                   f"Prefer precision_mode='fp32' for full numerical accuracy.")
    else:
        speedup      = 0.0
        overlap_frac = 0.0
        num_equiv    = False
        rec          = "Could not compare: one or both modes failed."

    return {
        "fp32":                       results.get("fp32",  {}),
        "mixed":                      results.get("mixed", {}),
        "speedup_mixed_over_fp32":    round(speedup, 4),
        "recommendation":             rec,
        "numerically_equivalent":     num_equiv,
        "top_nodes_overlap_fraction": round(overlap_frac, 4),
        "num_warmup":                 num_warmup,
        "num_runs":                   num_runs,
    }


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _gpu(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    """Runner entry point — returns ``{output, extra_params}`` envelope."""
    p    = _merge_params(params)
    full = rwr_gpu(graph_csr, p)
    return {"output": full, "extra_params": p}
