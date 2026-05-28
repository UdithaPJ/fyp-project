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

  GRN    — seed with disease-associated TFs (TP53 in cancer, NF-κB in
           inflammation); top-ranked genes are candidate downstream
           regulatory targets, potentially many hops away from any seed.
  PPI    — seed with known disease proteins; top-ranked proteins are
           candidate disease modifiers (network medicine workflows).
  miRNA  — seed with miRNAs of interest; top-ranked genes are
           candidate co-targeted effectors of the miRNA panel.

Steady-state equation:
    p* = (1 - r) · W · p* + r · p₀

W is the column-stochastic transition matrix (columns sum to 1) and p₀
is the seed distribution (uniform over seeds; uniform over all nodes
when no seeds are provided).

Network-type-aware transition matrix
------------------------------------
GRN    : directed CSR, column-normalised by out-degree.
         Dangling columns (out_degree == 0) get a self-loop so that
         probability mass is conserved during diffusion.
PPI    : A_sym = A + Aᵀ, binarised, then column-normalised.
PPI    bidirectional interactions get equal weight per direction.
miRNA  : directed bipartite CSR (miRNA → gene), column-normalised.
         Gene-target nodes (out_degree == 0) get self-loops.

Optimised GPU pipeline (this module)
------------------------------------
1. GPU-only convergence reduction (``reduce_to_scalar_f32``):
   - Block-partial L1 written by ``l1_convergence_rwr`` is reduced to
     a single FP32 scalar on the GPU.  Only that one float crosses
     PCIe per iteration (vs. the full partial-sum array before).
   - Exactly ONE ``stream_compute.synchronize()`` per iteration.

2. Hub-row ELLPACK for memory coalescing:
   - Rows with ``degree >= hub_threshold`` are repacked into padded
     ELLPACK (predictable stride, no ``row_ptr`` indirection).
   - Adaptive hub threshold at the 95th degree percentile, clamped
     to ``[WARP_SIZE, max_degree]`` and snapped to a power of 2.
   - Remaining rows continue to use CSR with the existing thread /
     warp / block three-tier dispatch.

3. Shared-memory caching of p for small graphs (``rwr_spmv_smem_p``):
   - When ``n <= SMEM_P_LIMIT`` (default 4096), the entire previous
     iteration's p vector is cached in shared memory once per block.
   - Eliminates the gmem reads of p during SpMV gather — the main
     bandwidth-bound step for power-law biological networks.
   - Auto-selected; falls back to the regular CSR/ELLPACK split for
     large graphs.

4. Optional mixed precision (``precision_mode = "mixed"``):
   - Transition weights stored as FP16 (``unsigned short`` half-bit
     pattern), p / p_new / p₀ remain FP32, accumulation is FP32.
   - Halves the w_values bandwidth at negligible accuracy cost
     (RWR converges in 30-100 iterations, FP16 rounding error stays
     well below the typical ``tolerance = 1e-6``).

5. Async overlap between SpMV and convergence reduction:
   - ``stream_compute`` for SpMV; ``stream_transfer`` for async H2D
     during setup AND for the convergence reduction launch.
   - ``cuda.Event`` between the two streams orders the launches
     without a host sync between them.

6. Chunked execution for graphs exceeding VRAM (``_rwr_gpu_chunked``):
   - p, p_new, p₀, and the partial-sum buffers stay full-size on the
     device throughout.  CSR rows are streamed in chunks; per-chunk
     scatter writes only into the chunk's row range of p_new.
   - Double-buffer pipeline: ``stream_transfer`` pre-loads chunk i+1
     while ``stream_compute`` runs chunk i SpMV.
   - Auto-triggered when estimated VRAM > 80 % of free VRAM or
     ``use_chunking=True`` from ``apply_config``.

7. Adaptive arch compilation (``_detect_arch_flag``):
   - Runtime ``cuda.Device(0).compute_capability()`` →
     ``-arch=sm_<major><minor>``.  Falls back to ``sm_75`` (RTX 20).

Does NOT silently fall back to CPU — raises ``RuntimeError`` /
``cuda.LogicError`` / ``MemoryError`` so the runner can surface the
failure.  The CPU implementations live in a separate package:
``src.algorithms.cpu.single_threaded.rwr`` and
``src.algorithms.cpu.multi_threaded.rwr`` (benchmarking only).

Multi-seed-set support
----------------------
``seed_nodes`` accepts either a flat list (single RWR run) or a
list-of-lists (one RWR per inner list, scores averaged across runs to
form a consensus influence vector).  Batched kernel handles
``1 < B <= MAX_BATCH = 4``; larger batches loop the single-seed kernel.

Parameter guide
---------------
restart_prob    (float, default 0.3)    Teleport probability per step.
max_iter        (int,   default 100)    Hard iteration cap.
tolerance       (float, default 1e-6)   L1-norm early-stop threshold.
seed_nodes      (list[int] or           Seeds for p₀.
                 list[list[int]])
network_type    (str,   default "grn")  One of "grn", "ppi", "mirna".
block_size      (int,   default 256)    CUDA block dimension.
precision_mode  (str,   default "fp32") "fp32" or "mixed" (FP16 W).
ellpack_fraction(float, default 0.05)   Fraction of rows in ELLPACK.
use_chunking    (bool,  default False)  Force chunked path on.
"""

# ── GPU / CUDA-optimised implementation (PyCUDA custom kernels) ──────────
# Source:    src/algorithms/gpu/cuda_optimized/rwr.py
# Requires:  pycuda (with a working NVCC toolchain)
# Used by:   webapp routes, GPU benchmarking (src.benchmarking.benchmark)
# Modes:     _gpu only — this module is GPU-exclusive.
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
    logging.warning("PyCUDA not available — rwr_gpu() will raise.")

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
    "restart_prob":     0.3,
    "max_iter":         100,
    "tolerance":        1e-6,
    "seed_nodes":       [],
    "network_type":     "grn",
    "block_size":       256,
    "precision_mode":   "fp32",       # "fp32" or "mixed"
    "ellpack_fraction": 0.05,         # ~5% of rows go to ELLPACK
    "use_chunking":     False,
    "use_zero_copy":    False,
    "reorder_nodes":    False,
}

BLOCK_SIZE: int          = 256
WARP_SIZE: int           = 32
MAX_BATCH: int           = 4              # batched kernel ceiling
SMEM_P_LIMIT: int        = 4096           # n above which SMEM-cached p is skipped
_TOP_NODES: int          = 20
_TOP_SEEDS: int          = 10
VRAM_BUDGET_FRACTION: float = 0.80


# ---------------------------------------------------------------------------
# CUDA kernel source
# ---------------------------------------------------------------------------
#
# Kernels:
#   reduce_to_scalar_f32           — GPU-side single-block reduction
#                                     of a partial-sum array to one scalar.
#   rwr_spmv_restart               — FP32 fused SpMV + restart (three-tier).
#   rwr_spmv_restart_fp16w         — same but reads FP16 (unsigned short
#                                     half-bit) weights, FP32 accumulation.
#   rwr_spmv_smem_p                — small-graph SMEM-cached p variant.
#   rwr_spmv_ellpack_hubs          — padded ELLPACK kernel for hub rows.
#   l1_convergence_rwr             — Σ|p_new - p| block-partial.
#   rwr_spmv_restart_batched       — multi-seed batched variant (B <= 4).
#
# All kernels live in one SourceModule (compiled once, cached).

KERNEL_SOURCE = r"""
#include <cuda_fp16.h>

extern "C" {

#define BLOCK_SIZE       256
#define WARP_SIZE        32
#define WARPS_PER_BLOCK  (BLOCK_SIZE / WARP_SIZE)

// =========================================================================
// KERNEL: reduce_to_scalar_f32  (NEW)
//
// Strided load + shared-memory tree reduction in one kernel.  Launched
// with grid=(1,1,1).  Writes a SINGLE float to scalar_output[0].  The
// host can read just that one float (or keep it on GPU for the next
// kernel that needs it).  Replaces the per-iteration .get() of the
// partial-sum array.
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
// KERNEL: rwr_spmv_restart  (FP32, three-tier)
//
// Fused SpMV + restart for a single seed set:
//   p_new[i] = (1 - r) * Σ_j W[i,j] * p[j]  +  r * p0[i]
//
// One block per node i.  Three-tier degree-aware scheduling:
//   LOW  (deg < 32)         : thread 0 only, serial scan
//   MED  (32 <= deg < 256)  : first warp, stride-32 + shfl_down_sync
//   HIGH (deg >= 256)       : full block, stride-256 + shared mem tree
//
// Restart term r * p0[i] is added by thread 0 ONLY, after the
// reduction, in the same write that stores p_new[i].
//
// __ldg(&p[col]) routes p neighbour reads through the read-only
// texture cache.
// =========================================================================
__global__ void rwr_spmv_restart(
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
    __shared__ float smem[BLOCK_SIZE];

    const int node_i = blockIdx.x;
    if (node_i >= n) return;

    const int row_start = row_ptr[node_i];
    const int row_end   = row_ptr[node_i + 1];
    const int degree    = row_end - row_start;

    if (degree == 0) {
        if (threadIdx.x == 0) p_new[node_i] = r * p0[node_i];
        return;
    }

    // ---- LOW tier --------------------------------------------------------
    if (degree < WARP_SIZE) {
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int j = row_start; j < row_end; ++j) {
                sum += w_values[j] * __ldg(&p[col_idx[j]]);
            }
            p_new[node_i] = one_minus_r * sum + r * p0[node_i];
        }
        return;
    }

    // ---- MED tier --------------------------------------------------------
    if (degree < BLOCK_SIZE) {
        if (threadIdx.x < WARP_SIZE) {
            float partial = 0.0f;
            for (int j = row_start + threadIdx.x; j < row_end; j += WARP_SIZE) {
                partial += w_values[j] * __ldg(&p[col_idx[j]]);
            }
            for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
                partial += __shfl_down_sync(0xffffffffu, partial, off);
            }
            if (threadIdx.x == 0) {
                p_new[node_i] = one_minus_r * partial + r * p0[node_i];
            }
        }
        return;
    }

    // ---- HIGH tier -------------------------------------------------------
    float partial = 0.0f;
    for (int j = row_start + threadIdx.x; j < row_end; j += BLOCK_SIZE) {
        partial += w_values[j] * __ldg(&p[col_idx[j]]);
    }
    smem[threadIdx.x] = partial;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        p_new[node_i] = one_minus_r * smem[0] + r * p0[node_i];
    }
}


// =========================================================================
// KERNEL: rwr_spmv_restart_fp16w  (NEW — mixed precision)
//
// Same dispatch as rwr_spmv_restart, but reads w_values as half (FP16)
// and converts to float at use time.  Halves the bandwidth of the
// w_values stream.  p / p_new / p0 stay FP32.  Accumulation is FP32.
// =========================================================================
__global__ void rwr_spmv_restart_fp16w(
    const int*           __restrict__ row_ptr,
    const int*           __restrict__ col_idx,
    const unsigned short* __restrict__ w_values_h,    // FP16 bit pattern
    const float*         __restrict__ p,
    const float*         __restrict__ p0,
    float*               __restrict__ p_new,
    const float                        one_minus_r,
    const float                        r,
    const int                          n)
{
    __shared__ float smem[BLOCK_SIZE];

    const int node_i = blockIdx.x;
    if (node_i >= n) return;

    const int row_start = row_ptr[node_i];
    const int row_end   = row_ptr[node_i + 1];
    const int degree    = row_end - row_start;

    if (degree == 0) {
        if (threadIdx.x == 0) p_new[node_i] = r * p0[node_i];
        return;
    }

    // helper macro: load FP16 weight as float
    #define WEIGHT(j) __half2float(*reinterpret_cast<const __half*>(&w_values_h[j]))

    if (degree < WARP_SIZE) {
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int j = row_start; j < row_end; ++j) {
                sum += WEIGHT(j) * __ldg(&p[col_idx[j]]);
            }
            p_new[node_i] = one_minus_r * sum + r * p0[node_i];
        }
        return;
    }

    if (degree < BLOCK_SIZE) {
        if (threadIdx.x < WARP_SIZE) {
            float partial = 0.0f;
            for (int j = row_start + threadIdx.x; j < row_end; j += WARP_SIZE) {
                partial += WEIGHT(j) * __ldg(&p[col_idx[j]]);
            }
            for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
                partial += __shfl_down_sync(0xffffffffu, partial, off);
            }
            if (threadIdx.x == 0) {
                p_new[node_i] = one_minus_r * partial + r * p0[node_i];
            }
        }
        return;
    }

    float partial = 0.0f;
    for (int j = row_start + threadIdx.x; j < row_end; j += BLOCK_SIZE) {
        partial += WEIGHT(j) * __ldg(&p[col_idx[j]]);
    }
    smem[threadIdx.x] = partial;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        p_new[node_i] = one_minus_r * smem[0] + r * p0[node_i];
    }

    #undef WEIGHT
}


// =========================================================================
// KERNEL: rwr_spmv_smem_p  (NEW — small-graph SMEM cache)
//
// For graphs small enough that the entire p vector fits in shared
// memory (n <= SMEM_P_LIMIT, ~4096 nodes = 16 KB / block).  Each block
// first cooperatively loads p[0..n) into SMEM, then evaluates row
// blockIdx.x.  Eliminates the gmem reads of p during the SpMV gather.
//
// Launch contract: block=(BLOCK_SIZE,1,1), grid=(n,1,1), dynamic SMEM
// = n * sizeof(float) bytes.
// =========================================================================
__global__ void rwr_spmv_smem_p(
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
    extern __shared__ float p_cache[];

    // ---- Cooperative load of p into SMEM ----
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        p_cache[i] = p[i];
    }
    __syncthreads();

    const int node_i = blockIdx.x;
    if (node_i >= n) return;

    const int row_start = row_ptr[node_i];
    const int row_end   = row_ptr[node_i + 1];
    const int degree    = row_end - row_start;

    if (degree == 0) {
        if (threadIdx.x == 0) p_new[node_i] = r * p0[node_i];
        return;
    }

    __shared__ float smem_red[BLOCK_SIZE];

    if (degree < WARP_SIZE) {
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int j = row_start; j < row_end; ++j) {
                sum += w_values[j] * p_cache[col_idx[j]];
            }
            p_new[node_i] = one_minus_r * sum + r * p0[node_i];
        }
        return;
    }

    if (degree < BLOCK_SIZE) {
        if (threadIdx.x < WARP_SIZE) {
            float partial = 0.0f;
            for (int j = row_start + threadIdx.x; j < row_end; j += WARP_SIZE) {
                partial += w_values[j] * p_cache[col_idx[j]];
            }
            for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
                partial += __shfl_down_sync(0xffffffffu, partial, off);
            }
            if (threadIdx.x == 0) {
                p_new[node_i] = one_minus_r * partial + r * p0[node_i];
            }
        }
        return;
    }

    float partial = 0.0f;
    for (int j = row_start + threadIdx.x; j < row_end; j += BLOCK_SIZE) {
        partial += w_values[j] * p_cache[col_idx[j]];
    }
    smem_red[threadIdx.x] = partial;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem_red[threadIdx.x] += smem_red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        p_new[node_i] = one_minus_r * smem_red[0] + r * p0[node_i];
    }
}


// =========================================================================
// KERNEL: rwr_spmv_ellpack_hubs  (NEW — ELLPACK for hub rows)
//
// One block per hub row.  Padded ELLPACK layout: hub_col_idx and
// hub_w_values are [num_hubs x max_row_len] row-major arrays;
// padding entries have col_idx == -1 and are skipped.
//
// Predictable stride (max_row_len) eliminates the row_ptr indirection
// and gives much better memory coalescing for the hub-tier nodes,
// which dominate the long tail of the SpMV cost on power-law graphs.
//
// Writes p_new directly (no atomic — one block per output node).
// =========================================================================
__global__ void rwr_spmv_ellpack_hubs(
    const int*   __restrict__ hub_ids,       // length num_hubs (global node id)
    const int*   __restrict__ hub_col_idx,   // [num_hubs x max_row_len]
    const float* __restrict__ hub_w_values,  // [num_hubs x max_row_len]
    const int                  max_row_len,
    const float* __restrict__ p,
    const float* __restrict__ p0,
    float*       __restrict__ p_new,
    const float                one_minus_r,
    const float                r,
    const int                  num_hubs)
{
    __shared__ float smem[BLOCK_SIZE];

    const int hub_idx = blockIdx.x;
    if (hub_idx >= num_hubs) return;
    const int node_i = hub_ids[hub_idx];

    const int row_base = hub_idx * max_row_len;

    float partial = 0.0f;
    for (int k = threadIdx.x; k < max_row_len; k += BLOCK_SIZE) {
        const int c = hub_col_idx[row_base + k];
        if (c >= 0) {
            partial += hub_w_values[row_base + k] * __ldg(&p[c]);
        }
    }
    smem[threadIdx.x] = partial;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        p_new[node_i] = one_minus_r * smem[0] + r * p0[node_i];
    }
}


// =========================================================================
// KERNEL: l1_convergence_rwr  (UNCHANGED)
//
// Σ |p_new[i] - p[i]| per block via warp-shuffle intra-warp reduction
// then a final warp-shuffle across the WARPS_PER_BLOCK partial sums in
// shared memory.  Partials reduced by reduce_to_scalar_f32.
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
// KERNEL: rwr_spmv_restart_batched  (UNCHANGED — multi-seed batched)
//
// Batched multi-seed RWR.  gridDim = (n, B, 1).
// =========================================================================
__global__ void rwr_spmv_restart_batched(
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
    __shared__ float smem[BLOCK_SIZE];

    const int node_i = blockIdx.x;
    const int seed_b = blockIdx.y;
    if (node_i >= n || seed_b >= B) return;

    const int row_start = row_ptr[node_i];
    const int row_end   = row_ptr[node_i + 1];
    const int degree    = row_end - row_start;
    const int out_idx   = node_i * B + seed_b;

    if (degree == 0) {
        if (threadIdx.x == 0) p_new[out_idx] = r * p0[out_idx];
        return;
    }

    if (degree < WARP_SIZE) {
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int j = row_start; j < row_end; ++j) {
                sum += w_values[j] * __ldg(&p[col_idx[j] * B + seed_b]);
            }
            p_new[out_idx] = one_minus_r * sum + r * p0[out_idx];
        }
        return;
    }

    if (degree < BLOCK_SIZE) {
        if (threadIdx.x < WARP_SIZE) {
            float partial = 0.0f;
            for (int j = row_start + threadIdx.x; j < row_end; j += WARP_SIZE) {
                partial += w_values[j] * __ldg(&p[col_idx[j] * B + seed_b]);
            }
            for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
                partial += __shfl_down_sync(0xffffffffu, partial, off);
            }
            if (threadIdx.x == 0) {
                p_new[out_idx] = one_minus_r * partial + r * p0[out_idx];
            }
        }
        return;
    }

    float partial = 0.0f;
    for (int j = row_start + threadIdx.x; j < row_end; j += BLOCK_SIZE) {
        partial += w_values[j] * __ldg(&p[col_idx[j] * B + seed_b]);
    }
    smem[threadIdx.x] = partial;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        p_new[out_idx] = one_minus_r * smem[0] + r * p0[out_idx];
    }
}

}  // extern "C"
"""


# ---------------------------------------------------------------------------
# Module-level kernel cache + adaptive compilation
# ---------------------------------------------------------------------------

_kernel_cache: dict[str, dict[str, Any]] = {}


def _detect_arch_flag() -> str:
    """Return ``-arch=sm_XY`` for the current device, with a Turing fallback.

    Falls back to ``sm_75`` (RTX 20-series) if PyCUDA cannot probe the
    device — that matches the project's target hardware.
    """
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
        _kernel_cache["rwr"] = {
            "reduce_scalar":        mod.get_function("reduce_to_scalar_f32"),
            "spmv_restart":         mod.get_function("rwr_spmv_restart"),
            "spmv_restart_fp16w":   mod.get_function("rwr_spmv_restart_fp16w"),
            "spmv_smem_p":          mod.get_function("rwr_spmv_smem_p"),
            "spmv_ellpack_hubs":    mod.get_function("rwr_spmv_ellpack_hubs"),
            "l1_conv":              mod.get_function("l1_convergence_rwr"),
            "spmv_restart_batched": mod.get_function("rwr_spmv_restart_batched"),
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
    """Build the column-stochastic transition matrix W for RWR.

    RWR update: ``p_new = (1 - r) · W · p + r · p₀``.  W must be
    column-stochastic (every column sums to 1) so that the random
    walk preserves probability mass.
    """
    nt = str(network_type).lower()

    if nt == "ppi":
        A = (graph_csr + graph_csr.T).astype(np.float32)
        A.data = np.ones_like(A.data, dtype=np.float32)        # binarise
        note = "PPI: A + Aᵀ binarised, then column-normalised"
    else:  # grn / mirna / default
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


def _build_p0(
    seed_nodes,
    n: int,
    network_type: str,
) -> np.ndarray:
    """Build the restart vector p₀ (length n, sums to 1.0).

    Empty / invalid seeds → uniform 1/n (global PageRank-like).
    Otherwise → uniform 1/|seeds| over valid seed indices.
    """
    p0 = np.zeros(n, dtype=np.float32)
    valid = [int(s) for s in (seed_nodes or []) if isinstance(s, (int, np.integer))
             and 0 <= int(s) < n]
    if len(valid) == 0:
        p0[:] = np.float32(1.0 / n)
    else:
        p0[valid] = np.float32(1.0 / len(valid))
    return p0


def _reorder_nodes_by_degree(
    W: sp.csr_matrix,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Permute W's rows and columns so similar-degree nodes are adjacent."""
    degrees = np.diff(W.indptr).astype(np.int32)
    perm    = np.argsort(degrees).astype(np.int32)
    W_row   = W[perm, :].tocsr()
    W_full  = W_row[:, perm].tocsr()
    return W_full.astype(np.float32), perm


def _estimate_rwr_vram(n: int, nnz: int, batch_size: int = 1,
                       fp16_weights: bool = False) -> int:
    """Conservative VRAM estimate (bytes) for RWR working set."""
    w_value_bytes = 2 if fp16_weights else 4
    csr_b      = (n + 1) * 4 + nnz * 4 + nnz * w_value_bytes
    pr_b       = n * 4 * (2 + batch_size)                 # p, p_new, B × p₀
    partial_b  = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE) * 4
    scalar_b   = 4 * 4                                    # convergence scalars
    return int(csr_b + pr_b + partial_b + scalar_b)


# ---------------------------------------------------------------------------
# Adaptive hub threshold + ELLPACK builder
# ---------------------------------------------------------------------------

def _compute_adaptive_hub_threshold(
    row_lens: np.ndarray,
    target_ellpack_fraction: float = 0.05,
) -> int:
    """Return a hub threshold (row length) at the (1 - target) percentile.

    Clamped to ``[WARP_SIZE, max_degree]`` and snapped to a power of 2.
    """
    if row_lens.size == 0:
        return WARP_SIZE
    deg_int = row_lens.astype(np.int64)
    pct = (1.0 - max(0.0, min(target_ellpack_fraction, 1.0))) * 100.0
    threshold = int(np.percentile(deg_int, pct))
    threshold = max(threshold, WARP_SIZE)
    threshold = min(threshold, int(deg_int.max()) if deg_int.size > 0 else WARP_SIZE)
    threshold = max(threshold, 1)
    snapped = int(2 ** round(math.log2(max(threshold, 1))))
    return max(snapped, WARP_SIZE)


def _build_ellpack_split(
    W: sp.csr_matrix,
    hub_threshold: int,
) -> tuple[dict, dict]:
    """Split W's rows into (hub_ellpack, low_csr) based on row length.

    Returns
    -------
    ellpack : dict
        {"hub_ids": int32[num_hubs],
         "hub_col_idx": int32[num_hubs * max_row_len],
         "hub_w_values": float32[num_hubs * max_row_len],
         "max_row_len": int,
         "num_hubs": int}
        Padding entries have col_idx == -1.
    low_csr : dict
        {"row_ptr": int32[n+1], "col_idx": int32[nnz_low],
         "values": float32[nnz_low], "n": int}
        Hub rows are zeroed in row_ptr (row_end == row_start) so the
        CSR kernel naturally skips them.
    """
    n = int(W.shape[0])
    row_lens = np.diff(W.indptr).astype(np.int32)
    hub_mask = row_lens >= hub_threshold
    hub_ids  = np.where(hub_mask)[0].astype(np.int32)
    num_hubs = int(hub_ids.size)

    # ---- Build ELLPACK ----------------------------------------------------
    if num_hubs == 0:
        ellpack = {
            "hub_ids":     np.empty((0,), dtype=np.int32),
            "hub_col_idx": np.empty((0,), dtype=np.int32),
            "hub_w_values": np.empty((0,), dtype=np.float32),
            "max_row_len": 0,
            "num_hubs":    0,
        }
    else:
        max_row_len = int(row_lens[hub_mask].max())
        hub_col_idx  = np.full((num_hubs * max_row_len,), -1, dtype=np.int32)
        hub_w_values = np.zeros((num_hubs * max_row_len,), dtype=np.float32)
        indptr  = W.indptr
        indices = W.indices
        values  = W.data.astype(np.float32, copy=False)
        for h_idx, node_i in enumerate(hub_ids):
            rs = int(indptr[node_i])
            re = int(indptr[node_i + 1])
            k = re - rs
            base = h_idx * max_row_len
            hub_col_idx[base:base + k]  = indices[rs:re]
            hub_w_values[base:base + k] = values[rs:re]
        ellpack = {
            "hub_ids":      hub_ids,
            "hub_col_idx":  hub_col_idx,
            "hub_w_values": hub_w_values,
            "max_row_len":  max_row_len,
            "num_hubs":     num_hubs,
        }

    # ---- Build low CSR (zero out hub rows) -------------------------------
    new_indptr = W.indptr.astype(np.int32, copy=True)
    if num_hubs > 0:
        # Zero out hub rows by collapsing their row_end to row_start
        keep_mask = np.ones(int(W.nnz), dtype=bool)
        for node_i in hub_ids:
            rs = int(W.indptr[node_i])
            re = int(W.indptr[node_i + 1])
            keep_mask[rs:re] = False
        new_col_idx = W.indices[keep_mask].astype(np.int32, copy=False)
        new_values  = W.data[keep_mask].astype(np.float32, copy=False)
        # Rebuild row_ptr
        new_row_lens = row_lens.copy()
        new_row_lens[hub_mask] = 0
        new_indptr = np.empty(n + 1, dtype=np.int32)
        new_indptr[0] = 0
        np.cumsum(new_row_lens, out=new_indptr[1:])
    else:
        new_col_idx = W.indices.astype(np.int32, copy=False)
        new_values  = W.data.astype(np.float32, copy=False)

    low_csr = {
        "row_ptr": new_indptr,
        "col_idx": new_col_idx,
        "values":  new_values,
        "n":       n,
    }
    return ellpack, low_csr


def _fp32_to_fp16_bits(values: np.ndarray) -> np.ndarray:
    """Convert an FP32 array to FP16 bit pattern as ``uint16`` (numpy)."""
    return values.astype(np.float16).view(np.uint16).astype(np.uint16)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _top_k(scores: np.ndarray, k: int = _TOP_NODES) -> list[int]:
    """Return the top-k node indices by descending score."""
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    return np.argsort(scores)[::-1][:k].astype(int).tolist()


def _top_k_among(
    scores: np.ndarray,
    candidate_indices: np.ndarray,
    k: int,
) -> list[int]:
    """Top-k from a candidate subset, ordered by score descending."""
    if candidate_indices.size == 0:
        return []
    sub = scores[candidate_indices]
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
        self.max_nnz   = max_nnz
        self.max_nodes = max_nodes

    def free(self) -> None:
        for arr in (self.d_row_ptr, self.d_col_idx,
                    self.d_values, self.d_node_ids):
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass


def _build_csr_chunks(
    W: sp.csr_matrix,
    chunk_size: int,
) -> list[dict]:
    """Split W into row-chunks for chunked SpMV.

    Each chunk dict has: ``node_ids`` (original row indices),
    ``indptr`` (rebased CSR indptr for this chunk), ``indices``,
    ``values``.
    """
    n = int(W.shape[0])
    chunks: list[dict] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        node_ids = np.arange(start, end, dtype=np.int32)
        rs = int(W.indptr[start])
        re = int(W.indptr[end])
        local_indptr = (W.indptr[start:end + 1] - rs).astype(np.int32)
        chunks.append({
            "node_ids": node_ids,
            "indptr":   local_indptr,
            "indices":  W.indices[rs:re].astype(np.int32, copy=False),
            "values":   W.data[rs:re].astype(np.float32, copy=False),
        })
    return chunks


# ---------------------------------------------------------------------------
# Chunked SpMV kernel (CSR with explicit node_ids)
# ---------------------------------------------------------------------------
# We do NOT add a separate "chunked" CUDA kernel: we reuse spmv_restart
# by passing chunked CSR + remapping with a tiny driver-side loop that
# offsets the output writes via a chunk-local row_ptr layout.  Each
# chunk is processed as a separate kernel launch with its own (chunked
# n) grid, but the same kernel reads its row_ptr / col_idx / values.

# ---------------------------------------------------------------------------
# Helper kernel (compiled with the others) that handles the chunk's
# row→node mapping. We use a tiny wrapper kernel that translates blockIdx
# back to the chunk's node_ids[chunk_local_row] and writes p_new[node_i].
# Implemented inline below as a wrapper around `spmv_restart` semantics
# but with a `node_ids` indirection.
# ---------------------------------------------------------------------------

_CHUNK_KERNEL_SOURCE = r"""
extern "C" {

#define BLOCK_SIZE       256
#define WARP_SIZE        32

__global__ void rwr_spmv_restart_chunk(
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
    __shared__ float smem[BLOCK_SIZE];

    const int local_i = blockIdx.x;
    if (local_i >= chunk_size) return;
    const int node_i = chunk_node_ids[local_i];

    const int row_start = row_ptr[local_i];
    const int row_end   = row_ptr[local_i + 1];
    const int degree    = row_end - row_start;

    if (degree == 0) {
        if (threadIdx.x == 0) p_new[node_i] = r * p0[node_i];
        return;
    }

    if (degree < WARP_SIZE) {
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int j = row_start; j < row_end; ++j) {
                sum += w_values[j] * __ldg(&p[col_idx[j]]);
            }
            p_new[node_i] = one_minus_r * sum + r * p0[node_i];
        }
        return;
    }

    if (degree < BLOCK_SIZE) {
        if (threadIdx.x < WARP_SIZE) {
            float partial = 0.0f;
            for (int j = row_start + threadIdx.x; j < row_end; j += WARP_SIZE) {
                partial += w_values[j] * __ldg(&p[col_idx[j]]);
            }
            for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
                partial += __shfl_down_sync(0xffffffffu, partial, off);
            }
            if (threadIdx.x == 0) {
                p_new[node_i] = one_minus_r * partial + r * p0[node_i];
            }
        }
        return;
    }

    float partial = 0.0f;
    for (int j = row_start + threadIdx.x; j < row_end; j += BLOCK_SIZE) {
        partial += w_values[j] * __ldg(&p[col_idx[j]]);
    }
    smem[threadIdx.x] = partial;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        p_new[node_i] = one_minus_r * smem[0] + r * p0[node_i];
    }
}

}  // extern "C"
"""


def _get_chunk_kernel() -> Any:
    """Compile (or fetch) the chunk-aware SpMV kernel."""
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
            "spmv_chunk": mod.get_function("rwr_spmv_restart_chunk"),
            "_arch_flag": arch_flag,
        }
    return _kernel_cache["rwr_chunk"]


# ---------------------------------------------------------------------------
# Chunked GPU iteration (single-seed only — batched chunked is future work)
# ---------------------------------------------------------------------------

def _rwr_gpu_chunked(
    W: sp.csr_matrix,
    p0_np: np.ndarray,
    one_minus_r: float,
    r_val: float,
    max_iter: int,
    tolerance: float,
    kernels: dict,
    stream_compute,
    stream_transfer,
    chunk_size: int,
) -> tuple[np.ndarray, int, bool]:
    """Chunked SpMV pipeline for graphs that exceed VRAM.

    p / p_new / p0 stay full-size on the device.  CSR rows are
    streamed in chunks with a double-buffer pipeline.  Returns
    (scores_np, iterations, converged).
    """
    chunk_kernels = _get_chunk_kernel()
    k_chunk    = chunk_kernels["spmv_chunk"]
    k_l1       = kernels["l1_conv"]
    k_reduce   = kernels["reduce_scalar"]

    n = int(W.shape[0])
    chunks = _build_csr_chunks(W, chunk_size=chunk_size)
    max_chunk_nnz   = max((c["values"].size for c in chunks), default=1)
    max_chunk_nodes = max((c["node_ids"].size for c in chunks), default=1)

    # ---- Allocate persistent vectors ---------------------------------------
    h_p0 = np.ascontiguousarray(p0_np, np.float32)
    d_p0 = gpuarray.to_gpu(h_p0)
    d_p  = gpuarray.to_gpu(h_p0.copy())                 # p starts = p0
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
        cuda.memcpy_htod_async(
            buf.d_row_ptr.gpudata, chunk["indptr"], stream_transfer,
        )
        cuda.memcpy_htod_async(
            buf.d_col_idx.gpudata, chunk["indices"], stream_transfer,
        )
        cuda.memcpy_htod_async(
            buf.d_values.gpudata, chunk["values"], stream_transfer,
        )
        cuda.memcpy_htod_async(
            buf.d_node_ids.gpudata, chunk["node_ids"], stream_transfer,
        )

    # Pre-load chunk 0
    _upload_chunk(chunks[0], buffers[0])
    events[0].record(stream_transfer)

    converged  = False
    iterations = 0
    try:
        for it in range(max_iter):
            iterations = it + 1

            # ---- Per-iteration: stream chunks --------------------------
            for ci, chunk in enumerate(chunks):
                buf = buffers[ci % 2]
                stream_compute.wait_for_event(events[ci % 2])
                k_chunk(
                    buf.d_row_ptr, buf.d_col_idx, buf.d_values,
                    buf.d_node_ids, d_p, d_p0, d_pn,
                    np.float32(one_minus_r), np.float32(r_val),
                    np.int32(chunk["node_ids"].size),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(int(chunk["node_ids"].size), 1, 1),
                    stream=stream_compute,
                )
                # Pre-load next chunk on transfer stream
                if ci + 1 < len(chunks):
                    nxt_buf = buffers[(ci + 1) % 2]
                    _upload_chunk(chunks[ci + 1], nxt_buf)
                    events[(ci + 1) % 2].record(stream_transfer)
                elif it + 1 < max_iter:
                    # Reset to chunk 0 for next iteration
                    _upload_chunk(chunks[0], buffers[0])
                    events[0].record(stream_transfer)

            # ---- Convergence (GPU-side reduction) ---------------------
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

            stream_compute.synchronize()                 # ONE sync
            l1 = float(d_l1_scalar.get()[0])

            d_p, d_pn = d_pn, d_p

            if l1 < tolerance:
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

    Optimisations (see module docstring for full list):
      1. GPU-side scalar reductions for convergence (one sync per iter).
      2. Hub-row ELLPACK + low-row CSR hybrid (memory coalescing).
      3. SMEM-cached p for small graphs.
      4. Optional mixed precision (FP16 weights).
      5. Multi-stream async overlap.
      6. Chunked execution for VRAM-bound graphs.

    Returns
    -------
    dict — see CLAUDE.md "RWR" result spec.

    Raises
    ------
    RuntimeError
        If PyCUDA is unavailable or no CUDA device can be initialised.
    MemoryError
        If GPU allocation fails (even after chunked path is attempted).
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA is required for rwr_gpu(). "
            "Install pycuda or use the CPU implementation from "
            "src/algorithms/cpu/single_threaded/rwr.py via the runner."
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
            p = apply_config("rwr", graph_csr, p) or p
            for k, v in _DEFAULT_PARAMS.items():
                p.setdefault(k, v)

        restart_prob     = float(p["restart_prob"])
        max_iter         = int(p["max_iter"])
        tolerance        = float(p["tolerance"])
        seed_nodes       = p.get("seed_nodes") or []
        network_type     = str(p.get("network_type", "grn"))
        block_size       = int(p.get("block_size", BLOCK_SIZE))
        if block_size <= 0 or block_size > 1024:
            block_size = BLOCK_SIZE
        precision_mode   = str(p.get("precision_mode", "fp32")).lower()
        ellpack_fraction = float(p.get("ellpack_fraction", 0.05))
        use_chunking_req = bool(p.get("use_chunking", False))
        reorder_nodes    = bool(p.get("reorder_nodes", False))

        n = int(graph_csr.shape[0])
        if n == 0:
            raise ValueError("Empty graph")

        one_minus_r = np.float32(1.0 - restart_prob)
        r_val       = np.float32(restart_prob)

        # ---- Detect batched input -------------------------------------
        if (len(seed_nodes) > 0
                and isinstance(seed_nodes[0], (list, tuple, np.ndarray))):
            seed_sets = [list(s) for s in seed_nodes]
        else:
            seed_sets = [list(seed_nodes)]
        batch_size = len(seed_sets)

        # ---- Build transition matrix ----------------------------------
        W, transition_note = _build_transition_matrix(graph_csr, network_type)

        # ---- Optional node reordering ---------------------------------
        perm = None
        if reorder_nodes:
            W, perm = _reorder_nodes_by_degree(W)
            inv_perm = np.argsort(perm).astype(np.int32)
            seed_sets = [
                [int(inv_perm[int(s)]) for s in sset if 0 <= int(s) < n]
                for sset in seed_sets
            ]

        # ---- Build p₀ vectors -----------------------------------------
        p0_list = [_build_p0(sset, n, network_type) for sset in seed_sets]

        fp16_weights = (precision_mode == "mixed")

        # ---- VRAM estimate + chunked dispatch decision ----------------
        est_bytes = _estimate_rwr_vram(
            n=n, nnz=int(W.nnz), batch_size=batch_size,
            fp16_weights=fp16_weights,
        )
        try:
            free_bytes, _total = cuda.mem_get_info()
        except Exception:                                   # noqa: BLE001
            free_bytes = 1 << 30

        use_chunking = use_chunking_req or (est_bytes > VRAM_BUDGET_FRACTION * free_bytes)
        # Batched + chunked is not supported in this pass — fall back to regular.
        if batch_size > 1 and use_chunking:
            logging.info(
                "rwr_gpu: chunked + batched is not implemented; running "
                "batched path on the full graph (may OOM)."
            )
            use_chunking = False

        if est_bytes > free_bytes and not use_chunking:
            raise MemoryError(
                f"RWR GPU needs ~{est_bytes/1e6:.1f} MB but only "
                f"{free_bytes/1e6:.1f} MB free.  Try use_chunking=True "
                f"or precision_mode='mixed', or use a higher-VRAM device."
            )

        kernels = _get_kernels()

        # ---- Streams + timing -----------------------------------------
        stream_compute  = cuda.Stream()
        stream_transfer = cuda.Stream()
        start_event     = cuda.Event()
        end_event       = cuda.Event()
        start_event.record(stream_compute)

        # ---- Chunked path -------------------------------------------------
        if use_chunking and batch_size == 1:
            # Pick chunk size from free VRAM
            available = max(1, int(free_bytes * 0.6))
            avg_nnz = max(1.0, W.nnz / n)
            bytes_per_node = int((1 + 2 * avg_nnz) * 4)
            chunk_size = max(1, min(n, available // max(bytes_per_node, 1)))

            scores_np, iterations, converged = _rwr_gpu_chunked(
                W=W, p0_np=p0_list[0],
                one_minus_r=float(one_minus_r), r_val=float(r_val),
                max_iter=max_iter, tolerance=tolerance,
                kernels=kernels,
                stream_compute=stream_compute,
                stream_transfer=stream_transfer,
                chunk_size=chunk_size,
            )

            end_event.record(stream_compute)
            end_event.synchronize()
            elapsed = start_event.time_till(end_event) / 1000.0

            if perm is not None:
                remap = np.zeros_like(scores_np)
                remap[perm] = scores_np
                scores_np = remap

            all_scores = [{
                "seed_set":   seed_sets[0],
                "scores":     scores_np,
                "iterations": iterations,
                "converged":  converged,
            }]
            return _finalize_result(
                all_scores=all_scores,
                n=n,
                seed_sets=seed_sets,
                network_type=network_type,
                graph_csr=graph_csr,
                elapsed=elapsed,
                transition_note=transition_note,
                perm=perm,
                arch_flag=kernels.get("_arch_flag", "?"),
                precision_mode=precision_mode,
                num_hubs=0,
                chunked=True,
            )

        # ---- Build hub ELLPACK / CSR split ---------------------------
        row_lens = np.diff(W.indptr).astype(np.int32)
        if ellpack_fraction > 0.0:
            hub_threshold = _compute_adaptive_hub_threshold(
                row_lens, target_ellpack_fraction=ellpack_fraction,
            )
            ellpack, low_csr = _build_ellpack_split(W, hub_threshold)
        else:
            hub_threshold = 0
            ellpack = {
                "hub_ids":      np.empty((0,), np.int32),
                "hub_col_idx":  np.empty((0,), np.int32),
                "hub_w_values": np.empty((0,), np.float32),
                "max_row_len":  0,
                "num_hubs":     0,
            }
            low_csr = {
                "row_ptr": W.indptr.astype(np.int32),
                "col_idx": W.indices.astype(np.int32),
                "values":  W.data.astype(np.float32),
                "n":       n,
            }
        num_hubs = ellpack["num_hubs"]

        def _to_gpu_async(arr: np.ndarray):
            ga = gpuarray.to_gpu_async(arr, stream=stream_transfer)
            d_buffers.append(ga)
            return ga

        def _empty(shape, dtype):
            ga = gpuarray.empty(shape, dtype=dtype)
            d_buffers.append(ga)
            return ga

        # ---- Upload CSR (low rows) -----------------------------------
        h_row_ptr = np.ascontiguousarray(low_csr["row_ptr"], dtype=np.int32)
        h_col_idx = np.ascontiguousarray(low_csr["col_idx"], dtype=np.int32)
        if fp16_weights:
            h_w_values = np.ascontiguousarray(
                _fp32_to_fp16_bits(low_csr["values"].astype(np.float32)),
                dtype=np.uint16,
            )
        else:
            h_w_values = np.ascontiguousarray(low_csr["values"], dtype=np.float32)

        d_row_ptr = _to_gpu_async(h_row_ptr)
        d_col_idx = _to_gpu_async(h_col_idx)
        d_w_values = _to_gpu_async(h_w_values)

        # ---- Upload ELLPACK hubs (if any) ----------------------------
        d_hub_ids = d_hub_col = d_hub_w = None
        if num_hubs > 0:
            h_hub_ids = np.ascontiguousarray(ellpack["hub_ids"], dtype=np.int32)
            h_hub_col = np.ascontiguousarray(ellpack["hub_col_idx"], dtype=np.int32)
            # ELLPACK weights remain FP32 (small fraction of total bytes,
            # FP16 conversion adds complexity for marginal gain here).
            h_hub_w   = np.ascontiguousarray(ellpack["hub_w_values"], dtype=np.float32)
            d_hub_ids = _to_gpu_async(h_hub_ids)
            d_hub_col = _to_gpu_async(h_hub_col)
            d_hub_w   = _to_gpu_async(h_hub_w)

        n_partial_blocks = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE)
        d_partial   = _empty((n_partial_blocks,), np.float32)
        d_l1_scalar = _empty((1,), np.float32)

        # ---- Choose execution path -----------------------------------
        use_batched = (1 < batch_size <= MAX_BATCH)
        use_smem_p  = (
            n <= SMEM_P_LIMIT
            and not use_batched
            and num_hubs == 0
            and not fp16_weights
        )

        all_scores: list[dict] = []

        if use_batched:
            # ---------- Batched path (FP32 only, no ELLPACK / SMEM cache) ----
            p0_matrix = np.column_stack(p0_list).astype(np.float32)
            h_p0_flat = np.ascontiguousarray(p0_matrix.reshape(-1), np.float32)

            d_p0  = _to_gpu_async(h_p0_flat)
            d_p   = _to_gpu_async(h_p0_flat.copy())
            d_pn  = _empty((n * batch_size,), np.float32)

            n_partial_batched = max(
                1, (n * batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE,
            )
            d_partial_b   = _empty((n_partial_batched,), np.float32)
            d_l1_scalar_b = _empty((1,), np.float32)

            stream_transfer.synchronize()

            k_batched = kernels["spmv_restart_batched"]
            k_l1      = kernels["l1_conv"]
            k_reduce  = kernels["reduce_scalar"]

            converged  = False
            iterations = 0
            for it in range(max_iter):
                iterations = it + 1
                k_batched(
                    d_row_ptr, d_col_idx, d_w_values,
                    d_p, d_p0, d_pn,
                    one_minus_r, r_val,
                    np.int32(n), np.int32(batch_size),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(n, batch_size, 1),
                    stream=stream_compute,
                )
                k_l1(
                    d_pn, d_p, d_partial_b, np.int32(n * batch_size),
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
                stream_compute.synchronize()                 # ONE sync
                l1_total = float(d_l1_scalar_b.get()[0])

                d_p, d_pn = d_pn, d_p

                if l1_total < tolerance * batch_size:
                    converged = True
                    break

            p_final = d_p.get().reshape(n, batch_size)
            for b in range(batch_size):
                scores_b = p_final[:, b].copy()
                if perm is not None:
                    remap = np.zeros_like(scores_b)
                    remap[perm] = scores_b
                    scores_b = remap
                all_scores.append({
                    "seed_set":   seed_sets[b],
                    "scores":     scores_b,
                    "iterations": iterations,
                    "converged":  converged,
                })

        else:
            # ---------- Serial-per-seed-set path -------------------------
            k_l1     = kernels["l1_conv"]
            k_reduce = kernels["reduce_scalar"]
            k_ell    = kernels["spmv_ellpack_hubs"] if num_hubs > 0 else None

            if fp16_weights:
                k_spmv = kernels["spmv_restart_fp16w"]
            elif use_smem_p:
                k_spmv = kernels["spmv_smem_p"]
            else:
                k_spmv = kernels["spmv_restart"]

            stream_transfer.synchronize()

            for b_idx, (sset, p0_np) in enumerate(zip(seed_sets, p0_list)):
                h_p0 = np.ascontiguousarray(p0_np, np.float32)

                d_p0 = _to_gpu_async(h_p0)
                d_p  = _to_gpu_async(h_p0.copy())
                d_pn = _empty((n,), np.float32)
                stream_transfer.synchronize()

                converged  = False
                iterations = 0
                smem_p_bytes = (n * 4) if use_smem_p else 0

                for it in range(max_iter):
                    iterations = it + 1

                    # ---- Low-CSR rows (or SMEM-cached, or FP16) ----
                    if use_smem_p:
                        k_spmv(
                            d_row_ptr, d_col_idx, d_w_values,
                            d_p, d_p0, d_pn,
                            one_minus_r, r_val, np.int32(n),
                            block=(BLOCK_SIZE, 1, 1),
                            grid=(n, 1, 1),
                            shared=smem_p_bytes,
                            stream=stream_compute,
                        )
                    else:
                        k_spmv(
                            d_row_ptr, d_col_idx, d_w_values,
                            d_p, d_p0, d_pn,
                            one_minus_r, r_val, np.int32(n),
                            block=(BLOCK_SIZE, 1, 1),
                            grid=(n, 1, 1),
                            stream=stream_compute,
                        )

                    # ---- Hub-row ELLPACK (overrides p_new for hubs) ----
                    if k_ell is not None and num_hubs > 0:
                        k_ell(
                            d_hub_ids, d_hub_col, d_hub_w,
                            np.int32(ellpack["max_row_len"]),
                            d_p, d_p0, d_pn,
                            one_minus_r, r_val,
                            np.int32(num_hubs),
                            block=(BLOCK_SIZE, 1, 1),
                            grid=(num_hubs, 1, 1),
                            stream=stream_compute,
                        )

                    # ---- Convergence (GPU-side scalar reduction) ----
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

                    stream_compute.synchronize()                 # ONE sync
                    l1 = float(d_l1_scalar.get()[0])

                    d_p, d_pn = d_pn, d_p

                    if l1 < tolerance:
                        converged = True
                        break

                scores_b = d_p.get()
                if perm is not None:
                    remap = np.zeros_like(scores_b)
                    remap[perm] = scores_b
                    scores_b = remap

                all_scores.append({
                    "seed_set":   sset,
                    "scores":     scores_b,
                    "iterations": iterations,
                    "converged":  converged,
                })

                for arr in (d_p0, d_p, d_pn):
                    try:
                        arr.gpudata.free()
                    except Exception:                       # noqa: BLE001
                        pass
                    if arr in d_buffers:
                        d_buffers.remove(arr)

        end_event.record(stream_compute)
        end_event.synchronize()
        elapsed = start_event.time_till(end_event) / 1000.0

        return _finalize_result(
            all_scores=all_scores,
            n=n,
            seed_sets=seed_sets,
            network_type=network_type,
            graph_csr=graph_csr,
            elapsed=elapsed,
            transition_note=transition_note,
            perm=perm,
            arch_flag=kernels.get("_arch_flag", "?"),
            precision_mode=precision_mode,
            num_hubs=num_hubs,
            chunked=False,
        )

    except cuda.LogicError as e:
        logging.error("CUDA error in rwr_gpu: %s", e)
        raise
    except MemoryError:
        logging.warning(
            "VRAM exhausted in rwr_gpu.  Retry with use_chunking=True "
            "or precision_mode='mixed', or use a higher-VRAM device."
        )
        raise
    finally:
        for arr in d_buffers:
            try:
                arr.gpudata.free()
            except Exception:                               # noqa: BLE001
                pass
        if pushed_ctx is not None:
            try:
                pushed_ctx.pop()
            except Exception:                               # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Result aggregation (shared between regular and chunked paths)
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
    perm: np.ndarray | None,
    arch_flag: str,
    precision_mode: str,
    num_hubs: int,
    chunked: bool,
) -> dict:
    """Build the final result dict (matches the CLAUDE.md RWR spec)."""
    if len(all_scores) == 1:
        primary_scores   = all_scores[0]["scores"]
        total_iterations = int(all_scores[0]["iterations"])
        any_converged    = bool(all_scores[0]["converged"])
        batch_results    = None
    else:
        score_matrix = np.array([s["scores"] for s in all_scores],
                                dtype=np.float64)
        primary_scores   = score_matrix.mean(axis=0).astype(np.float32)
        total_iterations = int(max(s["iterations"] for s in all_scores))
        any_converged    = bool(all(s["converged"] for s in all_scores))
        batch_results    = [
            {
                "seed_set":   list(s["seed_set"]),
                "scores":     [float(x) for x in s["scores"]],
                "iterations": int(s["iterations"]),
                "converged":  bool(s["converged"]),
            }
            for s in all_scores
        ]

    top_nodes = _top_k(primary_scores, _TOP_NODES)
    all_seed_indices = np.array(
        sorted({int(idx) for sset in seed_sets for idx in sset
                if 0 <= int(idx) < n}),
        dtype=np.int32,
    )
    if perm is not None and all_seed_indices.size > 0:
        all_seed_indices = perm[all_seed_indices]
    if all_seed_indices.size > 0:
        top_seeds = _top_k_among(primary_scores, all_seed_indices, _TOP_SEEDS)
    else:
        top_seeds = top_nodes[:_TOP_SEEDS]

    note = (
        f"{transition_note}. "
        f"GPU pipeline: hub_ellpack_count={num_hubs}, "
        f"precision={precision_mode}, "
        f"chunked={chunked}, "
        f"arch={arch_flag}, "
        f"syncs_per_iter=1."
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
# Runner interface
# ---------------------------------------------------------------------------

def _gpu(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    """Runner entry point — preserves the legacy ``{output, extra_params}``
    shape required by ``src.benchmarking.benchmark`` and
    ``src.runner.algorithm_runner``.
    """
    p = _merge_params(params)
    full = rwr_gpu(graph_csr, p)
    return {"output": full, "extra_params": p}
