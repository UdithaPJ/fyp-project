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
1. Fused SpMV + restart (rwr_spmv_restart / rwr_spmv_restart_fp16w):
   One block per node, three-tier degree-aware scheduling.  __ldg routes
   p reads through the read-only texture cache.

2. GPU-side convergence reduction:
   l1_convergence_rwr writes per-block L1 partials; reduce_to_scalar_f32
   collapses them to a single scalar on the GPU.  Only that one float
   crosses PCIe per convergence check.

3. Convergence checked every N iterations (default 5):
   SpMV kernels queue continuously; sync + scalar read happen only every N
   iterations.  Eliminates ~80 % of CPU-GPU synchronisation stalls.

4. Transition matrix W cached CPU-side:
   The scipy symmetrisation + column-normalisation is recomputed only
   when the graph object or network_type changes.

5. GPU CSR arrays cached device-side:
   row_ptr / col_idx / w_values are re-uploaded only when the graph or
   precision_mode changes.  Subsequent runs on the same graph skip H2D.

6. Optional mixed precision (FP16 weights, FP32 accumulation):
   rwr_spmv_restart_fp16w halves the w_values bandwidth.

7. Batched execution (rwr_spmv_restart_batched) for B ≤ MAX_BATCH = 4.

8. Chunked execution (_rwr_gpu_chunked) for graphs exceeding VRAM.

Removed optimisations (introduced more overhead than benefit):
  - Hub-row ELLPACK: preprocessing (percentile + split) + two kernel
    launches per iter outweighed coalescing gain on biological sparsity.
  - SMEM-cached p vector: n * 4 B of SMEM per block cut SM occupancy;
    L1/L2 + __ldg already handles this on Turing/Ampere.
  - Node reordering: O(n log n) argsort + two CSR constructions +
    seed/score remapping cost exceeded warp-divergence savings.

Does NOT silently fall back to CPU.  Target: adaptive (sm_75 fallback).

Multi-seed-set support
------------------------
seed_nodes accepts a flat list (single RWR) or list-of-lists (one RWR per
inner list, scores averaged to form a consensus influence vector).
Batched kernel handles 1 < B <= MAX_BATCH = 4; larger batches loop the
single-seed kernel.

Parameter guide
---------------
restart_prob       (float, default 0.3)   Teleport probability per step.
max_iter           (int,   default 100)   Hard iteration cap.
tolerance          (float, default 1e-6)  L1-norm early-stop threshold.
seed_nodes         (list)                 Seeds for p₀.
network_type       (str,   default "grn") One of "grn", "ppi", "mirna".
block_size         (int,   default 256)   CUDA block dimension.
precision_mode     (str,   default "fp32") "fp32" or "mixed" (FP16 W).
use_chunking       (bool,  default False) Force chunked path on.
conv_check_interval(int,   default 5)     Check convergence every N iters.
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
    "precision_mode":       "fp32",     # "fp32" or "mixed"
    "use_chunking":         False,
    "use_zero_copy":        False,
    "conv_check_interval":  5,          # check convergence every N iters
}

BLOCK_SIZE: int            = 256
WARP_SIZE: int             = 32
MAX_BATCH: int             = 4
_TOP_NODES: int            = 20
_TOP_SEEDS: int            = 10
VRAM_BUDGET_FRACTION: float = 0.80

# ---------------------------------------------------------------------------
# CPU-side transition matrix cache
# ---------------------------------------------------------------------------

_W_CACHE: dict      = {}   # key -> (W: csr_matrix, note: str, h_row_ptr, h_col_idx)
_W_CACHE_REFS: dict = {}   # key -> weakref to graph_csr
_W_CACHE_MAX: int   = 8    # max cached graphs

# ---------------------------------------------------------------------------
# GPU CSR cache (single entry — most recently used graph)
# ---------------------------------------------------------------------------

class _GPUCSRCache:
    """Cache the most recent GPU-side CSR arrays to avoid redundant H2D uploads.

    Thread safety: not thread-safe; intended for single-threaded benchmarking.
    All free() calls must occur while the CUDA primary context is active (i.e.
    from within rwr_gpu after the context push).
    """

    __slots__ = ("key", "d_row_ptr", "d_col_idx", "d_w_values")

    def __init__(self) -> None:
        self.key: Any       = None
        self.d_row_ptr      = None
        self.d_col_idx      = None
        self.d_w_values     = None

    def matches(self, key: Any) -> bool:
        return self.key == key and self.d_row_ptr is not None

    def store(self, key: Any, d_row_ptr, d_col_idx, d_w_values) -> None:
        self.key        = key
        self.d_row_ptr  = d_row_ptr
        self.d_col_idx  = d_col_idx
        self.d_w_values = d_w_values

    def invalidate(self) -> None:
        """Free cached GPU arrays.  Call only while the CUDA context is active."""
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
# CUDA kernel source
# ---------------------------------------------------------------------------
#
# Kernels:
#   reduce_to_scalar_f32       — GPU-side single-block partial-sum reduction.
#   rwr_spmv_restart           — FP32 fused SpMV + restart (three-tier).
#   rwr_spmv_restart_fp16w     — same, FP16 weights, FP32 accumulation.
#   l1_convergence_rwr         — |p_new - p| block-partial.
#   rwr_spmv_restart_batched   — multi-seed batched variant (B <= 4).
#
# All kernels live in one SourceModule (compiled once, cached in _kernel_cache).

KERNEL_SOURCE = r"""
#include <cuda_fp16.h>

extern "C" {

#define BLOCK_SIZE       256
#define WARP_SIZE        32
#define WARPS_PER_BLOCK  (BLOCK_SIZE / WARP_SIZE)


// =========================================================================
// KERNEL: reduce_to_scalar_f32
//
// Strided load + shared-memory tree reduction.  Launched with grid=(1,1,1).
// Writes a SINGLE float to scalar_output[0].  Replaces the per-iteration
// .get() of the partial-sum array.
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
// KERNEL: rwr_spmv_restart  (FP32, three-tier degree-aware)
//
// Fused SpMV + restart for a single seed set:
//   p_new[i] = (1-r) * sum_j W[i,j]*p[j]  +  r * p0[i]
//
// One block per node i.  Three-tier scheduling:
//   LOW  (deg < 32)    : thread 0 only, serial scan
//   MED  (32 <= d<256) : first warp, stride-32 + shfl_down_sync
//   HIGH (deg >= 256)  : full block, stride-256 + SMEM tree
//
// __ldg routes p neighbour reads through the read-only texture cache.
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
// KERNEL: rwr_spmv_restart_fp16w  (mixed precision)
//
// Same three-tier dispatch as rwr_spmv_restart, but reads w_values as
// FP16 (unsigned short bit pattern) and converts to float at use time.
// Halves the w_values bandwidth.  p / p_new / p0 stay FP32.
// =========================================================================
__global__ void rwr_spmv_restart_fp16w(
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
// KERNEL: l1_convergence_rwr
//
// Sum of |p_new[i] - p[i]| per block via warp-shuffle intra-warp reduction
// then a final warp-shuffle across the WARPS_PER_BLOCK partial sums.
// Partials reduced by reduce_to_scalar_f32.
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
// KERNEL: rwr_spmv_restart_batched  (multi-seed, B <= MAX_BATCH)
//
// gridDim = (n, B, 1); one block per (node_i, seed_b) pair.
// p / p₀ / p_new stored as [n × B] row-major flat arrays.
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
        _kernel_cache["rwr"] = {
            "reduce_scalar":        mod.get_function("reduce_to_scalar_f32"),
            "spmv_restart":         mod.get_function("rwr_spmv_restart"),
            "spmv_restart_fp16w":   mod.get_function("rwr_spmv_restart_fp16w"),
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

    W must be column-stochastic (every column sums to 1) so that the
    random walk preserves probability mass.
    """
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
    """Return the cached (W, note) for this graph, building it if needed.

    Key: (object id, shape, nnz, network_type).  A weakref guards against
    Python reusing the same id for a different graph object.
    """
    key = (id(graph_csr), graph_csr.shape, int(graph_csr.nnz), network_type)

    if key in _W_CACHE:
        ref = _W_CACHE_REFS.get(key)
        if ref is None or ref() is graph_csr:
            return _W_CACHE[key]
        # id was reused for a different object — invalidate
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
    """Build the restart vector p₀ (length n, sums to 1.0).

    Empty / invalid seeds → uniform 1/n.
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


def _fp32_to_fp16_bits(values: np.ndarray) -> np.ndarray:
    """Convert an FP32 array to FP16 bit pattern as uint16."""
    return values.astype(np.float16).view(np.uint16).astype(np.uint16)


def _estimate_rwr_vram(n: int, nnz: int, batch_size: int = 1,
                       fp16_weights: bool = False) -> int:
    """Conservative VRAM estimate (bytes) for RWR working set."""
    w_val_bytes = 2 if fp16_weights else 4
    csr_b     = (n + 1) * 4 + nnz * 4 + nnz * w_val_bytes
    pr_b      = n * 4 * (2 + batch_size)       # p, p_new, B × p₀
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
# Chunk-aware SpMV kernel (separate SourceModule)
# ---------------------------------------------------------------------------

_CHUNK_KERNEL_SOURCE = r"""
extern "C" {

#define BLOCK_SIZE  256
#define WARP_SIZE   32

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
    """Chunked SpMV for graphs that exceed VRAM.

    p / p_new / p0 stay full-size on the device.  CSR rows are streamed
    in chunks with a double-buffer pipeline.  Convergence is checked every
    conv_check_interval iterations.  Returns (scores_np, iterations, converged).
    """
    chunk_kernels = _get_chunk_kernel()
    k_chunk   = chunk_kernels["spmv_chunk"]
    k_l1      = kernels["l1_conv"]
    k_reduce  = kernels["reduce_scalar"]

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
                k_chunk(
                    buf.d_row_ptr, buf.d_col_idx, buf.d_values,
                    buf.d_node_ids, d_p, d_p0, d_pn,
                    np.float32(one_minus_r), np.float32(r_val),
                    np.int32(chunk["node_ids"].size),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(int(chunk["node_ids"].size), 1, 1),
                    stream=stream_compute,
                )
                if ci + 1 < len(chunks):
                    nxt_buf = buffers[(ci + 1) % 2]
                    _upload_chunk(chunks[ci + 1], nxt_buf)
                    events[(ci + 1) % 2].record(stream_transfer)
                elif it + 1 < max_iter:
                    _upload_chunk(chunks[0], buffers[0])
                    events[0].record(stream_transfer)

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
                l1 = float(d_l1_scalar.get()[0])
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
      1. GPU-side convergence reduction — one float per check crosses PCIe.
      2. Convergence checked every N iterations (default 5) — 5× fewer syncs.
      3. Transition matrix W cached CPU-side — no recompute on repeated runs.
      4. GPU CSR arrays cached device-side — no re-upload on repeated runs.
      5. Optional FP16 weights (precision_mode="mixed").
      6. Batched execution (B <= 4 seed sets).
      7. Chunked execution for VRAM-bound graphs.

    Raises
    ------
    RuntimeError   PyCUDA unavailable or CUDA device not found.
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
    # Cached GPU CSR arrays (d_row_ptr, d_col_idx, d_w_values) are NOT
    # added to d_buffers so they survive across calls.
    d_buffers: list = []

    try:
        # ---- Parameter merging -------------------------------------------
        p = _merge_params(params)
        if _GPU_CONFIG_AVAILABLE:
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
        conv_check_interval = max(1, int(p.get("conv_check_interval", 5)))

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

        use_chunking = use_chunking_req or (est_bytes > VRAM_BUDGET_FRACTION * free_bytes)
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
            )

        # ---- Prepare GPU CSR arrays (use cache if available) -------------
        gpu_cache_key = (id(graph_csr), graph_csr.shape,
                         int(graph_csr.nnz), network_type, precision_mode)

        if _GPU_CSR_CACHE.matches(gpu_cache_key):
            d_row_ptr  = _GPU_CSR_CACHE.d_row_ptr
            d_col_idx  = _GPU_CSR_CACHE.d_col_idx
            d_w_values = _GPU_CSR_CACHE.d_w_values
        else:
            # Free previous cached arrays (context is active — safe)
            _GPU_CSR_CACHE.invalidate()

            h_row_ptr = np.ascontiguousarray(W.indptr,  dtype=np.int32)
            h_col_idx = np.ascontiguousarray(W.indices, dtype=np.int32)
            if fp16_weights:
                h_w_values = np.ascontiguousarray(
                    _fp32_to_fp16_bits(W.data.astype(np.float32)), dtype=np.uint16,
                )
            else:
                h_w_values = np.ascontiguousarray(W.data, dtype=np.float32)

            d_row_ptr  = gpuarray.to_gpu_async(h_row_ptr,  stream=stream_transfer)
            d_col_idx  = gpuarray.to_gpu_async(h_col_idx,  stream=stream_transfer)
            d_w_values = gpuarray.to_gpu_async(h_w_values, stream=stream_transfer)
            _GPU_CSR_CACHE.store(gpu_cache_key, d_row_ptr, d_col_idx, d_w_values)

        # ---- Helper for temporary GPU arrays (freed in finally) ----------
        def _alloc_temp_from(arr: np.ndarray):
            ga = gpuarray.to_gpu_async(arr, stream=stream_transfer)
            d_buffers.append(ga)
            return ga

        def _alloc_temp_empty(shape, dtype):
            ga = gpuarray.empty(shape, dtype)
            d_buffers.append(ga)
            return ga

        n_partial_blocks = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE)
        d_partial   = _alloc_temp_empty((n_partial_blocks,), np.float32)
        d_l1_scalar = _alloc_temp_empty((1,),               np.float32)

        # ---- Select SpMV kernel ------------------------------------------
        k_spmv   = kernels["spmv_restart_fp16w"] if fp16_weights else kernels["spmv_restart"]
        k_l1     = kernels["l1_conv"]
        k_reduce = kernels["reduce_scalar"]

        use_batched = (1 < batch_size <= MAX_BATCH)
        all_scores: list[dict] = []

        # ---- Batched path (B <= MAX_BATCH seed sets) ---------------------
        if use_batched:
            p0_matrix = np.column_stack(p0_list).astype(np.float32)
            h_p0_flat = np.ascontiguousarray(p0_matrix.reshape(-1), np.float32)

            d_p0 = _alloc_temp_from(h_p0_flat)
            d_p  = _alloc_temp_from(h_p0_flat.copy())
            d_pn = _alloc_temp_empty((n * batch_size,), np.float32)

            n_partial_batched = max(1, (n * batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE)
            d_partial_b   = _alloc_temp_empty((n_partial_batched,), np.float32)
            d_l1_scalar_b = _alloc_temp_empty((1,),                 np.float32)

            k_batched = kernels["spmv_restart_batched"]

            stream_transfer.synchronize()
            start_event.record(stream_compute)

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
                do_check = (iterations % conv_check_interval == 0) or (iterations == max_iter)
                if do_check:
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

        # ---- Serial-per-seed-set path ------------------------------------
        else:
            stream_transfer.synchronize()
            start_event.record(stream_compute)

            for b_idx, (sset, p0_np) in enumerate(zip(seed_sets, p0_list)):
                h_p0 = np.ascontiguousarray(p0_np, np.float32)

                d_p0 = _alloc_temp_from(h_p0)
                d_p  = _alloc_temp_from(h_p0.copy())
                d_pn = _alloc_temp_empty((n,), np.float32)
                stream_transfer.synchronize()

                converged  = False
                iterations = 0

                for it in range(max_iter):
                    iterations = it + 1

                    k_spmv(
                        d_row_ptr, d_col_idx, d_w_values,
                        d_p, d_p0, d_pn,
                        one_minus_r, r_val, np.int32(n),
                        block=(BLOCK_SIZE, 1, 1),
                        grid=(n, 1, 1),
                        stream=stream_compute,
                    )

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

                scores_b = d_p.get()
                all_scores.append({
                    "seed_set":   sset,
                    "scores":     scores_b,
                    "iterations": iterations,
                    "converged":  converged,
                })

                # Free per-seed-set temporaries
                for arr in (d_p0, d_p, d_pn):
                    try:
                        arr.gpudata.free()
                    except Exception:                   # noqa: BLE001
                        pass
                    for idx in range(len(d_buffers) - 1, -1, -1):
                        if d_buffers[idx] is arr:
                            d_buffers.pop(idx)

        end_event.record(stream_compute)
        end_event.synchronize()
        elapsed = start_event.time_till(end_event) / 1000.0

        return _finalize_result(
            all_scores=all_scores,
            n=n, seed_sets=seed_sets, network_type=network_type,
            graph_csr=graph_csr, elapsed=elapsed,
            transition_note=transition_note,
            arch_flag=kernels.get("_arch_flag", "?"),
            precision_mode=precision_mode, chunked=False,
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
    if all_seed_indices.size > 0:
        top_seeds = _top_k_among(primary_scores, all_seed_indices, _TOP_SEEDS)
    else:
        top_seeds = top_nodes[:_TOP_SEEDS]

    note = (
        f"{transition_note}. "
        f"GPU pipeline: precision={precision_mode}, "
        f"chunked={chunked}, "
        f"arch={arch_flag}, "
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
    """Runner entry point — returns ``{output, extra_params}`` envelope."""
    p    = _merge_params(params)
    full = rwr_gpu(graph_csr, p)
    return {"output": full, "extra_params": p}
