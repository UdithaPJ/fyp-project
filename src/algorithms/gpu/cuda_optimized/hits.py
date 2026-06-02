"""
algorithms/hits.py — HITS (Hyperlink-Induced Topic Search) for Biological Networks
====================================================================================

Biological Context — Why HITS suits directed GRNs
--------------------------------------------------
GRNs are *directed* networks: edges encode TF → target-gene regulatory events.
HITS exploits this directionality in a way symmetric algorithms cannot:

  Hub score   — high for nodes (TFs) that regulate many high-authority genes.
                A TF with a high hub score is a *master regulator*: it sits at
                the top of broad regulatory hierarchies (e.g. TP53 in the
                DNA-damage response network).

  Authority score — high for nodes (genes) that receive regulatory input from
                    many high-hub TFs.  A gene with a high authority score is a
                    *convergence point* of regulatory signals — often a key
                    pathway effector (e.g. CDKN1A, targeted by multiple stress-
                    response TFs).

Network-type adaptation
-----------------------
GRN / miRNA  : directed adjacency used AS-IS; returns top_hubs, top_authorities
               and hub_authority_overlap (feedback regulators).
PPI          : adjacency is symmetrised (A ← A + Aᵀ) and binarised before HITS,
               so hub == authority == connectivity centrality; returns top_nodes.

Algorithm
---------
1. Initialise h[i] = a[i] = 1/n for all nodes.
2. Authority update : a ← Aᵀ h    (L2-normalise)
3. Hub update       : h ← A  a    (L2-normalise)
4. Convergence      : ‖Δh,Δa‖ < tolerance

Per-iteration pipeline (optimised — 10 kernel launches, 1 CPU-GPU sync):
  Authority update  (a_new = Aᵀ h):
    1. spmv_edge_parallel_low_degree  — packed-warp SpMV + partial norm (deg < 32)
    2. spmv_with_norm_sq              — fused SpMV + partial norm (deg >= 32)
    3. partial_reduce_to_scalar       — GPU sqrt(sum(partial_norm)) → scalar
    4. normalize_inplace              — divide a_new by GPU scalar (no CPU round-trip)
  Hub update  (h_new = A a_new):
    5. spmv_edge_parallel_low_degree  — same, for A
    6. spmv_with_norm_sq              — same, for A
    7. partial_reduce_to_scalar
    8. normalize_inplace
  Convergence:
    9. compute_convergence_partial    — FP64 partial sums of ||delta_h||^2 + ||delta_a||^2
   10. stream_compute.synchronize()  — ONLY CPU sync per iteration
       delta = sqrt(sum(d_partial_conv.get()))

Precision:
  FP32 — SpMV arithmetic, hub/authority scores, L2 norms.
  FP64 — convergence delta only (avoids false termination near tol=1e-6).
  Rationale: hub/authority scores are relative rankings; FP32 resolution
  is sufficient.  Convergence detection requires FP64.

Node reordering (default=True):
  Nodes are permuted by descending out-degree before GPU work.  High-degree
  nodes (hubs) land at low CSR row indices; similar-degree rows are adjacent,
  clustering their x[col_idx[j]] neighbour accesses in the L1/L2 caches.
  Original-index order is restored before building the result dict.

Parameter Guide
---------------
max_iter      (int,   default 100)    Hard iteration cap.
tolerance     (float, default 1e-6)   L2-norm convergence threshold.
network_type  (str,   default "grn")  One of "grn", "ppi", "mirna".
reorder_nodes (bool,  default True)   Sort nodes by descending degree for
                                      cache-locality improvement.
"""

# ── GPU / CUDA-optimised implementation (PyCUDA custom kernels) ──────────
# Source:    biological_network_framework/algorithms/hits.py
# Requires:  pycuda (with a working NVCC toolchain)
# Used by:   webapp routes, GPU benchmarking (src.benchmarking.benchmark)
# Modes:     _gpu only — this module is GPU-exclusive.
# CPU-only counterparts (benchmarking only — never import in webapp):
#   src.algorithms.cpu.single_threaded.hits
#   src.algorithms.cpu.multi_threaded.hits
#
# Active kernels (compiled once, cached in _kernel_cache["hits"]):
#   spmv_edge_parallel_low_degree  — packed-warp SpMV + partial norm (deg < WARP_SIZE)
#   spmv_with_norm_sq              — fused SpMV + per-node partial norm (deg >= WARP_SIZE)
#   partial_reduce_to_scalar       — single-block FP32 reduce + sqrt -> d_scalar[0]
#   normalize_inplace              — in-place divide by GPU scalar (no CPU round-trip)
#   compute_convergence_partial    — FP64 partial ||delta_h||^2 + ||delta_a||^2
#
# Deprecated (in KERNEL_SOURCE for reference, no longer called in hits_gpu()):
#   spmv_degree_aware, compute_partial_norm_sq, normalize_vector,
#   compute_convergence_delta
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
    logging.warning("PyCUDA not available — hits_gpu() will raise.")

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
    "max_iter":      100,
    "tolerance":     1e-6,
    "network_type":  "grn",
    "reorder_nodes": True,
}

BLOCK_SIZE: int      = 256
WARP_SIZE: int       = 32
NODES_PER_BLOCK: int = BLOCK_SIZE // WARP_SIZE   # 8 low-degree nodes per block
VRAM_SAFETY: float   = 0.80      # use at most 80 % of free VRAM before warning
_TOP_K: int          = 15


# ---------------------------------------------------------------------------
# CUDA kernel source
# ---------------------------------------------------------------------------

KERNEL_SOURCE = r"""
extern "C" {

#define BLOCK_SIZE      256
#define WARP_SIZE       32
#define NODES_PER_BLOCK 8       /* BLOCK_SIZE / WARP_SIZE */

// =========================================================================
// [DEPRECATED — kept for reference, not called by hits_gpu()]
// KERNEL: spmv_degree_aware — three-tier degree-aware SpMV (y = M * x)
// Replaced by spmv_with_norm_sq (fused) + spmv_edge_parallel_low_degree.
// =========================================================================
__global__ void spmv_degree_aware(
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float*       __restrict__ y,
    const int*   __restrict__ node_degrees,
    const int                  n)
{
    __shared__ float smem[BLOCK_SIZE];
    const int node_id = blockIdx.x;
    if (node_id >= n) return;
    const int degree    = node_degrees[node_id];
    const int row_start = row_ptr[node_id];
    const int row_end   = row_ptr[node_id + 1];
    if (degree == 0) {
        if (threadIdx.x == 0) y[node_id] = 0.0f;
        return;
    }
    if (degree < WARP_SIZE) {
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int j = row_start; j < row_end; ++j)
                sum += values[j] * x[col_idx[j]];
            y[node_id] = sum;
        }
        return;
    }
    if (degree < BLOCK_SIZE) {
        if (threadIdx.x < WARP_SIZE) {
            float partial = 0.0f;
            for (int j = row_start + threadIdx.x; j < row_end; j += WARP_SIZE)
                partial += values[j] * x[col_idx[j]];
            for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
                partial += __shfl_down_sync(0xffffffffu, partial, off);
            if (threadIdx.x == 0) y[node_id] = partial;
        }
        return;
    }
    float partial = 0.0f;
    for (int j = row_start + threadIdx.x; j < row_end; j += BLOCK_SIZE)
        partial += values[j] * x[col_idx[j]];
    smem[threadIdx.x] = partial;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[node_id] = smem[0];
}


// =========================================================================
// [DEPRECATED — kept for reference, not called by hits_gpu()]
// KERNEL: compute_partial_norm_sq — block-level partial L2-norm-squared
// Replaced by the fused spmv_with_norm_sq + partial_reduce_to_scalar pair.
// =========================================================================
__global__ void compute_partial_norm_sq(
    const float* __restrict__ vec,
    float*       __restrict__ partial_sums,
    const int                  n)
{
    __shared__ float smem[BLOCK_SIZE];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const float v = (tid < n) ? vec[tid] : 0.0f;
    smem[threadIdx.x] = v * v;
    __syncthreads();
    for (int s = (blockDim.x >> 1); s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial_sums[blockIdx.x] = smem[0];
}


// =========================================================================
// [DEPRECATED — kept for reference, not called by hits_gpu()]
// KERNEL: normalize_vector — in-place divide by a scalar L2 norm
// Replaced by normalize_inplace which reads norm from a GPU pointer.
// =========================================================================
__global__ void normalize_vector(
    float*      __restrict__ vec,
    const float              norm,
    const int                n)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    if (norm > 1e-10f) vec[tid] /= norm;
}


// =========================================================================
// [DEPRECATED — kept for reference, not called by hits_gpu()]
// KERNEL: compute_convergence_delta — FP32 convergence partial sums
// Replaced by compute_convergence_partial which uses FP64 accumulation.
// =========================================================================
__global__ void compute_convergence_delta(
    const float* __restrict__ h_new,
    const float* __restrict__ h_old,
    const float* __restrict__ a_new,
    const float* __restrict__ a_old,
    float*       __restrict__ partial_sums,
    const int                  n)
{
    __shared__ float smem[BLOCK_SIZE];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (tid < n) {
        const float dh = h_new[tid] - h_old[tid];
        const float da = a_new[tid] - a_old[tid];
        val = dh * dh + da * da;
    }
    smem[threadIdx.x] = val;
    __syncthreads();
    for (int s = (blockDim.x >> 1); s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial_sums[blockIdx.x] = smem[0];
}


// =========================================================================
// KERNEL 1: spmv_edge_parallel_low_degree — Improvement 3
//
// Packs NODES_PER_BLOCK (=8) low-degree nodes into one CTA, one warp each.
// All 256 threads stay busy => 8x SM occupancy vs. one-block-per-node
// where 255 threads sit idle for degree-1 nodes.
//
// Grid layout : gridDim.x = ceil(num_low_nodes / NODES_PER_BLOCK)
// Block layout: blockDim.x = BLOCK_SIZE  (= 8 warps)
//   warp_id   = threadIdx.x / WARP_SIZE   -- which node slot in this block
//   lane_id   = threadIdx.x % WARP_SIZE   -- lane within that warp
//
// Each warp computes its node's dot product via warp-level reduction,
// then lane 0 writes y[u] and partial_norm[u] = y[u]^2.
// partial_norm is sized n; partial_reduce_to_scalar reduces all n elements.
// =========================================================================
__global__ void spmv_edge_parallel_low_degree(
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float*       __restrict__ y,
    float*       __restrict__ partial_norm,
    const int*   __restrict__ node_ids,
    const int                  num_low_nodes)
{
    const int warp_id    = threadIdx.x / WARP_SIZE;
    const int lane_id    = threadIdx.x % WARP_SIZE;
    const int global_idx = blockIdx.x * NODES_PER_BLOCK + warp_id;

    if (global_idx >= num_low_nodes) return;

    const int u         = node_ids[global_idx];
    const int row_start = row_ptr[u];
    const int row_end   = row_ptr[u + 1];

    // Each lane accumulates its strided slice of the row.
    float partial = 0.0f;
    for (int j = row_start + lane_id; j < row_end; j += WARP_SIZE)
        partial += values[j] * x[col_idx[j]];

    // Warp-level reduction via shuffle.
    for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
        partial += __shfl_down_sync(0xffffffffu, partial, off);

    // Lane 0 writes y and the per-node squared value for the norm reduction.
    if (lane_id == 0) {
        y[u]            = partial;
        partial_norm[u] = partial * partial;
    }
}


// =========================================================================
// KERNEL 2: spmv_with_norm_sq — Improvement 1
//
// Fused kernel: SpMV (y = M * x) + per-node partial norm accumulation.
// Handles MED (32 <= deg < 256) and HIGH (deg >= 256) tier nodes.
// Low-degree nodes (deg < 32) are handled by spmv_edge_parallel_low_degree.
//
// Grid layout: gridDim.x = num_high_nodes  (one block per node in node_ids)
// After the dot product, thread 0 writes:
//   partial_norm[node_id] = y[node_id]^2
// partial_norm is sized n; partial_reduce_to_scalar reduces all n elements.
// =========================================================================
__global__ void spmv_with_norm_sq(
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float*       __restrict__ y,
    float*       __restrict__ partial_norm,
    const int*   __restrict__ node_ids,
    const int                  num_high_nodes)
{
    __shared__ float smem[BLOCK_SIZE];

    if (blockIdx.x >= num_high_nodes) return;
    const int node_id   = node_ids[blockIdx.x];
    const int row_start = row_ptr[node_id];
    const int row_end   = row_ptr[node_id + 1];
    const int degree    = row_end - row_start;

    float result = 0.0f;

    if (degree == 0) {
        // Isolated node — nothing to accumulate.
        if (threadIdx.x == 0) {
            y[node_id]            = 0.0f;
            partial_norm[node_id] = 0.0f;
        }
        return;
    }

    // ---- MED tier: 32 <= degree < 256 ----------------------------------
    if (degree < BLOCK_SIZE) {
        if (threadIdx.x < WARP_SIZE) {
            float partial = 0.0f;
            for (int j = row_start + threadIdx.x; j < row_end; j += WARP_SIZE)
                partial += values[j] * x[col_idx[j]];
            for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
                partial += __shfl_down_sync(0xffffffffu, partial, off);
            if (threadIdx.x == 0) result = partial;
        }
        if (threadIdx.x == 0) {
            y[node_id]            = result;
            partial_norm[node_id] = result * result;
        }
        return;
    }

    // ---- HIGH tier: degree >= 256 --------------------------------------
    float partial = 0.0f;
    for (int j = row_start + threadIdx.x; j < row_end; j += BLOCK_SIZE)
        partial += values[j] * x[col_idx[j]];
    smem[threadIdx.x] = partial;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        result                = smem[0];
        y[node_id]            = result;
        partial_norm[node_id] = result * result;
    }
}


// =========================================================================
// KERNEL 3: partial_reduce_to_scalar — Improvement 2
//
// Reduces an n-element FP32 partial_norm array to a single scalar and
// writes sqrt(sum) to scalar_out[0] — entirely on the GPU.
//
// Grid:  (1, 1, 1) — single block, BLOCK_SIZE threads.
// The strided loop handles n > BLOCK_SIZE without extra kernel launches.
// Keeps the result on the GPU so normalize_inplace can consume it
// without a CPU round-trip.
// =========================================================================
__global__ void partial_reduce_to_scalar(
    const float* __restrict__ partial_norm,
    float*       __restrict__ scalar_out,
    const int                  n)
{
    __shared__ float smem[BLOCK_SIZE];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE)
        sum += partial_norm[i];

    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        scalar_out[0] = sqrtf(smem[0] > 0.0f ? smem[0] : 0.0f);
}


// =========================================================================
// KERNEL 4: normalize_inplace — Improvement 2
//
// Reads the L2 norm from a GPU scalar pointer (written by
// partial_reduce_to_scalar) and divides each element of vec in-place.
// All threads read the same scalar address — it is broadcast through
// the L1 constant cache (single-address load optimisation).
//
// Grid: (ceil(n / BLOCK_SIZE), 1, 1)
// =========================================================================
__global__ void normalize_inplace(
    float*       __restrict__ vec,
    const float* __restrict__ scalar_norm,
    const int                  n)
{
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const float norm = *scalar_norm;
    if (norm > 1e-10f) vec[tid] /= norm;
}


// =========================================================================
// KERNEL 5: compute_convergence_partial — Improvement 5 (FP64)
//
// Computes per-block partial sums of (delta_h^2 + delta_a^2) using
// FP64 accumulation in shared memory.  The host sums the small
// norm_blocks array and takes sqrt — a trivially fast CPU operation.
//
// Precision policy: FP32 inputs, FP64 smem/output to reliably detect
// delta < 1e-6 without cancellation error near convergence.
// =========================================================================
__global__ void compute_convergence_partial(
    const float*  __restrict__ h_new,
    const float*  __restrict__ h_old,
    const float*  __restrict__ a_new,
    const float*  __restrict__ a_old,
    double*        __restrict__ partial_sums,
    const int                   n)
{
    __shared__ double smem[BLOCK_SIZE];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double val = 0.0;
    if (tid < n) {
        const double dh = (double)(h_new[tid] - h_old[tid]);
        const double da = (double)(a_new[tid] - a_old[tid]);
        val = dh * dh + da * da;
    }
    smem[threadIdx.x] = val;
    __syncthreads();
    for (int s = (blockDim.x >> 1); s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial_sums[blockIdx.x] = smem[0];
}

}  // extern "C"
"""

# ---------------------------------------------------------------------------
# Module-level kernel cache
# ---------------------------------------------------------------------------

_kernel_cache: dict[str, dict[str, Any]] = {}


def _get_kernels() -> dict[str, Any]:
    """Compile (or fetch from cache) all HITS device kernels.

    Active kernels
    --------------
    spmv_low_deg   : spmv_edge_parallel_low_degree  — packed-warp low-degree SpMV + norm
    spmv_high      : spmv_with_norm_sq              — fused high-degree SpMV + norm
    reduce_scalar  : partial_reduce_to_scalar        — single-block FP32 reduce + sqrt
    norm_div       : normalize_inplace               — in-place divide by GPU scalar
    conv_partial   : compute_convergence_partial     — FP64 convergence partials

    Deprecated (compiled but not used in main iteration):
    spmv           : spmv_degree_aware
    norm_sq        : compute_partial_norm_sq
    normalize      : normalize_vector
    conv_delta_f32 : compute_convergence_delta
    """
    if "hits" not in _kernel_cache:
        if not PYCUDA_AVAILABLE:
            raise RuntimeError(
                "PyCUDA is required to compile HITS kernels — "
                "install pycuda and ensure NVCC is on PATH."
            )
        # Adaptive arch detection — falls back to sm_75 (Turing) if the
        # device cannot be probed.
        try:
            cuda.init()
            cc_major, cc_minor = cuda.Device(0).compute_capability()
            arch_flag = f"-arch=sm_{cc_major}{cc_minor}"
        except Exception:                               # noqa: BLE001
            arch_flag = "-arch=sm_75"
        mod = SourceModule(
            KERNEL_SOURCE,
            options=[arch_flag, "-O3"],
            no_extern_c=True,
        )
        _kernel_cache["hits"] = {
            # --- Active kernels ---
            "spmv_low_deg":   mod.get_function("spmv_edge_parallel_low_degree"),
            "spmv_high":      mod.get_function("spmv_with_norm_sq"),
            "reduce_scalar":  mod.get_function("partial_reduce_to_scalar"),
            "norm_div":       mod.get_function("normalize_inplace"),
            "conv_partial":   mod.get_function("compute_convergence_partial"),
            # --- Deprecated (kept for debugging / reference) ---
            "spmv":           mod.get_function("spmv_degree_aware"),
            "norm_sq":        mod.get_function("compute_partial_norm_sq"),
            "normalize":      mod.get_function("normalize_vector"),
            "conv_delta_f32": mod.get_function("compute_convergence_delta"),
        }
    return _kernel_cache["hits"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


def _top_k(scores: np.ndarray, k: int = _TOP_K) -> list[int]:
    return np.argsort(scores)[::-1][:k].tolist()


def _pack_result(
    hub: np.ndarray,
    auth: np.ndarray,
    iterations: int,
    converged: bool,
    network_type: str,
) -> dict:
    """Build the inner result dict in the network-type-specific shape.

    Parameters
    ----------
    hub, auth     : Arrays in *original* node-index space (reordering
                    must have been reversed before calling this).
    """
    if network_type == "ppi":
        combined = hub + auth
        return {
            "hub_scores":       hub.tolist(),
            "authority_scores": auth.tolist(),
            "top_nodes":        _top_k(combined),
            "iterations":       iterations,
            "converged":        converged,
        }
    # grn / mirna
    top_h = _top_k(hub)
    top_a = _top_k(auth)
    overlap = sorted(set(top_h) & set(top_a))
    return {
        "hub_scores":            hub.tolist(),
        "authority_scores":      auth.tolist(),
        "iterations":            iterations,
        "converged":             converged,
        "top_hubs":              top_h,
        "top_authorities":       top_a,
        "hub_authority_overlap": overlap,
    }


def _reorder_by_degree_hits(
    A: sp.csr_matrix,
    A_T: sp.csr_matrix,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray]:
    """Sort nodes by descending out-degree for improved CSR cache locality.

    High-degree hubs are placed at the front of the CSR layout so that
    nearby rows share more column-index neighbours, reducing cache misses
    in ``x[col_idx[j]]`` accesses during SpMV.

    Parameters
    ----------
    A, A_T : CSR adjacency and its transpose (must have equal shape).

    Returns
    -------
    A_reord  : A with rows and columns permuted by ``perm``.
    A_T_reord: A_T with rows and columns permuted by ``perm``.
    perm     : int32 array; ``perm[new_idx] = old_idx``.
    inv_perm : int32 array; ``inv_perm[old_idx] = new_idx``.

    Notes
    -----
    Scores produced on the reordered graph are in *reordered* index space.
    Restore original order via ``scores_orig[perm] = scores_reord``.
    """
    degrees = np.diff(A.indptr)                        # out-degrees
    perm = np.argsort(-degrees).astype(np.int32)       # descending
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm), dtype=np.int32)

    # scipy fancy indexing: A_reord[i, j] = A[perm[i], perm[j]]
    A_reord   = A[perm,   :][:, perm].tocsr().astype(np.float32)
    A_T_reord = A_T[perm, :][:, perm].tocsr().astype(np.float32)
    return A_reord, A_T_reord, perm, inv_perm


# ---------------------------------------------------------------------------
# Main GPU implementation
# ---------------------------------------------------------------------------

def hits_gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """HITS — GPU-accelerated via custom PyCUDA kernels (optimised).

    Optimisations over the initial implementation
    ---------------------------------------------
    1. Fused SpMV + norm : ``spmv_with_norm_sq`` computes y = M*x and
       writes per-node squared values in one kernel pass, eliminating a
       separate ``compute_partial_norm_sq`` launch.
    2. GPU-side norm reduction : ``partial_reduce_to_scalar`` reduces the
       partial-norm array to a scalar and writes sqrt to a GPU pointer.
       ``normalize_inplace`` reads that pointer — no CPU round-trip.
    3. Single CPU sync per iteration : only the convergence delta check
       (``d_partial_conv.get()``) ever breaks to the CPU.
    4. Edge-parallel low-degree kernel : ``spmv_edge_parallel_low_degree``
       packs 8 nodes per block (one warp each) for ~8x better SM occupancy
       on degree-< WARP_SIZE nodes (the majority in biological networks).
    5. Node reordering : nodes sorted by descending degree before GPU work
       (default ``reorder_nodes=True``).  Scores remapped to original indices
       before result construction.

    Precision
    ---------
    FP32 for SpMV, hub/authority scores, and L2 norms.
    FP64 for the convergence delta only (avoids false early termination).

    Parameters
    ----------
    graph_csr    : scipy.sparse.csr_matrix
        Directed adjacency (rows = sources for GRN / miRNA;
        will be symmetrised for PPI).
    params       : dict
        Keys: ``max_iter``, ``tolerance``, ``network_type``,
        ``reorder_nodes``.  Missing keys use module defaults.

    Returns
    -------
    dict — outer envelope + inner result per CLAUDE.md spec.

    Raises
    ------
    RuntimeError
        If PyCUDA is unavailable or no CUDA device is found.
    MemoryError
        If the graph + working set exceeds free VRAM.
    cuda.LogicError
        On PyCUDA driver-API errors.
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA is required for hits_gpu(). "
            "Install it or use hits_cpu_single() from "
            "src/algorithms/cpu/single_threaded/hits.py"
        )

    # Push the device PRIMARY context onto the PyCUDA driver-API stack.
    # _ensure_cuda_context() only activates the CuPy *runtime-API* context;
    # PyCUDA's cuModuleLoadDataEx requires a *driver-API* context on the
    # current thread.  retain_primary_context().push() is idempotent and
    # reference-counted, so it coexists safely with CuPy in the same process.
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
        # ── Parameter merging ──────────────────────────────────────────────
        p = _merge_params(params)
        if _GPU_CONFIG_AVAILABLE:
            p = apply_config("hits", graph_csr, p) or p
            for k, v in _DEFAULT_PARAMS.items():
                p.setdefault(k, v)

        max_iter      = int(p["max_iter"])
        tolerance     = float(p["tolerance"])
        network_type  = str(p.get("network_type", "grn"))
        reorder_nodes = bool(p.get("reorder_nodes", True))

        n = int(graph_csr.shape[0])
        if n == 0:
            raise ValueError("Empty graph")

        # ── Network-type adaptation (CPU, not timed) ───────────────────────
        if network_type == "ppi":
            A = (graph_csr + graph_csr.T).tocsr()
            A.data = np.ones_like(A.data, dtype=np.float32)
        else:
            A = graph_csr.astype(np.float32)
        A.sum_duplicates()
        A = A.tocsr().astype(np.float32)
        A_T = A.T.tocsr().astype(np.float32)

        # ── Node reordering by descending degree (CPU, not timed) ──────────
        perm: np.ndarray | None = None
        inv_perm: np.ndarray | None = None
        if reorder_nodes:
            A, A_T, perm, inv_perm = _reorder_by_degree_hits(A, A_T)

        # ── CSR host arrays ────────────────────────────────────────────────
        row_ptr_h   = np.ascontiguousarray(A.indptr,    dtype=np.int32)
        col_idx_h   = np.ascontiguousarray(A.indices,   dtype=np.int32)
        values_h    = np.ascontiguousarray(A.data,      dtype=np.float32)
        row_ptr_T_h = np.ascontiguousarray(A_T.indptr,  dtype=np.int32)
        col_idx_T_h = np.ascontiguousarray(A_T.indices, dtype=np.int32)
        values_T_h  = np.ascontiguousarray(A_T.data,    dtype=np.float32)

        # Per-row degrees (for node classification).
        degrees_h   = np.diff(row_ptr_h).astype(np.int32)    # out-degree of A
        degrees_T_h = np.diff(row_ptr_T_h).astype(np.int32)  # out-degree of A_T

        # ── Node classification (CPU, not timed) ───────────────────────────
        # For each matrix (A and A_T) separately, because transposition
        # changes degree distributions.
        low_mask_A   = degrees_h   < WARP_SIZE
        low_ids_A    = np.where( low_mask_A)[0].astype(np.int32)
        high_ids_A   = np.where(~low_mask_A)[0].astype(np.int32)
        n_low_A,  n_high_A  = len(low_ids_A),  len(high_ids_A)

        low_mask_AT  = degrees_T_h < WARP_SIZE
        low_ids_AT   = np.where( low_mask_AT)[0].astype(np.int32)
        high_ids_AT  = np.where(~low_mask_AT)[0].astype(np.int32)
        n_low_AT, n_high_AT = len(low_ids_AT), len(high_ids_AT)

        # Blocks for convergence partial reduction (FP64).
        norm_blocks = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE)

        # ── VRAM estimate ──────────────────────────────────────────────────
        required_bytes = (
            row_ptr_h.nbytes + col_idx_h.nbytes + values_h.nbytes +
            row_ptr_T_h.nbytes + col_idx_T_h.nbytes + values_T_h.nbytes +
            4 * n * 4 +                         # h, h_new, a, a_new  (float32)
            n * 4 +                             # partial_norm         (float32)
            4 +                                 # scalar_norm          (float32)
            norm_blocks * 8 +                   # partial_conv         (float64)
            (n_low_A + n_high_A +
             n_low_AT + n_high_AT) * 4          # node-id arrays       (int32)
        )
        free_vram_mb = 0
        try:
            free_vram_mb = int(get_gpu_config().get("free_vram_mb", 0))
        except Exception:                                   # noqa: BLE001
            pass
        free_vram_bytes = free_vram_mb * 1024 * 1024
        use_pagelocked = False
        if free_vram_bytes > 0:
            if required_bytes > free_vram_bytes:
                raise MemoryError(
                    f"HITS GPU needs ~{required_bytes/1e6:.1f} MB "
                    f"but only {free_vram_mb} MB VRAM free."
                )
            if required_bytes > VRAM_SAFETY * free_vram_bytes:
                use_pagelocked = True
                logging.warning(
                    "Graph near VRAM limit (%.1f / %.1f MB) — "
                    "using page-locked host buffers for convergence check.",
                    required_bytes / 1e6, free_vram_bytes / 1e6,
                )

        # ── Kernel compilation ─────────────────────────────────────────────
        kernels       = _get_kernels()
        k_spmv_low    = kernels["spmv_low_deg"]   # packed-warp low-degree
        k_spmv_high   = kernels["spmv_high"]       # fused high-degree
        k_reduce      = kernels["reduce_scalar"]   # FP32 partial_norm → scalar
        k_norm_inplace = kernels["norm_div"]       # in-place normalize
        k_conv        = kernels["conv_partial"]    # FP64 convergence partials

        # ── CUDA streams + timing events ───────────────────────────────────
        stream_compute  = cuda.Stream()
        stream_transfer = cuda.Stream()
        start_event     = cuda.Event()
        end_event       = cuda.Event()

        # ── Device allocation ──────────────────────────────────────────────
        d_buffers: list = []

        def _alloc(shape, dtype):
            arr = gpuarray.empty(shape, dtype=dtype)
            d_buffers.append(arr)
            return arr

        def _upload(host_arr: np.ndarray):
            """Allocate and immediately copy host_arr to device."""
            arr = gpuarray.to_gpu(host_arr)
            d_buffers.append(arr)
            return arr

        # Matrix CSR arrays.
        d_row_ptr   = _alloc(row_ptr_h.shape,   np.int32)
        d_col_idx   = _alloc(col_idx_h.shape,   np.int32)
        d_values    = _alloc(values_h.shape,    np.float32)
        d_row_ptr_T = _alloc(row_ptr_T_h.shape, np.int32)
        d_col_idx_T = _alloc(col_idx_T_h.shape, np.int32)
        d_values_T  = _alloc(values_T_h.shape,  np.float32)

        # Score vectors (ping-pong pairs).
        d_h     = _alloc((n,), np.float32)
        d_a     = _alloc((n,), np.float32)
        d_h_new = _alloc((n,), np.float32)
        d_a_new = _alloc((n,), np.float32)

        # Auxiliary buffers.
        d_partial_norm  = _alloc((n,),           np.float32)   # per-node y^2
        d_scalar_norm   = _alloc((1,),           np.float32)   # GPU scalar norm
        d_partial_conv  = _alloc((norm_blocks,), np.float64)   # FP64 convergence

        # Node-ID arrays (conditional — skip allocation when count == 0).
        d_low_ids_A   = _upload(low_ids_A)   if n_low_A  > 0 else None
        d_high_ids_A  = _upload(high_ids_A)  if n_high_A > 0 else None
        d_low_ids_AT  = _upload(low_ids_AT)  if n_low_AT > 0 else None
        d_high_ids_AT = _upload(high_ids_AT) if n_high_AT > 0 else None

        # Optional page-locked staging for the convergence D2H transfer.
        if use_pagelocked:
            pl_conv = cuda.pagelocked_empty(norm_blocks, dtype=np.float64)
        else:
            pl_conv = None

        try:
            # ── H2D: matrix CSR arrays (async on transfer stream) ──────────
            cuda.memcpy_htod_async(d_row_ptr.gpudata,   row_ptr_h,   stream_transfer)
            cuda.memcpy_htod_async(d_col_idx.gpudata,   col_idx_h,   stream_transfer)
            cuda.memcpy_htod_async(d_values.gpudata,    values_h,    stream_transfer)
            cuda.memcpy_htod_async(d_row_ptr_T.gpudata, row_ptr_T_h, stream_transfer)
            cuda.memcpy_htod_async(d_col_idx_T.gpudata, col_idx_T_h, stream_transfer)
            cuda.memcpy_htod_async(d_values_T.gpudata,  values_T_h,  stream_transfer)

            # ── H2D: initial score vectors ─────────────────────────────────
            init_val  = np.float32(1.0 / n)
            init_host = np.full(n, init_val, dtype=np.float32)
            cuda.memcpy_htod_async(d_h.gpudata, init_host, stream_transfer)
            cuda.memcpy_htod_async(d_a.gpudata, init_host, stream_transfer)
            stream_transfer.synchronize()

            # MEMORY_FIX (timing audit): record start AFTER all H2D bytes
            # so reported execution_time reflects algorithm work, not the
            # ~80 ms PCIe transfer for the 6 CSR arrays.
            start_event.record(stream_compute)

            # ── Precomputed grid/block dimensions ──────────────────────────
            spmv_block    = (BLOCK_SIZE, 1, 1)
            scalar_grid   = (1, 1, 1)               # single block for reduce
            norm_full_grid = (
                max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE), 1, 1
            )
            conv_grid = (norm_blocks, 1, 1)

            # Low/high grid dimensions (handle zero-count cases).
            low_A_grid   = (max(1, (n_low_A  + NODES_PER_BLOCK - 1) // NODES_PER_BLOCK), 1, 1)
            high_A_grid  = (max(1, n_high_A), 1, 1)
            low_AT_grid  = (max(1, (n_low_AT + NODES_PER_BLOCK - 1) // NODES_PER_BLOCK), 1, 1)
            high_AT_grid = (max(1, n_high_AT), 1, 1)

            # Scalar args (avoid repeated boxing in the loop).
            n_i32         = np.int32(n)
            n_low_A_i32   = np.int32(n_low_A)
            n_high_A_i32  = np.int32(n_high_A)
            n_low_AT_i32  = np.int32(n_low_AT)
            n_high_AT_i32 = np.int32(n_high_AT)
            nb_i32        = np.int32(norm_blocks)

            # Ping-pong pointers (no data copy on swap).
            cur_h, nxt_h = d_h, d_h_new
            cur_a, nxt_a = d_a, d_a_new

            # ── Iteration loop ─────────────────────────────────────────────
            iterations = 0
            converged  = False

            for it in range(1, max_iter + 1):
                iterations = it

                # =========================================================
                # AUTHORITY UPDATE:  a_new = A_T * h
                # =========================================================

                # — Low-degree nodes (deg < WARP_SIZE) —
                if n_low_AT > 0:
                    k_spmv_low(
                        d_row_ptr_T, d_col_idx_T, d_values_T,
                        cur_h, nxt_a, d_partial_norm,
                        d_low_ids_AT, n_low_AT_i32,
                        block=spmv_block, grid=low_AT_grid,
                        stream=stream_compute,
                    )

                # — High-degree nodes (deg >= WARP_SIZE) —
                if n_high_AT > 0:
                    k_spmv_high(
                        d_row_ptr_T, d_col_idx_T, d_values_T,
                        cur_h, nxt_a, d_partial_norm,
                        d_high_ids_AT, n_high_AT_i32,
                        block=spmv_block, grid=high_AT_grid,
                        stream=stream_compute,
                    )

                # — Reduce partial_norm → GPU scalar, then normalise —
                k_reduce(
                    d_partial_norm, d_scalar_norm, n_i32,
                    block=spmv_block, grid=scalar_grid,
                    stream=stream_compute,
                )
                k_norm_inplace(
                    nxt_a, d_scalar_norm, n_i32,
                    block=spmv_block, grid=norm_full_grid,
                    stream=stream_compute,
                )

                # =========================================================
                # HUB UPDATE:  h_new = A * a_new
                # =========================================================

                # — Low-degree nodes —
                if n_low_A > 0:
                    k_spmv_low(
                        d_row_ptr, d_col_idx, d_values,
                        nxt_a, nxt_h, d_partial_norm,
                        d_low_ids_A, n_low_A_i32,
                        block=spmv_block, grid=low_A_grid,
                        stream=stream_compute,
                    )

                # — High-degree nodes —
                if n_high_A > 0:
                    k_spmv_high(
                        d_row_ptr, d_col_idx, d_values,
                        nxt_a, nxt_h, d_partial_norm,
                        d_high_ids_A, n_high_A_i32,
                        block=spmv_block, grid=high_A_grid,
                        stream=stream_compute,
                    )

                # — Reduce partial_norm → GPU scalar, then normalise —
                k_reduce(
                    d_partial_norm, d_scalar_norm, n_i32,
                    block=spmv_block, grid=scalar_grid,
                    stream=stream_compute,
                )
                k_norm_inplace(
                    nxt_h, d_scalar_norm, n_i32,
                    block=spmv_block, grid=norm_full_grid,
                    stream=stream_compute,
                )

                # =========================================================
                # CONVERGENCE CHECK — exactly ONE CPU sync per iteration
                # FP64 partial sums keep precision near tol=1e-6.
                # =========================================================
                k_conv(
                    nxt_h, cur_h, nxt_a, cur_a, d_partial_conv, n_i32,
                    block=spmv_block, grid=conv_grid,
                    stream=stream_compute,
                )
                stream_compute.synchronize()           # ← the single sync

                if pl_conv is not None:
                    cuda.memcpy_dtoh(pl_conv, d_partial_conv.gpudata)
                    delta = float(np.sqrt(np.sum(pl_conv)))
                else:
                    delta = float(np.sqrt(np.sum(d_partial_conv.get())))

                # Pointer swap — reuse GPU buffers without data copy.
                cur_h, nxt_h = nxt_h, cur_h
                cur_a, nxt_a = nxt_a, cur_a

                if delta < tolerance:
                    converged = True
                    break

            # ── Timing ends ────────────────────────────────────────────────
            end_event.record(stream_compute)
            end_event.synchronize()
            elapsed = start_event.time_till(end_event) / 1000.0  # seconds

            # ── Pull scores to host ────────────────────────────────────────
            hub_host  = cur_h.get()   # in (possibly) reordered index space
            auth_host = cur_a.get()

            # ── Restore original node order ────────────────────────────────
            # perm[new_idx] = old_idx  ⟹  scores_orig[perm] = scores_reord
            if perm is not None:
                hub_orig  = np.empty(n, dtype=np.float32)
                auth_orig = np.empty(n, dtype=np.float32)
                hub_orig[perm]  = hub_host
                auth_orig[perm] = auth_host
                hub_host  = hub_orig
                auth_host = auth_orig

            inner = _pack_result(hub_host, auth_host, iterations, converged,
                                 network_type)

            return {
                "algorithm":      "hits",
                "mode":           "gpu",
                "network_type":   network_type,
                "execution_time": elapsed,
                "num_nodes":      n,
                "num_edges":      int(graph_csr.nnz),
                "result":         inner,
            }

        finally:
            # Release all device allocations (including node-id arrays).
            for arr in d_buffers:
                try:
                    arr.gpudata.free()
                except Exception:                       # noqa: BLE001
                    pass
            # MEMORY_FIX (H-5): explicitly release the page-locked host
            # buffer allocated for the convergence D2H transfer.  Old
            # code relied on Python GC, which on Linux+CUDA can leak the
            # pinned allocation across repeated benchmark runs and
            # eventually surfaces as cuMemHostAlloc OUT_OF_MEMORY.
            if pl_conv is not None:
                try:
                    pl_conv.base.free()
                except Exception:                       # noqa: BLE001
                    pass

    except cuda.LogicError as e:
        logging.warning("CUDA error in hits_gpu: %s", e)
        raise
    except MemoryError:
        logging.warning(
            "VRAM exhausted in hits_gpu. "
            "Try a smaller graph or use a higher-VRAM device."
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
    """Runner entry point — wraps ``hits_gpu()`` in the standard
    ``{output, extra_params}`` envelope required by
    ``src.benchmarking.benchmark`` and ``src.runner.algorithm_runner``.
    """
    p = _merge_params(params)
    full = hits_gpu(graph_csr, p)
    return {"output": full, "extra_params": p}
