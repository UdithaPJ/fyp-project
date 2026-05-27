"""
algorithms/hits.py — HITS (Hyperlink-Induced Topic Search) for Biological Networks
=====================================================================================

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

Parameter Guide
---------------
max_iter     (int,   default 100)    Hard iteration cap.
tolerance    (float, default 1e-6)   L2-norm convergence threshold.
network_type (str,   default "grn")  One of "grn", "ppi", "mirna".
block_size   (int,   default 256)    CUDA block dimension (overridden by
                                     gpu_config.apply_config for the tier).
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
# Four PyCUDA kernels (one SourceModule, compiled once, cached):
#   spmv_degree_aware        — three-tier SpMV (1 thread / 1 warp / 1 block)
#   compute_partial_norm_sq  — block-level partial sum of squares for L2 norm
#   normalize_vector         — in-place division by scalar L2 norm
#   compute_convergence_delta— block-level partial sum of (Δh² + Δa²)
#
# Per-iteration sequence:
#   spmv(Aᵀ, h, a_new) → norm_sq → CPU-reduce → normalize(a_new)
#   spmv(A , a_new, h_new) → norm_sq → CPU-reduce → normalize(h_new)
#   conv_delta(h_new, h, a_new, a) → CPU-reduce → delta
#   pointer-swap, check tol.
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
    "max_iter":     100,
    "tolerance":    1e-6,
    "network_type": "grn",
    "block_size":   256,
}

BLOCK_SIZE: int  = 256
WARP_SIZE: int   = 32
VRAM_SAFETY: float = 0.80      # use at most 80 % of free VRAM before warning
_TOP_K: int      = 15


# ---------------------------------------------------------------------------
# CUDA kernel source (all four kernels, single SourceModule)
# ---------------------------------------------------------------------------

KERNEL_SOURCE = r"""
extern "C" {

#define BLOCK_SIZE 256
#define WARP_SIZE  32

// =========================================================================
// KERNEL 1: degree-aware SpMV (y = M * x)
//
// Grid layout: one block per row of M (gridDim.x = n).
// Inside the block, work is split by the row's nnz:
//   LOW  (degree < 32)         : thread 0 does a serial scan
//   MED  (32 <= degree < 256)  : the first warp strides through the row
//                                and reduces with __shfl_down_sync
//   HIGH (degree >= 256)       : all 256 threads stride, reduce in shared
//                                memory via tree reduction
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

    // ---- LOW tier ------------------------------------------------------
    if (degree < WARP_SIZE) {
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int j = row_start; j < row_end; ++j) {
                sum += values[j] * x[col_idx[j]];
            }
            y[node_id] = sum;
        }
        return;
    }

    // ---- MED tier ------------------------------------------------------
    if (degree < BLOCK_SIZE) {
        if (threadIdx.x < WARP_SIZE) {
            float partial = 0.0f;
            for (int j = row_start + threadIdx.x; j < row_end; j += WARP_SIZE) {
                partial += values[j] * x[col_idx[j]];
            }
            // Warp-level reduction
            for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
                partial += __shfl_down_sync(0xffffffffu, partial, off);
            }
            if (threadIdx.x == 0) y[node_id] = partial;
        }
        return;
    }

    // ---- HIGH tier -----------------------------------------------------
    float partial = 0.0f;
    for (int j = row_start + threadIdx.x; j < row_end; j += BLOCK_SIZE) {
        partial += values[j] * x[col_idx[j]];
    }
    smem[threadIdx.x] = partial;
    __syncthreads();
    // Tree reduction in shared memory.
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) y[node_id] = smem[0];
}


// =========================================================================
// KERNEL 2: partial L2-norm-squared (one float per block)
//
// Each block reduces a chunk of vec into smem[0] = sum(vec[i] * vec[i]).
// CPU then sums the per-block partials and takes sqrt.
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
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) partial_sums[blockIdx.x] = smem[0];
}


// =========================================================================
// KERNEL 3: in-place divide by scalar L2 norm
// =========================================================================
__global__ void normalize_vector(
    float* __restrict__ vec,
    const float          norm,
    const int            n)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    if (norm > 1e-10f) {
        vec[tid] /= norm;
    }
}


// =========================================================================
// KERNEL 4: combined convergence partial — sum( (dh*dh + da*da) ) per block
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
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
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
    """Compile (or fetch from cache) the four HITS device kernels."""
    if "hits" not in _kernel_cache:
        if not PYCUDA_AVAILABLE:
            raise RuntimeError(
                "PyCUDA is required to compile HITS kernels — "
                "install pycuda and ensure NVCC is on PATH."
            )
        mod = SourceModule(
            KERNEL_SOURCE,
            options=["-arch=sm_75"],        # RTX 20-series Turing
            no_extern_c=True,
        )
        _kernel_cache["hits"] = {
            "spmv":       mod.get_function("spmv_degree_aware"),
            "norm_sq":    mod.get_function("compute_partial_norm_sq"),
            "normalize":  mod.get_function("normalize_vector"),
            "conv_delta": mod.get_function("compute_convergence_delta"),
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
    """Build the inner result dict in the network-type-specific shape."""
    if network_type == "ppi":
        # Symmetrised graph → hub == authority; return combined ranking.
        combined = (hub + auth)
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


# ---------------------------------------------------------------------------
# Main GPU implementation
# ---------------------------------------------------------------------------

def hits_gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """HITS — GPU-accelerated via custom PyCUDA kernels.

    No CuPy dependency.  Both ``A`` and ``Aᵀ`` are stored on-device in CSR
    form for the full iteration.  Four kernels (compiled once, cached) run
    a two-phase normalised power iteration.

    Returns the same inner-result shape required by the CLAUDE.md spec —
    branched by ``network_type``.

    Parameters
    ----------
    graph_csr    : scipy.sparse.csr_matrix
        Directed adjacency (rows = sources for GRN / miRNA;
        will be symmetrised for PPI).
    params       : dict
        Keys: ``max_iter``, ``tolerance``, ``network_type``, ``block_size``.
        Missing keys use module defaults.

    Returns
    -------
    dict — see CLAUDE.md "HITS" result spec.

    Raises
    ------
    RuntimeError
        If PyCUDA is unavailable or no CUDA device can be initialised.
    MemoryError
        If the graph + working set exceeds free VRAM.
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA is required for hits_gpu(). "
            "Install it or use hits_cpu_single() from "
            "src/algorithms/cpu/single_threaded/hits.py"
        )

    # Push the device PRIMARY context onto the PyCUDA driver-API stack
    # unconditionally.  We cannot rely on `_ensure_cuda_context()` — that
    # path activates the CuPy runtime-API context, which PyCUDA's
    # `cuModuleLoadDataEx` does not recognise as a current driver-API
    # context.  Pushing the primary context here is idempotent: it is
    # reference-counted and shared with CuPy under the hood, so both
    # libraries cooperate cleanly.
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
        # ---------- Parameter merging ---------------------------------------
        p = _merge_params(params)
        if _GPU_CONFIG_AVAILABLE:
            p = apply_config("hits", graph_csr, p) or p
            for k, v in _DEFAULT_PARAMS.items():
                p.setdefault(k, v)

        max_iter     = int(p["max_iter"])
        tolerance    = float(p["tolerance"])
        network_type = str(p.get("network_type", "grn"))
        block_size   = int(p.get("block_size", BLOCK_SIZE))
        if block_size <= 0 or block_size > 1024:
            block_size = BLOCK_SIZE
        # Kernels are hard-wired to BLOCK_SIZE=256 for shared-mem layout;
        # block_size only governs the normalisation / partial-sum launches.
        norm_block = BLOCK_SIZE  # always 256 to match smem[256]

        n = int(graph_csr.shape[0])
        if n == 0:
            raise ValueError("Empty graph")

        # ---------- Network-type adaptation ---------------------------------
        if network_type == "ppi":
            A = (graph_csr + graph_csr.T).tocsr()
            A.data = np.ones_like(A.data, dtype=np.float32)
        else:
            A = graph_csr.astype(np.float32)
        A.sum_duplicates()

        # Build Aᵀ for the authority update.
        A_T = A.T.tocsr().astype(np.float32)

        # CSR arrays (host).
        row_ptr_h   = np.ascontiguousarray(A.indptr,   dtype=np.int32)
        col_idx_h   = np.ascontiguousarray(A.indices,  dtype=np.int32)
        values_h    = np.ascontiguousarray(A.data,     dtype=np.float32)
        row_ptr_T_h = np.ascontiguousarray(A_T.indptr, dtype=np.int32)
        col_idx_T_h = np.ascontiguousarray(A_T.indices, dtype=np.int32)
        values_T_h  = np.ascontiguousarray(A_T.data,   dtype=np.float32)

        # Per-row degrees for the SpMV dispatcher.
        degrees_h   = np.diff(row_ptr_h).astype(np.int32)
        degrees_T_h = np.diff(row_ptr_T_h).astype(np.int32)

        # Number of partial-sum blocks for norm reductions.
        norm_blocks = max(1, (n + norm_block - 1) // norm_block)

        # ---------- VRAM check ---------------------------------------------
        required_bytes = (
            row_ptr_h.nbytes + col_idx_h.nbytes + values_h.nbytes +
            row_ptr_T_h.nbytes + col_idx_T_h.nbytes + values_T_h.nbytes +
            4 * n * 4 +                              # h, h_new, a, a_new
            degrees_h.nbytes + degrees_T_h.nbytes +
            norm_blocks * 4                          # partial_sums
        )
        free_vram_mb = 0
        try:
            free_vram_mb = int(get_gpu_config().get("free_vram_mb", 0))
        except Exception:                               # noqa: BLE001
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
                    "using page-locked host buffers.",
                    required_bytes / 1e6, free_vram_bytes / 1e6,
                )

        # ---------- Kernel compilation -------------------------------------
        kernels    = _get_kernels()
        k_spmv     = kernels["spmv"]
        k_norm_sq  = kernels["norm_sq"]
        k_normalize = kernels["normalize"]
        k_conv     = kernels["conv_delta"]

        # ---------- CUDA streams + events ----------------------------------
        stream_compute  = cuda.Stream()
        stream_transfer = cuda.Stream()
        start_event = cuda.Event()
        end_event   = cuda.Event()

        # ---------- Device allocation --------------------------------------
        # Allocate GPUArrays so we can use the .gpudata pointer with the
        # raw kernel.prepared_call API.
        d_buffers: list = []

        def _empty(shape, dtype):
            arr = gpuarray.empty(shape, dtype=dtype)
            d_buffers.append(arr)
            return arr

        d_row_ptr   = _empty(row_ptr_h.shape,   np.int32)
        d_col_idx   = _empty(col_idx_h.shape,   np.int32)
        d_values    = _empty(values_h.shape,    np.float32)
        d_row_ptr_T = _empty(row_ptr_T_h.shape, np.int32)
        d_col_idx_T = _empty(col_idx_T_h.shape, np.int32)
        d_values_T  = _empty(values_T_h.shape,  np.float32)
        d_degrees   = _empty(degrees_h.shape,   np.int32)
        d_degrees_T = _empty(degrees_T_h.shape, np.int32)

        d_h     = _empty((n,), np.float32)
        d_a     = _empty((n,), np.float32)
        d_h_new = _empty((n,), np.float32)
        d_a_new = _empty((n,), np.float32)
        d_partial = _empty((norm_blocks,), np.float32)

        # Optional page-locked host staging for H↔D transfers near VRAM limit.
        if use_pagelocked:
            pl_partial = cuda.pagelocked_empty(norm_blocks, dtype=np.float32)
        else:
            pl_partial = None

        try:
            # ---------- Timing starts ---------------------------------------
            start_event.record(stream_compute)

            # H2D copies on the dedicated transfer stream.
            cuda.memcpy_htod_async(d_row_ptr.gpudata,   row_ptr_h,   stream_transfer)
            cuda.memcpy_htod_async(d_col_idx.gpudata,   col_idx_h,   stream_transfer)
            cuda.memcpy_htod_async(d_values.gpudata,    values_h,    stream_transfer)
            cuda.memcpy_htod_async(d_row_ptr_T.gpudata, row_ptr_T_h, stream_transfer)
            cuda.memcpy_htod_async(d_col_idx_T.gpudata, col_idx_T_h, stream_transfer)
            cuda.memcpy_htod_async(d_values_T.gpudata,  values_T_h,  stream_transfer)
            cuda.memcpy_htod_async(d_degrees.gpudata,   degrees_h,   stream_transfer)
            cuda.memcpy_htod_async(d_degrees_T.gpudata, degrees_T_h, stream_transfer)

            init_val = np.float32(1.0 / n)
            init_host = np.full(n, init_val, dtype=np.float32)
            cuda.memcpy_htod_async(d_h.gpudata, init_host, stream_transfer)
            cuda.memcpy_htod_async(d_a.gpudata, init_host, stream_transfer)
            stream_transfer.synchronize()

            # ---------- Iteration loop --------------------------------------
            iterations = 0
            converged = False
            spmv_grid  = (n, 1, 1)
            spmv_block = (BLOCK_SIZE, 1, 1)
            norm_grid  = (norm_blocks, 1, 1)
            norm_block_dim = (norm_block, 1, 1)
            norm_n = np.int32(n)
            norm_blocks_i32 = np.int32(norm_blocks)
            n_i32 = np.int32(n)

            # Pointers we ping-pong every iteration.
            cur_h, nxt_h = d_h, d_h_new
            cur_a, nxt_a = d_a, d_a_new

            for it in range(1, max_iter + 1):
                iterations = it

                # ---- Authority update: a_new = Aᵀ * h ----------------------
                k_spmv(
                    d_row_ptr_T, d_col_idx_T, d_values_T,
                    cur_h, nxt_a, d_degrees_T, n_i32,
                    block=spmv_block, grid=spmv_grid, stream=stream_compute,
                )

                # ---- ||a_new||₂ via partial-norm² + CPU reduce -------------
                k_norm_sq(
                    nxt_a, d_partial, n_i32,
                    block=norm_block_dim, grid=norm_grid, stream=stream_compute,
                )
                stream_compute.synchronize()
                if pl_partial is not None:
                    cuda.memcpy_dtoh(pl_partial, d_partial.gpudata)
                    a_norm = float(np.sqrt(np.sum(pl_partial)))
                else:
                    a_norm = float(np.sqrt(float(gpuarray.sum(d_partial).get())))

                # ---- Normalise a_new in place -----------------------------
                norm_vec_grid = (max(1, (n + block_size - 1) // block_size), 1, 1)
                k_normalize(
                    nxt_a, np.float32(a_norm), n_i32,
                    block=(block_size, 1, 1), grid=norm_vec_grid,
                    stream=stream_compute,
                )

                # ---- Hub update: h_new = A * a_new -------------------------
                k_spmv(
                    d_row_ptr, d_col_idx, d_values,
                    nxt_a, nxt_h, d_degrees, n_i32,
                    block=spmv_block, grid=spmv_grid, stream=stream_compute,
                )

                # ---- ||h_new||₂ via partial-norm² + CPU reduce -------------
                k_norm_sq(
                    nxt_h, d_partial, n_i32,
                    block=norm_block_dim, grid=norm_grid, stream=stream_compute,
                )
                stream_compute.synchronize()
                if pl_partial is not None:
                    cuda.memcpy_dtoh(pl_partial, d_partial.gpudata)
                    h_norm = float(np.sqrt(np.sum(pl_partial)))
                else:
                    h_norm = float(np.sqrt(float(gpuarray.sum(d_partial).get())))

                # ---- Normalise h_new in place -----------------------------
                k_normalize(
                    nxt_h, np.float32(h_norm), n_i32,
                    block=(block_size, 1, 1), grid=norm_vec_grid,
                    stream=stream_compute,
                )

                # ---- Convergence ‖Δh,Δa‖ ----------------------------------
                k_conv(
                    nxt_h, cur_h, nxt_a, cur_a, d_partial, n_i32,
                    block=norm_block_dim, grid=norm_grid, stream=stream_compute,
                )
                stream_compute.synchronize()
                if pl_partial is not None:
                    cuda.memcpy_dtoh(pl_partial, d_partial.gpudata)
                    delta = float(np.sqrt(np.sum(pl_partial)))
                else:
                    delta = float(np.sqrt(float(gpuarray.sum(d_partial).get())))

                # Pointer swap: this level's "new" becomes next level's "cur".
                cur_h, nxt_h = nxt_h, cur_h
                cur_a, nxt_a = nxt_a, cur_a

                if delta < tolerance:
                    converged = True
                    break

            end_event.record(stream_compute)
            end_event.synchronize()
            elapsed = start_event.time_till(end_event) / 1000.0

            # ---------- Pull results back to host ---------------------------
            hub_host  = cur_h.get()
            auth_host = cur_a.get()

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
            # Explicit device-memory release.
            for arr in d_buffers:
                try:
                    arr.gpudata.free()
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
    """Runner entry point — preserves the legacy ``{output, extra_params}``
    shape required by ``src.benchmarking.benchmark`` and
    ``src.runner.algorithm_runner``.
    """
    p = _merge_params(params)
    full = hits_gpu(graph_csr, p)
    # `full` already has the outer envelope.  The benchmark runner only
    # consumes "output" and "extra_params"; keep both layers available.
    return {"output": full, "extra_params": p}
