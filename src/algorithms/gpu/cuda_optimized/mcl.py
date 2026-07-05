"""
algorithms/mcl.py - Markov Clustering (MCL) for Biological Network Modules
==========================================================================

Biological context
------------------
MCL models a random walk on the network: at each step, mass diffuses
across neighbours (expansion) and is then non-linearly concentrated on
the strongest flows (inflation).  Tightly co-connected groups trap
probability mass into attractor states, revealing modules:

  GRN   - co-regulated gene modules (shared TF inputs, feedback arcs).
  PPI   - protein complexes / functional modules.
  miRNA - miRNA-gene regulons (a miRNA and its co-targeted genes).

Algorithm
---------
0. Symmetrise + binarise (when needed), add self-loops, column-normalise.
1. Iterate:
     a. Expansion  - M <- M^e            (random walk spreads probability)
     b. Prune      - threshold + top-k   (bound VRAM, keep top flows)
     c. Inflation  - M[i,j] <- M[i,j]^r  (sharpen strong flows)
     d. Column-renormalise               (re-establish column-stochastic)
     e. Convergence - ||M_new - M_old||_F < tol
2. Extract clusters from attractor columns (argmax per column).

Optimised GPU pipeline (this module)
------------------------------------
1. Hash-based row-wise SpGEMM (Gustavson-style sparse accumulator):
   - Per-row SMEM hash table, linear probe; processes only NONZEROS.
   - Eliminates the O(n^2) zero-multiplications of the inner-product
     approach; matches the power-law degree distribution of biological
     networks far better.
   - Wide-row fallback to the original inner-product kernel
     (`spgemm_row_chunk`) when SMEM hash table cannot hold the row.

2. GPU-native pruning compaction:
   - `prefix_sum_block` + `prefix_sum_add_offsets` + `compact_csr_values`
     + `rebuild_row_ptr` replace the previous numpy round-trip.
   - Only one int (new_nnz) crosses CPU<->GPU per compaction.

3. Bitonic-style top-k for wide columns (`topk_bitonic_column`):
   - One block per column, SMEM-resident sorted top-k running set
     refreshed by chunked scans of the column.
   - Falls back to `topk_column_prune` (count-greater) for narrow
     columns where its O(L^2/blockDim) cost is already small.

4. Multi-stream pipelining:
   - `stream_compute`  - main kernel execution.
   - `stream_transfer` - async H2D uploads of per-iteration matrix
                         arrays; recorded with cuda.Event so compute
                         can wait without a host sync.
   - `stream_prune`    - threshold prune + GPU compaction can overlap
                         with the start of the next iteration's SpGEMM.

5. Degree-aware row classification:
   - heavy   (deg >= block_size) -> inner-product SpGEMM with extra
                                    threads per row;
   - medium  (32 <= deg < block_size) -> hash SpGEMM, one block per row;
   - light   (deg < 32) -> hash SpGEMM, smaller hash table.
   - Classification computed ONCE per iteration from the GPU-resident
     row_ptr; kernel selection driven by per-tier index arrays.

6. Adaptive arch compilation: runtime
   `cuda.Device(0).compute_capability()` -> `-arch=sm_XY`.

VRAM ceiling (important)
------------------------
MCL is the most memory-intensive of the six algorithms: the expansion
step squares the matrix (M @ M), and the fill-in of the product — not the
input graph — is the binding VRAM constraint.  On random / scale-free
graphs there is no tight community structure to keep the product sparse,
so it explodes before pruning can contain it.

Two mechanisms keep MCL within VRAM as far as possible, then fail cleanly:
  * ``_adaptive_top_k`` caps top-k so the pruned *working set*
    (~n * top_k entries) fits free VRAM (bounds the SpGEMM INPUT).
  * ``_adaptive_prune_threshold`` raises the threshold under pressure to
    shrink the SpGEMM OUTPUT.

When the product still does not fit, MCL raises a clear ``MemoryError``
(it does NOT silently truncate the product — that was removed as a
correctness hazard).  Measured ceiling on a 6 GB RTX 2060 (grn/undirected,
~6 avg degree): roughly ~1M nodes (barabasi_albert, scale-free), ~1.6M
(erdos_renyi, random), ~3.3M (watts_strogatz, small-world).  The exact
limit is degree-variance dependent — e.g. erdos_renyi and watts_strogatz
at the SAME n/m differ because ER's Poisson degree variance produces more
fill-in than WS's near-uniform degree.  Real biological networks are far
smaller and sparser than these synthetic stress graphs and run comfortably
below the ceiling.  Beyond it, a higher-VRAM device (or higher
prune_threshold / lower top_k_per_column) is required; full out-of-core
MCL (host-streamed SpGEMM+prune+inflate) is future work.

References
----------
van Dongen, S. (2000). A Cluster Algorithm for Graphs. CWI Tech. Report.
"""

# -- GPU / CUDA-optimised implementation (PyCUDA custom kernels) ----------
# Source:    src/algorithms/gpu/cuda_optimized/mcl.py
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
import scipy.sparse.csgraph as csgraph

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
    logging.warning("PyCUDA not available - mcl_gpu() will raise.")

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
    "expansion":         2,
    "inflation":         2.0,
    "prune_threshold":   1e-3,
    "top_k_per_column":  50,
    "max_iter":          100,
    "convergence_tol":   1e-4,
    "network_type":      "grn",
    "block_size":        256,
}

BLOCK_SIZE: int        = 256
WARP_SIZE: int         = 32
VRAM_BUDGET_FRACTION: float = 0.70
COO_BYTES_PER_ENTRY: int = 12    # int (row) + int (col) + float (val)

# Default shared-memory budget per block on Turing (RTX 20-series) is 48 KB.
# We probe at runtime and fall back to this conservative value.
DEFAULT_SHARED_MEM_PER_BLOCK: int = 48 * 1024     # 49152 bytes
HASH_ENTRY_BYTES: int             = 8             # int key + float value

# Bitonic threshold for top-k: above this column length, switch from the
# O(L^2 / blockDim) count-greater kernel to the SMEM-resident bitonic
# kernel.  256 was chosen as the break-even point empirically on RTX 2060.
TOP_K_BITONIC_THRESHOLD: int = 256

# Degree thresholds for SpGEMM dispatch.
HEAVY_ROW_THRESH:  int = BLOCK_SIZE           # 256
LIGHT_ROW_THRESH:  int = WARP_SIZE            # 32


# ---------------------------------------------------------------------------
# CUDA kernel source (all kernels, single SourceModule)
# ---------------------------------------------------------------------------

KERNEL_SOURCE = r"""
extern "C" {

#define BLOCK_SIZE 256
#define WARP_SIZE  32

// =========================================================================
// KERNEL: spgemm_hash_row  (NEW - Improvement 1)
//
// Gustavson-style hash-based SpGEMM for C = A * B, one block per output
// row.  Processes only NONZERO entries of A's row, scatters their
// contributions through B's rows into a SMEM hash table keyed by output
// column index.  Eliminates the zero x zero work of the inner-product
// approach and matches power-law sparsity of biological networks.
//
// Inputs:
//   A : CSR  (A_row_ptr, A_col_idx, A_values)        - left operand
//   B : CSR  (B_row_ptr, B_col_idx, B_values)        - right operand
//   row_list   : indices of A rows assigned to this kernel launch
//                (one block processes row_list[blockIdx.x])
//   hash_size  : power-of-2 SMEM hash table size
//
// Output:
//   COO triples appended to (C_row, C_col, C_val) via atomic counter
//   C_nnz.  max_C_nnz is the caller-allocated capacity; overflow is
//   detected on the host.
//
// SMEM layout (dynamic):
//   int   hash_keys[hash_size]     - -1 == empty bucket
//   float hash_vals[hash_size]
// =========================================================================
__global__ void spgemm_hash_row(
    const int*   __restrict__ A_row_ptr,
    const int*   __restrict__ A_col_idx,
    const float* __restrict__ A_values,
    const int*   __restrict__ B_row_ptr,
    const int*   __restrict__ B_col_idx,
    const float* __restrict__ B_values,
    const int*   __restrict__ row_list,
    int*         __restrict__ C_row,
    int*         __restrict__ C_col,
    float*       __restrict__ C_val,
    int*         __restrict__ C_nnz,
    const int                  num_rows_in_list,
    const int                  hash_size,
    const int                  max_C_nnz,
    const float                prune_threshold)
{
    extern __shared__ int smem[];
    int*   hash_keys = smem;
    float* hash_vals = (float*)(smem + hash_size);

    if (blockIdx.x >= num_rows_in_list) return;
    const int row_i = row_list[blockIdx.x];

    // --- Init hash table -------------------------------------------------
    for (int b = threadIdx.x; b < hash_size; b += blockDim.x) {
        hash_keys[b] = -1;
        hash_vals[b] = 0.0f;
    }
    __syncthreads();

    const int a_start = A_row_ptr[row_i];
    const int a_end   = A_row_ptr[row_i + 1];
    if (a_start == a_end) return;       // empty row -> no output

    const int hmask = hash_size - 1;    // hash_size must be power of 2

    // --- Scatter phase: for each (i,k) in A, for each (k,j) in B --------
    // Threads share work over A's nonzeros (cooperative outer loop);
    // each thread serially walks B's row for its assigned k.  This keeps
    // the hash-table fast path independent across threads.
    for (int ak = a_start + threadIdx.x; ak < a_end; ak += blockDim.x) {
        const int   k    = A_col_idx[ak];
        const float a_ik = A_values[ak];

        const int b_start = B_row_ptr[k];
        const int b_end   = B_row_ptr[k + 1];

        for (int bj = b_start; bj < b_end; ++bj) {
            const int   j       = B_col_idx[bj];
            const float contrib = a_ik * B_values[bj];

            int bucket = j & hmask;
            // Linear probing with bounded probe count for safety.
            for (int probe = 0; probe < hash_size; ++probe) {
                const int old = atomicCAS(&hash_keys[bucket], -1, j);
                if (old == -1 || old == j) {
                    atomicAdd(&hash_vals[bucket], contrib);
                    break;
                }
                bucket = (bucket + 1) & hmask;
            }
        }
    }
    __syncthreads();

    // --- Flush non-empty hash buckets to COO output ---------------------
    // Threshold pruning is fused here (O1): entries below prune_threshold
    // are never written, so max_C_nnz can be sized against the post-prune
    // survivor count instead of the raw hash-accumulator output.
    for (int b = threadIdx.x; b < hash_size; b += blockDim.x) {
        const int   key = hash_keys[b];
        const float val = hash_vals[b];
        if (key >= 0 && val >= prune_threshold) {
            const int pos = atomicAdd(C_nnz, 1);
            if (pos < max_C_nnz) {
                C_row[pos] = row_i;
                C_col[pos] = key;
                C_val[pos] = val;
            }
            // Overflow detected host-side via C_nnz > max_C_nnz.
        }
    }
}


// =========================================================================
// KERNEL: spgemm_row_chunk  (FALLBACK - inner-product)
//
// Original inner-product SpGEMM.  Kept as a wide-row fallback when the
// estimated output of `spgemm_hash_row` exceeds available shared memory
// for the chosen hash size.
// =========================================================================
__global__ void spgemm_row_chunk(
    const int*   __restrict__ A_row_ptr,
    const int*   __restrict__ A_col_idx,
    const float* __restrict__ A_values,
    const int*   __restrict__ B_col_ptr,    // CSC
    const int*   __restrict__ B_row_idx,    // CSC
    const float* __restrict__ B_values,
    const int*   __restrict__ row_list,
    int*         __restrict__ C_row,
    int*         __restrict__ C_col,
    float*       __restrict__ C_val,
    int*         __restrict__ C_nnz,
    const int                  num_rows_in_list,
    const int                  n,
    const int                  max_C_nnz,
    const float                prune_threshold)
{
    if (blockIdx.x >= num_rows_in_list) return;
    const int row = row_list[blockIdx.x];

    const int a_start = A_row_ptr[row];
    const int a_end   = A_row_ptr[row + 1];
    if (a_start == a_end) return;

    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const int b_start = B_col_ptr[j];
        const int b_end   = B_col_ptr[j + 1];
        if (b_start == b_end) continue;

        int   ai = a_start;
        int   bi = b_start;
        float dot = 0.0f;
        while (ai < a_end && bi < b_end) {
            const int ak = A_col_idx[ai];
            const int bk = B_row_idx[bi];
            if (ak == bk) {
                dot += A_values[ai] * B_values[bi];
                ++ai; ++bi;
            } else if (ak < bk) {
                ++ai;
            } else {
                ++bi;
            }
        }
        // Threshold pruning fused (O1) — see spgemm_hash_row.
        if (dot >= prune_threshold) {
            const int pos = atomicAdd(C_nnz, 1);
            if (pos < max_C_nnz) {
                C_row[pos] = row;
                C_col[pos] = j;
                C_val[pos] = dot;
            }
        }
    }
}


// =========================================================================
// KERNEL: threshold_prune  (UNCHANGED - now feeds GPU compactor)
//
// One thread per nonzero.  Marks entries with value < threshold as
// dropped (keep_flag = 0) and survivors as kept (keep_flag = 1).
// =========================================================================
__global__ void threshold_prune(
    float*       __restrict__ values,
    int*         __restrict__ keep_flag,
    const float                threshold,
    const int                  nnz)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;
    if (values[tid] < threshold) {
        values[tid]    = 0.0f;
        keep_flag[tid] = 0;
    } else {
        keep_flag[tid] = 1;
    }
}


// =========================================================================
// KERNEL: prefix_sum_block  (NEW - Improvement 2)
//
// Per-block exclusive prefix sum of an int array (keep_flag in our use).
// Writes the per-block total to block_sums[blockIdx.x] for the host
// (or a follow-up kernel) to scan.  Uses the Hillis-Steele in-place
// scan in shared memory.
// =========================================================================
__global__ void prefix_sum_block(
    const int* __restrict__ input,
    int*       __restrict__ output,
    int*       __restrict__ block_sums,
    const int                n)
{
    __shared__ int smem[BLOCK_SIZE];
    __shared__ int smem_buf[BLOCK_SIZE];

    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int v   = (gid < n) ? input[gid] : 0;
    smem[threadIdx.x] = v;
    __syncthreads();

    // Hillis-Steele inclusive scan.
    for (int off = 1; off < blockDim.x; off <<= 1) {
        if (threadIdx.x >= off) {
            smem_buf[threadIdx.x] = smem[threadIdx.x] + smem[threadIdx.x - off];
        } else {
            smem_buf[threadIdx.x] = smem[threadIdx.x];
        }
        __syncthreads();
        smem[threadIdx.x] = smem_buf[threadIdx.x];
        __syncthreads();
    }

    // Convert inclusive -> exclusive (shift right by one).
    const int incl = smem[threadIdx.x];
    const int excl = incl - v;
    if (gid < n) output[gid] = excl;

    // Last thread writes block total.
    if (threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = incl;
    }
}


// =========================================================================
// KERNEL: prefix_sum_add_offsets  (NEW - Improvement 2)
//
// Adds the scanned block_sums[blockIdx.x] to every entry of the per-block
// exclusive prefix sums, producing a globally correct exclusive prefix
// sum.  `block_offsets` is the EXCLUSIVE scan of the original
// `block_sums` (computed on host - tiny array).
// =========================================================================
__global__ void prefix_sum_add_offsets(
    int*       __restrict__ output,
    const int* __restrict__ block_offsets,
    const int                n)
{
    const int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const int offset = block_offsets[blockIdx.x];
    if (offset != 0) output[gid] += offset;
}


// =========================================================================
// KERNEL: compact_csr_values  (NEW - Improvement 2)
//
// Stream-compacts (col_idx_in, values_in) under keep_flag, writing to
// (col_idx_out, values_out) at positions given by the exclusive prefix
// sum of keep_flag.  One thread per input nonzero.
// =========================================================================
__global__ void compact_csr_values(
    const int*   __restrict__ col_idx_in,
    const float* __restrict__ values_in,
    const int*   __restrict__ keep_flag,
    const int*   __restrict__ prefix_sum,
    int*         __restrict__ col_idx_out,
    float*       __restrict__ values_out,
    const int                  nnz)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;
    if (keep_flag[tid] != 0) {
        const int pos = prefix_sum[tid];
        col_idx_out[pos] = col_idx_in[tid];
        values_out[pos]  = values_in[tid];
    }
}


// =========================================================================
// KERNEL: rebuild_row_ptr  (NEW - Improvement 2)
//
// Maps each old row_ptr boundary to its compacted position via the
// global exclusive prefix sum.  Output has length n+1.
// =========================================================================
__global__ void rebuild_row_ptr(
    const int* __restrict__ old_row_ptr,
    const int* __restrict__ prefix_sum,
    const int                old_nnz,
    const int                new_nnz,
    int*       __restrict__ new_row_ptr,
    const int                n_rows)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > n_rows) return;
    const int boundary = old_row_ptr[tid];
    if (boundary >= old_nnz) {
        new_row_ptr[tid] = new_nnz;
    } else {
        new_row_ptr[tid] = prefix_sum[boundary];
    }
}


// =========================================================================
// KERNEL: topk_column_prune  (UPDATED - bitonic dispatch)
//
// Exact top-k per column with index tie-break.  Uses the original
// count-greater approach when col_len <= bitonic_threshold; wide
// columns are handled by `topk_bitonic_column` (host-side dispatch).
// =========================================================================
__global__ void topk_column_prune(
    const int*   __restrict__ col_ptr,
    float*       __restrict__ values,
    int*         __restrict__ keep_flag,
    const int                  top_k,
    const int                  n_cols,
    const int                  bitonic_threshold)
{
    const int col = blockIdx.x;
    if (col >= n_cols) return;

    const int s = col_ptr[col];
    const int e = col_ptr[col + 1];
    const int len = e - s;
    if (len <= top_k) return;
    // Wide columns are routed to topk_bitonic_column from the host -
    // those launches simply skip narrow columns and vice versa.
    if (len > bitonic_threshold) return;

    for (int i = s + threadIdx.x; i < e; i += blockDim.x) {
        const float v = values[i];
        int greater = 0;
        for (int j = s; j < e; ++j) {
            const float u = values[j];
            if (u > v || (u == v && j < i)) {
                greater++;
                if (greater >= top_k) break;
            }
        }
        if (greater >= top_k) {
            values[i]    = 0.0f;
            keep_flag[i] = 0;
        }
    }
}


// =========================================================================
// KERNEL: topk_bitonic_column  (NEW - Improvement 3)
//
// SMEM-based top-k for wide columns.  One block per column.
//
// Strategy: maintain a sorted (ascending) buffer of size 2*BLOCK_SIZE in
// shared memory.  Each iteration loads a fresh BLOCK_SIZE-chunk of the
// column into the LOWER half (overwriting the previously-smaller
// values), then runs a full bitonic sort over all 2*BLOCK_SIZE entries.
// After the sort, the UPPER half always holds the running top BLOCK_SIZE
// candidates.  At the end, threshold = buf[2*BLOCK_SIZE - K] (the K-th
// largest), and any column entry below that is pruned.
//
// Launch contract: block=(2*BLOCK_SIZE, 1, 1) = (512, 1, 1) threads per
// block.  The kernel uses 4 KB of SMEM (512 floats) plus the bitonic
// sort's __syncthreads() steps.
//
// Complexity per column: O(L) loads + O((log 2K)^2) sort stages per
// chunk = linear-in-L with a small log-squared sorting constant, vs.
// O(L^2 / blockDim.x) for the count-greater kernel.
// =========================================================================
__global__ void topk_bitonic_column(
    const int*   __restrict__ col_ptr,
    float*       __restrict__ values,
    int*         __restrict__ keep_flag,
    const int                  top_k_raw,
    const int                  n_cols,
    const int                  bitonic_threshold)
{
    const int col = blockIdx.x;
    if (col >= n_cols) return;

    const int s = col_ptr[col];
    const int e = col_ptr[col + 1];
    const int len = e - s;
    if (len <= top_k_raw) return;
    if (len <= bitonic_threshold) return;   // routed to count-greater

    // Clamp K to BLOCK_SIZE so the running-top buffer always holds it.
    const int K     = (top_k_raw < BLOCK_SIZE) ? top_k_raw : BLOCK_SIZE;
    const int TOTAL = 2 * BLOCK_SIZE;        // 512

    __shared__ float buf[2 * BLOCK_SIZE];

    // Initialise entire buffer to -INF.  Upper half acts as the
    // running top-BLOCK_SIZE; starts empty (-INF).
    buf[threadIdx.x] = -INFINITY;
    __syncthreads();

    // -- Scan column in chunks of BLOCK_SIZE -----------------------------
    for (int base = 0; base < len; base += BLOCK_SIZE) {
        // Load next chunk into LOWER half (threads 0..BLOCK_SIZE-1).
        if (threadIdx.x < BLOCK_SIZE) {
            const int local = base + threadIdx.x;
            buf[threadIdx.x] = (local < len)
                ? values[s + local]
                : -INFINITY;
        }
        __syncthreads();

        // Full bitonic sort, ASCENDING, over all TOTAL = 2*BLOCK_SIZE
        // entries.  This is the standard Batcher network.  After the
        // sort, the largest values are at the high end of buf, so the
        // upper half preserves the running top BLOCK_SIZE candidates
        // for the next chunk to merge against.
        for (int stage = 2; stage <= TOTAL; stage <<= 1) {
            for (int step = stage >> 1; step > 0; step >>= 1) {
                const int idx     = threadIdx.x;
                const int partner = idx ^ step;
                if (partner > idx && partner < TOTAL) {
                    const float a = buf[idx];
                    const float b = buf[partner];
                    const bool ascending = ((idx & stage) == 0);
                    if (ascending) {
                        if (a > b) {
                            buf[idx]     = b;
                            buf[partner] = a;
                        }
                    } else {
                        if (a < b) {
                            buf[idx]     = b;
                            buf[partner] = a;
                        }
                    }
                }
                __syncthreads();
            }
        }
        // Post-sort: buf is ascending.  Top BLOCK_SIZE in buf[BLOCK_SIZE..TOTAL).
    }

    // K-th largest value sits at index TOTAL - K (ascending sort).
    const float threshold = buf[TOTAL - K];
    __syncthreads();

    // Apply threshold.  Use only the first BLOCK_SIZE threads for the
    // write-back loop; threads BLOCK_SIZE..TOTAL-1 idle here.
    if (threadIdx.x < BLOCK_SIZE) {
        for (int i = s + threadIdx.x; i < e; i += BLOCK_SIZE) {
            const float v = values[i];
            if (v < threshold) {
                values[i]    = 0.0f;
                keep_flag[i] = 0;
            }
        }
    }
}


// =========================================================================
// KERNEL: inflate_values  (UNCHANGED)
// =========================================================================
__global__ void inflate_values(
    float*       __restrict__ values,
    const float                r,
    const int                  nnz)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nnz) {
        values[tid] = powf(values[tid], r);
    }
}


// =========================================================================
// KERNEL: column_sum_segmented  (UNCHANGED)
// =========================================================================
__global__ void column_sum_segmented(
    const int*   __restrict__ col_ptr,
    const float* __restrict__ values,
    float*       __restrict__ col_sums,
    const int                  n_cols)
{
    const int warp_id_in_block = threadIdx.x / WARP_SIZE;
    const int warps_per_block  = blockDim.x  / WARP_SIZE;
    const int lane             = threadIdx.x & (WARP_SIZE - 1);
    const int col              = blockIdx.x * warps_per_block + warp_id_in_block;
    if (col >= n_cols) return;

    const int s = col_ptr[col];
    const int e = col_ptr[col + 1];

    float partial = 0.0f;
    for (int i = s + lane; i < e; i += WARP_SIZE) {
        partial += values[i];
    }
    for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, off);
    }
    if (lane == 0) col_sums[col] = partial;
}


// =========================================================================
// KERNEL: normalize_columns  (UNCHANGED)
// =========================================================================
__global__ void normalize_columns(
    const int*   __restrict__ col_ptr,
    float*       __restrict__ values,
    const float* __restrict__ col_sums,
    const float                epsilon,
    const int                  n_cols)
{
    const int warp_id_in_block = threadIdx.x / WARP_SIZE;
    const int warps_per_block  = blockDim.x  / WARP_SIZE;
    const int lane             = threadIdx.x & (WARP_SIZE - 1);
    const int col              = blockIdx.x * warps_per_block + warp_id_in_block;
    if (col >= n_cols) return;

    const float s = col_sums[col];
    if (s < epsilon) return;

    const int cs = col_ptr[col];
    const int ce = col_ptr[col + 1];
    const float inv = 1.0f / s;
    for (int i = cs + lane; i < ce; i += WARP_SIZE) {
        values[i] *= inv;
    }
}


// =========================================================================
// KERNEL: convergence_frobenius  (UNCHANGED)
// =========================================================================
__global__ void convergence_frobenius(
    const float* __restrict__ M_new_vals,
    const float* __restrict__ M_old_vals,
    double*      __restrict__ partial_sums,
    const int                  nnz)
{
    __shared__ double smem[BLOCK_SIZE];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (tid < nnz) {
        const float diff = M_new_vals[tid] - M_old_vals[tid];
        val = (double)diff * (double)diff;
    }
    smem[threadIdx.x] = val;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
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


def _detect_arch_flag() -> str:
    """Return ``-arch=sm_XY`` for the current device, with a Turing fallback.

    Falls back to ``sm_75`` (RTX 20-series) if PyCUDA cannot probe the
    device - that matches the project's target hardware.
    """
    try:
        cuda.init()
        cc_major, cc_minor = cuda.Device(0).compute_capability()
        return f"-arch=sm_{cc_major}{cc_minor}"
    except Exception:                                   # noqa: BLE001
        return "-arch=sm_75"


def _get_kernels() -> dict[str, Any]:
    """Compile (or fetch from cache) all MCL device kernels."""
    if "mcl" not in _kernel_cache:
        if not PYCUDA_AVAILABLE:
            raise RuntimeError(
                "PyCUDA is required to compile MCL kernels - "
                "install pycuda and ensure NVCC is on PATH."
            )
        arch_flag = _detect_arch_flag()
        mod = SourceModule(
            KERNEL_SOURCE,
            options=[arch_flag, "-O3"],
            no_extern_c=True,
        )
        _kernel_cache["mcl"] = {
            # SpGEMM
            "spgemm_hash":  mod.get_function("spgemm_hash_row"),
            "spgemm":       mod.get_function("spgemm_row_chunk"),
            # Pruning + compaction
            "thresh_prune": mod.get_function("threshold_prune"),
            "prefix_sum":   mod.get_function("prefix_sum_block"),
            "scan_offsets": mod.get_function("prefix_sum_add_offsets"),
            "compact":      mod.get_function("compact_csr_values"),
            "rebuild_rptr": mod.get_function("rebuild_row_ptr"),
            "topk_prune":   mod.get_function("topk_column_prune"),
            "topk_bitonic": mod.get_function("topk_bitonic_column"),
            # Inflate + normalise + convergence
            "inflate":      mod.get_function("inflate_values"),
            "col_sum":      mod.get_function("column_sum_segmented"),
            "col_norm":     mod.get_function("normalize_columns"),
            "convergence":  mod.get_function("convergence_frobenius"),
            "_arch_flag":   arch_flag,
        }
    return _kernel_cache["mcl"]


# ---------------------------------------------------------------------------
# Parameter merging
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# CPU preprocessing helpers
# ---------------------------------------------------------------------------

def _symmetrize_mcl(
    graph_csr: sp.csr_matrix,
    network_type: str,
) -> tuple[sp.csr_matrix, str]:
    """Network-type-aware undirected conversion + self-loops for MCL."""
    nt = str(network_type).lower()
    n  = int(graph_csr.shape[0])

    if nt == "ppi":
        A = graph_csr.astype(np.float32).tocsr()
        note = "PPI: graph used as-is + self-loops"
    else:
        A = (graph_csr + graph_csr.T).astype(np.float32)
        A.data = np.ones_like(A.data, dtype=np.float32)
        A = A.tocsr()
        note = f"{nt.upper()}: binarise(A + Aᵀ) + self-loops"

    eye = sp.eye(n, format="csr", dtype=np.float32)
    M   = (A + eye).tocsr()
    M.sum_duplicates()
    M.data = np.minimum(M.data, np.float32(1.0))
    return M, note


def _to_column_stochastic(csr: sp.csr_matrix) -> sp.csr_matrix:
    """Normalise each column to sum to 1.0, with zero-sum self-loop guard."""
    M_csc = csr.tocsc().astype(np.float64)
    col_sums = np.asarray(M_csc.sum(axis=0)).flatten()

    zero_cols = np.where(col_sums == 0.0)[0]
    if zero_cols.size > 0:
        M_lil = M_csc.tolil()
        for j in zero_cols:
            M_lil[j, j] = 1.0
        M_csc = M_lil.tocsc()
        col_sums = np.asarray(M_csc.sum(axis=0)).flatten()

    inv = sp.diags(1.0 / col_sums, format="csc")
    return (M_csc @ inv).tocsr().astype(np.float32)


# ---------------------------------------------------------------------------
# Row classification (Improvement 5)
# ---------------------------------------------------------------------------

def _classify_rows_by_degree(
    indptr: np.ndarray,
    heavy_thresh: int = HEAVY_ROW_THRESH,
    light_thresh: int = LIGHT_ROW_THRESH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (heavy_rows, medium_rows, light_rows) by row-length tier.

    heavy  : >= heavy_thresh nonzeros  -> inner-product fallback
                                          (hash SMEM would not hold output)
    medium : light_thresh <= L < heavy -> hash SpGEMM, medium hash size
    light  : < light_thresh            -> hash SpGEMM, smallest hash
    Empty rows are dropped from all three lists.
    """
    lens = np.diff(indptr).astype(np.int32)
    heavy = np.where(lens >= heavy_thresh)[0].astype(np.int32)
    medium = np.where(
        (lens >= light_thresh) & (lens < heavy_thresh)
    )[0].astype(np.int32)
    light = np.where((lens > 0) & (lens < light_thresh))[0].astype(np.int32)
    return heavy, medium, light


def _next_pow2(x: int) -> int:
    """Round up to the next power of two (minimum 1)."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


# ---------------------------------------------------------------------------
# Fix 2: Hub-corrected SpGEMM output size estimator
# ---------------------------------------------------------------------------

def _estimate_spgemm_output_size(
    M_csr: sp.csr_matrix,
    safety_factor: float = 1.5,
) -> int:
    """Estimate the output nnz of ``M @ M``.

    With threshold pruning fused into the SpGEMM hash flush (O1), the
    per-row output is bounded by the number of hash buckets whose value
    survived ``prune_threshold`` — typically 10–30 % of the raw
    accumulator entries.  The old hub-correction factor was compensating
    for the fact that ALL accumulator entries had to be materialized;
    with fusion, hub rows still expand but overwhelmingly into
    below-threshold noise that never gets written.

    The estimator therefore no longer applies a p95/avg hub correction
    and defaults to a modest ``safety_factor=1.5`` (O13).  The floor is
    ``nnz // 4`` rather than ``nnz`` because after pruning the output
    can be significantly smaller than the input.

    Parameters
    ----------
    M_csr : sp.csr_matrix
        Input matrix (will be squared).
    safety_factor : float
        Multiplier applied to the estimate.  Default 1.5.

    Returns
    -------
    int
        Estimated output nnz.
    """
    n   = int(M_csr.shape[0])
    nnz = int(M_csr.nnz)
    if nnz == 0 or n == 0:
        return 0
    avg_row = max(1.0, float(nnz) / max(1, n))
    est = int(nnz * avg_row * safety_factor)
    return max(est, nnz // 4)


# ---------------------------------------------------------------------------
# O11: adaptive prune threshold under VRAM pressure
# ---------------------------------------------------------------------------

def _adaptive_prune_threshold(
    M: sp.csr_matrix,
    base_threshold: float,
    max_multiplier: float = 32.0,
) -> tuple[float, str]:
    """Bump ``prune_threshold`` when estimated SpGEMM output exceeds safe VRAM.

    Called at the top of every MCL iteration.  Compares the projected
    SpGEMM output footprint (COO bytes) against a fraction of currently
    free VRAM.  If the estimate exceeds the budget, doubles the
    threshold until the estimate fits or ``max_multiplier`` is reached.

    Returns
    -------
    (adjusted_threshold, note)
        ``note`` is an empty string when no adjustment was needed.
    """
    if not PYCUDA_AVAILABLE:
        return base_threshold, ""
    try:
        free_bytes, _ = cuda.mem_get_info()
    except Exception:                                   # noqa: BLE001
        return base_threshold, ""

    est_nnz   = _estimate_spgemm_output_size(M)
    est_bytes = est_nnz * COO_BYTES_PER_ENTRY
    budget    = max(1, int(free_bytes * VRAM_BUDGET_FRACTION))
    if est_bytes <= budget:
        return base_threshold, ""

    pressure_ratio = est_bytes / float(budget)
    # Each doubling of the threshold roughly halves survivors on the
    # power-law degree distribution typical of biological networks.
    steps = int(math.ceil(math.log2(pressure_ratio)))
    steps = max(1, min(steps, int(math.log2(max_multiplier))))
    # Never bump ABOVE 1.0 — that would prune everything.
    ceiling = min(1.0, max(base_threshold, 1e-6) * max_multiplier)
    new_threshold = max(base_threshold, 1e-6) * (2.0 ** steps)
    new_threshold = min(new_threshold, ceiling)

    note = (
        f"adaptive prune @nnz={M.nnz}: threshold "
        f"{base_threshold:g} -> {new_threshold:g} "
        f"(est {est_bytes/1e6:.0f}MB vs budget {budget/1e6:.0f}MB)"
    )
    return new_threshold, note


# ---------------------------------------------------------------------------
# Out-of-core Stage A: size-adaptive top-k working-set cap
# ---------------------------------------------------------------------------

# Bytes held per surviving matrix entry across the SpGEMM working set:
# the pruned M is uploaded as CSR (int32 col + float32 val = 8 B) and, when
# heavy rows are present, transposed to CSC (another 8 B), so ~16 B/entry
# must fit alongside the output buffer.  Conservative — undercounting would
# defeat the guard.
_WORKING_BYTES_PER_ENTRY: int = 16


def _adaptive_top_k(
    n: int,
    user_top_k: int,
    free_bytes: int,
    vram_fraction: float = 0.35,
    bytes_per_entry: int = _WORKING_BYTES_PER_ENTRY,
) -> tuple[int, str]:
    """Cap ``top_k`` so the pruned working set fits a fraction of free VRAM.

    MCL keeps every iteration's matrix bounded by top-k-per-column: after
    pruning, ``nnz(M) <= n * top_k``.  That bound is what dictates the SpGEMM
    input (CSR + optional CSC) footprint next iteration.  On very large
    graphs a fixed ``top_k`` (default 50) makes the working set exceed VRAM
    before the SpGEMM even starts — the ``cuMemAlloc failed`` seen on the
    multi-million-node scalability graphs.  Shrinking ``top_k`` to the VRAM
    budget lets those graphs run; it costs cluster granularity (fewer
    survivors per column) but only ever binds when the graph is too large for
    the user's value — realistic biological networks are far below the cap
    and are returned unchanged.

    Returns ``(effective_top_k, note)`` where ``note`` is empty when no
    reduction was applied.
    """
    if n <= 0 or user_top_k <= 0 or free_bytes <= 0:
        return user_top_k, ""
    budget = int(free_bytes * vram_fraction)
    max_k  = max(1, budget // (bytes_per_entry * n))
    if user_top_k <= max_k:
        return user_top_k, ""
    note = (
        f"top_k {user_top_k} -> {int(max_k)} to fit working set "
        f"(~{n} x {int(max_k)} entries at {bytes_per_entry} B) in "
        f"{budget/1e6:.0f} MB VRAM budget"
    )
    return int(max_k), note


# ---------------------------------------------------------------------------
# GPU exclusive prefix sum (Improvement 2)
# ---------------------------------------------------------------------------

def _gpu_exclusive_scan(
    d_input,
    n: int,
    kernels: dict[str, Any],
    stream,
) -> tuple[Any, int]:
    """Run an exclusive prefix sum on a length-n int array, GPU-side.

    Returns (d_output, total) where ``d_output`` is the prefix-sum array
    on the device and ``total`` is the sum of all elements (int).

    The single host transfer is a copy of the per-block sums array
    (small: ceil(n / BLOCK_SIZE) ints) which the host scans serially
    (np.cumsum) before re-uploading.  The big arrays (input/output)
    never cross the bus.
    """
    if n <= 0:
        d_out = gpuarray.zeros((1,), np.int32)
        return d_out, 0

    nblocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    d_output    = gpuarray.zeros((n,),       np.int32)
    d_block_sum = gpuarray.zeros((nblocks,), np.int32)

    # Per-block exclusive scan; writes per-block totals.
    kernels["prefix_sum"](
        d_input, d_output, d_block_sum, np.int32(n),
        block=(BLOCK_SIZE, 1, 1),
        grid=(nblocks, 1, 1),
        stream=stream,
    )

    # Host-side exclusive scan over block sums (tiny array).
    stream.synchronize()
    block_sums_host = d_block_sum.get()
    total = int(block_sums_host.sum())
    block_offsets = np.zeros_like(block_sums_host)
    block_offsets[1:] = np.cumsum(block_sums_host[:-1]).astype(np.int32)
    d_block_off = gpuarray.to_gpu(block_offsets.astype(np.int32))

    # Add the per-block offsets back.
    kernels["scan_offsets"](
        d_output, d_block_off, np.int32(n),
        block=(BLOCK_SIZE, 1, 1),
        grid=(nblocks, 1, 1),
        stream=stream,
    )

    # Free intermediate before return.
    try:
        d_block_sum.gpudata.free()
        d_block_off.gpudata.free()
    except Exception:                                   # noqa: BLE001
        pass

    return d_output, total


def _gpu_compact_csr(
    d_row_ptr_in,
    d_col_idx_in,
    d_values_in,
    d_keep_flag,
    n_rows: int,
    nnz: int,
    kernels: dict[str, Any],
    stream,
) -> tuple[Any, Any, Any, int]:
    """GPU-side stream compaction of a CSR triple under ``keep_flag``.

    Returns (d_row_ptr_new, d_col_idx_new, d_values_new, new_nnz).
    Existing input gpuarrays are NOT freed - the caller manages lifetime.
    """
    if nnz == 0:
        d_rp = gpuarray.zeros((n_rows + 1,), np.int32)
        d_ci = gpuarray.zeros((0,), np.int32)
        d_vv = gpuarray.zeros((0,), np.float32)
        return d_rp, d_ci, d_vv, 0

    d_prefix, new_nnz = _gpu_exclusive_scan(d_keep_flag, nnz, kernels, stream)
    if new_nnz == 0:
        d_rp = gpuarray.zeros((n_rows + 1,), np.int32)
        d_ci = gpuarray.zeros((0,), np.int32)
        d_vv = gpuarray.zeros((0,), np.float32)
        try:
            d_prefix.gpudata.free()
        except Exception:                               # noqa: BLE001
            pass
        return d_rp, d_ci, d_vv, 0

    d_col_idx_out = gpuarray.zeros((new_nnz,), np.int32)
    d_values_out  = gpuarray.zeros((new_nnz,), np.float32)
    d_row_ptr_out = gpuarray.zeros((n_rows + 1,), np.int32)

    grid_nnz = ((nnz + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
    kernels["compact"](
        d_col_idx_in, d_values_in, d_keep_flag, d_prefix,
        d_col_idx_out, d_values_out, np.int32(nnz),
        block=(BLOCK_SIZE, 1, 1), grid=grid_nnz, stream=stream,
    )

    grid_rp = (((n_rows + 1) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
    kernels["rebuild_rptr"](
        d_row_ptr_in, d_prefix,
        np.int32(nnz), np.int32(new_nnz),
        d_row_ptr_out, np.int32(n_rows),
        block=(BLOCK_SIZE, 1, 1), grid=grid_rp, stream=stream,
    )

    try:
        d_prefix.gpudata.free()
    except Exception:                                   # noqa: BLE001
        pass
    return d_row_ptr_out, d_col_idx_out, d_values_out, new_nnz


# ---------------------------------------------------------------------------
# Hash-based + inner-product SpGEMM dispatcher (Improvements 1 + 5)
# ---------------------------------------------------------------------------

def _estimate_hash_size(
    avg_row_len_A: float,
    avg_row_len_B: float,
    max_hash_entries: int,
) -> int:
    """Choose a power-of-two SMEM hash size for `spgemm_hash_row`.

    Target: 2 x estimated output nonzeros per A-row (load factor ~= 0.5).
    Clamped to [64, max_hash_entries].
    """
    est = max(1.0, avg_row_len_A * avg_row_len_B * 2.0)
    return max(64, min(_next_pow2(int(est)), max_hash_entries))


def _spgemm_gpu(
    M_csr: sp.csr_matrix,
    kernels: dict[str, Any],
    block_size: int,
    stream_compute,
    stream_transfer,
    shared_mem_per_block: int = DEFAULT_SHARED_MEM_PER_BLOCK,
    prune_threshold: float = 1e-3,
) -> tuple[sp.csr_matrix, str]:
    """Compute ``C = M @ M`` on the GPU.

    Strategy:
      - Classify rows into heavy / medium / light by source row length.
      - heavy rows -> inner-product `spgemm_row_chunk` (needs CSC of B).
      - medium + light rows -> hash `spgemm_hash_row` (CSR of B suffices).
      - Output COO accumulated GPU-side, downloaded once at end of the
        SpGEMM step, then assembled into CSR on CPU.  (A full GPU-side
        COO->CSR sort is left as a future optimisation; for typical MCL
        nnz the single download is bandwidth-cheap relative to the
        compute saved by the hash kernel.)

    The transfer of A's CSR is dispatched on ``stream_transfer`` and
    consumed by the compute kernels on ``stream_compute`` via an
    inter-stream event - no host sync is required between upload and
    kernel launch.

    Returns
    -------
    tuple[sp.csr_matrix, str]
        The product matrix and an overflow warning string (empty when
        no overflow occurred).  Overflow is recovered from internally
        (Fix 1) rather than raised as MemoryError.
    """
    n   = int(M_csr.shape[0])
    nnz = int(M_csr.nnz)
    if nnz == 0:
        return sp.csr_matrix((n, n), dtype=np.float32), ""

    # ---- Async H2D of A=B (we square M, so the same arrays serve as both)
    h_row_ptr = np.ascontiguousarray(M_csr.indptr,  dtype=np.int32)
    h_col_idx = np.ascontiguousarray(M_csr.indices, dtype=np.int32)
    h_values  = np.ascontiguousarray(M_csr.data,    dtype=np.float32)

    d_local: list = []

    def _empty(shape, dtype):
        ga = gpuarray.empty(shape, dtype=dtype)
        d_local.append(ga)
        return ga

    try:
        d_row_ptr = _empty(h_row_ptr.shape, np.int32)
        d_col_idx = _empty(h_col_idx.shape, np.int32)
        d_values  = _empty(h_values.shape,  np.float32)
        cuda.memcpy_htod_async(d_row_ptr.gpudata, h_row_ptr, stream_transfer)
        cuda.memcpy_htod_async(d_col_idx.gpudata, h_col_idx, stream_transfer)
        cuda.memcpy_htod_async(d_values.gpudata,  h_values,  stream_transfer)
        transfer_done = cuda.Event()
        transfer_done.record(stream_transfer)

        # ---- Classify rows --------------------------------------------
        heavy, medium, light = _classify_rows_by_degree(h_row_ptr)

        # ---- Output COO buffer (Fix 2 + Fix 4) ----------------------------
        # Hub-corrected estimate (safety_factor=3.0) prevents the overflow
        # seen on power-law networks where the naive nnz*avg_row formula
        # under-counts fill-in.  The capacity is recalculated from the
        # CURRENT M_csr.nnz on every call (Fix 4 — dynamic per-iteration
        # sizing; nnz shrinks after each prune, so the buffer shrinks too).
        try:
            free_bytes, _ = cuda.mem_get_info()
        except Exception:                               # noqa: BLE001
            free_bytes = 1 << 30
        vram_budget   = int(free_bytes * VRAM_BUDGET_FRACTION)
        cap_estimate  = _estimate_spgemm_output_size(M_csr)
        max_cap       = max(1024, vram_budget // COO_BYTES_PER_ENTRY)
        capacity      = int(min(cap_estimate, max_cap))
        # Floor at nnz // 4 (post-prune output can shrink well below nnz);
        # overflow recovery below handles the rare underestimate.
        capacity      = max(capacity, max(1024, nnz // 4))

        d_C_row = _empty((capacity,), np.int32)
        d_C_col = _empty((capacity,), np.int32)
        d_C_val = _empty((capacity,), np.float32)
        d_C_nnz = _empty((1,),         np.int32)
        cuda.memset_d32(d_C_nnz.gpudata, 0, 1)

        # ---- Kernel dispatch -----------------------------------------
        max_hash_entries = max(64, shared_mem_per_block // HASH_ENTRY_BYTES)
        avg_row_len_full = max(1.0, nnz / max(1, n))

        # Compute hash size tier for medium rows (default).
        hash_size_medium = _estimate_hash_size(
            avg_row_len_A=avg_row_len_full,
            avg_row_len_B=avg_row_len_full,
            max_hash_entries=max_hash_entries,
        )
        hash_size_light  = max(
            64,
            min(_next_pow2(int(avg_row_len_full * avg_row_len_full * 2)),
                max_hash_entries // 2),
        )

        # Wait for A transfer before launching any kernel.
        stream_compute.wait_for_event(transfer_done)

        k_hash  = kernels["spgemm_hash"]
        k_inner = kernels["spgemm"]

        # --- Hash kernel for medium rows ------------------------------
        # prune_threshold is fused (O1): buckets < threshold are never
        # written to the COO output, shrinking peak capacity.
        if medium.size > 0:
            d_list = gpuarray.to_gpu(medium)
            d_local.append(d_list)
            smem_bytes = hash_size_medium * HASH_ENTRY_BYTES
            k_hash(
                d_row_ptr, d_col_idx, d_values,    # A (= M)
                d_row_ptr, d_col_idx, d_values,    # B (= M)
                d_list,
                d_C_row, d_C_col, d_C_val, d_C_nnz,
                np.int32(medium.size),
                np.int32(hash_size_medium),
                np.int32(capacity),
                np.float32(prune_threshold),
                block=(block_size, 1, 1),
                grid=(medium.size, 1, 1),
                shared=smem_bytes,
                stream=stream_compute,
            )

        # --- Hash kernel for light rows (smaller hash) ----------------
        if light.size > 0:
            d_list = gpuarray.to_gpu(light)
            d_local.append(d_list)
            smem_bytes = hash_size_light * HASH_ENTRY_BYTES
            k_hash(
                d_row_ptr, d_col_idx, d_values,
                d_row_ptr, d_col_idx, d_values,
                d_list,
                d_C_row, d_C_col, d_C_val, d_C_nnz,
                np.int32(light.size),
                np.int32(hash_size_light),
                np.int32(capacity),
                np.float32(prune_threshold),
                block=(block_size, 1, 1),
                grid=(light.size, 1, 1),
                shared=smem_bytes,
                stream=stream_compute,
            )

        # --- Inner-product fallback for heavy rows --------------------
        if heavy.size > 0:
            # Build a CSC view of M for the inner-product path.
            M_csc = M_csr.tocsc()
            h_B_col_ptr = np.ascontiguousarray(M_csc.indptr,  dtype=np.int32)
            h_B_row_idx = np.ascontiguousarray(M_csc.indices, dtype=np.int32)
            h_B_values  = np.ascontiguousarray(M_csc.data,    dtype=np.float32)
            d_B_col_ptr = _empty(h_B_col_ptr.shape, np.int32)
            d_B_row_idx = _empty(h_B_row_idx.shape, np.int32)
            d_B_values  = _empty(h_B_values.shape,  np.float32)
            cuda.memcpy_htod_async(d_B_col_ptr.gpudata, h_B_col_ptr,
                                    stream_transfer)
            cuda.memcpy_htod_async(d_B_row_idx.gpudata, h_B_row_idx,
                                    stream_transfer)
            cuda.memcpy_htod_async(d_B_values.gpudata,  h_B_values,
                                    stream_transfer)
            csc_done = cuda.Event()
            csc_done.record(stream_transfer)
            stream_compute.wait_for_event(csc_done)

            d_list = gpuarray.to_gpu(heavy)
            d_local.append(d_list)
            k_inner(
                d_row_ptr, d_col_idx, d_values,
                d_B_col_ptr, d_B_row_idx, d_B_values,
                d_list,
                d_C_row, d_C_col, d_C_val, d_C_nnz,
                np.int32(heavy.size),
                np.int32(n),
                np.int32(capacity),
                np.float32(prune_threshold),
                block=(block_size, 1, 1),
                grid=(heavy.size, 1, 1),
                stream=stream_compute,
            )

        stream_compute.synchronize()
        written = int(d_C_nnz.get()[0])

        # Fast-fail on SpGEMM output overflow.  This previously TRUNCATED the
        # product (kept the first ``capacity`` entries, dropped the rest) and
        # continued with "approximate" results — a silent correctness hazard
        # that produced a wrong clustering rather than an error.  MCL's
        # expansion (M @ M) fill-in genuinely does not fit VRAM in this case,
        # so we stop with a clear, actionable message instead.  See the module
        # docstring "VRAM ceiling" note for the per-topology limits.
        if written > capacity:
            need_mb = written * COO_BYTES_PER_ENTRY / (1024 * 1024)
            have_mb = capacity * COO_BYTES_PER_ENTRY / (1024 * 1024)
            raise MemoryError(
                "MCL gpu: SpGEMM expansion output exceeds VRAM — the pruned "
                f"M @ M product needs {written:,} entries (~{need_mb:.0f} MB) "
                f"but only {capacity:,} (~{have_mb:.0f} MB) fit the VRAM "
                "budget, so the graph is too large for MCL at this VRAM. "
                "MCL squares the matrix each iteration and is the most "
                "memory-intensive of the six algorithms. Options: use a "
                "higher-VRAM device, raise prune_threshold, or lower "
                "top_k_per_column."
            )

        if written == 0:
            return sp.csr_matrix((n, n), dtype=np.float32), ""

        rows_host = d_C_row.get()[:written]
        cols_host = d_C_col.get()[:written]
        vals_host = d_C_val.get()[:written]

        out = sp.coo_matrix(
            (vals_host, (rows_host, cols_host)),
            shape=(n, n), dtype=np.float32,
        ).tocsr()
        out.sum_duplicates()
        return out, ""

    finally:
        for arr in d_local:
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# GPU-native prune (threshold + GPU compaction + top-k dispatch)
# ---------------------------------------------------------------------------

def _prune_gpu(
    M: sp.csr_matrix,
    kernels: dict[str, Any],
    prune_threshold: float,
    top_k: int,
    block_size: int,
    stream_compute,
    stream_transfer,
) -> sp.csr_matrix:
    """Threshold prune + GPU compaction + top-k prune.

    Compaction is GPU-native: ``threshold_prune`` writes a keep_flag,
    ``_gpu_compact_csr`` builds the new CSR with one int crossing the
    bus.  Top-k pruning then runs in CSC space (one block per column),
    using ``topk_bitonic_column`` for wide columns and
    ``topk_column_prune`` for narrow ones.
    """
    n   = int(M.shape[0])
    nnz = int(M.nnz)
    if nnz == 0:
        return M

    # ---- H2D of CSR (async on transfer stream) ---------------------------
    h_row_ptr = np.ascontiguousarray(M.indptr,  dtype=np.int32)
    h_col_idx = np.ascontiguousarray(M.indices, dtype=np.int32)
    h_values  = np.ascontiguousarray(M.data,    dtype=np.float32)
    d_local: list = []

    def _empty(shape, dtype):
        ga = gpuarray.empty(shape, dtype=dtype)
        d_local.append(ga)
        return ga

    try:
        d_row_ptr   = _empty(h_row_ptr.shape, np.int32)
        d_col_idx   = _empty(h_col_idx.shape, np.int32)
        d_values    = _empty(h_values.shape,  np.float32)
        d_keep_flag = _empty((nnz,),          np.int32)

        cuda.memcpy_htod_async(d_row_ptr.gpudata, h_row_ptr, stream_transfer)
        cuda.memcpy_htod_async(d_col_idx.gpudata, h_col_idx, stream_transfer)
        cuda.memcpy_htod_async(d_values.gpudata,  h_values,  stream_transfer)
        tr_done = cuda.Event()
        tr_done.record(stream_transfer)
        stream_compute.wait_for_event(tr_done)

        # ---- Threshold prune --------------------------------------------
        grid_nnz = ((nnz + block_size - 1) // block_size, 1, 1)
        kernels["thresh_prune"](
            d_values, d_keep_flag,
            np.float32(prune_threshold), np.int32(nnz),
            block=(block_size, 1, 1), grid=grid_nnz,
            stream=stream_compute,
        )

        # ---- GPU compaction --------------------------------------------
        (d_row_ptr_new, d_col_idx_new,
         d_values_new, new_nnz) = _gpu_compact_csr(
            d_row_ptr, d_col_idx, d_values, d_keep_flag,
            n, nnz, kernels, stream_compute,
        )
        # Track for cleanup.
        d_local.extend([d_row_ptr_new, d_col_idx_new, d_values_new])

        if new_nnz == 0:
            return sp.csr_matrix((n, n), dtype=np.float32)

        # ---- Download compacted CSR (for CSC build + top-k) -------------
        # The CSR -> CSC transpose still runs on CPU; this single D2H is
        # the price of staying numpy-portable.  Values + indices are
        # smaller than before threshold (by definition).
        new_row_ptr = d_row_ptr_new.get()
        new_col_idx = d_col_idx_new.get()
        new_values  = d_values_new.get()

        M_pruned = sp.csr_matrix(
            (new_values, new_col_idx, new_row_ptr),
            shape=(n, n), dtype=np.float32,
        )

        if top_k <= 0:
            return M_pruned

        # ---- Top-k prune on CSC ----------------------------------------
        M_csc = M_pruned.tocsc()
        nnz_csc = int(M_csc.nnz)
        if nnz_csc == 0:
            return M_pruned

        h_csc_col_ptr = np.ascontiguousarray(M_csc.indptr,  dtype=np.int32)
        h_csc_row_idx = np.ascontiguousarray(M_csc.indices, dtype=np.int32)
        h_csc_vals    = np.ascontiguousarray(M_csc.data,    dtype=np.float32)

        d_col_ptr2 = _empty(h_csc_col_ptr.shape, np.int32)
        d_vals2    = _empty(h_csc_vals.shape,    np.float32)
        d_keep2    = _empty((nnz_csc,),          np.int32)

        cuda.memcpy_htod_async(d_col_ptr2.gpudata, h_csc_col_ptr,
                                stream_transfer)
        cuda.memcpy_htod_async(d_vals2.gpudata,    h_csc_vals,
                                stream_transfer)
        cuda.memset_d32(d_keep2.gpudata, 1, nnz_csc)        # default keep
        tr_done2 = cuda.Event()
        tr_done2.record(stream_transfer)
        stream_compute.wait_for_event(tr_done2)

        # Dispatch BOTH topk kernels; each kernel skips columns outside
        # its threshold bucket, so they cover the column space jointly.
        kernels["topk_prune"](
            d_col_ptr2, d_vals2, d_keep2,
            np.int32(top_k), np.int32(n),
            np.int32(TOP_K_BITONIC_THRESHOLD),
            block=(block_size, 1, 1), grid=(n, 1, 1),
            stream=stream_compute,
        )
        # Bitonic kernel launches with 2*BLOCK_SIZE = 512 threads per
        # block (see kernel comment) so its SMEM buffer is fully
        # populated.
        kernels["topk_bitonic"](
            d_col_ptr2, d_vals2, d_keep2,
            np.int32(top_k), np.int32(n),
            np.int32(TOP_K_BITONIC_THRESHOLD),
            block=(2 * BLOCK_SIZE, 1, 1), grid=(n, 1, 1),
            stream=stream_compute,
        )

        # ---- GPU compaction of CSC -------------------------------------
        # _gpu_compact_csr operates on flat arrays (it is CSR-name-only;
        # the math is identical for CSC).  Inputs here:
        #   d_col_ptr2     -> "row_ptr" in compaction terms (column-ptr)
        #   d_row_idx_csc  -> "col_idx" in compaction terms (row indices)
        #   d_vals2        -> values
        d_row_idx_csc = _empty(h_csc_row_idx.shape, np.int32)
        cuda.memcpy_htod_async(d_row_idx_csc.gpudata, h_csc_row_idx,
                                stream_transfer)
        tr_done3 = cuda.Event()
        tr_done3.record(stream_transfer)
        stream_compute.wait_for_event(tr_done3)

        (d_csc_cp_new, d_csc_row_new,
         d_csc_vals_new, new_nnz_csc) = _gpu_compact_csr(
            d_col_ptr2, d_row_idx_csc,
            d_vals2, d_keep2,
            n, nnz_csc, kernels, stream_compute,
        )
        d_local.extend([d_csc_cp_new, d_csc_row_new, d_csc_vals_new])

        if new_nnz_csc == 0:
            return sp.csr_matrix((n, n), dtype=np.float32)

        new_csc_cp   = d_csc_cp_new.get()
        new_csc_rows = d_csc_row_new.get()
        new_csc_vals = d_csc_vals_new.get()

        M_csc_pruned = sp.csc_matrix(
            (new_csc_vals, new_csc_rows, new_csc_cp),
            shape=(n, n), dtype=np.float32,
        )
        return M_csc_pruned.tocsr()

    finally:
        for arr in d_local:
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# GPU inflate + column normalize step
# ---------------------------------------------------------------------------

def _inflate_normalize_gpu(
    M: sp.csr_matrix,
    kernels: dict[str, Any],
    inflation: float,
    block_size: int,
    stream_compute,
    stream_transfer,
) -> sp.csr_matrix:
    """In-place ``M[i,j] <- M[i,j]^r`` then column-renormalise."""
    n   = int(M.shape[0])
    nnz = int(M.nnz)
    if nnz == 0:
        return M

    M_csc = M.tocsc()
    h_col_ptr = np.ascontiguousarray(M_csc.indptr,  dtype=np.int32)
    h_row_idx = np.ascontiguousarray(M_csc.indices, dtype=np.int32)
    h_vals    = np.ascontiguousarray(M_csc.data,    dtype=np.float32)

    d_local: list = []
    try:
        d_col_ptr = gpuarray.empty(h_col_ptr.shape, np.int32)
        d_vals    = gpuarray.empty(h_vals.shape,    np.float32)
        d_csums   = gpuarray.zeros((n,),            np.float32)
        d_local.extend([d_col_ptr, d_vals, d_csums])

        cuda.memcpy_htod_async(d_col_ptr.gpudata, h_col_ptr, stream_transfer)
        cuda.memcpy_htod_async(d_vals.gpudata,    h_vals,    stream_transfer)
        tr_done = cuda.Event()
        tr_done.record(stream_transfer)
        stream_compute.wait_for_event(tr_done)

        nnz_csc = h_vals.size
        grid_e = ((nnz_csc + block_size - 1) // block_size, 1, 1)
        kernels["inflate"](
            d_vals, np.float32(inflation), np.int32(nnz_csc),
            block=(block_size, 1, 1), grid=grid_e,
            stream=stream_compute,
        )

        warps_per_block = max(1, block_size // WARP_SIZE)
        grid_c = ((n + warps_per_block - 1) // warps_per_block, 1, 1)
        kernels["col_sum"](
            d_col_ptr, d_vals, d_csums, np.int32(n),
            block=(block_size, 1, 1), grid=grid_c,
            stream=stream_compute,
        )
        kernels["col_norm"](
            d_col_ptr, d_vals, d_csums,
            np.float32(1e-30), np.int32(n),
            block=(block_size, 1, 1), grid=grid_c,
            stream=stream_compute,
        )
        stream_compute.synchronize()

        new_vals = d_vals.get()
    finally:
        for arr in d_local:
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass

    M_new_csc = sp.csc_matrix(
        (new_vals, h_row_idx, h_col_ptr),
        shape=(n, n), dtype=np.float32,
    )
    return M_new_csc.tocsr()


# ---------------------------------------------------------------------------
# GPU Frobenius convergence check
# ---------------------------------------------------------------------------

def _frobenius_diff_gpu(
    M_new: sp.csr_matrix,
    M_old: sp.csr_matrix,
    kernels: dict[str, Any],
    block_size: int,
    stream_compute,
) -> float:
    """``||M_new - M_old||_F`` on GPU (FP64 reduction), inf on pattern mismatch."""
    if (M_new.nnz != M_old.nnz
            or not np.array_equal(M_new.indptr, M_old.indptr)
            or not np.array_equal(M_new.indices, M_old.indices)):
        return float("inf")

    nnz = int(M_new.nnz)
    if nnz == 0:
        return 0.0

    d_local: list = []
    try:
        d_new = gpuarray.to_gpu(np.ascontiguousarray(M_new.data, np.float32))
        d_old = gpuarray.to_gpu(np.ascontiguousarray(M_old.data, np.float32))
        d_local.extend([d_new, d_old])

        nblocks = max(1, (nnz + BLOCK_SIZE - 1) // BLOCK_SIZE)
        d_partial = gpuarray.zeros((nblocks,), np.float64)
        d_local.append(d_partial)

        kernels["convergence"](
            d_new, d_old, d_partial, np.int32(nnz),
            block=(BLOCK_SIZE, 1, 1), grid=(nblocks, 1, 1),
            stream=stream_compute,
        )
        stream_compute.synchronize()

        partial_host = d_partial.get()
    finally:
        for arr in d_local:
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass

    return float(np.sqrt(float(np.sum(partial_host))))


# ---------------------------------------------------------------------------
# Cluster extraction (attractor method)
# ---------------------------------------------------------------------------

def _extract_clusters(M: sp.csr_matrix) -> np.ndarray:
    """Extract cluster labels from a converged MCL matrix.

    Attractor method: each node j is assigned to the attractor row i
    (i.e. `M[i,i] > 0`) whose `M[i, j]` value is largest.  Iterates the
    CSC representation column-by-column so we never densify — the old
    ``todense()`` path allocated `n_attractors × n` floats which OOM'd
    on graphs with >10k attractors.
    """
    n = M.shape[0]
    M_csr = M.tocsr()
    diag = np.asarray(M_csr.diagonal()).flatten()
    attractors = np.where(diag > 0)[0]

    if attractors.size == 0:
        _, labels = csgraph.connected_components(
            M_csr, directed=False, connection="weak"
        )
        return labels.astype(np.int32)

    # Sparse per-column argmax over the attractor rows only.
    # `attr_rank[i] = k` iff row i is the k-th attractor; other rows -1.
    attr_rank = np.full(n, -1, dtype=np.int64)
    attr_rank[attractors] = np.arange(attractors.size, dtype=np.int64)

    M_csc  = M_csr.tocsc()
    indptr = M_csc.indptr
    idx    = M_csc.indices
    data   = M_csc.data

    labels = np.empty(n, dtype=np.int32)
    for j in range(n):
        s, e = int(indptr[j]), int(indptr[j + 1])
        if e == s:
            # No incoming flow — fall back to nearest attractor by index.
            labels[j] = int(attractors[0])
            continue
        col_rows = idx[s:e]
        col_vals = data[s:e]
        # Restrict to entries whose row is an attractor.
        rank = attr_rank[col_rows]
        keep = rank >= 0
        if keep.any():
            kept_vals = col_vals[keep]
            kept_rows = col_rows[keep]
            labels[j] = int(kept_rows[int(np.argmax(kept_vals))])
        else:
            # No attractor row hit this column; assign the overall
            # column-maximum row and rely on the cluster renumbering
            # to fold it in.
            labels[j] = int(col_rows[int(np.argmax(col_vals))])
    return labels


def _renumber_clusters(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary cluster IDs to a compact 0..K-1 range."""
    _, compact = np.unique(labels, return_inverse=True)
    return compact.astype(np.int32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def mcl_gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """MCL - GPU-accelerated via custom PyCUDA kernels.

    Network-type handling
    ---------------------
    GRN / miRNA : binarise(A + Aᵀ) + self-loops + column-stochastic.
    PPI         : graph used as-is  + self-loops + column-stochastic.

    Returns
    -------
    dict - see CLAUDE.md "MCL" result spec.

    Raises
    ------
    RuntimeError
        If PyCUDA is unavailable or no CUDA device can be initialised.
    MemoryError
        If SpGEMM output exceeds the per-chunk capacity.
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA is required for mcl_gpu(). "
            "Install it or use mcl_cpu_single() from "
            "src/algorithms/cpu/single_threaded/mcl.py"
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

    try:
        # ---- Parameter merging ----------------------------------------
        p = _merge_params(params)
        if _GPU_CONFIG_AVAILABLE:
            p = apply_config("mcl", graph_csr, p) or p
            for k, v in _DEFAULT_PARAMS.items():
                p.setdefault(k, v)

        expansion        = int(p["expansion"])
        inflation        = float(p["inflation"])
        prune_threshold  = float(p["prune_threshold"])
        top_k            = int(p["top_k_per_column"])
        max_iter         = int(p["max_iter"])
        convergence_tol  = float(p["convergence_tol"])
        network_type     = str(p.get("network_type", "grn"))
        block_size       = int(p.get("block_size", BLOCK_SIZE))
        if block_size <= 0 or block_size > 1024:
            block_size = BLOCK_SIZE

        n_original = int(graph_csr.shape[0])
        if n_original == 0:
            raise ValueError("Empty graph")

        # Out-of-core Stage A: shrink top_k so the pruned working set fits
        # free VRAM.  On multi-million-node graphs the default top_k makes the
        # SpGEMM input (CSR + CSC) exceed VRAM before the product is even
        # computed; capping it here is what lets those graphs run at all.
        try:
            _free_b0, _ = cuda.mem_get_info()
        except Exception:                               # noqa: BLE001
            _free_b0 = 1 << 30
        top_k, top_k_note = _adaptive_top_k(n_original, top_k, _free_b0)
        if top_k_note:
            logging.info("mcl_gpu: %s", top_k_note)

        # Probe shared-memory budget per block (Turing = 48 KB by default).
        try:
            dev = cuda.Device(0)
            shared_mem_per_block = int(
                dev.get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
            )
        except Exception:                               # noqa: BLE001
            shared_mem_per_block = DEFAULT_SHARED_MEM_PER_BLOCK

        # ---- CPU preprocessing ----------------------------------------
        A_sym, sym_note = _symmetrize_mcl(graph_csr, network_type)
        M = _to_column_stochastic(A_sym)
        if M.dtype != np.float32:
            M = M.astype(np.float32)

        kernels = _get_kernels()

        # ---- Streams + timing -----------------------------------------
        stream_compute  = cuda.Stream()
        stream_transfer = cuda.Stream()
        # stream_prune reserved for future use; kept declared for parity
        # with the documented pipeline.  Threshold prune currently runs
        # on stream_compute because its output feeds the compaction
        # immediately, making a separate stream not useful.
        start_event     = cuda.Event()
        end_event       = cuda.Event()
        start_event.record(stream_compute)

        # Fix 5: track SpGEMM overflow warnings across iterations.
        overflow_warning: str = ""
        # O11: track adaptive-threshold decisions across iterations.
        adaptive_notes: list[str] = []

        # ---- Main iteration loop --------------------------------------
        converged = False
        iterations = 0
        for it in range(max_iter):
            iterations = it + 1
            M_old = M

            # O11: raise prune_threshold this iteration if the current M
            # would blow the SpGEMM VRAM budget.  Reset every iteration so
            # a shrinking M can return to the user's threshold.
            iter_threshold, adapt_note = _adaptive_prune_threshold(
                M, prune_threshold
            )
            if adapt_note:
                adaptive_notes.append(f"iter {iterations}: {adapt_note}")

            # Fix 3 (tightened, O14): Pre-SpGEMM pruning when M is already
            # dense.  Lowered trigger from n*10 to n*5 so we prune earlier
            # on power-law biological networks, keeping the SpGEMM output
            # buffer allocation small.
            M_new = M
            if M_new.nnz > n_original * 5:
                M_new = _prune_gpu(
                    M_new, kernels,
                    prune_threshold=iter_threshold,
                    top_k=top_k,
                    block_size=block_size,
                    stream_compute=stream_compute,
                    stream_transfer=stream_transfer,
                )

            # ---- Expansion: M_new = M ^ e ----
            # For e=2 this is one SpGEMM; e>2 chains successive squares.
            # Buffer capacity is recalculated from current nnz each call
            # (Fix 4 — dynamic per-iteration sizing via _spgemm_gpu).
            # O1: threshold pruning is fused into the SpGEMM hash flush,
            # so iter_threshold directly shrinks the output buffer.
            for _ in range(expansion - 1):
                M_new, ow = _spgemm_gpu(
                    M_new, kernels, block_size,
                    stream_compute=stream_compute,
                    stream_transfer=stream_transfer,
                    shared_mem_per_block=shared_mem_per_block,
                    prune_threshold=iter_threshold,
                )
                if ow:
                    overflow_warning = ow   # keep last non-empty message

            # ---- Prune (threshold + top-k) ----
            M_new = _prune_gpu(
                M_new, kernels,
                prune_threshold=iter_threshold,
                top_k=top_k,
                block_size=block_size,
                stream_compute=stream_compute,
                stream_transfer=stream_transfer,
            )

            # ---- Inflate + column-renormalise ----
            if M_new.nnz > 0:
                M_new = _inflate_normalize_gpu(
                    M_new, kernels,
                    inflation=inflation,
                    block_size=block_size,
                    stream_compute=stream_compute,
                    stream_transfer=stream_transfer,
                )

            # ---- Convergence check ----
            frob = _frobenius_diff_gpu(
                M_new, M_old, kernels, block_size, stream_compute
            )
            M = M_new
            if frob < convergence_tol and frob != float("inf"):
                converged = True
                break

        end_event.record(stream_compute)
        end_event.synchronize()
        elapsed = start_event.time_till(end_event) / 1000.0   # ms -> s

        # ---- Cluster extraction (CPU, NOT timed) ----
        raw_labels = _extract_clusters(M)
        cluster_assignments = _renumber_clusters(raw_labels).tolist()
        num_clusters = (
            int(max(cluster_assignments) + 1) if cluster_assignments else 0
        )

        adaptive_summary = ""
        if adaptive_notes:
            adaptive_summary = (
                f" Adaptive prune fired on {len(adaptive_notes)} iteration(s): "
                f"last='{adaptive_notes[-1]}'."
            )
        top_k_summary = f" {top_k_note}." if top_k_note else ""
        note = (
            f"Graph symmetrised for MCL ({sym_note}). "
            f"Column-stochastic normalisation applied. "
            f"Pruning: threshold={prune_threshold:g}, top_k={top_k}.{top_k_summary} "
            f"Precision: FP32 computation, FP64 convergence check. "
            f"GPU pipeline: threshold-fused hash SpGEMM + GPU-native "
            f"compaction + bitonic top-k "
            f"(arch {kernels.get('_arch_flag', '?')})."
            f"{adaptive_summary}"
        )

        return {
            "algorithm":      "mcl",
            "mode":           "gpu",
            "network_type":   network_type,
            "execution_time": elapsed,
            "num_nodes":      n_original,
            "num_edges":      int(graph_csr.nnz),
            "result": {
                "cluster_assignments": cluster_assignments,
                "num_clusters":        num_clusters,
                "iterations":          iterations,
                "converged":           converged,
                "note":                note,
                # Fix 5: user-visible overflow guidance (empty string = clean run).
                "overflow_warning":    overflow_warning,
                # O11: per-iteration adaptive-prune events (empty if none).
                "adaptive_prune_events": adaptive_notes,
            },
        }

    except cuda.LogicError as e:
        logging.error("CUDA error in mcl_gpu: %s", e)
        raise
    except MemoryError as mem_exc:
        # Two ways to get here: (1) the explicit fast-fail raise on SpGEMM
        # output overflow (message already actionable), or (2) a raw
        # gpuarray.empty() / cuMemAlloc failure when an intermediate genuinely
        # exceeds VRAM.  For (2) the underlying "cuMemAlloc failed: out of
        # memory" is cryptic, so re-raise with the concrete MCL VRAM-ceiling
        # guidance.  MCL squares the matrix each iteration; its fill-in is the
        # binding constraint, and the reachable size depends on degree
        # variance (see module docstring "VRAM ceiling").
        msg = str(mem_exc)
        if "MCL gpu:" in msg:
            raise                                       # already actionable
        raise MemoryError(
            "MCL gpu: VRAM exhausted during the expansion (M @ M) — MCL is "
            "the most memory-intensive of the six algorithms and its fill-in "
            "does not fit this GPU for a graph of this size/density. On a 6 GB "
            "GPU it scales to roughly ~1M nodes (scale-free), ~1.6M (random), "
            "~3.3M (small-world); the exact limit depends on degree variance. "
            "Use a higher-VRAM device, raise prune_threshold, or lower "
            f"top_k_per_column. (Underlying error: {msg})"
        ) from mem_exc
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
    """Runner entry point - preserves the legacy ``{output, extra_params}`` shape."""
    p = _merge_params(params)
    full = mcl_gpu(graph_csr, p)
    return {"output": full, "extra_params": p}


# ---------------------------------------------------------------------------
# Overflow-recovery verification test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import networkx as nx

    print("=" * 60)
    print("MCL SpGEMM overflow recovery test")
    print("=" * 60)

    # Build a Barabasi-Albert (scale-free) graph with m=8 new edges per node.
    # n=500, m=8 -> avg_degree ~16, heavy hubs -> was reliably overflowing.
    G = nx.barabasi_albert_graph(500, 8, seed=42)
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float32)
    n = A.shape[0]

    print(f"Graph: n={n}, nnz={A.nnz}, avg_degree={A.nnz / n:.1f}")

    try:
        result = mcl_gpu(A, {"prune_threshold": 0.001, "network_type": "ppi"})
        res = result["result"]
        assignments = res["cluster_assignments"]
        overflow_warning = res.get("overflow_warning", "")

        assert len(assignments) == n, (
            f"cluster_assignments length {len(assignments)} != n={n}"
        )
        num_clusters = res["num_clusters"]
        assert num_clusters >= 1, f"Expected at least 1 cluster, got {num_clusters}"

        if overflow_warning:
            print(f"\noverflow_warning triggered:\n  {overflow_warning}\n")
        else:
            print("\nNo overflow occurred (estimate was sufficient).")

        print(
            f"clusters={num_clusters}, "
            f"iterations={res['iterations']}, "
            f"converged={res['converged']}"
        )
        print("\nOverflow recovery test: PASS")

    except MemoryError as e:
        print(f"\nOverflow recovery test: FAIL — MemoryError was raised: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print(f"\nOverflow recovery test: FAIL — unexpected error: {e}")
        raise
