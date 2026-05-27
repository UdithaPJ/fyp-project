"""
algorithms/mcl.py — Markov Clustering (MCL) for Biological Network Modules
===========================================================================

Biological context
------------------
MCL models a random walk on the network: at each step, mass diffuses
across neighbours (expansion) and is then non-linearly concentrated on
the strongest flows (inflation).  Tightly co-connected groups trap
probability mass into attractor states, revealing modules:

  GRN   — co-regulated gene modules (shared TF inputs, feedback arcs).
  PPI   — protein complexes / functional modules.
  miRNA — miRNA–gene regulons (a miRNA and its co-targeted genes).

The ``top_communities``-style ``cluster_assignments`` output is the
natural input to GO-term enrichment tools (Enrichr, g:Profiler) — each
cluster can be tested for functional coherence.

Why the graph must be undirected and stochastic
------------------------------------------------
MCL operates on a column-stochastic Markov matrix.  For directed
networks the matrix is symmetrised before normalisation:

  GRN / miRNA : A_sym = binarise(A + Aᵀ)  +  self-loops on every node
  PPI         : graph used as-is          +  self-loops on every node

Self-loops guarantee that every column has at least one positive entry
(no absorbing dead-ends), keeping the Markov chain irreducible.

Algorithm
---------
0. Symmetrise + binarise (when needed), add self-loops, column-normalise.
1. Iterate:
     a. Expansion  — M ← M^e            (random walk spreads probability)
     b. Prune      — threshold + top-k  (bound VRAM, keep top flows)
     c. Inflation  — M[i,j] ← M[i,j]^r  (sharpen strong flows)
     d. Column-renormalise              (re-establish column-stochastic)
     e. Convergence — ‖M_new − M_old‖_F < tol
2. Extract clusters from attractor columns (argmax per column).

Parameter guide
---------------
expansion       (int,   default 2)    Matrix power per iteration.
inflation       (float, default 2.0)  Inflation exponent.
                                      Range 1.4 (coarse) … 6.0 (fine).
prune_threshold (float, default 1e-3) Flow values below this are zeroed.
top_k_per_column (int,  default 50)   Keep at most this many entries
                                      per column after threshold prune.
max_iter        (int,   default 100)  Hard iteration cap.
convergence_tol (float, default 1e-4) Frobenius early-stop threshold.
network_type    (str,   default "grn") One of "grn", "ppi", "mirna".
block_size      (int,   default 256)  CUDA block dimension.

References
----------
van Dongen, S. (2000). A Cluster Algorithm for Graphs. CWI Tech. Report.
"""

# ── GPU / CUDA-optimised implementation (PyCUDA custom kernels) ──────────
# Source:    biological_network_framework/algorithms/mcl.py
# Requires:  pycuda (with a working NVCC toolchain)
# Used by:   webapp routes, GPU benchmarking (src.benchmarking.benchmark)
# Modes:     _gpu only — this module is GPU-exclusive.
# CPU-only counterparts (benchmarking only — never import in webapp):
#   src.algorithms.cpu.single_threaded.mcl
#   src.algorithms.cpu.multi_threaded.mcl
#
# Seven PyCUDA kernels (one SourceModule, compiled once, cached):
#   spgemm_row_chunk        — inner-product SpGEMM (CSR × CSC → COO),
#                              chunked to bound VRAM growth from fill-in.
#                              Each output column computed via two-pointer
#                              merge of sorted index lists.
#   threshold_prune         — mark entries < prune_threshold for removal.
#                              CPU compaction follows (avoids GPU prefix
#                              scan dependency).
#   topk_column_prune       — exact top-k per column via count-greater
#                              with index tie-break.  One block per column.
#   inflate_values          — element-wise powf(x, r) in FP32.
#   column_sum_segmented    — warp-per-column reduction via shfl_down_sync.
#   normalize_columns       — warp-per-column in-place divide.
#   convergence_frobenius   — partial ‖M_new − M_old‖_F², FP64
#                              accumulation, one partial per block.
#
# Sparsity-pattern check (CPU, O(nnz) array-equal on indptr+indices) gates
# the Frobenius launch — element-wise diff is only valid when both matrices
# share structure.  Early iterations with drifting patterns skip
# convergence and continue.
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
    logging.warning("PyCUDA not available — mcl_gpu() will raise.")

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
ELLPACK_THRESHOLD: int = 32      # documented; CSR+CSC path is used instead
VRAM_BUDGET_FRACTION: float = 0.70
COO_BYTES_PER_ENTRY: int = 12    # int (row) + int (col) + float (val)


# ---------------------------------------------------------------------------
# CUDA kernel source (all seven kernels, single SourceModule)
# ---------------------------------------------------------------------------

KERNEL_SOURCE = r"""
extern "C" {

#define BLOCK_SIZE 256
#define WARP_SIZE  32

// =========================================================================
// KERNEL 1: spgemm_row_chunk
//
// Inner-product SpGEMM for C = A * B over a row-chunk of A.
//
//   A : CSR  (A_row_ptr, A_col_idx, A_values)
//   B : CSC  (B_col_ptr, B_row_idx, B_values)
//   C : COO  (C_row, C_col, C_val) with atomic-counter C_nnz
//
// One block per output row.  Threads in the block iterate over output
// columns j = 0..n-1; each thread computes the sparse dot product
// dot(A_row_i, B_col_j) using a two-pointer merge of the two sorted
// index lists.  Nonzero results are appended to the COO output via
// atomicAdd on C_nnz.
//
// max_C_nnz is the caller-allocated capacity of the output arrays.
// The host checks C_nnz after sync; if it exceeded max_C_nnz the run
// raises MemoryError with guidance to tighten pruning.
// =========================================================================
__global__ void spgemm_row_chunk(
    const int*   __restrict__ A_row_ptr,
    const int*   __restrict__ A_col_idx,
    const float* __restrict__ A_values,
    const int*   __restrict__ B_col_ptr,
    const int*   __restrict__ B_row_idx,
    const float* __restrict__ B_values,
    int*         __restrict__ C_row,
    int*         __restrict__ C_col,
    float*       __restrict__ C_val,
    int*         __restrict__ C_nnz,
    const int                  chunk_start,
    const int                  chunk_end,
    const int                  n,
    const int                  max_C_nnz)
{
    const int row = blockIdx.x + chunk_start;
    if (row >= chunk_end) return;

    const int a_start = A_row_ptr[row];
    const int a_end   = A_row_ptr[row + 1];
    if (a_start == a_end) return;        // row is all-zero → no output

    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const int b_start = B_col_ptr[j];
        const int b_end   = B_col_ptr[j + 1];
        if (b_start == b_end) continue;  // column j of B is empty

        // Two-pointer merge over two sorted index lists.
        int ai = a_start;
        int bi = b_start;
        float dot = 0.0f;
        while (ai < a_end && bi < b_end) {
            const int ak = A_col_idx[ai];
            const int bk = B_row_idx[bi];
            if (ak == bk) {
                dot += A_values[ai] * B_values[bi];
                ai++; bi++;
            } else if (ak < bk) {
                ai++;
            } else {
                bi++;
            }
        }

        if (dot > 0.0f) {
            const int pos = atomicAdd(C_nnz, 1);
            if (pos < max_C_nnz) {
                C_row[pos] = row;
                C_col[pos] = j;
                C_val[pos] = dot;
            }
            // pos >= max_C_nnz: silently dropped; host detects overflow
            // and raises MemoryError after the kernel completes.
        }
    }
}


// =========================================================================
// KERNEL 2: threshold_prune
//
// One thread per nonzero.  Marks entries with value < threshold as
// dropped (value=0, keep_flag=0) and survivors as kept (keep_flag=1).
// Host-side compaction follows the kernel.
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
// KERNEL 3: topk_column_prune
//
// Exact top-k per column with index tie-break.  One block per column.
// For each entry i in the column, the thread counts how many entries in
// the same column are strictly greater (or equal-and-earlier-by-index).
// If count >= top_k, the entry is below the k-th best and is zeroed.
//
//   keep_flag is updated to 0 for pruned entries (1 for survivors).
//
// O(col_len^2 / blockDim.x) per column.  For typical post-threshold
// column lengths (a few hundred) this is fast; very wide columns are
// rare because threshold_prune ran first.
// =========================================================================
__global__ void topk_column_prune(
    const int*   __restrict__ col_ptr,
    float*       __restrict__ values,
    int*         __restrict__ keep_flag,
    const int                  top_k,
    const int                  n_cols)
{
    const int col = blockIdx.x;
    if (col >= n_cols) return;

    const int s = col_ptr[col];
    const int e = col_ptr[col + 1];
    const int len = e - s;
    if (len <= top_k) return;            // nothing to prune

    for (int i = s + threadIdx.x; i < e; i += blockDim.x) {
        const float v = values[i];
        int greater = 0;
        for (int j = s; j < e; ++j) {
            const float u = values[j];
            // strict > beats; equal-and-earlier-index also beats (tie-break)
            if (u > v || (u == v && j < i)) {
                greater++;
                if (greater >= top_k) break;   // early exit
            }
        }
        if (greater >= top_k) {
            values[i]    = 0.0f;
            keep_flag[i] = 0;
        }
    }
}


// =========================================================================
// KERNEL 4: inflate_values
//
// Element-wise powf(value, r).  FP32 throughout.
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
// KERNEL 5: column_sum_segmented
//
// One warp per column.  Computes Σ values[j] over column j and writes to
// col_sums[col].  Warp-stride read + shfl_down_sync reduction.
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
// KERNEL 6: normalize_columns
//
// One warp per column.  Divides every entry by its column sum, with an
// epsilon guard against degenerate (zero-sum) columns.
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
    if (s < epsilon) return;             // leave column unchanged

    const int cs = col_ptr[col];
    const int ce = col_ptr[col + 1];
    const float inv = 1.0f / s;
    for (int i = cs + lane; i < ce; i += WARP_SIZE) {
        values[i] *= inv;
    }
}


// =========================================================================
// KERNEL 7: convergence_frobenius
//
// Partial ‖M_new − M_old‖_F² per block (FP64 accumulation, FP32 inputs).
// Host sums the partials and takes sqrt.
//
// PRECONDITION: M_new and M_old must share the same sparsity pattern
// (same row_ptr and col_idx).  Host enforces this with an array-equal
// check before launching.
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


def _get_kernels() -> dict[str, Any]:
    """Compile (or fetch from cache) the seven MCL device kernels."""
    if "mcl" not in _kernel_cache:
        if not PYCUDA_AVAILABLE:
            raise RuntimeError(
                "PyCUDA is required to compile MCL kernels — "
                "install pycuda and ensure NVCC is on PATH."
            )
        mod = SourceModule(
            KERNEL_SOURCE,
            options=["-arch=sm_75"],        # RTX 20-series Turing
            no_extern_c=True,
        )
        _kernel_cache["mcl"] = {
            "spgemm":       mod.get_function("spgemm_row_chunk"),
            "thresh_prune": mod.get_function("threshold_prune"),
            "topk_prune":   mod.get_function("topk_column_prune"),
            "inflate":      mod.get_function("inflate_values"),
            "col_sum":      mod.get_function("column_sum_segmented"),
            "col_norm":     mod.get_function("normalize_columns"),
            "convergence":  mod.get_function("convergence_frobenius"),
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
    """Network-type-aware undirected conversion + self-loops for MCL.

    MCL requires a symmetric stochastic matrix.  Self-loops are added on
    every node so each column has at least one positive entry (no
    absorbing dead-ends, Markov chain irreducible).

    Behaviour
    ---------
    GRN    : binarise(A + Aᵀ), add identity, return as float32 CSR.
             Biological note: mutual TF↔gene regulation collapses to a
             single edge (the iteration smooths the structure anyway).
    PPI    : graph used as-is (already undirected), add identity.
    miRNA  : same as GRN — binarise(A + Aᵀ), add identity.
    """
    nt = str(network_type).lower()
    n  = int(graph_csr.shape[0])

    if nt == "ppi":
        A = graph_csr.astype(np.float32).tocsr()
        note = "PPI: graph used as-is + self-loops"
    else:  # grn / mirna / default
        A = (graph_csr + graph_csr.T).astype(np.float32)
        A.data = np.ones_like(A.data, dtype=np.float32)   # binarise
        A = A.tocsr()
        note = f"{nt.upper()}: binarise(A + Aᵀ) + self-loops"

    eye = sp.eye(n, format="csr", dtype=np.float32)
    M   = (A + eye).tocsr()
    M.sum_duplicates()
    # Re-binarise after self-loop addition so existing diagonal entries
    # do not blow up to 2.0 (pre-stochastic-normalisation cleanup).
    M.data = np.minimum(M.data, np.float32(1.0))
    return M, note


def _to_column_stochastic(csr: sp.csr_matrix) -> sp.csr_matrix:
    """Normalise each column to sum to 1.0.

    Columns whose sum is zero (impossible after :func:`_symmetrize_mcl`
    adds self-loops, but kept as a safety net) get a 1.0 placed on their
    diagonal to make them absorbing states.

    Returns
    -------
    float32 CSR matrix.
    """
    M_csc = csr.tocsc().astype(np.float64)
    col_sums = np.asarray(M_csc.sum(axis=0)).flatten()

    # Detect zero-sum columns and inject a diagonal entry (absorbing).
    zero_cols = np.where(col_sums == 0.0)[0]
    if zero_cols.size > 0:
        M_lil = M_csc.tolil()
        for j in zero_cols:
            M_lil[j, j] = 1.0
        M_csc = M_lil.tocsc()
        col_sums = np.asarray(M_csc.sum(axis=0)).flatten()

    inv = sp.diags(1.0 / col_sums, format="csc")
    return (M_csc @ inv).tocsr().astype(np.float32)


def _to_ellpack_r(
    csr: sp.csr_matrix,
    pad_to: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Convert CSR to ELLPACK-R format for coalesced GPU access.

    NOTE: this implementation does NOT use ELLPACK-R for the main
    iteration — CSR + CSC are the canonical SciPy round-trip pair and
    the row-length variance in biological networks (hubs vs. leaves)
    makes ELLPACK padding wasteful.  This helper is kept for future
    work where a narrow-row subset is dispatched to an ELLPACK kernel.

    Returns
    -------
    (ellpack_cols, ellpack_vals, max_row_len)
    """
    indptr  = csr.indptr
    indices = csr.indices
    data    = csr.data
    n       = csr.shape[0]
    row_lens = np.diff(indptr)
    max_row_len = int(row_lens.max()) if row_lens.size else 0
    if pad_to is not None:
        max_row_len = max(max_row_len, int(pad_to))
    ellpack_cols = np.full((n, max_row_len), -1,  dtype=np.int32)
    ellpack_vals = np.zeros((n, max_row_len),     dtype=np.float32)
    for r in range(n):
        s, e = int(indptr[r]), int(indptr[r + 1])
        L = e - s
        if L > 0:
            ellpack_cols[r, :L] = indices[s:e]
            ellpack_vals[r, :L] = data[s:e]
    return ellpack_cols, ellpack_vals, max_row_len


# ---------------------------------------------------------------------------
# CSR / CSC compaction utilities (CPU, post-prune)
# ---------------------------------------------------------------------------

def _compact_csr_by_keep_flag(
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    keep_flag: np.ndarray,
    n: int,
) -> sp.csr_matrix:
    """Drop pruned entries from a CSR triple and rebuild row_ptr.

    Parameters
    ----------
    indptr / indices / values : CSR arrays (host)
    keep_flag                 : per-entry 0/1 mask in CSR order
    n                         : number of rows
    """
    keep_bool = keep_flag.astype(bool)
    new_data  = values[keep_bool]
    new_cols  = indices[keep_bool]
    # Per-row kept counts via np.add.reduceat on the keep_flag.
    starts    = indptr[:-1]
    ends      = indptr[1:]
    # Vectorised per-row sum:
    if values.size > 0:
        # Use cumulative sums of keep_flag, then differences.
        cum = np.concatenate(([0], np.cumsum(keep_flag.astype(np.int64))))
        kept_per_row = (cum[ends] - cum[starts]).astype(np.int32)
    else:
        kept_per_row = np.zeros(n, dtype=np.int32)
    new_indptr = np.concatenate(([0], np.cumsum(kept_per_row))).astype(np.int32)
    return sp.csr_matrix(
        (new_data.astype(np.float32),
         new_cols.astype(np.int32),
         new_indptr),
        shape=(n, n),
    )


def _compact_csc_by_keep_flag(
    col_ptr: np.ndarray,
    row_idx: np.ndarray,
    values: np.ndarray,
    keep_flag: np.ndarray,
    n: int,
) -> sp.csc_matrix:
    """Same as :func:`_compact_csr_by_keep_flag` for CSC."""
    keep_bool = keep_flag.astype(bool)
    new_data  = values[keep_bool]
    new_rows  = row_idx[keep_bool]
    starts    = col_ptr[:-1]
    ends      = col_ptr[1:]
    if values.size > 0:
        cum = np.concatenate(([0], np.cumsum(keep_flag.astype(np.int64))))
        kept_per_col = (cum[ends] - cum[starts]).astype(np.int32)
    else:
        kept_per_col = np.zeros(n, dtype=np.int32)
    new_colptr = np.concatenate(([0], np.cumsum(kept_per_col))).astype(np.int32)
    return sp.csc_matrix(
        (new_data.astype(np.float32),
         new_rows.astype(np.int32),
         new_colptr),
        shape=(n, n),
    )


# ---------------------------------------------------------------------------
# Cluster extraction (attractor method)
# ---------------------------------------------------------------------------

def _extract_clusters(M: sp.csr_matrix) -> np.ndarray:
    """Extract cluster labels from a converged MCL matrix.

    Strategy
    --------
    1. Identify attractor nodes: rows i where M[i, i] > 0 (a strong
       self-flow ⇒ i is a cluster centre).
    2. For each node j, assign it to the attractor i = argmax_{i∈attr} M[i, j].
    3. Fallback: if no attractor exists (degenerate convergence), use
       weakly-connected components of the binarised matrix.
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

    # M[attractors, :] gives, for each attractor row, its flow into every node.
    # Argmax over rows = best attractor for each node (each column).
    M_att_rows = M_csr[attractors, :].tocsc()    # (K, n) — fast column access
    # Dense extraction is acceptable here because K (number of attractors)
    # is typically much smaller than n after MCL converges.
    dense_att = np.asarray(M_att_rows.todense())  # (K, n)
    if dense_att.size == 0:
        return np.zeros(n, dtype=np.int32)
    best_local = np.argmax(dense_att, axis=0).flatten()
    return attractors[best_local].astype(np.int32)


def _renumber_clusters(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary cluster IDs to a compact 0..K-1 range."""
    _, compact = np.unique(labels, return_inverse=True)
    return compact.astype(np.int32)


def _estimate_spgemm_vram(nnz: int, n: int) -> int:
    """Conservative VRAM estimate for one SpGEMM output (bytes).

    Worst-case output is n² entries, but biological networks fill in to
    roughly nnz × avg_degree.  Used to decide whether a single shot or
    multiple chunks are needed.
    """
    avg_deg = max(1, nnz // max(n, 1))
    return int(nnz * avg_deg * COO_BYTES_PER_ENTRY)


# ---------------------------------------------------------------------------
# GPU one-shot SpGEMM (handles chunking internally)
# ---------------------------------------------------------------------------

def _spgemm_gpu(
    M_csr: sp.csr_matrix,
    M_csc: sp.csc_matrix,
    kernels: dict[str, Any],
    block_size: int,
    stream_compute,
) -> sp.csr_matrix:
    """Compute M @ M on the GPU using the inner-product SpGEMM kernel.

    Output starts as a COO buffer sized to the available VRAM budget.
    If the actual output nnz exceeds capacity the function raises
    MemoryError with guidance to tighten pruning parameters.

    Chunking
    --------
    chunk_rows is selected so that the COO output buffer for the chunk
    plus the matrix transfers fits in 70 % of free VRAM.  For graphs
    that easily fit, chunk_rows = n (one shot).
    """
    n   = int(M_csr.shape[0])
    nnz = int(M_csr.nnz)
    if nnz == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)

    # ---- Free VRAM query (driver mem_get_info) ------------------------
    try:
        free_bytes, _total = cuda.mem_get_info()
    except Exception:                                   # noqa: BLE001
        free_bytes = 1 << 30                            # 1 GB fallback
    vram_budget = int(free_bytes * VRAM_BUDGET_FRACTION)

    # Per-chunk COO output capacity: bound by VRAM budget.
    # Worst-case fill: chunk_rows * n entries; biological networks rarely
    # densify that much, so we cap at min(chunk_rows * top_k_used, n).
    # A simpler conservative bound: vram_budget / COO_BYTES_PER_ENTRY.
    max_chunk_output = max(1024, vram_budget // COO_BYTES_PER_ENTRY)
    # Estimate chunk_rows: assume ~density = nnz / (n*n).  Pessimistic.
    density       = max(1.0, nnz / max(1, n))
    chunk_rows    = max(1, min(n, int(max_chunk_output / max(density, 1.0))))

    # ---- Upload A (CSR) and B (CSC) -----------------------------------
    h_A_row_ptr = np.ascontiguousarray(M_csr.indptr,  dtype=np.int32)
    h_A_col_idx = np.ascontiguousarray(M_csr.indices, dtype=np.int32)
    h_A_values  = np.ascontiguousarray(M_csr.data,    dtype=np.float32)
    h_B_col_ptr = np.ascontiguousarray(M_csc.indptr,  dtype=np.int32)
    h_B_row_idx = np.ascontiguousarray(M_csc.indices, dtype=np.int32)
    h_B_values  = np.ascontiguousarray(M_csc.data,    dtype=np.float32)

    d_local: list = []

    def _to_gpu(arr):
        ga = gpuarray.to_gpu(arr)
        d_local.append(ga)
        return ga

    def _empty(shape, dtype):
        ga = gpuarray.empty(shape, dtype=dtype)
        d_local.append(ga)
        return ga

    try:
        d_A_row_ptr = _to_gpu(h_A_row_ptr)
        d_A_col_idx = _to_gpu(h_A_col_idx)
        d_A_values  = _to_gpu(h_A_values)
        d_B_col_ptr = _to_gpu(h_B_col_ptr)
        d_B_row_idx = _to_gpu(h_B_row_idx)
        d_B_values  = _to_gpu(h_B_values)

        # Accumulate COO triples chunk-by-chunk on host.
        coo_rows: list[np.ndarray] = []
        coo_cols: list[np.ndarray] = []
        coo_vals: list[np.ndarray] = []

        k_spgemm = kernels["spgemm"]

        for chunk_start in range(0, n, chunk_rows):
            chunk_end = min(chunk_start + chunk_rows, n)
            # Per-chunk output capacity: heuristic on (rows × density).
            this_chunk = chunk_end - chunk_start
            capacity = min(
                max_chunk_output,
                max(1024, this_chunk * max(int(density * 4), 4)),
            )
            d_C_row = _empty((capacity,), np.int32)
            d_C_col = _empty((capacity,), np.int32)
            d_C_val = _empty((capacity,), np.float32)
            d_C_nnz = _empty((1,),         np.int32)
            cuda.memset_d32(d_C_nnz.gpudata, 0, 1)

            grid = (this_chunk, 1, 1)
            k_spgemm(
                d_A_row_ptr, d_A_col_idx, d_A_values,
                d_B_col_ptr, d_B_row_idx, d_B_values,
                d_C_row, d_C_col, d_C_val, d_C_nnz,
                np.int32(chunk_start), np.int32(chunk_end),
                np.int32(n), np.int32(capacity),
                block=(block_size, 1, 1), grid=grid,
                stream=stream_compute,
            )
            stream_compute.synchronize()

            written = int(d_C_nnz.get()[0])
            if written > capacity:
                raise MemoryError(
                    f"SpGEMM output overflow: {written} entries needed "
                    f"but only {capacity} allocated for rows "
                    f"[{chunk_start}, {chunk_end}).  Raise prune_threshold "
                    f"or lower top_k_per_column."
                )

            if written > 0:
                # D2H of just the populated prefix
                rows_host = d_C_row.get()[:written]
                cols_host = d_C_col.get()[:written]
                vals_host = d_C_val.get()[:written]
                coo_rows.append(rows_host)
                coo_cols.append(cols_host)
                coo_vals.append(vals_host)

            # Free per-chunk output buffers eagerly.
            for arr in (d_C_row, d_C_col, d_C_val, d_C_nnz):
                try:
                    arr.gpudata.free()
                except Exception:                       # noqa: BLE001
                    pass
                d_local.remove(arr)

        # Assemble CSR from accumulated COO triples on CPU.
        if not coo_rows:
            return sp.csr_matrix((n, n), dtype=np.float32)
        rows = np.concatenate(coo_rows)
        cols = np.concatenate(coo_cols)
        vals = np.concatenate(coo_vals)
        out = sp.coo_matrix(
            (vals, (rows, cols)), shape=(n, n), dtype=np.float32
        ).tocsr()
        out.sum_duplicates()
        return out

    finally:
        for arr in d_local:
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# GPU prune step (threshold + top-k)
# ---------------------------------------------------------------------------

def _prune_gpu(
    M: sp.csr_matrix,
    kernels: dict[str, Any],
    prune_threshold: float,
    top_k: int,
    block_size: int,
    stream_compute,
) -> sp.csr_matrix:
    """Apply threshold + top-k pruning on the GPU, return pruned CSR."""
    n   = int(M.shape[0])
    nnz = int(M.nnz)
    if nnz == 0:
        return M

    # -------- Threshold prune (on CSR data) ---------------------------
    h_values  = np.ascontiguousarray(M.data,    dtype=np.float32)
    h_indices = np.ascontiguousarray(M.indices, dtype=np.int32)
    h_indptr  = np.ascontiguousarray(M.indptr,  dtype=np.int32)

    d_local: list = []
    try:
        d_values    = gpuarray.to_gpu(h_values);  d_local.append(d_values)
        d_keep_flag = gpuarray.empty((nnz,), np.int32); d_local.append(d_keep_flag)

        grid = ((nnz + block_size - 1) // block_size, 1, 1)
        kernels["thresh_prune"](
            d_values, d_keep_flag,
            np.float32(prune_threshold), np.int32(nnz),
            block=(block_size, 1, 1), grid=grid,
            stream=stream_compute,
        )
        stream_compute.synchronize()

        keep_host = d_keep_flag.get()
    finally:
        for arr in d_local:
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass

    M_pruned = _compact_csr_by_keep_flag(
        h_indptr, h_indices, h_values, keep_host, n
    )
    if M_pruned.nnz == 0:
        return M_pruned

    # -------- Top-k prune (needs CSC) ---------------------------------
    M_csc = M_pruned.tocsc()
    nnz_csc = int(M_csc.nnz)
    if nnz_csc == 0 or top_k <= 0:
        return M_pruned

    h_csc_col_ptr = np.ascontiguousarray(M_csc.indptr,  dtype=np.int32)
    h_csc_row_idx = np.ascontiguousarray(M_csc.indices, dtype=np.int32)
    h_csc_vals    = np.ascontiguousarray(M_csc.data,    dtype=np.float32)

    d_local2: list = []
    try:
        d_col_ptr = gpuarray.to_gpu(h_csc_col_ptr); d_local2.append(d_col_ptr)
        d_vals    = gpuarray.to_gpu(h_csc_vals);    d_local2.append(d_vals)
        d_keep2   = gpuarray.empty((nnz_csc,), np.int32); d_local2.append(d_keep2)
        # Initialise keep_flag = 1 (everything survives unless kernel zeros it)
        cuda.memset_d32(d_keep2.gpudata, 1, nnz_csc)

        # One block per column.
        kernels["topk_prune"](
            d_col_ptr, d_vals, d_keep2,
            np.int32(top_k), np.int32(n),
            block=(block_size, 1, 1), grid=(n, 1, 1),
            stream=stream_compute,
        )
        stream_compute.synchronize()

        keep2_host = d_keep2.get()
        vals_host  = d_vals.get()
    finally:
        for arr in d_local2:
            try:
                arr.gpudata.free()
            except Exception:                           # noqa: BLE001
                pass

    M_csc_pruned = _compact_csc_by_keep_flag(
        h_csc_col_ptr, h_csc_row_idx, vals_host, keep2_host, n
    )
    return M_csc_pruned.tocsr()


# ---------------------------------------------------------------------------
# GPU inflate + column normalize step
# ---------------------------------------------------------------------------

def _inflate_normalize_gpu(
    M: sp.csr_matrix,
    kernels: dict[str, Any],
    inflation: float,
    block_size: int,
    stream_compute,
) -> sp.csr_matrix:
    """Element-wise ``M[i,j] ← M[i,j] ** r`` then column-renormalise."""
    n   = int(M.shape[0])
    nnz = int(M.nnz)
    if nnz == 0:
        return M

    # Work in CSC for column operations.
    M_csc = M.tocsc()
    h_col_ptr = np.ascontiguousarray(M_csc.indptr,  dtype=np.int32)
    h_row_idx = np.ascontiguousarray(M_csc.indices, dtype=np.int32)
    h_vals    = np.ascontiguousarray(M_csc.data,    dtype=np.float32)

    d_local: list = []
    try:
        d_col_ptr = gpuarray.to_gpu(h_col_ptr);     d_local.append(d_col_ptr)
        d_vals    = gpuarray.to_gpu(h_vals);        d_local.append(d_vals)
        d_csums   = gpuarray.zeros((n,), np.float32); d_local.append(d_csums)

        # ---- Inflate (element-wise) ---------------------------------
        nnz_csc = h_vals.size
        grid_e = ((nnz_csc + block_size - 1) // block_size, 1, 1)
        kernels["inflate"](
            d_vals, np.float32(inflation), np.int32(nnz_csc),
            block=(block_size, 1, 1), grid=grid_e,
            stream=stream_compute,
        )

        # ---- Column sum --------------------------------------------
        warps_per_block = max(1, block_size // WARP_SIZE)
        grid_c = ((n + warps_per_block - 1) // warps_per_block, 1, 1)
        kernels["col_sum"](
            d_col_ptr, d_vals, d_csums, np.int32(n),
            block=(block_size, 1, 1), grid=grid_c,
            stream=stream_compute,
        )

        # ---- Normalize columns -------------------------------------
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

    # Reassemble CSC then convert to CSR for next iteration's SpGEMM left
    # operand.  Pattern is unchanged, only values updated.
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
    """Compute ‖M_new − M_old‖_F on the GPU (FP64 reduction).

    Returns ``inf`` if the two matrices do not share the same sparsity
    pattern.  Caller should treat ``inf`` as "not converged this iter".
    """
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
# Main entry point
# ---------------------------------------------------------------------------

def mcl_gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """MCL — GPU-accelerated via custom PyCUDA kernels.

    No CuPy dependency.  Each iteration uploads the working CSR/CSC views
    from the CPU master copy, runs SpGEMM + prune + inflate + normalize +
    convergence on the GPU, then downloads the updated matrix.

    Network-type handling
    ---------------------
    GRN / miRNA : binarise(A + Aᵀ) + self-loops + column-stochastic.
    PPI         : graph used as-is  + self-loops + column-stochastic.

    Returns
    -------
    dict — see CLAUDE.md "MCL" result spec (outer envelope + ``result``
    sub-dict with cluster_assignments, num_clusters, iterations,
    converged, note).

    Raises
    ------
    RuntimeError
        If PyCUDA is unavailable or no CUDA device can be initialised.
    MemoryError
        If SpGEMM output exceeds the per-chunk capacity; suggests
        raising ``prune_threshold`` or lowering ``top_k_per_column``.
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA is required for mcl_gpu(). "
            "Install it or use mcl_cpu_single() from "
            "src/algorithms/cpu/single_threaded/mcl.py"
        )

    # Push the device PRIMARY context unconditionally — same rationale as
    # hits.py / louvain.py.  retain_primary_context() is ref-counted and
    # coexists with any CuPy code that may be active in the same process.
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

        # ---- CPU preprocessing ----------------------------------------
        A_sym, sym_note = _symmetrize_mcl(graph_csr, network_type)
        M = _to_column_stochastic(A_sym)
        if M.dtype != np.float32:
            M = M.astype(np.float32)

        kernels = _get_kernels()

        # ---- Streams + timing -----------------------------------------
        stream_compute  = cuda.Stream()
        stream_transfer = cuda.Stream()
        start_event     = cuda.Event()
        end_event       = cuda.Event()
        start_event.record(stream_compute)

        # ---- Main iteration loop --------------------------------------
        converged = False
        iterations = 0
        for it in range(max_iter):
            iterations = it + 1
            M_old = M

            # ---- Expansion: M_new = M @ M (... e-1 times for e > 2) ----
            # For e=2 this is one SpGEMM; e=3 chains two.  Each SpGEMM
            # builds the CSC view on demand from the current CSR.
            M_new = M
            for _ in range(expansion - 1):
                M_new_csc = M_new.tocsc()
                M_new = _spgemm_gpu(
                    M_new, M_new_csc, kernels, block_size, stream_compute,
                )

            # ---- Prune (threshold + top-k) ----
            M_new = _prune_gpu(
                M_new, kernels,
                prune_threshold=prune_threshold,
                top_k=top_k,
                block_size=block_size,
                stream_compute=stream_compute,
            )

            # ---- Inflate + column-renormalise ----
            if M_new.nnz > 0:
                M_new = _inflate_normalize_gpu(
                    M_new, kernels,
                    inflation=inflation,
                    block_size=block_size,
                    stream_compute=stream_compute,
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
        elapsed = start_event.time_till(end_event) / 1000.0   # ms → s

        # ---- Cluster extraction (CPU, NOT timed) ----
        raw_labels = _extract_clusters(M)
        cluster_assignments = _renumber_clusters(raw_labels).tolist()
        num_clusters = int(max(cluster_assignments) + 1) if cluster_assignments else 0

        note = (
            f"Graph symmetrised for MCL ({sym_note}). "
            f"Column-stochastic normalisation applied. "
            f"Pruning: threshold={prune_threshold:g}, top_k={top_k}. "
            "Precision: FP32 computation, FP64 convergence check."
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
            },
        }

    except cuda.LogicError as e:
        logging.error("CUDA error in mcl_gpu: %s", e)
        raise
    except MemoryError:
        logging.warning(
            "VRAM exhausted in mcl_gpu. "
            "Raise prune_threshold or lower top_k_per_column to keep "
            "the matrix sparse, or use a higher-VRAM device."
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
    full = mcl_gpu(graph_csr, p)
    # `full` already has the outer envelope; the benchmark runner only
    # consumes "output" and "extra_params" — keep both layers available.
    return {"output": full, "extra_params": p}
