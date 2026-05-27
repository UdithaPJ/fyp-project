"""
algorithms/rwr.py — Random Walk with Restart for Biological Network Diffusion
==============================================================================

Biological context
------------------
Random Walk with Restart propagates probability mass from a seed set
across a network.  At each step the walker follows an out-edge with
probability (1 − r) or teleports back to a seed with probability r.
The steady-state vector ranks every node by its diffusion-influence
relative to the seeds:

  GRN    — seed with disease-associated TFs (TP53 in cancer, NF-κB in
           inflammation); top-ranked genes are candidate downstream
           regulatory targets, potentially many hops away from any seed.
  PPI    — seed with known disease proteins; top-ranked proteins are
           candidate disease modifiers (network medicine workflows).
  miRNA  — seed with miRNAs of interest; top-ranked genes are
           candidate co-targeted effectors of the miRNA panel.

The restart probability r is the key locality knob:
  r ≈ 0.7 → tight local neighbourhood
  r ≈ 0.3 → standard compromise (default)
  r ≈ 0.1 → broad diffusion across the network

Steady-state equation:
    p* = (1 − r) · W · p* + r · p₀

W is the column-stochastic transition matrix (columns sum to 1) and p₀
is the seed distribution (uniform over seeds; uniform over all nodes
when no seeds are provided).

Network-type-aware transition matrix
------------------------------------
GRN    : directed CSR, column-normalised by out-degree.
         Dangling columns (out_degree == 0) get a self-loop so that
         probability mass is conserved during diffusion.
PPI    : A_sym = A + Aᵀ, binarised, then column-normalised.
         Bidirectional interactions get equal weight per direction.
miRNA  : directed bipartite CSR (miRNA → gene), column-normalised.
         Gene-target nodes (out_degree == 0) get self-loops.

Multi-seed-set support
----------------------
``seed_nodes`` accepts either a flat list (single RWR run) or a
list-of-lists (one RWR per inner list, scores averaged across runs to
form a consensus influence vector).  This averaging is biologically
useful when comparing multiple disease-associated panels — genes that
are convergently influenced by all panels rank highest.

Parameter guide
---------------
restart_prob (float, default 0.3)   Teleport probability per step.
max_iter     (int,   default 100)   Hard iteration cap.
tolerance    (float, default 1e-6)  L1-norm early-stop threshold.
seed_nodes   (list[int] or          Seeds for p₀.
              list[list[int]])
network_type (str,   default "grn") One of "grn", "ppi", "mirna".
block_size   (int,   default 256)   CUDA block dimension.
"""

# ── GPU / CUDA-optimised implementation (PyCUDA custom kernels) ──────────
# Source:    biological_network_framework/algorithms/rwr.py
# Requires:  pycuda (with a working NVCC toolchain)
# Used by:   webapp routes, GPU benchmarking (src.benchmarking.benchmark)
# Modes:     _gpu only — this module is GPU-exclusive.
# CPU-only counterparts (benchmarking only — never import in webapp):
#   src.algorithms.cpu.single_threaded.rwr
#   src.algorithms.cpu.multi_threaded.rwr
#
# Three PyCUDA kernels (one SourceModule, compiled once, cached):
#   rwr_spmv_restart           — fused SpMV + restart update for single
#                                seed set.  Three-tier degree-aware
#                                scheduling (thread / warp / block);
#                                restart term added by thread 0 only,
#                                after the reduction.  Uses __ldg() on
#                                p reads for read-only cache routing.
#   l1_convergence_rwr         — Σ |p_new − p| per block via warp
#                                shuffles + final warp reduction in
#                                shared memory (same pattern as
#                                pagerank.compute_l1_convergence).
#   rwr_spmv_restart_batched   — batched multi-seed kernel.
#                                gridDim = (n, B, 1); p / p₀ / p_new
#                                stored as [n × B] row-major; used only
#                                when 1 < B <= MAX_BATCH = 4.
#
# Compilation: -arch=sm_75 (RTX 20-series, Turing).
# Context: retain_primary_context().push() pattern.
# Does NOT silently fall back to CPU — raises RuntimeError /
# MemoryError / cuda.LogicError so the runner can surface the failure.
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
    "restart_prob": 0.3,
    "max_iter":     100,
    "tolerance":    1e-6,
    "seed_nodes":   [],
    "network_type": "grn",
    "block_size":   256,
}

BLOCK_SIZE: int = 256
WARP_SIZE: int  = 32
MAX_BATCH: int  = 4                # batched kernel ceiling
_TOP_NODES: int = 20
_TOP_SEEDS: int = 10
VRAM_BUDGET_FRACTION: float = 0.80


# ---------------------------------------------------------------------------
# CUDA kernel source (three kernels, single SourceModule)
# ---------------------------------------------------------------------------

KERNEL_SOURCE = r"""
extern "C" {

#define BLOCK_SIZE       256
#define WARP_SIZE        32
#define WARPS_PER_BLOCK  (BLOCK_SIZE / WARP_SIZE)

// =========================================================================
// KERNEL 1: rwr_spmv_restart
//
// Fused SpMV + restart for a single seed set:
//   p_new[i] = (1 − r) * Σ_j W[i,j] * p[j]  +  r * p0[i]
//
// One block per node i.  Three-tier degree-aware scheduling:
//   LOW  (deg < 32)         : thread 0 only, serial scan
//   MED  (32 <= deg < 256)  : first warp, stride-32 + shfl_down_sync
//   HIGH (deg >= 256)       : full block, stride-256 + shared mem tree
//
// Restart term r * p0[i] is added by thread 0 ONLY, after the
// reduction, in the same write that stores p_new[i].  Adding it from
// every thread would multiply the contribution by the active-thread
// count — a silent correctness bug.
//
// __ldg(&p[col]) routes the p neighbour reads through the read-only
// texture cache, reducing pressure on the L1 data cache (p is read
// many times per iteration but never written by this kernel).
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

    // Zero-degree row: SpMV contribution is 0; emit pure restart.
    if (degree == 0) {
        if (threadIdx.x == 0) {
            p_new[node_i] = r * p0[node_i];
        }
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
// KERNEL 2: l1_convergence_rwr
//
// Σ |p_new[i] − p[i]| per block via warp-shuffle intra-warp reduction
// then a final warp-shuffle across the WARPS_PER_BLOCK partial sums in
// shared memory.  Same pattern as pagerank.compute_l1_convergence.
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
// KERNEL 3: rwr_spmv_restart_batched
//
// Batched multi-seed RWR.  gridDim = (n, B, 1).  Each block handles
// one (node_i, seed_b) pair; the same three-tier dispatch as the
// single-seed kernel runs inside.  p / p₀ / p_new are stored as
// [n × B] row-major flat arrays: entry (node, seed) lives at
// node * B + seed.
//
// Used only when 1 < B <= MAX_BATCH (= 4).  For larger batches the
// Python driver loops the single-seed kernel serially to avoid the
// O(n · B) VRAM cost of stacked p vectors.
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
        if (threadIdx.x == 0) {
            p_new[out_idx] = r * p0[out_idx];
        }
        return;
    }

    // ---- LOW tier --------------------------------------------------------
    if (degree < WARP_SIZE) {
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int j = row_start; j < row_end; ++j) {
                const int c = col_idx[j];
                sum += w_values[j] * __ldg(&p[c * B + seed_b]);
            }
            p_new[out_idx] = one_minus_r * sum + r * p0[out_idx];
        }
        return;
    }

    // ---- MED tier --------------------------------------------------------
    if (degree < BLOCK_SIZE) {
        if (threadIdx.x < WARP_SIZE) {
            float partial = 0.0f;
            for (int j = row_start + threadIdx.x; j < row_end; j += WARP_SIZE) {
                const int c = col_idx[j];
                partial += w_values[j] * __ldg(&p[c * B + seed_b]);
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

    // ---- HIGH tier -------------------------------------------------------
    float partial = 0.0f;
    for (int j = row_start + threadIdx.x; j < row_end; j += BLOCK_SIZE) {
        const int c = col_idx[j];
        partial += w_values[j] * __ldg(&p[c * B + seed_b]);
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
# Module-level kernel cache
# ---------------------------------------------------------------------------

_kernel_cache: dict[str, dict[str, Any]] = {}


def _get_kernels() -> dict[str, Any]:
    """Compile (or fetch from cache) the three RWR device kernels."""
    if "rwr" not in _kernel_cache:
        if not PYCUDA_AVAILABLE:
            raise RuntimeError(
                "PyCUDA is required to compile RWR kernels — "
                "install pycuda and ensure NVCC is on PATH."
            )
        mod = SourceModule(
            KERNEL_SOURCE,
            options=["-arch=sm_75"],        # RTX 20-series Turing
            no_extern_c=True,
        )
        _kernel_cache["rwr"] = {
            "spmv_restart":         mod.get_function("rwr_spmv_restart"),
            "l1_conv":              mod.get_function("l1_convergence_rwr"),
            "spmv_restart_batched": mod.get_function("rwr_spmv_restart_batched"),
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

    RWR update: ``p_new = (1 − r) · W · p + r · p₀``.  W must be
    column-stochastic (every column sums to 1) so that the random
    walk preserves probability mass.

    Behaviour
    ---------
    GRN    : directed CSR as-is; ``W = (D⁻¹ · A).T``.
    PPI    : ``A_sym = A + Aᵀ``, binarised, then column-normalised.
    miRNA  : directed bipartite CSR as-is; same as GRN.

    Dangling-column fix
    -------------------
    For directed graphs (GRN / miRNA), nodes with out_degree == 0
    produce all-zero columns in W (mass disappears under SpMV).
    These columns get a self-loop ``W[j,j] = 1`` — the absorbing
    state preserves mass; the restart term still works on it.
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

    # Column-stochastic: each row of A is divided by its row sum (out-degree),
    # then transposed so that column j of W carries node j's outgoing weights.
    out_degrees = np.asarray(A.sum(axis=1), dtype=np.float64).flatten()
    safe_degs   = np.where(out_degrees == 0.0, 1.0, out_degrees)
    D_inv       = sp.diags(1.0 / safe_degs, format="csr")
    W           = (D_inv @ A).T.tocsr().astype(np.float32)

    # Repair dangling columns: zero-sum columns → self-loop = 1.0.
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
    """Permute W's rows and columns so similar-degree nodes are adjacent.

    Returns ``(W_reord, perm)`` where ``perm[reord_i] = orig_i``.
    Reduces warp divergence inside the three-tier dispatch and
    improves cache locality.

    NOTE: optional, ``reorder_nodes=False`` by default for correctness
    simplicity.  Future optimisation: integrate Cuthill-McKee or RCM
    bandwidth-reduction for an even better access pattern.
    """
    degrees = np.diff(W.indptr).astype(np.int32)
    perm    = np.argsort(degrees).astype(np.int32)        # perm[reord] = orig
    # Permute rows then columns.
    W_row   = W[perm, :].tocsr()
    W_full  = W_row[:, perm].tocsr()
    return W_full.astype(np.float32), perm


def _estimate_rwr_vram(n: int, nnz: int, batch_size: int = 1) -> int:
    """Conservative VRAM estimate (bytes) for RWR working set."""
    csr_b      = (n + 1 + 2 * nnz) * 4
    pr_b       = n * 4 * (2 + batch_size)                 # p, p_new, B × p₀
    partial_b  = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE) * 4
    return int(csr_b + pr_b + partial_b)


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
# Main entry point
# ---------------------------------------------------------------------------

def rwr_gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """RWR — GPU-accelerated via custom PyCUDA kernels.

    No CuPy dependency.  Fused SpMV + restart in a single kernel,
    with three-tier degree-aware scheduling and a batched variant
    for multi-seed-set workloads (B ≤ MAX_BATCH).

    Returns
    -------
    dict — see CLAUDE.md "RWR" result spec (outer envelope +
    ``result`` sub-dict with scores, top_nodes, top_seeds,
    iterations, plus optional batch_results when multi-seed).

    Raises
    ------
    RuntimeError
        If PyCUDA is unavailable or no CUDA device can be initialised.
    MemoryError
        If GPU allocation fails.
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA is required for rwr_gpu(). "
            "Install it or use rwr_cpu_single() from "
            "src/algorithms/cpu/single_threaded/rwr.py"
        )

    # Push the device PRIMARY context unconditionally — same rationale as
    # hits.py / louvain.py / mcl.py / pagerank.py.
    pushed_ctx = None
    try:
        cuda.init()
        if cuda.Device.count() <= 0:
            raise RuntimeError("No CUDA device available")
        pushed_ctx = cuda.Device(0).retain_primary_context()
        pushed_ctx.push()
    except cuda.LogicError as e:
        raise RuntimeError(f"CUDA initialisation failed: {e}") from e

    d_buffers: list = []   # tracked for guaranteed cleanup

    try:
        # ---- Parameter merging ----------------------------------------
        p = _merge_params(params)
        if _GPU_CONFIG_AVAILABLE:
            p = apply_config("rwr", graph_csr, p) or p
            for k, v in _DEFAULT_PARAMS.items():
                p.setdefault(k, v)

        restart_prob  = float(p["restart_prob"])
        max_iter      = int(p["max_iter"])
        tolerance     = float(p["tolerance"])
        seed_nodes    = p.get("seed_nodes") or []
        network_type  = str(p.get("network_type", "grn"))
        block_size    = int(p.get("block_size", BLOCK_SIZE))
        if block_size <= 0 or block_size > 1024:
            block_size = BLOCK_SIZE
        use_chunking  = bool(p.get("use_chunking", False))
        use_zero_copy = bool(p.get("use_zero_copy", False))
        reorder_nodes = bool(p.get("reorder_nodes", False))

        n = int(graph_csr.shape[0])
        if n == 0:
            raise ValueError("Empty graph")

        one_minus_r = np.float32(1.0 - restart_prob)
        r_val       = np.float32(restart_prob)

        # ---- Detect batched input -------------------------------------
        # seed_nodes = [1, 2, 3]      → single seed set
        # seed_nodes = [[1,2], [3,4]] → multiple seed sets
        if (len(seed_nodes) > 0
                and isinstance(seed_nodes[0], (list, tuple, np.ndarray))):
            seed_sets = [list(s) for s in seed_nodes]
        else:
            seed_sets = [list(seed_nodes)]
        batch_size = len(seed_sets)

        # ---- Build transition matrix (CPU, network-type aware) --------
        W, transition_note = _build_transition_matrix(graph_csr, network_type)

        # ---- Optional node reordering ---------------------------------
        perm = None
        if reorder_nodes:
            W, perm = _reorder_nodes_by_degree(W)
            inv_perm = np.argsort(perm).astype(np.int32)   # orig → reord
            seed_sets = [
                [int(inv_perm[int(s)]) for s in sset if 0 <= int(s) < n]
                for sset in seed_sets
            ]

        # ---- Build p₀ vectors (one per seed set) ----------------------
        p0_list = [_build_p0(sset, n, network_type) for sset in seed_sets]

        # ---- VRAM estimate ---------------------------------------------
        est_bytes = _estimate_rwr_vram(n=n, nnz=int(W.nnz), batch_size=batch_size)
        try:
            free_bytes, _total = cuda.mem_get_info()
        except Exception:                                   # noqa: BLE001
            free_bytes = 1 << 30                            # 1 GB fallback
        if est_bytes > free_bytes:
            raise MemoryError(
                f"RWR GPU needs ~{est_bytes/1e6:.1f} MB but only "
                f"{free_bytes/1e6:.1f} MB free.  Try use_chunking=True "
                f"or a higher-VRAM device."
            )
        if est_bytes > VRAM_BUDGET_FRACTION * free_bytes and use_zero_copy:
            logging.warning(
                "RWR near VRAM limit (%.1f / %.1f MB) — zero-copy "
                "requested but the page-locked path is a TODO; running "
                "regular path.",
                est_bytes / 1e6, free_bytes / 1e6,
            )
        if use_chunking:
            logging.info(
                "rwr_gpu: use_chunking=True requested but chunked path "
                "is a TODO — running regular path."
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

        # ---- Upload W (shared across all seed sets) -------------------
        h_row_ptr  = np.ascontiguousarray(W.indptr,  dtype=np.int32)
        h_col_idx  = np.ascontiguousarray(W.indices, dtype=np.int32)
        h_w_values = np.ascontiguousarray(W.data,    dtype=np.float32)

        d_row_ptr  = _to_gpu(h_row_ptr)
        d_col_idx  = _to_gpu(h_col_idx)
        d_w_values = _to_gpu(h_w_values)

        n_partial_blocks = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE)
        d_partial        = _empty((n_partial_blocks,), np.float32)

        # ---- Choose execution path -----------------------------------
        use_batched = (1 < batch_size <= MAX_BATCH)
        all_scores: list[dict] = []

        if use_batched:
            # ---------- Batched path -----------------------------------
            # Stack p₀ vectors into [n × B] row-major flat array.
            p0_matrix = np.column_stack(p0_list).astype(np.float32)
            h_p0_flat = np.ascontiguousarray(p0_matrix.reshape(-1), np.float32)

            d_p0  = _to_gpu(h_p0_flat)
            d_p   = _to_gpu(h_p0_flat.copy())              # start: p = p₀
            d_pn  = _empty((n * batch_size,), np.float32)

            # Larger partial-sum buffer for n*B-sized convergence reduction.
            n_partial_batched = max(
                1, (n * batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE,
            )
            d_partial_b = _empty((n_partial_batched,), np.float32)

            stream_transfer.synchronize()

            k_batched = kernels["spmv_restart_batched"]
            k_l1      = kernels["l1_conv"]

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
                # Convergence on the entire [n*B] flat array.
                k_l1(
                    d_pn, d_p, d_partial_b, np.int32(n * batch_size),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(n_partial_batched, 1, 1),
                    stream=stream_compute,
                )
                stream_compute.synchronize()
                l1_total = float(np.sum(d_partial_b.get()))

                # Pointer swap (no data copy).
                d_p, d_pn = d_pn, d_p

                # Per-seed-equivalent threshold = tolerance * batch_size.
                if l1_total < tolerance * batch_size:
                    converged = True
                    break

            # Pull final scores: reshape [n*B] → (n, B) and split.
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
            # ---------- Serial-per-seed-set path (B == 1 or B > MAX_BATCH)
            k_spmv = kernels["spmv_restart"]
            k_l1   = kernels["l1_conv"]

            for b_idx, (sset, p0_np) in enumerate(zip(seed_sets, p0_list)):
                h_p0 = np.ascontiguousarray(p0_np, np.float32)

                d_p0 = _to_gpu(h_p0)
                d_p  = _to_gpu(h_p0.copy())
                d_pn = _empty((n,), np.float32)
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
                    k_l1(
                        d_pn, d_p, d_partial, np.int32(n),
                        block=(BLOCK_SIZE, 1, 1),
                        grid=(n_partial_blocks, 1, 1),
                        stream=stream_compute,
                    )
                    stream_compute.synchronize()
                    l1 = float(np.sum(d_partial.get()))

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

                # Release per-seed-set buffers eagerly to bound peak VRAM.
                for arr in (d_p0, d_p, d_pn):
                    try:
                        arr.gpudata.free()
                    except Exception:                       # noqa: BLE001
                        pass
                    if arr in d_buffers:
                        d_buffers.remove(arr)

        end_event.record(stream_compute)
        end_event.synchronize()
        elapsed = start_event.time_till(end_event) / 1000.0   # ms → s

        # ---- Result aggregation (NOT timed) ---------------------------
        if len(all_scores) == 1:
            primary_scores   = all_scores[0]["scores"]
            total_iterations = int(all_scores[0]["iterations"])
            any_converged    = bool(all_scores[0]["converged"])
            batch_results    = None
        else:
            score_matrix = np.array([s["scores"] for s in all_scores],
                                    dtype=np.float64)        # (B, n)
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

        # Top-k nodes / seeds.
        top_nodes = _top_k(primary_scores, _TOP_NODES)
        all_seed_indices = np.array(
            sorted({int(idx) for sset in seed_sets for idx in sset
                    if 0 <= int(idx) < n}),
            dtype=np.int32,
        )
        # If reorder was applied, seed_sets are in reord-space; remap to
        # original space for the reported top_seeds.
        if perm is not None and all_seed_indices.size > 0:
            all_seed_indices = perm[all_seed_indices]
        if all_seed_indices.size > 0:
            top_seeds = _top_k_among(primary_scores, all_seed_indices, _TOP_SEEDS)
        else:
            top_seeds = top_nodes[:_TOP_SEEDS]

        inner: dict = {
            "scores":     [float(x) for x in primary_scores],
            "top_nodes":  top_nodes,
            "top_seeds":  top_seeds,
            "iterations": total_iterations,
            "converged":  any_converged,
            "note":       transition_note,
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

    except cuda.LogicError as e:
        logging.error("CUDA error in rwr_gpu: %s", e)
        raise
    except MemoryError:
        logging.warning(
            "VRAM exhausted in rwr_gpu.  Retry with use_chunking=True "
            "or use_zero_copy=True, or use a higher-VRAM device."
        )
        raise
    finally:
        # Explicit buffer release before context pop.
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
# Runner interface
# ---------------------------------------------------------------------------

def _gpu(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    """Runner entry point — preserves the legacy ``{output, extra_params}``
    shape required by ``src.benchmarking.benchmark`` and
    ``src.runner.algorithm_runner``.
    """
    p = _merge_params(params)
    full = rwr_gpu(graph_csr, p)
    # `full` already has the outer envelope; the benchmark runner only
    # consumes "output" and "extra_params" — keep both layers available.
    return {"output": full, "extra_params": p}
