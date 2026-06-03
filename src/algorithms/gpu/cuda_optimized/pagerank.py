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
damping              (float, default 0.85)   Probability of following an edge.
max_iter             (int,   default 100)    Hard iteration cap.
tolerance            (float, default 1e-6)   L1-norm early-stop threshold.
network_type         (str,   default "grn")  One of "grn", "ppi", "mirna".
block_size           (int,   default 256)    CUDA block dimension.
ellpack_fraction     (float, default 0.001)  Fraction of nodes routed to ELLPACK.
                                             REDUCED from 0.05: on power-law
                                             graphs 5 % of 1M nodes × max_degree
                                             50K padded slots × 8 B = 20 GB ELLPACK.
                                             0.1 % keeps it under 400 MB even on
                                             the most skewed BA graphs.
ellpack_max_mb       (float, default 256)    Disable ELLPACK if it would exceed
                                             this many MB (safeguard against
                                             extreme hub degrees).
ellpack_vram_pct     (float, default 0.10)   Also disable ELLPACK if it would
                                             exceed this fraction of free VRAM.
pull_threshold       (int,   default -1)     -1 = auto-detect scale-free graphs
                                             and set threshold at top pull_fraction
                                             in-degree nodes. 0 = disabled.
                                             >0 = explicit in-degree threshold.
pull_fraction        (float, default 0.01)   When pull_threshold=-1, top this
                                             fraction of nodes (by in-degree)
                                             become pull targets (default 1 %).
chunk_vram_pct       (float, default 0.70)   Auto-chunk when estimated VRAM
                                             exceeds this fraction of free VRAM.
use_chunking         (bool,  default False)  Force chunked path.
enable_diagnostics   (bool,  default False)  Log per-phase timing breakdown.
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
import time
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
    "damping":            0.85,
    "max_iter":           100,
    "tolerance":          1e-6,
    "network_type":       "grn",
    "block_size":         256,
    # CHANGE 1: ellpack_fraction 0.05 → 0.001.
    # Root cause of the 13–29 GB VRAM estimates: 5 % of 1M nodes = 50k hubs,
    # and on BA graphs max_degree can be 50k+ → 50k×50k×8 B = 20 GB ELLPACK.
    # 0.1 % keeps the hub set tiny (≈1k nodes on 1M-node graphs) so even with
    # extreme max degrees the ELLPACK stays under the safeguard limits.
    "ellpack_fraction":   0.001,
    # CHANGE 1 (safeguards): disable ELLPACK if it would exceed either of:
    "ellpack_max_mb":     256.0,   # absolute cap in MB
    "ellpack_vram_pct":   0.10,    # fraction of free VRAM
    # CHANGE 2: pull_threshold -1 = auto-detect (was 0 = always disabled).
    # Auto mode enables pull for the top pull_fraction of in-degree nodes
    # on scale-free graphs (detected by max/avg in-degree ratio > 10).
    "pull_threshold":     -1,
    "pull_fraction":      0.01,    # top 1 % in-degree nodes become pull targets
    # CHANGE 4: auto-chunk when estimate > chunk_vram_pct of free VRAM.
    "chunk_vram_pct":     0.70,
    "use_chunking":       False,
    # CHANGE 6: optional per-phase timing diagnostic output.
    "enable_diagnostics": False,
}

BLOCK_SIZE: int     = 256
WARP_SIZE: int      = 32          # hardware-fixed (Turing+); not a tunable
SMEM_BUCKETS: int   = 256         # = BLOCK_SIZE so each thread flushes one bucket
_TOP_REG: int       = 15          # top regulators returned (GRN / miRNA)
_TOP_TGT: int       = 15          # top targets returned   (GRN / miRNA)
_TOP_NODES: int     = 20          # top nodes returned     (PPI)


# ---------------------------------------------------------------------------
# CUDA kernel source  (UNCHANGED — all improvements are in the Python layer)
# ---------------------------------------------------------------------------

KERNEL_SOURCE = r"""
extern "C" {

#define BLOCK_SIZE      256
#define WARP_SIZE       32
#define SMEM_BUCKETS    256
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

// =========================================================================
// KERNEL: initialize_pr (BACKUP - kept for chunked path, not on hot loop)
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
// KERNEL: initialize_pr_with_dangling
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
// KERNEL: scatter_contributions_csr
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
// KERNEL: scatter_contributions_ellpack
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
// KERNEL: distribute_dangling_mass  (BACKUP)
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
// KERNEL: sum_dangling_pr
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
// KERNEL: compute_l1_convergence
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
// KERNEL: reduce_to_scalar_f32
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
// KERNEL: gather_contributions_pull
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
# CHANGE 6: Lightweight per-phase profiler
# ---------------------------------------------------------------------------

class _PerfProfiler:
    """Phase-level wall-clock timer for preprocessing vs GPU execution."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._phases: dict[str, float] = {}
        self._start:  float | None     = None
        self._phase:  str | None       = None

    def begin(self, phase: str) -> None:
        if not self.enabled:
            return
        if self._phase is not None:
            self.end()
        self._phase = phase
        self._start = time.perf_counter()

    def end(self) -> None:
        if not self.enabled or self._phase is None:
            return
        elapsed = time.perf_counter() - (self._start or 0.0)
        self._phases[self._phase] = elapsed
        self._phase = None
        self._start = None

    def record(self, phase: str, elapsed_s: float) -> None:
        """Record a pre-measured duration (e.g. from CUDA events)."""
        if self.enabled:
            self._phases[phase] = elapsed_s

    def summary(self) -> dict[str, float]:
        if self._phase:
            self.end()
        return dict(self._phases)

    def log(self, extra: str = "") -> None:
        if not self.enabled:
            return
        phases = self.summary()
        total = sum(phases.values())
        lines = [f"[PageRank Profiler] total={total*1000:.1f} ms  {extra}"]
        for ph, t in phases.items():
            lines.append(f"  {ph:<28}: {t*1000:8.2f} ms  "
                         f"({100*t/max(total, 1e-12):5.1f} %)")
        logging.info("\n".join(lines))


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
    n  = int(graph_csr.shape[0])
    nt = str(network_type).lower()
    mask = np.zeros(n, dtype=np.uint8)
    if nt == "grn":
        mask[out_degrees > 0.0] = 1
        note = "dangling mass -> regulator nodes only (out_degree > 0)"
    elif nt == "mirna":
        mask[out_degrees > 0.0] = 1
        note = "dangling mass -> miRNA nodes only (out_degree > 0)"
    else:  # ppi (and default)
        mask[:] = 1
        note = "dangling mass -> all nodes (uniform)"
    return mask, note


def _compute_adaptive_hub_threshold(
    out_degrees: np.ndarray,
    target_ellpack_fraction: float = 0.001,   # CHANGE 1: was 0.05
) -> int:
    """Compute a hub threshold so ~target_ellpack_fraction of nodes go to ELLPACK."""
    if out_degrees.size == 0:
        return WARP_SIZE
    deg_int = out_degrees.astype(np.int64)
    pct = (1.0 - max(0.0, min(target_ellpack_fraction, 1.0))) * 100.0
    threshold = int(np.percentile(deg_int, pct))
    threshold = max(threshold, WARP_SIZE)
    max_deg = int(deg_int.max()) if deg_int.size > 0 else WARP_SIZE
    threshold = min(threshold, max_deg)
    threshold = max(threshold, 1)
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


# ---------------------------------------------------------------------------
# CHANGE 1: ELLPACK safeguard helper
# ---------------------------------------------------------------------------

def _build_ellpack_safe(
    graph_csr: sp.csr_matrix,
    out_degrees: np.ndarray,
    hub_threshold: int,
    ellpack_max_bytes: float,
    free_bytes: int,
    ellpack_vram_pct: float,
) -> tuple[dict, dict, bool, float, str]:
    """Build ELLPACK + CSR split with automatic disabling on memory excess.

    CHANGE 1 rationale
    ------------------
    On power-law biological graphs (BA with m=5, n=1M) the naive 5 % hub
    fraction selects 50k nodes.  If any hub has out-degree 50k, the padded
    ELLPACK array is 50k × 50k × 8 B = 20 GB — larger than the whole VRAM
    budget.  This helper disables ELLPACK and falls all nodes back to CSR
    whenever the projected ELLPACK size exceeds either the absolute MB cap
    or the VRAM-fraction cap.  The CSR scatter kernel handles high-degree
    nodes via its HIGH tier (full block + SMEM hash), so correctness is
    preserved.

    Returns
    -------
    ellpack_data, csr_remainder, ellpack_disabled, ellpack_bytes, notes
    """
    ellpack_data, csr_remainder = _build_ellpack(
        graph_csr, out_degrees, hub_threshold,
    )
    num_hubs    = ellpack_data["num_hubs"]
    max_row_len = ellpack_data["max_row_len"]
    ellpack_bytes = float(num_hubs * max_row_len * 8)   # int32 cols + float32 vals

    # Effective cap: smaller of absolute MB limit and VRAM-fraction limit.
    vram_cap = float(free_bytes) * max(0.0, min(ellpack_vram_pct, 1.0))
    limit    = min(ellpack_max_bytes, vram_cap)

    max_hub_degree = 0
    if num_hubs > 0:
        deg_int = np.diff(graph_csr.indptr).astype(np.int32)
        max_hub_degree = int(deg_int[ellpack_data["hub_ids"]].max())

    notes = (
        f"ELLPACK: {num_hubs} hubs, max_hub_degree={max_hub_degree}, "
        f"padded_slots={max_row_len}, "
        f"estimated={ellpack_bytes/1e6:.1f} MB"
    )

    ellpack_disabled = (num_hubs > 0) and (ellpack_bytes > limit)

    if ellpack_disabled:
        notes += (
            f" — DISABLED (exceeds {limit/1e6:.1f} MB limit; "
            f"all nodes fall to CSR high-tier)"
        )
        logging.info("[PageRank] %s", notes)
        # Move ALL non-dangling nodes into the CSR remainder.
        deg_int = np.diff(graph_csr.indptr).astype(np.int32)
        all_nodes = np.where(deg_int > 0)[0].astype(np.int32)
        indptr   = np.ascontiguousarray(graph_csr.indptr,  dtype=np.int32)
        indices  = np.ascontiguousarray(graph_csr.indices, dtype=np.int32)
        data     = np.ascontiguousarray(graph_csr.data,    dtype=np.float32)
        ellpack_data = {
            "hub_ids":     np.zeros((0,), dtype=np.int32),
            "cols":        np.zeros((0,), dtype=np.int32),
            "vals":        np.zeros((0,), dtype=np.float32),
            "max_row_len": 0,
            "num_hubs":    0,
        }
        csr_remainder = {
            "node_ids": all_nodes,
            "num_low":  int(all_nodes.size),
            "row_ptr":  indptr,
            "col_idx":  indices,
            "values":   data,
        }
    else:
        notes += " — ACTIVE"
        logging.info("[PageRank] %s", notes)

    return ellpack_data, csr_remainder, ellpack_disabled, ellpack_bytes, notes


# ---------------------------------------------------------------------------
# CHANGE 2: Automatic pull-threshold detection
# ---------------------------------------------------------------------------

def _auto_pull_threshold(
    in_degrees: np.ndarray,
    pull_fraction: float = 0.01,
    min_threshold: int = 32,
) -> tuple[int, str]:
    """Set pull threshold at the (1 - pull_fraction) in-degree percentile.

    CHANGE 2 rationale
    ------------------
    Pull mode converts atomicAdd contention for high-in-degree "convergence
    hub" nodes into a contention-free sequential gather.  It is only worth
    the overhead (transpose CSR + extra kernel) when the in-degree
    distribution is strongly skewed (power-law), which is the normal case
    for biological networks.

    Detection heuristic: if max_in_degree / mean_in_degree > 10 the graph
    is considered scale-free and pull mode is enabled.  The threshold is
    set at the (1 - pull_fraction) percentile so exactly pull_fraction of
    nodes become pull targets.

    Returns (threshold, note_string).  threshold == 0 means disabled.
    """
    if in_degrees.size == 0:
        return 0, "pull disabled (empty graph)"

    nonzero = in_degrees[in_degrees > 0]
    if nonzero.size < 50:
        return 0, "pull disabled (too few nodes)"

    max_deg = float(nonzero.max())
    avg_deg = float(nonzero.mean())
    ratio   = max_deg / max(avg_deg, 1.0)

    if ratio < 10.0:
        return 0, (f"pull disabled (max/avg in-degree ratio={ratio:.1f} < 10; "
                   f"not a scale-free graph)")

    pct = (1.0 - max(0.0, min(pull_fraction, 0.5))) * 100.0
    threshold = int(np.percentile(in_degrees, pct))
    threshold = max(threshold, min_threshold)

    pull_ids = np.where(in_degrees >= threshold)[0]
    n_pull   = int(pull_ids.size)
    n_edges  = int((in_degrees[pull_ids]).sum()) if n_pull > 0 else 0
    total_edges = int(in_degrees.sum())
    edge_pct = 100.0 * n_edges / max(total_edges, 1)

    note = (
        f"pull AUTO-ENABLED (max/avg={ratio:.1f}); "
        f"threshold={threshold} (top {pull_fraction*100:.1f}% in-degree), "
        f"{n_pull} pull nodes, "
        f"{edge_pct:.1f}% of edges handled via atomic-free gather"
    )
    return threshold, note


# ---------------------------------------------------------------------------
# CHANGE 3: Improved VRAM estimator with detailed breakdown
# ---------------------------------------------------------------------------

def _estimate_pagerank_vram(
    n: int,
    nnz: int,
    num_hubs: int,
    max_row_len: int,
    has_pull: bool = False,
    pull_nnz: int | None = None,
) -> tuple[int, dict[str, float]]:
    """VRAM estimate in bytes plus a per-component breakdown dict.

    CHANGE 3 rationale
    ------------------
    The old signature returned a single int; callers had no visibility
    into *which* component was eating memory.  The new signature returns
    (total_bytes, breakdown_mb_dict) so the log can show:

        CSR:     44.0 MB    ELLPACK: 0.0 MB    PR vecs: 8.0 MB
        aux:     5.0 MB     pull:    44.0 MB   total:   101.0 MB
    """
    # Row-pointer + col-idx + values
    csr_b    = (n + 1) * 4 + nnz * 4 + nnz * 4

    # ELLPACK: int32 cols + float32 vals per slot
    ellpack_b = int(num_hubs) * int(max_row_len) * 8

    # PR_old + PR_new
    pr_b     = 2 * n * 4

    # Partial-sum reduction buffers + 2 scalar outputs
    n_blocks  = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE)
    partial_b = n_blocks * 4 + 2 * 4

    # Auxiliary per-node arrays:
    #   out_degrees (FP32), dangling_flags (INT32),
    #   eligible_mask (UINT8), pull_target_mask (UINT8)
    aux_b = n * 4 + n * 4 + n * 1 + n * 1

    # Node-ID arrays: hub_ids + low_ids (upper bound = n)
    node_id_b = (num_hubs + n) * 4

    # Pull-mode: transposed CSR (row_ptr + col_idx + values) + pull_ids
    _pull_nnz = pull_nnz if pull_nnz is not None else nnz
    pull_b    = ((n + 1) * 4 + _pull_nnz * 4 + _pull_nnz * 4) if has_pull else 0

    total = (csr_b + ellpack_b + pr_b + partial_b
             + aux_b + node_id_b + pull_b)

    breakdown = {
        "csr_mb":      csr_b     / 1e6,
        "ellpack_mb":  ellpack_b / 1e6,
        "pr_mb":       pr_b      / 1e6,
        "partial_mb":  partial_b / 1e6,
        "aux_mb":      aux_b     / 1e6,
        "node_id_mb":  node_id_b / 1e6,
        "pull_mb":     pull_b    / 1e6,
        "total_mb":    total     / 1e6,
    }
    return int(total), breakdown


def _log_vram_breakdown(
    breakdown: dict[str, float],
    free_mb: float,
    label: str = "",
) -> None:
    """Log a one-line VRAM breakdown summary."""
    parts = (
        f"CSR={breakdown['csr_mb']:.1f} "
        f"ELLPACK={breakdown['ellpack_mb']:.1f} "
        f"PR={breakdown['pr_mb']:.1f} "
        f"aux={breakdown['aux_mb']:.1f} "
        f"pull={breakdown['pull_mb']:.1f} "
        f"TOTAL={breakdown['total_mb']:.1f} "
        f"| free={free_mb:.1f} MB"
    )
    logging.info("[PageRank VRAM%s] %s", f" ({label})" if label else "", parts)


# ---------------------------------------------------------------------------
# Network-type-aware result packing
# ---------------------------------------------------------------------------

def _top_k_among(
    scores: np.ndarray,
    candidate_indices: np.ndarray,
    k: int,
) -> list[int]:
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

    return {
        "scores":          scores_list,
        "top_regulators":  _top_k_among(scores, regulators, _TOP_REG),
        "top_targets":     _top_k_among(scores, targets,    _TOP_TGT),
        "iterations":      iterations,
        "converged":       converged,
    }


# ---------------------------------------------------------------------------
# Chunked-path helpers
# ---------------------------------------------------------------------------

class _ChunkBuffer:
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
    n = int(graph_csr.shape[0])
    indptr  = graph_csr.indptr
    indices = graph_csr.indices
    values  = graph_csr.data.astype(np.float32, copy=False)

    chunks: list[dict] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        node_ids = np.arange(start, end, dtype=np.int32)
        mask     = out_degrees[start:end] > 0.0
        node_ids = node_ids[mask]
        if node_ids.size == 0:
            continue

        row_starts = indptr[node_ids]
        row_ends   = indptr[node_ids + 1]
        row_lens   = (row_ends - row_starts).astype(np.int32)
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
    """PageRank — GPU-accelerated via custom PyCUDA kernels.

    Changes from the previous version
    ----------------------------------
    C1  ELLPACK safeguard: default fraction 0.05 → 0.001; additional
        MB-cap and VRAM-pct-cap disables ELLPACK entirely when it would
        exceed the limit, routing all nodes through the CSR HIGH-tier.
    C2  Pull-mode auto-enabled for scale-free graphs (max/avg > 10).
    C3  VRAM estimator returns per-component breakdown dict + detailed log.
    C4  Auto-chunking at chunk_vram_pct (default 70 %) of free VRAM.
        MemoryError is no longer raised for graphs that fit via chunking.
    C5  Timing already excludes H2D (previous fix); chunked path now also
        excludes the pull-transpose rebuild (was erroneously rebuilding it
        inside the chunked function).
    C6  Optional _PerfProfiler reports preprocessing vs GPU time split.
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

        damping            = float(p["damping"])
        max_iter           = int(p["max_iter"])
        tolerance          = float(p["tolerance"])
        network_type       = str(p.get("network_type", "grn"))
        block_size         = int(p.get("block_size", BLOCK_SIZE))
        if block_size <= 0 or block_size > 1024:
            block_size = BLOCK_SIZE
        use_chunking       = bool(p.get("use_chunking", False))
        ellpack_fraction   = float(p.get("ellpack_fraction", 0.001))
        ellpack_max_mb     = float(p.get("ellpack_max_mb", 256.0))
        ellpack_vram_pct   = float(p.get("ellpack_vram_pct", 0.10))
        pull_threshold_p   = int(p.get("pull_threshold", -1))
        pull_fraction      = float(p.get("pull_fraction", 0.01))
        chunk_vram_pct     = float(p.get("chunk_vram_pct", 0.70))
        enable_diag        = bool(p.get("enable_diagnostics", False))
        node_index_map     = p.get("node_index_map")

        n = int(graph_csr.shape[0])
        if n == 0:
            raise ValueError("Empty graph")

        prof = _PerfProfiler(enabled=enable_diag)
        teleport_val = np.float32((1.0 - damping) / n)
        init_val     = np.float32(1.0 / n)

        # ---- Get free VRAM once (used for all budget decisions) --------
        prof.begin("vram_query")
        try:
            free_bytes, total_bytes = cuda.mem_get_info()
        except Exception:                               # noqa: BLE001
            free_bytes = 1 << 30
            total_bytes = free_bytes
        free_mb = free_bytes / 1e6
        prof.end()

        # ---- CPU preprocessing: degrees & masks -----------------------
        prof.begin("preprocess_degrees")
        out_degrees    = _compute_out_degrees(graph_csr)
        dangling_mask  = _identify_dangling_nodes(out_degrees)
        dangling_flags = dangling_mask.astype(np.int32)
        in_degrees     = _compute_in_degrees(graph_csr).astype(np.float32)

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
        prof.end()

        # ---- CHANGE 1: ELLPACK with safeguard -------------------------
        prof.begin("ellpack_build")
        hub_threshold = _compute_adaptive_hub_threshold(
            out_degrees, target_ellpack_fraction=ellpack_fraction,
        )
        ellpack_data, csr_remainder, ellpack_disabled, ellpack_bytes, ell_note = \
            _build_ellpack_safe(
                graph_csr, out_degrees, hub_threshold,
                ellpack_max_bytes=ellpack_max_mb * 1e6,
                free_bytes=free_bytes,
                ellpack_vram_pct=ellpack_vram_pct,
            )
        prof.end()

        # ---- CHANGE 2: Auto pull-threshold ----------------------------
        prof.begin("pull_setup")
        if pull_threshold_p == -1:
            # Auto-detect
            pull_threshold, pull_note = _auto_pull_threshold(
                in_degrees,
                pull_fraction=pull_fraction,
            )
        elif pull_threshold_p == 0:
            pull_threshold = 0
            pull_note = "pull disabled (pull_threshold=0)"
        else:
            pull_threshold = pull_threshold_p
            pull_note = f"pull threshold={pull_threshold} (explicit)"

        pull_enabled = pull_threshold > 0
        if pull_enabled:
            pull_ids = np.where(in_degrees >= pull_threshold)[0].astype(np.int32)
            pull_enabled = pull_ids.size > 0
            if not pull_enabled:
                pull_note += " (no nodes found at threshold — disabled)"
        else:
            pull_ids = np.zeros((0,), dtype=np.int32)

        pull_target_mask_host = np.zeros((n,), dtype=np.uint8)
        if pull_enabled:
            pull_target_mask_host[pull_ids] = 1

        logging.info("[PageRank pull] %s", pull_note)
        prof.end()

        # ---- Build pull-mode transposed CSR (BEFORE timing) -----------
        # CHANGE 5: always built here in the main function and passed to
        # the chunked path.  The chunked path no longer rebuilds it.
        prof.begin("transpose_build")
        pull_indptr_h = pull_indices_h = pull_values_h = None
        if pull_enabled:
            graph_csr_T = graph_csr.T.tocsr()
            if graph_csr_T.dtype != np.float32:
                graph_csr_T = graph_csr_T.astype(np.float32)
            pull_indptr_h  = np.ascontiguousarray(graph_csr_T.indptr,  np.int32)
            pull_indices_h = np.ascontiguousarray(graph_csr_T.indices, np.int32)
            pull_values_h  = np.ascontiguousarray(graph_csr_T.data,    np.float32)
            del graph_csr_T
        prof.end()

        # ---- CHANGE 3: VRAM estimation with breakdown -----------------
        prof.begin("vram_estimate")
        pull_nnz = int(graph_csr.nnz) if pull_enabled else None
        est_bytes, breakdown = _estimate_pagerank_vram(
            n=n, nnz=int(graph_csr.nnz),
            num_hubs=ellpack_data["num_hubs"],
            max_row_len=ellpack_data["max_row_len"],
            has_pull=pull_enabled,
            pull_nnz=pull_nnz,
        )
        _log_vram_breakdown(breakdown, free_mb)
        prof.end()

        # ---- CHANGE 4: Auto-chunking at chunk_vram_pct of free VRAM --
        # Previously: raise MemoryError when est > free.
        # Now: auto-chunk at chunk_vram_pct threshold; only raise if even
        # the per-node cost alone exceeds free VRAM (degenerate case).
        auto_chunk = (est_bytes > chunk_vram_pct * free_bytes)

        if auto_chunk and not use_chunking:
            logging.info(
                "[PageRank] Auto-chunking: estimated %.1f MB > %.0f%% of "
                "%.1f MB free VRAM.",
                est_bytes / 1e6, chunk_vram_pct * 100, free_mb,
            )

        # Hard-fail only when even minimal (per-node) working set > free VRAM.
        min_bytes = (2 * n * 4) + (n * 4 * 4)   # PR_old/new + 4 aux arrays
        if min_bytes > free_bytes:
            raise MemoryError(
                f"PageRank GPU: even minimal working set ({min_bytes/1e6:.1f} MB) "
                f"exceeds free VRAM ({free_mb:.1f} MB).  "
                f"n={n} nodes is too large for this device."
            )

        if use_chunking or auto_chunk:
            prof.begin("chunked_path")
            result = _pagerank_gpu_chunked(
                graph_csr, p, out_degrees, dangling_flags,
                eligible_mask_host, eligible_note, num_eligible,
                pull_enabled, pull_ids, pull_target_mask_host,
                pull_indptr_h, pull_indices_h, pull_values_h,   # CHANGE 5
                ellpack_data, csr_remainder,
                damping, max_iter, tolerance, network_type, block_size,
                teleport_val, init_val, n,
            )
            prof.end()
            prof.record("gpu_execution", result["execution_time"])
            prof.log(extra=f"n={n} nnz={graph_csr.nnz} chunked=True")
            return result

        # ---- Compile kernels (not timed — kernel cache means only
        #      first call pays compilation cost) -------------------------
        kernels = _get_kernels()

        # ---- Streams + events ------------------------------------------
        stream_compute  = cuda.Stream()
        stream_transfer = cuda.Stream()
        start_event     = cuda.Event()
        end_event       = cuda.Event()

        def _to_gpu(arr: np.ndarray):
            ga = gpuarray.to_gpu_async(arr, stream=stream_transfer)
            d_buffers.append(ga)
            return ga

        def _empty(shape, dtype):
            ga = gpuarray.empty(shape, dtype=dtype)
            d_buffers.append(ga)
            return ga

        # ---- Device allocation + async H2D (NOT timed) ----------------
        prof.begin("h2d_transfer")
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

        # Pull mode arrays (pre-built above — CHANGE 5).
        d_row_ptr_T = d_col_idx_T = d_values_T = d_pull_ids = None
        if pull_enabled:
            d_row_ptr_T = _to_gpu(pull_indptr_h)
            d_col_idx_T = _to_gpu(pull_indices_h)
            d_values_T  = _to_gpu(pull_values_h)
            d_pull_ids  = _to_gpu(pull_ids)

        d_PR_old = _empty((n,), np.float32)
        d_PR_new = _empty((n,), np.float32)

        n_partial_blocks  = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE)
        d_partial          = _empty((n_partial_blocks,), np.float32)
        d_dangling_scalar  = _empty((1,), np.float32)
        d_l1_scalar        = _empty((1,), np.float32)

        init_host = np.full(n, init_val, dtype=np.float32)
        cuda.memcpy_htod_async(d_PR_old.gpudata, init_host, stream_transfer)
        stream_transfer.synchronize()
        prof.end()

        # Null pull-mask (needed for the kernel pointer check).
        if d_pull_target_mask is None:
            d_pull_target_mask = _to_gpu(np.zeros((n,), dtype=np.uint8))
            stream_transfer.synchronize()

        # ---- Iteration loop (TIMED) ------------------------------------
        start_event.record(stream_compute)

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

        for it in range(max_iter):
            iterations = it + 1

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
            k_init_dang(
                d_PR_new, teleport_val, d_dangling_scalar,
                d_eligible_mask, np.int32(num_eligible),
                np.float32(damping), np.int32(n),
                block=n_block_dim, grid=n_init_grid,
                stream=stream_compute,
            )

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

        # ---- Result extraction (NOT timed) ----------------------------
        scores_host = d_PR_old.get()

        inner = _pack_result(
            scores_host, out_degrees, iterations, converged, network_type,
        )
        inner["note"] = (
            f"{eligible_note}; "
            f"{ell_note}; "
            f"{pull_note}; "
            f"arch={kernels.get('_arch_flag', '?')}, "
            f"syncs_per_iter=1, "
            f"vram_est={breakdown['total_mb']:.1f}MB"
        )

        # CHANGE 6: profiler summary
        prof.record("gpu_execution", elapsed)
        prof.log(extra=f"n={n} nnz={graph_csr.nnz} iters={iterations}")

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
            "VRAM exhausted in pagerank_gpu (even minimal footprint too large). "
            "Use a higher-VRAM device or reduce graph size."
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
# Chunked execution path
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
    pull_indptr_h: np.ndarray | None,       # CHANGE 5: pre-built, not rebuilt here
    pull_indices_h: np.ndarray | None,
    pull_values_h: np.ndarray | None,
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

    CHANGE 5: The pull-mode transposed CSR is now received as pre-built
    host arrays (pull_indptr_h, pull_indices_h, pull_values_h) rather than
    being rebuilt inside this function.  The previous rebuild was an
    O(nnz log nnz) CPU operation that was erroneously included inside the
    GPU timed region and paid twice when auto-chunking triggered.
    """
    kernels = _get_kernels()

    try:
        free_bytes, _ = cuda.mem_get_info()
    except Exception:                                   # noqa: BLE001
        free_bytes = 1 << 30

    avg_nnz = max(1.0, graph_csr.nnz / max(1, n))
    bytes_per_node = (1 + 2 * avg_nnz) * 4
    available  = int(free_bytes * 0.60)
    chunk_size = max(1, min(n, int(available / max(1.0, bytes_per_node))))

    logging.info(
        "[PageRank chunked] chunk_size=%d (%d chunks approx), "
        "bytes_per_node=%.0f, available=%.0f MB",
        chunk_size,
        max(1, n // chunk_size),
        bytes_per_node,
        available / 1e6,
    )

    chunks = _build_csr_chunks(graph_csr, out_degrees, chunk_size)
    if not chunks:
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

        # CHANGE 5: use the pre-built transpose arrays — no rebuild here.
        d_row_ptr_T = d_col_idx_T = d_values_T = d_pull_ids = None
        if pull_enabled and pull_indptr_h is not None:
            d_row_ptr_T = _to_gpu(pull_indptr_h)
            d_col_idx_T = _to_gpu(pull_indices_h)
            d_values_T  = _to_gpu(pull_values_h)
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
            k_init_dang(
                d_PR_new, teleport_val, d_dangling_scalar,
                d_eligible_mask, np.int32(num_eligible),
                np.float32(damping), np.int32(n),
                block=n_block_dim, grid=n_init_grid,
                stream=stream_compute,
            )

            # Double-buffer chunked scatter.
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
            f"num_hubs={ellpack_data['num_hubs']}; "
            f"arch={kernels.get('_arch_flag', '?')}, syncs_per_iter=1"
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
