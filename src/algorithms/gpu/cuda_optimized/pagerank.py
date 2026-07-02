"""
algorithms/pagerank.py - PageRank for Biological Network Hub Identification
============================================================================

Biological context
------------------
PageRank models the propagation of "influence" through a directed network.
At each step a fraction d of the mass at each node follows an outgoing
edge; the remaining fraction (1 - d) teleports.

  GRN   - top_regulators are master TFs; top_targets are convergence points.
  PPI   - top_nodes are hub proteins central to the interaction network.
  miRNA - top_mirnas are master post-transcriptional regulators;
          top_target_genes are heavily co-targeted effector genes.

Dangling-node handling (network-type aware)
-------------------------------------------
  GRN / miRNA : redistribute ONLY to nodes with out_degree > 0.
  PPI         : redistribute UNIFORMLY across all nodes.

Improvement log
---------------
I1  ELLPACK pre-allocation OOM fix:
    _build_ellpack_safe now computes the projected ELLPACK size from hub
    counts and max_row_len WITHOUT allocating the numpy arrays first.  The
    size check runs before any allocation.  The old code called _build_ellpack
    (which did the full np.full allocation) before the limit check, causing
    the "Unable to allocate 36.0 GiB" errors on large BA graphs.
    Added use_ellpack=False parameter to bypass ELLPACK entirely (fast path).

I2  Conditional pull mode with edge-fraction threshold:
    Pull mode is enabled only when:
      (a) max/avg in-degree ratio > 10 (scale-free heuristic)
      AND
      (b) edges to pull nodes >= pull_edge_pct_threshold (default 5 %) of
          all edges.
    Condition (b) prevents enabling pull for graphs where only a tiny
    fraction of edges pass through high-in-degree nodes — in those cases
    the transposed CSR overhead exceeds any atomic contention savings.
    Detailed diagnostics: pull_nodes, pull_edges, edge_pct, decision.

I3  Convergence check interval:
    conv_check_interval (default 5) controls how often L1 is computed and
    the CPU synchronises.  Iterations 1..N-1 run fully on the GPU without
    any CPU-GPU sync.  Only iteration N (and the final iteration) sync and
    read the scalar.  The L1 and reduce kernels are skipped on non-check
    iterations, saving ~2 kernel launches per skip.

I4  Per-phase CUDA-event iteration statistics:
    When enable_diagnostics=True the first min(5, max_iter) iterations are
    bracketed with CUDA events.  After the final stream sync, per-phase
    average GPU times are computed and attached to result["diagnostics"].

I5  Optimization benchmark utility:
    benchmark_pagerank_optimizations() runs five parameter configurations
    (CSR-only, ELLPACK-on, pull-on, interval-5, all-on) and returns a
    ranked report.  Use this to confirm which optimizations actually help
    on a specific graph.

I6  CSR-only fast path:
    pagerank_gpu_simple() is a one-line wrapper that disables ELLPACK,
    pull mode, and sets conv_check_interval=5.  Use it as a simpler,
    lower-overhead baseline for benchmarking.

Parameter guide
---------------
damping              (float, default 0.85)
max_iter             (int,   default 100)
tolerance            (float, default 1e-6)
network_type         (str,   default "grn")
block_size           (int,   default 256)
use_ellpack          (bool,  default True)   I1: set False to skip ELLPACK
ellpack_fraction     (float, default 0.001)  fraction of nodes routed to ELLPACK
ellpack_max_mb       (float, default 256)    disable ELLPACK above this many MB
ellpack_vram_pct     (float, default 0.10)   disable ELLPACK above this VRAM pct
pull_threshold       (int,   default -1)     -1=auto, 0=disabled, >0=explicit
pull_fraction        (float, default 0.01)   top 1% in-degree -> pull targets
pull_edge_pct_threshold (float, default 0.05) I2: min edge fraction for pull
chunk_vram_pct       (float, default 0.70)   auto-chunk above this VRAM fraction
use_chunking         (bool,  default False)  force chunked path
conv_check_interval  (int,   default 5)      I3: check convergence every N iters
enable_diagnostics   (bool,  default False)  I4: per-phase CUDA-event profiling
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

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
    "damping":                  0.85,
    "max_iter":                 100,
    "tolerance":                1e-6,
    "network_type":             "grn",
    "block_size":               256,
    # I1: use_ellpack=False completely bypasses ELLPACK preprocessing.
    "use_ellpack":              True,
    # I1: ellpack_fraction 0.001 keeps hub set tiny on power-law graphs.
    "ellpack_fraction":         0.001,
    "ellpack_max_mb":           256.0,
    "ellpack_vram_pct":         0.10,
    # I2: pull mode: -1=auto, 0=off, >0=explicit threshold.
    "pull_threshold":           -1,
    "pull_fraction":            0.01,
    # I2: minimum fraction of all edges that must reach pull nodes.
    "pull_edge_pct_threshold":  0.05,
    "chunk_vram_pct":           0.70,
    "use_chunking":             False,
    # I3: check convergence every N iterations (saves N-1 out of N syncs).
    "conv_check_interval":      5,
    # I4: per-phase CUDA-event profiling (first 5 iters).
    "enable_diagnostics":       False,
}

BLOCK_SIZE: int   = 256
WARP_SIZE: int    = 32
SMEM_BUCKETS: int = 256
_TOP_REG: int     = 15
_TOP_TGT: int     = 15
_TOP_NODES: int   = 20


# ---------------------------------------------------------------------------
# CUDA kernel source  (kernels are unchanged; all improvements are Python-side)
# ---------------------------------------------------------------------------

KERNEL_SOURCE = r"""
extern "C" {

#define BLOCK_SIZE      256
#define WARP_SIZE       32
#define SMEM_BUCKETS    256
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

// =========================================================================
// KERNEL: initialize_pr
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
// KERNEL: scatter_contributions_csr_low  (thread-per-node)
//
// The block-per-node scatter_contributions_csr above dedicates a whole
// 256-thread block to ONE node and, for low-degree nodes (deg < WARP_SIZE),
// lets only threadIdx.x == 0 do the work — 0.4% thread utilisation.  On
// uniform-degree graphs (Erdős–Rényi, Watts–Strogatz; avg degree ~6 → every
// node is low-degree) that cripples the whole iteration.
//
// This kernel assigns ONE THREAD per node, so a 256-thread block scatters
// 256 different nodes concurrently (100% utilisation).  Each thread walks
// its node's full (short) neighbour list with direct global atomicAdd.
// The host routes only low-degree nodes here; medium/high-degree nodes stay
// on the block-per-node kernel (where the warp/block tiers + SMEM hash pay
// off).  Grid: ceil(num_nodes / BLOCK_SIZE) blocks.
// =========================================================================
__global__ void scatter_contributions_csr_low(
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
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    const int   u     = node_ids[idx];
    const float deg_u = out_degree[u];
    if (deg_u <= 0.0f) return;

    const int   row_start    = row_ptr[u];
    const int   row_end      = row_ptr[u + 1];
    const float contribution = damping * PR_old[u] / deg_u;

    for (int j = row_start; j < row_end; ++j) {
        const int v = col_idx[j];
        if (pull_target_mask != 0 && pull_target_mask[v]) continue;
        atomicAdd(&PR_new[v], contribution * edge_weights[j]);
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
// KERNEL: distribute_dangling_mass  (used in chunked path)
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
        if (deg_v > 0.0f)
            partial += values_T[j] * PR_old[v] / deg_v;
    }

    smem[threadIdx.x] = partial;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        PR_new[u] += damping * smem[0];
}

}  // extern "C"
"""


# ---------------------------------------------------------------------------
# Module-level kernel cache + adaptive compilation
# ---------------------------------------------------------------------------

_kernel_cache: dict[str, dict[str, Any]] = {}


def _detect_arch_flag() -> str:
    try:
        cuda.init()
        cc_major, cc_minor = cuda.Device(0).compute_capability()
        return f"-arch=sm_{cc_major}{cc_minor}"
    except Exception:                                   # noqa: BLE001
        return "-arch=sm_75"


def _get_kernels() -> dict[str, Any]:
    if "pagerank" not in _kernel_cache:
        if not PYCUDA_AVAILABLE:
            raise RuntimeError(
                "PyCUDA is required to compile PageRank kernels."
            )
        arch_flag = _detect_arch_flag()
        mod = SourceModule(KERNEL_SOURCE, options=[arch_flag, "-O3"],
                           no_extern_c=True)
        _kernel_cache["pagerank"] = {
            "init":          mod.get_function("initialize_pr"),
            "init_dangling": mod.get_function("initialize_pr_with_dangling"),
            "scatter_csr":   mod.get_function("scatter_contributions_csr"),
            "scatter_csr_low": mod.get_function("scatter_contributions_csr_low"),
            "scatter_ell":   mod.get_function("scatter_contributions_ellpack"),
            "dangling":      mod.get_function("distribute_dangling_mass"),
            "sum_dangling":  mod.get_function("sum_dangling_pr"),
            "l1_conv":       mod.get_function("compute_l1_convergence"),
            "reduce_scalar": mod.get_function("reduce_to_scalar_f32"),
            "pull_gather":   mod.get_function("gather_contributions_pull"),
            "_arch_flag":    arch_flag,
        }
    return _kernel_cache["pagerank"]


# ---------------------------------------------------------------------------
# Lightweight per-phase wall-clock profiler
# ---------------------------------------------------------------------------

class _PerfProfiler:
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
        self._phases[self._phase] = self._phases.get(self._phase, 0.0) + elapsed
        self._phase = None
        self._start = None

    def record(self, phase: str, elapsed_s: float) -> None:
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
            lines.append(
                f"  {ph:<28}: {t*1000:8.2f} ms  "
                f"({100*t/max(total, 1e-12):5.1f} %)"
            )
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
    return np.asarray(graph_csr.sum(axis=1), dtype=np.float32).flatten()


def _compute_in_degrees(graph_csr: sp.csr_matrix) -> np.ndarray:
    return np.asarray(graph_csr.sum(axis=0), dtype=np.float32).flatten()


def _identify_dangling_nodes(out_degrees: np.ndarray) -> np.ndarray:
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
    else:
        mask[:] = 1
        note = "dangling mass -> all nodes (uniform)"
    return mask, note


def _compute_adaptive_hub_threshold(
    out_degrees: np.ndarray,
    target_ellpack_fraction: float = 0.001,
) -> int:
    if out_degrees.size == 0:
        return WARP_SIZE
    deg_int   = out_degrees.astype(np.int64)
    pct       = (1.0 - max(0.0, min(target_ellpack_fraction, 1.0))) * 100.0
    threshold = int(np.percentile(deg_int, pct))
    threshold = max(threshold, WARP_SIZE)
    max_deg   = int(deg_int.max()) if deg_int.size > 0 else WARP_SIZE
    threshold = min(threshold, max_deg)
    threshold = max(threshold, 1)
    snapped   = int(2 ** round(math.log2(max(threshold, 1))))
    return max(snapped, WARP_SIZE)


# ---------------------------------------------------------------------------
# I1: ELLPACK with pre-allocation size check
# ---------------------------------------------------------------------------

def _build_ellpack_safe(
    graph_csr: sp.csr_matrix,
    out_degrees: np.ndarray,
    hub_threshold: int,
    ellpack_max_bytes: float,
    free_bytes: int,
    ellpack_vram_pct: float,
) -> tuple[dict, dict, bool, float, str]:
    """Build ELLPACK + CSR split, checking size BEFORE allocating arrays.

    I1 root-cause fix
    -----------------
    The previous implementation called _build_ellpack() (which allocated
    np.full((hub_ids.size, max_row_len), -1) and np.zeros(...)) BEFORE
    the size check.  On a 1M-node BA graph with ellpack_fraction=0.05 and
    max_hub_degree=50k this allocated 50k×50k×8 B = 20 GB of numpy arrays
    on the CPU, causing the "Unable to allocate 36.0 GiB" MemoryError
    BEFORE the safeguard could disable ELLPACK.

    Fix: compute num_hubs and max_row_len from the CSR indptr (O(n), no
    large allocation), check against limits, THEN allocate the ELLPACK
    arrays only if the projected size is safe.
    """
    n       = int(graph_csr.shape[0])
    indptr  = np.ascontiguousarray(graph_csr.indptr,  dtype=np.int32)
    indices = np.ascontiguousarray(graph_csr.indices, dtype=np.int32)
    data    = np.ascontiguousarray(graph_csr.data,    dtype=np.float32)
    deg_int = np.diff(indptr).astype(np.int32)

    hub_mask = deg_int >= hub_threshold
    hub_ids  = np.where(hub_mask)[0].astype(np.int32)
    low_mask = (deg_int > 0) & (~hub_mask)
    low_ids  = np.where(low_mask)[0].astype(np.int32)

    num_hubs    = int(hub_ids.size)
    max_row_len = int(deg_int[hub_ids].max()) if num_hubs > 0 else 0
    # I1: compute projected ELLPACK bytes WITHOUT allocating anything.
    ellpack_bytes = float(num_hubs * max_row_len * 8)

    vram_cap = float(free_bytes) * max(0.0, min(ellpack_vram_pct, 1.0))
    limit    = min(ellpack_max_bytes, vram_cap)

    max_hub_deg = max_row_len
    notes = (
        f"ELLPACK: {num_hubs} hubs, max_hub_degree={max_hub_deg}, "
        f"padded_slots={max_row_len}, "
        f"estimated={ellpack_bytes/1e6:.1f} MB"
    )

    # I1: size check BEFORE allocation.
    ellpack_disabled = (num_hubs > 0) and (ellpack_bytes > limit)

    if ellpack_disabled:
        notes += (
            f" -- DISABLED (exceeds {limit/1e6:.1f} MB limit; "
            f"all nodes fall to CSR high-tier)"
        )
        logging.info("[PageRank] %s", notes)
        all_nodes = np.where(deg_int > 0)[0].astype(np.int32)
        ellpack_data = {
            "hub_ids": np.zeros((0,), dtype=np.int32),
            "cols":    np.zeros((0,), dtype=np.int32),
            "vals":    np.zeros((0,), dtype=np.float32),
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
        # I1: safe to allocate now.
        if num_hubs > 0:
            cols = np.full((num_hubs, max_row_len), -1, dtype=np.int32)
            vals = np.zeros((num_hubs, max_row_len),    dtype=np.float32)
            for h, u in enumerate(hub_ids):
                s = int(indptr[u]); e = int(indptr[u + 1])
                L = e - s
                cols[h, :L] = indices[s:e]
                vals[h, :L] = data[s:e]
        else:
            max_row_len = 0
            cols = np.zeros((0, 0), dtype=np.int32)
            vals = np.zeros((0, 0), dtype=np.float32)

        notes += " -- ACTIVE"
        logging.info("[PageRank] %s", notes)
        ellpack_data = {
            "hub_ids":     hub_ids,
            "cols":        cols.reshape(-1),
            "vals":        vals.reshape(-1),
            "max_row_len": max_row_len,
            "num_hubs":    num_hubs,
        }
        csr_remainder = {
            "node_ids": low_ids,
            "num_low":  int(low_ids.size),
            "row_ptr":  indptr,
            "col_idx":  indices,
            "values":   data,
        }

    return ellpack_data, csr_remainder, ellpack_disabled, ellpack_bytes, notes


def _make_csr_only_remainder(
    graph_csr: sp.csr_matrix,
    out_degrees: np.ndarray,
) -> tuple[dict, dict]:
    """Return empty ELLPACK + full CSR remainder (use_ellpack=False path)."""
    indptr  = np.ascontiguousarray(graph_csr.indptr,  dtype=np.int32)
    indices = np.ascontiguousarray(graph_csr.indices, dtype=np.int32)
    data    = np.ascontiguousarray(graph_csr.data,    dtype=np.float32)
    deg_int = np.diff(indptr).astype(np.int32)
    all_nodes = np.where(deg_int > 0)[0].astype(np.int32)
    ellpack_data = {
        "hub_ids": np.zeros((0,), dtype=np.int32),
        "cols":    np.zeros((0,), dtype=np.int32),
        "vals":    np.zeros((0,), dtype=np.float32),
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
    return ellpack_data, csr_remainder


# ---------------------------------------------------------------------------
# I2: Conditional pull-mode with edge-fraction threshold
# ---------------------------------------------------------------------------

def _auto_pull_threshold(
    in_degrees: np.ndarray,
    pull_fraction: float = 0.01,
    min_threshold: int = 32,
    pull_edge_pct_threshold: float = 0.05,
) -> tuple[int, str, dict]:
    """Decide pull threshold with scale-free + edge-fraction gating.

    I2: Two conditions must both be true before pull mode activates:
      (a) max_in_degree / mean_in_degree > 10  (scale-free heuristic)
      (b) edges reaching pull nodes >= pull_edge_pct_threshold of all edges

    Condition (b) prevents enabling pull for graphs where pull nodes attract
    only a tiny fraction of edges — in those cases the transposed CSR
    (~nnz×8 bytes extra VRAM) and pull_gather kernel overhead exceed the
    atomic contention savings.

    Returns (threshold, note, diagnostics_dict).
    threshold == 0 means pull mode disabled.
    """
    diag: dict = {}

    if in_degrees.size == 0:
        return 0, "pull disabled (empty graph)", diag

    nonzero = in_degrees[in_degrees > 0]
    if nonzero.size < 50:
        return 0, "pull disabled (too few nodes for scale-free test)", diag

    max_deg = float(nonzero.max())
    avg_deg = float(nonzero.mean())
    ratio   = max_deg / max(avg_deg, 1.0)
    diag["max_in_degree"] = max_deg
    diag["avg_in_degree"] = avg_deg
    diag["scale_free_ratio"] = ratio

    if ratio < 10.0:
        return 0, (
            f"pull disabled (max/avg in-degree={ratio:.1f} < 10; "
            f"not scale-free)"
        ), diag

    pct       = (1.0 - max(0.0, min(pull_fraction, 0.5))) * 100.0
    threshold = int(np.percentile(in_degrees, pct))
    threshold = max(threshold, min_threshold)

    pull_ids     = np.where(in_degrees >= threshold)[0]
    n_pull       = int(pull_ids.size)
    pull_edges   = int(in_degrees[pull_ids].sum()) if n_pull > 0 else 0
    total_edges  = int(in_degrees.sum())
    edge_pct     = 100.0 * pull_edges / max(total_edges, 1)

    diag.update({
        "threshold":     threshold,
        "pull_nodes":    n_pull,
        "pull_edges":    pull_edges,
        "total_edges":   total_edges,
        "pull_edge_pct": edge_pct,
        "edge_pct_threshold": pull_edge_pct_threshold * 100.0,
    })

    # I2: second gate — enough edges must flow through pull nodes.
    if edge_pct < pull_edge_pct_threshold * 100.0:
        return 0, (
            f"pull disabled: only {edge_pct:.1f}% of edges reach pull nodes "
            f"(threshold {pull_edge_pct_threshold*100:.0f}%); "
            f"pull overhead would exceed atomic contention savings"
        ), diag

    note = (
        f"pull AUTO-ENABLED (max/avg={ratio:.1f}, scale-free); "
        f"threshold={threshold} (top {pull_fraction*100:.1f}% in-degree); "
        f"{n_pull} pull nodes, {pull_edges:,} pull edges ({edge_pct:.1f}% of total)"
    )
    return threshold, note, diag


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------

def _estimate_pagerank_vram(
    n: int,
    nnz: int,
    num_hubs: int,
    max_row_len: int,
    has_pull: bool = False,
    pull_nnz: int | None = None,
) -> tuple[int, dict[str, float]]:
    csr_b    = (n + 1) * 4 + nnz * 4 + nnz * 4
    ellpack_b = int(num_hubs) * int(max_row_len) * 8
    pr_b     = 2 * n * 4
    n_blocks  = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE)
    partial_b = n_blocks * 4 + 2 * 4
    aux_b    = n * 4 + n * 4 + n * 1 + n * 1
    node_id_b = (num_hubs + n) * 4
    _pull_nnz = pull_nnz if pull_nnz is not None else nnz
    pull_b   = ((n + 1) * 4 + _pull_nnz * 4 + _pull_nnz * 4) if has_pull else 0
    total    = csr_b + ellpack_b + pr_b + partial_b + aux_b + node_id_b + pull_b
    breakdown = {
        "csr_mb":     csr_b     / 1e6,
        "ellpack_mb": ellpack_b / 1e6,
        "pr_mb":      pr_b      / 1e6,
        "partial_mb": partial_b / 1e6,
        "aux_mb":     aux_b     / 1e6,
        "node_id_mb": node_id_b / 1e6,
        "pull_mb":    pull_b    / 1e6,
        "total_mb":   total     / 1e6,
    }
    return int(total), breakdown


def _log_vram_breakdown(breakdown: dict[str, float], free_mb: float,
                        label: str = "") -> None:
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
# Result packing
# ---------------------------------------------------------------------------

def _top_k_among(scores: np.ndarray, candidate_indices: np.ndarray,
                 k: int) -> list[int]:
    if candidate_indices.size == 0:
        return []
    sub = scores[candidate_indices]
    order = np.argsort(sub)[::-1][:k]
    return candidate_indices[order].astype(int).tolist()


def _pack_result(scores: np.ndarray, out_degrees: np.ndarray,
                 iterations: int, converged: bool,
                 network_type: str) -> dict:
    nt = str(network_type).lower()
    scores_list = scores.astype(float).tolist()
    if nt == "ppi":
        return {
            "scores":     scores_list,
            "top_nodes":  np.argsort(scores)[::-1][:_TOP_NODES].astype(int).tolist(),
            "iterations": iterations,
            "converged":  converged,
        }
    regulators = np.where(out_degrees > 0.0)[0]
    targets    = np.where(out_degrees == 0.0)[0]
    if nt == "mirna":
        return {
            "scores":           scores_list,
            "top_mirnas":       _top_k_among(scores, regulators, _TOP_REG),
            "top_target_genes": _top_k_among(scores, targets,    _TOP_TGT),
            "iterations":       iterations,
            "converged":        converged,
        }
    return {
        "scores":         scores_list,
        "top_regulators": _top_k_among(scores, regulators, _TOP_REG),
        "top_targets":    _top_k_among(scores, targets,    _TOP_TGT),
        "iterations":     iterations,
        "converged":      converged,
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


def _transfer_chunk_async(
    chunk_indptr: np.ndarray,
    chunk_indices: np.ndarray,
    chunk_values: np.ndarray,
    chunk_node_ids: np.ndarray,
    buf: _ChunkBuffer,
    stream,
) -> None:
    cuda.memcpy_htod_async(
        buf.d_row_ptr.gpudata,
        np.ascontiguousarray(chunk_indptr,   np.int32),   stream)
    cuda.memcpy_htod_async(
        buf.d_col_idx.gpudata,
        np.ascontiguousarray(chunk_indices,  np.int32),   stream)
    cuda.memcpy_htod_async(
        buf.d_values.gpudata,
        np.ascontiguousarray(chunk_values,   np.float32), stream)
    cuda.memcpy_htod_async(
        buf.d_node_ids.gpudata,
        np.ascontiguousarray(chunk_node_ids, np.int32),   stream)


def _build_csr_chunks(
    graph_csr: sp.csr_matrix,
    out_degrees: np.ndarray,
    chunk_size: int,
) -> list[dict]:
    n      = int(graph_csr.shape[0])
    indptr = graph_csr.indptr
    indices= graph_csr.indices
    values = graph_csr.data.astype(np.float32, copy=False)
    chunks: list[dict] = []
    for start in range(0, n, chunk_size):
        end      = min(start + chunk_size, n)
        node_ids = np.arange(start, end, dtype=np.int32)
        mask     = out_degrees[start:end] > 0.0
        node_ids = node_ids[mask]
        if node_ids.size == 0:
            continue
        row_starts = indptr[node_ids]
        row_ends   = indptr[node_ids + 1]
        row_lens   = (row_ends - row_starts).astype(np.int32)
        nnz        = int(row_lens.sum())
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

    Improvements in this version vs the previous:
      I1  ELLPACK size check runs BEFORE np allocation (OOM fix).
          use_ellpack=False bypasses ELLPACK entirely (fast path).
      I2  Pull mode requires both scale-free heuristic AND edge-fraction
          threshold (default 5%) to be met.
      I3  Convergence checked every conv_check_interval iters (default 5),
          saving ~80% of CPU-GPU syncs.
      I4  Per-phase CUDA-event profiling attached to result when
          enable_diagnostics=True.
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA is required for pagerank_gpu(). "
            "Install it or use pagerank_cpu_single()."
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
        # Capture whether the USER explicitly forced chunking, before
        # apply_config can inject use_chunking=True from a coarse estimate.
        user_forced_chunking = bool((params or {}).get("use_chunking", False))
        p = _merge_params(params)
        if _GPU_CONFIG_AVAILABLE:
            p = apply_config("pagerank", graph_csr, p) or p
            for k, v in _DEFAULT_PARAMS.items():
                p.setdefault(k, v)

        damping               = float(p["damping"])
        max_iter              = int(p["max_iter"])
        tolerance             = float(p["tolerance"])
        network_type          = str(p.get("network_type", "grn"))
        block_size            = int(p.get("block_size", BLOCK_SIZE))
        if block_size <= 0 or block_size > 1024:
            block_size = BLOCK_SIZE
        use_ellpack           = bool(p.get("use_ellpack", True))
        use_chunking          = bool(p.get("use_chunking", False))
        ellpack_fraction      = float(p.get("ellpack_fraction", 0.001))
        ellpack_max_mb        = float(p.get("ellpack_max_mb", 256.0))
        ellpack_vram_pct      = float(p.get("ellpack_vram_pct", 0.10))
        pull_threshold_p      = int(p.get("pull_threshold", -1))
        pull_fraction         = float(p.get("pull_fraction", 0.01))
        pull_edge_pct_thresh  = float(p.get("pull_edge_pct_threshold", 0.05))
        chunk_vram_pct        = float(p.get("chunk_vram_pct", 0.70))
        conv_check_interval   = max(1, int(p.get("conv_check_interval", 5)))
        enable_diag           = bool(p.get("enable_diagnostics", False))
        node_index_map        = p.get("node_index_map")

        n = int(graph_csr.shape[0])
        if n == 0:
            raise ValueError("Empty graph")

        prof          = _PerfProfiler(enabled=enable_diag)
        teleport_val  = np.float32((1.0 - damping) / n)
        init_val      = np.float32(1.0 / n)

        # ---- VRAM query -----------------------------------------------
        prof.begin("vram_query")
        try:
            free_bytes, _ = cuda.mem_get_info()
        except Exception:                               # noqa: BLE001
            free_bytes = 1 << 30
        free_mb = free_bytes / 1e6
        prof.end()

        # ---- Degrees + dangling masks ---------------------------------
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
                "pagerank_gpu: no eligible nodes (network_type=%s) — "
                "falling back to uniform.", network_type,
            )
            eligible_mask_host[:] = 1
            num_eligible = n
            eligible_note += " (fallback to uniform)"
        prof.end()

        # ---- I1: ELLPACK (with pre-allocation size check) -------------
        prof.begin("ellpack_build")
        if not use_ellpack:
            ellpack_data, csr_remainder = _make_csr_only_remainder(
                graph_csr, out_degrees)
            ellpack_disabled = True
            ellpack_bytes    = 0.0
            ell_note         = "ELLPACK disabled (use_ellpack=False)"
            logging.info("[PageRank] %s", ell_note)
        else:
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

        # ---- I2: Conditional pull mode --------------------------------
        prof.begin("pull_setup")
        pull_diag: dict = {}
        if pull_threshold_p == -1:
            pull_threshold, pull_note, pull_diag = _auto_pull_threshold(
                in_degrees,
                pull_fraction=pull_fraction,
                pull_edge_pct_threshold=pull_edge_pct_thresh,
            )
        elif pull_threshold_p == 0:
            pull_threshold = 0
            pull_note      = "pull disabled (pull_threshold=0)"
        else:
            pull_threshold = pull_threshold_p
            pull_note      = f"pull threshold={pull_threshold} (explicit)"

        pull_enabled = pull_threshold > 0
        if pull_enabled:
            pull_ids     = np.where(in_degrees >= pull_threshold)[0].astype(np.int32)
            pull_enabled = pull_ids.size > 0
            if not pull_enabled:
                pull_note += " (no nodes at threshold — disabled)"
        else:
            pull_ids = np.zeros((0,), dtype=np.int32)

        pull_target_mask_host = np.zeros((n,), dtype=np.uint8)
        if pull_enabled:
            pull_target_mask_host[pull_ids] = 1
        logging.info("[PageRank pull] %s", pull_note)
        prof.end()

        # ---- Build transposed CSR for pull (before timed region) ------
        prof.begin("transpose_build")
        pull_indptr_h = pull_indices_h = pull_values_h = None
        if pull_enabled:
            graph_csr_T   = graph_csr.T.tocsr().astype(np.float32)
            pull_indptr_h  = np.ascontiguousarray(graph_csr_T.indptr,  np.int32)
            pull_indices_h = np.ascontiguousarray(graph_csr_T.indices, np.int32)
            pull_values_h  = np.ascontiguousarray(graph_csr_T.data,    np.float32)
            del graph_csr_T
        prof.end()

        # ---- VRAM estimate + auto-chunking ----------------------------
        prof.begin("vram_estimate")
        pull_nnz  = int(graph_csr.nnz) if pull_enabled else None
        est_bytes, breakdown = _estimate_pagerank_vram(
            n=n, nnz=int(graph_csr.nnz),
            num_hubs=ellpack_data["num_hubs"],
            max_row_len=ellpack_data["max_row_len"],
            has_pull=pull_enabled, pull_nnz=pull_nnz,
        )
        _log_vram_breakdown(breakdown, free_mb)
        prof.end()

        auto_chunk = est_bytes > chunk_vram_pct * free_bytes

        # The chunked path re-streams the ENTIRE CSR host→device every
        # iteration (see _transfer_chunk_async inside the iteration loop).
        # That is only worthwhile when the graph genuinely does not fit in
        # VRAM; for a graph that fits it is catastrophic (e.g. 100x re-upload
        # of the full CSR).  apply_config's MemoryManager can inject
        # use_chunking=True from a coarse estimate even when the working set
        # fits comfortably, so pagerank's OWN precise estimate (auto_chunk) is
        # authoritative.  A user who *explicitly* passes use_chunking=True can
        # still force the path (e.g. to trade speed for lower peak memory).
        needs_chunk = auto_chunk or user_forced_chunking
        if use_chunking and not user_forced_chunking and not auto_chunk:
            logging.info(
                "[PageRank] apply_config use_chunking ignored — working set "
                "%.1f MB fits in %.1f MB free (chunking would re-stream the "
                "CSR every iteration).", est_bytes / 1e6, free_mb,
            )
        if auto_chunk and not user_forced_chunking:
            logging.info(
                "[PageRank] Auto-chunking: %.1f MB > %.0f%% of %.1f MB free.",
                est_bytes / 1e6, chunk_vram_pct * 100, free_mb,
            )

        min_bytes = (2 * n * 4) + (n * 4 * 4)
        if min_bytes > free_bytes:
            raise MemoryError(
                f"PageRank GPU: minimal working set ({min_bytes/1e6:.1f} MB) "
                f"exceeds free VRAM ({free_mb:.1f} MB). n={n} too large."
            )

        if needs_chunk:
            prof.begin("chunked_path")
            result = _pagerank_gpu_chunked(
                graph_csr, p, out_degrees, dangling_flags,
                eligible_mask_host, eligible_note, num_eligible,
                pull_enabled, pull_ids, pull_target_mask_host,
                pull_indptr_h, pull_indices_h, pull_values_h,
                ellpack_data, csr_remainder,
                damping, max_iter, tolerance, network_type, block_size,
                teleport_val, init_val, n, conv_check_interval,
            )
            prof.end()
            prof.record("gpu_execution", result["execution_time"])
            prof.log(extra=f"n={n} nnz={graph_csr.nnz} chunked=True")
            return result

        # ---- Compile kernels ------------------------------------------
        kernels = _get_kernels()

        # ---- Streams + events -----------------------------------------
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

        # ---- Split remainder nodes by degree for scatter dispatch ------
        # Low-degree nodes (deg < WARP_SIZE) go to the thread-per-node
        # scatter kernel (100% thread utilisation); the rest stay on the
        # block-per-node kernel whose warp/block tiers + SMEM hash pay off
        # only for higher-degree rows.  csr_remainder["row_ptr"] is the full
        # graph indptr, so degree is diff(indptr) indexed by global node id.
        _rem_ids  = csr_remainder["node_ids"]
        _full_deg = np.diff(csr_remainder["row_ptr"]).astype(np.int64)
        _rem_deg  = _full_deg[_rem_ids] if _rem_ids.size > 0 else _rem_ids
        _lowdeg_mask   = _rem_deg < WARP_SIZE
        lowdeg_ids     = np.ascontiguousarray(_rem_ids[_lowdeg_mask],  dtype=np.int32)
        highdeg_ids    = np.ascontiguousarray(_rem_ids[~_lowdeg_mask], dtype=np.int32)
        num_lowdeg     = int(lowdeg_ids.size)
        num_highdeg    = int(highdeg_ids.size)

        # ---- H2D transfers (not timed) --------------------------------
        prof.begin("h2d_transfer")
        d_csr_row_ptr = _to_gpu(csr_remainder["row_ptr"])
        d_csr_col_idx = _to_gpu(csr_remainder["col_idx"])
        d_csr_values  = _to_gpu(csr_remainder["values"])
        d_lowdeg_ids  = _to_gpu(lowdeg_ids)  if num_lowdeg  > 0 else None
        d_highdeg_ids = _to_gpu(highdeg_ids) if num_highdeg > 0 else None

        num_hubs    = ellpack_data["num_hubs"]
        max_row_len = ellpack_data["max_row_len"]
        if num_hubs > 0:
            d_ellpack_cols = _to_gpu(ellpack_data["cols"])
            d_ellpack_vals = _to_gpu(ellpack_data["vals"])
            d_hub_ids      = _to_gpu(ellpack_data["hub_ids"])
        else:
            d_ellpack_cols = d_ellpack_vals = d_hub_ids = None

        d_out_degree      = _to_gpu(out_degrees)
        d_dangling_flags  = _to_gpu(dangling_flags)
        d_eligible_mask   = _to_gpu(eligible_mask_host)
        d_pull_target_mask = (
            _to_gpu(pull_target_mask_host) if pull_enabled
            else _to_gpu(np.zeros((n,), dtype=np.uint8))
        )

        d_row_ptr_T = d_col_idx_T = d_values_T = d_pull_ids = None
        if pull_enabled:
            d_row_ptr_T = _to_gpu(pull_indptr_h)
            d_col_idx_T = _to_gpu(pull_indices_h)
            d_values_T  = _to_gpu(pull_values_h)
            d_pull_ids  = _to_gpu(pull_ids)

        d_PR_old         = _empty((n,), np.float32)
        d_PR_new         = _empty((n,), np.float32)
        n_partial_blocks = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE)
        d_partial        = _empty((n_partial_blocks,), np.float32)
        d_dangling_scalar= _empty((1,), np.float32)
        d_l1_scalar      = _empty((1,), np.float32)

        init_host = np.full(n, init_val, dtype=np.float32)
        cuda.memcpy_htod_async(d_PR_old.gpudata, init_host, stream_transfer)
        stream_transfer.synchronize()
        prof.end()

        # ---- I4: Pre-allocate diagnostic CUDA events ------------------
        _diag_n  = min(max_iter, 5) if enable_diag else 0
        _di      = 0
        if enable_diag:
            _ev_dang_s = [cuda.Event() for _ in range(_diag_n)]
            _ev_dang_e = [cuda.Event() for _ in range(_diag_n)]
            _ev_scat_s = [cuda.Event() for _ in range(_diag_n)]
            _ev_scat_e = [cuda.Event() for _ in range(_diag_n)]
            _ev_pull_s = [cuda.Event() for _ in range(_diag_n)]
            _ev_pull_e = [cuda.Event() for _ in range(_diag_n)]
            _ev_l1_s   = [cuda.Event() for _ in range(_diag_n)]
            _ev_l1_e   = [cuda.Event() for _ in range(_diag_n)]
            _diag_l1_iters: list[int] = []   # which iterations had l1 checks

        # ---- Fixed grid dimensions ------------------------------------
        n_init_grid    = ((n + block_size - 1) // block_size, 1, 1)
        n_block_dim    = (block_size, 1, 1)
        partial_grid   = (n_partial_blocks, 1, 1)
        partial_block  = (BLOCK_SIZE, 1, 1)
        n_pull_nodes   = int(pull_ids.size) if pull_enabled else 0
        # Grid for the thread-per-node low-degree scatter kernel.
        lowdeg_grid    = (((num_lowdeg + block_size - 1) // block_size), 1, 1)

        k_init_dang  = kernels["init_dangling"]
        k_sc_csr     = kernels["scatter_csr"]
        k_sc_csr_low = kernels["scatter_csr_low"]
        k_sc_ell     = kernels["scatter_ell"]
        k_sum_dang  = kernels["sum_dangling"]
        k_l1        = kernels["l1_conv"]
        k_reduce    = kernels["reduce_scalar"]
        k_pull      = kernels["pull_gather"]

        # ---- I3+I4: Iteration loop with convergence interval ----------
        start_event.record(stream_compute)
        converged  = False
        iterations = 0

        for it in range(max_iter):
            iterations = it + 1
            _profiling_this_iter = enable_diag and _di < _diag_n

            # ---- Dangling mass sum + dangling scalar ------------------
            if _profiling_this_iter:
                _ev_dang_s[_di].record(stream_compute)

            k_sum_dang(
                d_PR_old, d_dangling_flags, d_partial, np.int32(n),
                block=partial_block, grid=partial_grid,
                stream=stream_compute,
            )
            k_reduce(
                d_partial, d_dangling_scalar, np.int32(n_partial_blocks),
                block=(BLOCK_SIZE, 1, 1), grid=(1, 1, 1),
                stream=stream_compute,
            )

            if _profiling_this_iter:
                _ev_dang_e[_di].record(stream_compute)

            # ---- Init PR_new + CSR scatter + ELLPACK scatter ----------
            if _profiling_this_iter:
                _ev_scat_s[_di].record(stream_compute)

            k_init_dang(
                d_PR_new, teleport_val, d_dangling_scalar,
                d_eligible_mask, np.int32(num_eligible),
                np.float32(damping), np.int32(n),
                block=n_block_dim, grid=n_init_grid,
                stream=stream_compute,
            )

            # Low-degree nodes: thread-per-node kernel (one thread per node,
            # 256 nodes per block) — 100% utilisation on uniform-degree graphs.
            if num_lowdeg > 0 and d_lowdeg_ids is not None:
                k_sc_csr_low(
                    d_csr_row_ptr, d_csr_col_idx, d_csr_values,
                    d_PR_old, d_PR_new, d_out_degree, d_lowdeg_ids,
                    d_pull_target_mask, np.int32(num_lowdeg),
                    np.float32(damping), np.int32(n),
                    block=(block_size, 1, 1),
                    grid=lowdeg_grid,
                    stream=stream_compute,
                )

            # Higher-degree nodes: block-per-node kernel (warp/block tiers +
            # SMEM hash) — few nodes, so one block per node is fine here.
            if num_highdeg > 0 and d_highdeg_ids is not None:
                k_sc_csr(
                    d_csr_row_ptr, d_csr_col_idx, d_csr_values,
                    d_PR_old, d_PR_new, d_out_degree, d_highdeg_ids,
                    d_pull_target_mask, np.int32(num_highdeg),
                    np.float32(damping), np.int32(n),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(num_highdeg, 1, 1),
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

            if _profiling_this_iter:
                _ev_scat_e[_di].record(stream_compute)

            # ---- Pull gather -----------------------------------------
            if pull_enabled and d_pull_ids is not None:
                if _profiling_this_iter:
                    _ev_pull_s[_di].record(stream_compute)

                k_pull(
                    d_row_ptr_T, d_col_idx_T, d_values_T,
                    d_PR_old, d_PR_new, d_out_degree,
                    d_pull_ids, np.int32(n_pull_nodes),
                    np.float32(damping),
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(n_pull_nodes, 1, 1),
                    stream=stream_compute,
                )

                if _profiling_this_iter:
                    _ev_pull_e[_di].record(stream_compute)

            # ---- I3: L1 convergence (only every conv_check_interval) --
            do_check = (
                (iterations % conv_check_interval == 0)
                or (iterations == max_iter)
            )

            if do_check:
                if _profiling_this_iter:
                    _ev_l1_s[_di].record(stream_compute)
                    _diag_l1_iters.append(_di)

                k_l1(
                    d_PR_new, d_PR_old, d_partial, np.int32(n),
                    block=partial_block, grid=partial_grid,
                    stream=stream_compute,
                )
                k_reduce(
                    d_partial, d_l1_scalar, np.int32(n_partial_blocks),
                    block=(BLOCK_SIZE, 1, 1), grid=(1, 1, 1),
                    stream=stream_compute,
                )

                if _profiling_this_iter:
                    _ev_l1_e[_di].record(stream_compute)

            # Pointer swap (every iteration, zero GPU cost)
            d_PR_old, d_PR_new = d_PR_new, d_PR_old

            if _profiling_this_iter:
                _di += 1

            # I3: single sync only when checking convergence
            if do_check:
                stream_compute.synchronize()
                l1_norm = float(d_l1_scalar.get()[0])
                if l1_norm < tolerance:
                    converged = True
                    break

        # ---- Timing ends ---------------------------------------------
        end_event.record(stream_compute)
        end_event.synchronize()
        elapsed = start_event.time_till(end_event) / 1000.0

        # ---- I4: Collect per-phase GPU times -------------------------
        diagnostics: dict = {}
        if enable_diag and _di > 0:
            def _ms(ev_list_s, ev_list_e, n_iters):
                return sum(
                    ev_list_s[i].time_till(ev_list_e[i])
                    for i in range(n_iters)
                ) / max(1, n_iters)

            n_l1 = len(_diag_l1_iters)
            diagnostics = {
                "profiled_iters":     _di,
                "avg_dangling_ms":    _ms(_ev_dang_s, _ev_dang_e, _di),
                "avg_scatter_ms":     _ms(_ev_scat_s, _ev_scat_e, _di),
                "avg_pull_ms":        (
                    _ms(_ev_pull_s, _ev_pull_e, _di)
                    if pull_enabled else 0.0
                ),
                "avg_l1_ms":          (
                    _ms(_ev_l1_s, _ev_l1_e, len(_diag_l1_iters))
                    if n_l1 > 0 else 0.0
                ),
                "conv_check_interval": conv_check_interval,
            }
            logging.info(
                "[PageRank Diagnostics] avg per-iter (first %d iters): "
                "dangling=%.3f ms  scatter=%.3f ms  pull=%.3f ms  "
                "l1=%.3f ms (every %d iters)",
                _di,
                diagnostics["avg_dangling_ms"],
                diagnostics["avg_scatter_ms"],
                diagnostics["avg_pull_ms"],
                diagnostics["avg_l1_ms"],
                conv_check_interval,
            )

        # ---- Result --------------------------------------------------
        scores_host = d_PR_old.get()
        inner = _pack_result(scores_host, out_degrees, iterations,
                             converged, network_type)
        inner["note"] = (
            f"{eligible_note}; {ell_note}; {pull_note}; "
            f"arch={kernels.get('_arch_flag','?')}, "
            f"conv_interval={conv_check_interval}, "
            f"vram_est={breakdown['total_mb']:.1f}MB"
        )
        if diagnostics:
            inner["diagnostics"] = diagnostics
        if pull_diag:
            inner["pull_diagnostics"] = pull_diag

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
        logging.warning("VRAM exhausted in pagerank_gpu.")
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
    pull_indptr_h: np.ndarray | None,
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
    conv_check_interval: int = 5,   # I3
) -> dict:
    """Double-buffer chunked scatter with convergence interval."""
    kernels = _get_kernels()

    try:
        free_bytes, _ = cuda.mem_get_info()
    except Exception:                                   # noqa: BLE001
        free_bytes = 1 << 30

    avg_nnz        = max(1.0, graph_csr.nnz / max(1, n))
    bytes_per_node = (1 + 2 * avg_nnz) * 4
    available      = int(free_bytes * 0.60)
    chunk_size     = max(1, min(n, int(available / max(1.0, bytes_per_node))))

    logging.info(
        "[PageRank chunked] chunk_size=%d (~%d chunks), "
        "bytes_per_node=%.0f, available=%.0f MB",
        chunk_size, max(1, n // chunk_size),
        bytes_per_node, available / 1e6,
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
        _ChunkBuffer(max_chunk_nnz, max_chunk_nodes),
        _ChunkBuffer(max_chunk_nnz, max_chunk_nodes),
    ]
    events  = [cuda.Event(), cuda.Event()]

    d_buffers: list = []

    def _to_gpu(arr):
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

        d_out_degree      = _to_gpu(out_degrees)
        d_dangling_flags  = _to_gpu(dangling_flags)
        d_eligible_mask   = _to_gpu(eligible_mask_host)
        d_pull_target_mask = _to_gpu(
            pull_target_mask_host if pull_enabled
            else np.zeros((n,), dtype=np.uint8)
        )

        d_row_ptr_T = d_col_idx_T = d_values_T = d_pull_ids = None
        if pull_enabled and pull_indptr_h is not None:
            d_row_ptr_T = _to_gpu(pull_indptr_h)
            d_col_idx_T = _to_gpu(pull_indices_h)
            d_values_T  = _to_gpu(pull_values_h)
            d_pull_ids  = _to_gpu(pull_ids)

        d_PR_old          = _empty((n,), np.float32)
        d_PR_new          = _empty((n,), np.float32)
        n_partial_blocks  = max(1, (n + BLOCK_SIZE - 1) // BLOCK_SIZE)
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
        n_pull_nodes  = int(pull_ids.size) if pull_enabled else 0

        k_init_dang = kernels["init_dangling"]
        k_sc_csr    = kernels["scatter_csr"]
        k_sc_ell    = kernels["scatter_ell"]
        k_sum_dang  = kernels["sum_dangling"]
        k_l1        = kernels["l1_conv"]
        k_reduce    = kernels["reduce_scalar"]
        k_pull      = kernels["pull_gather"]

        converged  = False
        iterations = 0

        for it in range(max_iter):
            iterations = it + 1

            k_sum_dang(
                d_PR_old, d_dangling_flags, d_partial, np.int32(n),
                block=partial_block, grid=partial_grid,
                stream=stream_compute,
            )
            k_reduce(
                d_partial, d_dangling_scalar, np.int32(n_partial_blocks),
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

            # Double-buffer chunk scatter
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
                    ns = (i + 1) % 2
                    nc = chunks[i + 1]
                    _transfer_chunk_async(
                        nc["indptr"], nc["indices"],
                        nc["values"], nc["node_ids"],
                        buffers[ns], stream_transfer,
                    )
                    events[ns].record(stream_transfer)

            if num_hubs > 0 and d_ellpack_cols is not None:
                k_sc_ell(
                    d_ellpack_cols, d_ellpack_vals, d_hub_ids,
                    d_PR_old, d_PR_new, d_out_degree, d_pull_target_mask,
                    np.int32(num_hubs), np.int32(max_row_len),
                    np.float32(damping),
                    block=(BLOCK_SIZE, 1, 1), grid=(num_hubs, 1, 1),
                    stream=stream_compute,
                )

            if pull_enabled and d_pull_ids is not None:
                k_pull(
                    d_row_ptr_T, d_col_idx_T, d_values_T,
                    d_PR_old, d_PR_new, d_out_degree,
                    d_pull_ids, np.int32(n_pull_nodes),
                    np.float32(damping),
                    block=(BLOCK_SIZE, 1, 1), grid=(n_pull_nodes, 1, 1),
                    stream=stream_compute,
                )

            # I3: convergence every N iters
            do_check = (
                (iterations % conv_check_interval == 0)
                or (iterations == max_iter)
            )
            if do_check:
                k_l1(
                    d_PR_new, d_PR_old, d_partial, np.int32(n),
                    block=partial_block, grid=partial_grid,
                    stream=stream_compute,
                )
                k_reduce(
                    d_partial, d_l1_scalar, np.int32(n_partial_blocks),
                    block=(BLOCK_SIZE, 1, 1), grid=(1, 1, 1),
                    stream=stream_compute,
                )

            d_PR_old, d_PR_new = d_PR_new, d_PR_old

            if do_check:
                stream_compute.synchronize()
                if float(d_l1_scalar.get()[0]) < tolerance:
                    converged = True
                    break

        end_event.record(stream_compute)
        end_event.synchronize()
        elapsed = start_event.time_till(end_event) / 1000.0

        scores_host = d_PR_old.get()
        inner = _pack_result(scores_host, out_degrees, iterations,
                             converged, network_type)
        inner["note"] = (
            f"{eligible_note}; chunked ({len(chunks)} chunks ~{chunk_size} rows); "
            f"num_hubs={ellpack_data['num_hubs']}; "
            f"conv_interval={conv_check_interval}; "
            f"arch={kernels.get('_arch_flag','?')}"
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
# I6: CSR-only lightweight fast path
# ---------------------------------------------------------------------------

def pagerank_gpu_simple(
    graph_csr: sp.csr_matrix,
    params: dict | None = None,
) -> dict:
    """Lightweight PageRank: CSR only, no ELLPACK, no pull, convergence every 5 iters.

    I6 rationale
    ------------
    On power-law biological graphs the preprocessing overhead for ELLPACK
    (hub detection + array construction) and pull mode (transpose CSR build,
    extra GPU arrays) can exceed their runtime benefits.  This wrapper
    disables both and sets conv_check_interval=5.

    Use this as a fast baseline when benchmarking or when the graph is too
    large for ELLPACK (which would have been auto-disabled anyway via I1).

    Benchmarking tip: call benchmark_pagerank_optimizations() to see whether
    this simple path is actually faster than the full implementation on your
    specific graph.
    """
    simple_p = {**(params or {})}
    simple_p.update({
        "use_ellpack":        False,
        "pull_threshold":     0,
        "conv_check_interval": 5,
    })
    return pagerank_gpu(graph_csr, simple_p)


# ---------------------------------------------------------------------------
# I5: Optimization effectiveness benchmark
# ---------------------------------------------------------------------------

def benchmark_pagerank_optimizations(
    graph_csr: sp.csr_matrix,
    params: dict | None = None,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> dict:
    """Measure the runtime contribution of each PageRank optimization.

    I5: Runs five configurations, each differing by a single variable:
      simple_csr     : CSR only, no ELLPACK, no pull, check every iter
      conv_interval5 : CSR only, no ELLPACK, no pull, check every 5 iters
      ellpack_on     : ELLPACK enabled, no pull, check every iter
      pull_on        : no ELLPACK, pull auto, check every iter
      all_on         : ELLPACK + pull auto + conv_interval=5

    The fastest configuration is identified and a recommendation is returned.
    Use this to confirm which optimizations help on your graph before
    enabling them in production.

    Parameters
    ----------
    graph_csr  : graph to benchmark.
    params     : base parameters (network_type, damping, etc.).  Each
                 configuration overrides the optimization-specific keys.
    n_warmup   : warm-up runs per configuration (kernel compile, VRAM alloc).
    n_runs     : timed runs per configuration.

    Returns
    -------
    dict with keys: configurations (per-config timing), ranked (list of
    config names by mean time), fastest, recommendation, graph_info.
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError("PyCUDA required for benchmark_pagerank_optimizations().")

    base = _merge_params(params)

    configs = {
        "simple_csr": {
            **base,
            "use_ellpack":         False,
            "pull_threshold":      0,
            "conv_check_interval": 1,
        },
        "conv_interval5": {
            **base,
            "use_ellpack":         False,
            "pull_threshold":      0,
            "conv_check_interval": 5,
        },
        "ellpack_on": {
            **base,
            "use_ellpack":         True,
            "pull_threshold":      0,
            "conv_check_interval": 1,
        },
        "pull_on": {
            **base,
            "use_ellpack":         False,
            "pull_threshold":      -1,
            "conv_check_interval": 1,
        },
        "all_on": {
            **base,
            "use_ellpack":         True,
            "pull_threshold":      -1,
            "conv_check_interval": 5,
        },
    }

    results: dict[str, dict] = {}

    for name, cfg in configs.items():
        logging.info("[benchmark_pagerank] running config '%s' ...", name)
        for _ in range(max(1, n_warmup)):
            try:
                pagerank_gpu(graph_csr, cfg)
            except Exception as exc:               # noqa: BLE001
                logging.debug("warmup failed for %s: %s", name, exc)

        times: list[float] = []
        iters_list: list[int] = []
        conv_list:  list[bool] = []
        for _ in range(max(1, n_runs)):
            try:
                r = pagerank_gpu(graph_csr, cfg)
                times.append(float(r["execution_time"]) * 1000.0)  # ms
                iters_list.append(int(r["result"]["iterations"]))
                conv_list.append(bool(r["result"]["converged"]))
            except Exception as exc:               # noqa: BLE001
                logging.warning("[benchmark_pagerank] %s run failed: %s", name, exc)
                times.append(float("inf"))

        valid = [t for t in times if t < float("inf")]
        arr   = np.asarray(valid) if valid else np.array([float("inf")])
        results[name] = {
            "mean_ms":    float(np.mean(arr)),
            "std_ms":     float(np.std(arr)),
            "min_ms":     float(np.min(arr)),
            "max_ms":     float(np.max(arr)),
            "valid_runs": len(valid),
            "iterations": int(np.median(iters_list)) if iters_list else 0,
            "converged":  bool(all(conv_list)) if conv_list else False,
            "config_keys": {
                k: cfg[k]
                for k in ("use_ellpack", "pull_threshold", "conv_check_interval")
            },
        }
        logging.info(
            "[benchmark_pagerank] %s: mean=%.2f ms  std=%.2f ms  iters=%d",
            name, results[name]["mean_ms"], results[name]["std_ms"],
            results[name]["iterations"],
        )

    # Rank by mean time (ascending)
    ranked = sorted(results.keys(),
                    key=lambda k: results[k]["mean_ms"])
    fastest = ranked[0] if ranked else None

    # Recommendation
    simple_ms = results.get("simple_csr", {}).get("mean_ms", float("inf"))
    best_ms   = results.get(fastest or "", {}).get("mean_ms", float("inf"))

    if fastest == "simple_csr":
        rec = (
            "CSR-only (no ELLPACK, no pull) is fastest. "
            "Set use_ellpack=False, pull_threshold=0 for production."
        )
    elif best_ms < simple_ms * 0.95:
        gain = simple_ms / max(best_ms, 1e-9)
        rec = (
            f"Configuration '{fastest}' is {gain:.2f}x faster than CSR-only. "
            f"Enable its settings for production."
        )
    else:
        rec = (
            "All configurations perform similarly. "
            "Use simple_csr for lowest complexity."
        )

    return {
        "configurations": results,
        "ranked":         ranked,
        "fastest":        fastest,
        "recommendation": rec,
        "graph_info": {
            "n":   int(graph_csr.shape[0]),
            "nnz": int(graph_csr.nnz),
            "avg_degree": float(graph_csr.nnz / max(1, graph_csr.shape[0])),
        },
        "benchmark_params": {"n_warmup": n_warmup, "n_runs": n_runs},
    }


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _gpu(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    """Runner entry point — {output, extra_params} envelope."""
    p    = _merge_params(params)
    full = pagerank_gpu(graph_csr, p)
    return {"output": full, "extra_params": p}
