"""
src/optimization/memory_manager.py — Central memory-aware execution planner
============================================================================

The ``MemoryManager`` decides — once per algorithm call — *how* a GPU
kernel pipeline should access its inputs.  Five execution modes are
supported:

  ============================  =============================================
    ``normal_gpu``                fits comfortably in VRAM -> standard device
                                allocation.
  ``chunked_gpu``               graph too big for one upload but small enough
                                that streaming row-chunks works.  Used by the
                                three algorithms that already implement a
                                ``_*_gpu_chunked`` path (pagerank, rwr,
                                louvain).
  ``zero_copy``                 page-locked host memory mapped into the device
                                — no upload at all, but every access pays
                                PCIe.  Used when ``chunked_gpu`` would still
                                miss because the *score vectors* are also too
                                big.
  ``unified_memory``            ``cuda.managed_empty`` — pages migrate on
                                demand.  Preferred over ``zero_copy`` when
                                the driver / device support it (Pascal+).
  ``partitioned_gpu``           graph won't fit even with chunking + score
                                vectors held off-device.  Caller must consume
                                ``GraphPartitioner`` output.
  ============================  =============================================

Decision order (highest-priority first, first match wins):

    estimate < 70 % of free VRAM                    -> normal_gpu
    algorithm supports chunking                     -> chunked_gpu
    driver supports cudaMallocManaged               -> unified_memory
    driver supports pagelocked + DEVICEMAP          -> zero_copy
    otherwise                                       -> partitioned_gpu

The plan is a *recommendation* — algorithm files inspect the plan and
either honour it (PageRank/RWR/Louvain do) or raise ``MemoryError``
(HITS/MCL/BFS until they implement out-of-core support).  No path
silently falls back to CPU.

This module intentionally has zero algorithm-specific kernel knowledge.
It uses :class:`MemoryEstimator` from ``gpu_config`` for the size
estimate and :mod:`unified_memory` for the capability probe.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional PyCUDA — for the live ``mem_get_info`` query
# ---------------------------------------------------------------------------

try:
    import pycuda.driver as cuda
    PYCUDA_AVAILABLE = True
except Exception:                                       # noqa: BLE001
    cuda = None                                         # type: ignore[assignment]
    PYCUDA_AVAILABLE = False


# Algorithms that already implement a chunked GPU path.  Listed here so
# the planner doesn't recommend a path the algorithm cannot follow.
_CHUNK_CAPABLE_ALGORITHMS: frozenset[str] = frozenset({
    "pagerank", "rwr", "louvain",
})

# Algorithms that accept partitioned input metadata in their kernel
# pipeline.  None today — populating this set is the next milestone.
_PARTITION_CAPABLE_ALGORITHMS: frozenset[str] = frozenset()

# Algorithms whose primary working set is well-suited to unified
# memory (scores stay device-resident, sparse data migrates).
_UNIFIED_FRIENDLY_ALGORITHMS: frozenset[str] = frozenset({
    "pagerank", "rwr", "hits",
})

# Pressure thresholds for the decision matrix.
_NORMAL_THRESHOLD: float   = 0.70   # < 70 % free VRAM -> normal_gpu
_CHUNKING_THRESHOLD: float = 1.50   # < 150 % free VRAM -> chunked_gpu works


# ---------------------------------------------------------------------------
# Memory query helpers
# ---------------------------------------------------------------------------

def get_available_vram() -> dict:
    """Return ``{"free_mb": int, "total_mb": int}`` for the current device.

    Falls back to ``{free_mb: 0, total_mb: 0}`` when PyCUDA is missing
    or no context is active — callers MUST handle the zero case.
    """
    if not PYCUDA_AVAILABLE:
        return {"free_mb": 0, "total_mb": 0}
    try:
        free_b, total_b = cuda.mem_get_info()
        return {
            "free_mb":  int(free_b // (1024 * 1024)),
            "total_mb": int(total_b // (1024 * 1024)),
        }
    except Exception:                                   # noqa: BLE001
        return {"free_mb": 0, "total_mb": 0}


# ---------------------------------------------------------------------------
# Estimator + planner
# ---------------------------------------------------------------------------

def estimate_algorithm_memory(
    algorithm_name: str,
    graph_csr,
    params: dict | None = None,
) -> dict:
    """Estimate VRAM working set for an algorithm + graph.

    Delegates to :class:`MemoryEstimator` from ``gpu_config`` so the
    multipliers (SpGEMM fill-in for MCL, A+A^T for HITS, etc.) stay in
    one place.  Returns the estimator dict with an extra
    ``free_mb`` and ``total_mb`` from the live VRAM query.
    """
    # Lazy import to avoid the gpu_config <-> memory_manager cycle.
    from src.optimization.gpu_config import (   # noqa: PLC0415
        GraphProfiler, MemoryEstimator, get_gpu_config,
    )

    cfg = get_gpu_config()
    try:
        profile = GraphProfiler.profile(graph_csr)
    except Exception:                                   # noqa: BLE001
        # Non-CSR input — coarse hand-rolled estimate.
        try:
            n   = int(graph_csr.shape[0])
            nnz = int(graph_csr.nnz)
        except Exception:                               # noqa: BLE001
            n, nnz = 0, 0
        csr_b = (n + 1 + nnz + nnz) * 4
        profile = {
            "n":              n,
            "m":              nnz,
            "density":        (nnz / max(n * n, 1)) if n else 0.0,
            "avg_degree":     (nnz / max(n, 1)) if n else 0.0,
            "max_degree":     0,
            "vram_estimate_mb": csr_b / (1024 * 1024),
            "sparsity_class": "sparse",
            "degree_class":   "uniform",
        }

    est = MemoryEstimator.estimate(algorithm_name, profile, cfg)
    vram = get_available_vram()
    est["free_mb"]  = vram["free_mb"]
    est["total_mb"] = vram["total_mb"]
    est["graph_profile"] = profile
    return est


def compute_chunk_size(
    graph_csr,
    available_vram_mb: int,
    algorithm_name: str,
    *,
    safety_fraction: float = 0.50,
) -> int:
    """Compute a chunk size (in nodes) that fits ``safety_fraction`` of free VRAM.

    Uses average bytes-per-row = ``avg_degree * (4 indptr + 4 col + 4 val)``.
    Clamped to ``[1, n]``.
    """
    try:
        n   = int(graph_csr.shape[0])
        nnz = int(graph_csr.nnz)
    except Exception:                                   # noqa: BLE001
        return 0

    if n <= 0:
        return 0
    avg_deg = max(1.0, nnz / n)
    # Per row: 1 indptr int + avg_deg * (col_idx int + value float) = 4 + 8*avg_deg
    bytes_per_row = 4 + int(avg_deg * 8)
    budget_bytes  = max(1, int(available_vram_mb * safety_fraction * 1024 * 1024))
    chunk = max(1, budget_bytes // bytes_per_row)
    return min(chunk, n)


def should_use_unified_memory(memory_estimate: dict, available_vram_mb: int) -> bool:
    """Recommend unified memory when above the chunking ceiling AND the
    driver/device support it.
    """
    from src.optimization.unified_memory import capabilities   # noqa: PLC0415
    caps = capabilities()
    if not caps["managed_memory"]:
        return False
    total = float(memory_estimate.get("total_mb") or 0.0)
    avail = float(available_vram_mb or 0.0)
    if avail <= 0:
        return False
    return total > _CHUNKING_THRESHOLD * avail


def should_use_zero_copy(memory_estimate: dict, available_vram_mb: int) -> bool:
    """Recommend zero-copy only when unified isn't available but PCIe-
    mapped pinned memory is, AND we're above the chunking ceiling.
    """
    from src.optimization.unified_memory import capabilities   # noqa: PLC0415
    caps = capabilities()
    if not caps["pagelocked_memory"]:
        return False
    if caps["managed_memory"]:
        # Unified always preferred when available.
        return False
    total = float(memory_estimate.get("total_mb") or 0.0)
    avail = float(available_vram_mb or 0.0)
    if avail <= 0:
        return False
    return total > _CHUNKING_THRESHOLD * avail


def should_partition_graph(memory_estimate: dict, available_vram_mb: int) -> bool:
    """Recommend partitioned execution as the last resort."""
    total = float(memory_estimate.get("total_mb") or 0.0)
    avail = float(available_vram_mb or 0.0)
    if avail <= 0:
        return total > 0.0
    return total > _CHUNKING_THRESHOLD * avail


# ---------------------------------------------------------------------------
# The planner itself
# ---------------------------------------------------------------------------

class MemoryManager:
    """Central memory-aware execution planner for GPU algorithms.

    The class is a namespace of ``@staticmethod`` helpers — no per-instance
    state.  All caching (graph profile, strategy) lives in
    ``gpu_config``; we re-use it here so the planner is consistent with
    the rest of the optimisation layer.
    """

    @staticmethod
    def get_available_vram() -> dict:
        return get_available_vram()

    @staticmethod
    def estimate_algorithm_memory(
        algorithm_name: str, graph_csr, params: dict | None = None,
    ) -> dict:
        return estimate_algorithm_memory(algorithm_name, graph_csr, params)

    @staticmethod
    def compute_chunk_size(
        graph_csr, available_vram_mb: int, algorithm_name: str,
    ) -> int:
        return compute_chunk_size(graph_csr, available_vram_mb, algorithm_name)

    @staticmethod
    def should_use_unified_memory(memory_estimate, available_vram_mb) -> bool:
        return should_use_unified_memory(memory_estimate, available_vram_mb)

    @staticmethod
    def should_use_zero_copy(memory_estimate, available_vram_mb) -> bool:
        return should_use_zero_copy(memory_estimate, available_vram_mb)

    @staticmethod
    def should_partition_graph(memory_estimate, available_vram_mb) -> bool:
        return should_partition_graph(memory_estimate, available_vram_mb)

    @staticmethod
    def select_execution_mode(
        algorithm_name: str,
        graph_csr,
        params: dict | None = None,
    ) -> dict:
        """Pick an execution mode and return a complete strategy dict.

        Returns
        -------
        dict
            Keys (consumed by ``apply_config`` and algorithm files):
              * ``execution_mode``        — one of the five mode names
              * ``estimated_memory_mb``   — total estimated working set
              * ``available_vram_mb``     — live free VRAM
              * ``memory_pressure``       — low / medium / high / critical
              * ``use_chunking``          — recommendation flag
              * ``chunk_size``            — int (nodes per chunk) or 0
              * ``use_unified_memory``    — recommendation flag
              * ``use_zero_copy``         — recommendation flag
              * ``use_partitioning``      — recommendation flag
              * ``partition_strategy``    — "by_edges" | "by_degree" | None
              * ``mode_reason``           — short human-readable why
        """
        est = estimate_algorithm_memory(algorithm_name, graph_csr, params)
        free_mb = int(est.get("free_mb") or 0)
        total_mb = float(est.get("total_mb_alloc")
                         or est.get("total_mb")
                         or 0.0)
        # ``MemoryEstimator.estimate`` already classifies pressure.
        pressure = str(est.get("pressure", "low"))

        algo = algorithm_name.lower().strip()
        chunk_capable = algo in _CHUNK_CAPABLE_ALGORITHMS
        unified_ok    = should_use_unified_memory(est, free_mb)
        zero_copy_ok  = should_use_zero_copy(est, free_mb)
        partition_ok  = should_partition_graph(est, free_mb)

        # ---- Decision matrix --------------------------------------------
        if pressure == "low" or (
            free_mb > 0 and total_mb < _NORMAL_THRESHOLD * free_mb
        ):
            mode   = "normal_gpu"
            reason = "estimate < 70% of free VRAM"
        elif chunk_capable:
            mode   = "chunked_gpu"
            reason = f"{algo} supports chunked path, pressure={pressure}"
        elif unified_ok:
            mode   = "unified_memory"
            reason = "CUDA managed memory available; no chunked path"
        elif zero_copy_ok:
            mode   = "zero_copy"
            reason = "DEVICEMAP pagelocked available; no unified memory"
        else:
            mode   = "partitioned_gpu"
            reason = "no chunked / unified / zero-copy path available"

        # Independent flags (kept additive so multiple paths can co-apply)
        chunk_size = (
            compute_chunk_size(graph_csr, free_mb, algorithm_name)
            if mode == "chunked_gpu" else 0
        )

        partition_strategy: Optional[str]
        if mode == "partitioned_gpu":
            # Power-law / skewed graphs benefit from edge-balanced.
            deg_class = est.get("graph_profile", {}).get("degree_class", "uniform")
            partition_strategy = (
                "by_edges" if deg_class in ("skewed", "power_law")
                else "by_degree"
            )
        else:
            partition_strategy = None

        plan = {
            "execution_mode":      mode,
            "estimated_memory_mb": total_mb,
            "available_vram_mb":   free_mb,
            "memory_pressure":     pressure,
            "use_chunking":        (mode == "chunked_gpu"),
            "chunk_size":          chunk_size,
            "use_unified_memory":  (mode == "unified_memory"),
            "use_zero_copy":       (mode == "zero_copy"),
            "use_partitioning":    (mode == "partitioned_gpu"),
            "partition_strategy":  partition_strategy,
            "mode_reason":         reason,
        }
        return plan

    @staticmethod
    def plan(algorithm_name: str, graph_csr, params: dict | None = None) -> dict:
        """Alias of :meth:`select_execution_mode`."""
        return MemoryManager.select_execution_mode(
            algorithm_name, graph_csr, params,
        )


__all__ = [
    "MemoryManager",
    "get_available_vram",
    "estimate_algorithm_memory",
    "compute_chunk_size",
    "should_use_unified_memory",
    "should_use_zero_copy",
    "should_partition_graph",
]
