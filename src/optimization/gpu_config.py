"""
optimization/gpu_config.py — Graph-, Algorithm-, and Runtime-Aware GPU Config
=============================================================================

Overview
--------
This module is the optimisation brain that sits between
``algorithm_runner.py`` and the six GPU algorithm implementations.  It is
called *before* every GPU run as ``apply_config(algo, graph_csr, params)``
and produces a merged params dict that combines:

  1. Hardware capabilities             (``get_gpu_config``)
  2. Graph-structural properties       (``GraphProfiler``)
  3. Algorithm-specific memory model   (``MemoryEstimator``)
  4. Algorithm-specific strategy choice(``AlgorithmStrategySelector``)
  5. Runtime feedback from past runs   (``RuntimeProfiler``)
  6. Caller-supplied user overrides

Every algorithm file (``pagerank.py``, ``bfs.py``, ``hits.py``,
``louvain.py``, ``rwr.py``, ``mcl.py``) consumes the returned dict via
``setdefault`` / `.get()` — unknown keys are silently ignored, so
this module can grow without touching any of the algorithm files.

Public API (backward compatible)
--------------------------------
get_gpu_config()        -> dict
    Detect the host GPU once per process and return the structured
    hardware config dict.  Cached at module level.

apply_config(name, csr, params, override=False, enable_profiling=False)
    Merge hardware + graph + algorithm + runtime params into a single
    dict.  Backward compatible signature — the two new keyword args
    default to the old behaviour.

print_gpu_summary(graph_csr=None, algorithm=None)
    Print a boxed human-readable summary.  When ``graph_csr`` and
    ``algorithm`` are provided, also prints graph profile and
    strategy choices.

New components (additive, all CPU-only — no CUDA calls inside)
--------------------------------------------------------------
GraphProfiler           — structural analysis (degree distribution,
                          symmetry, bipartite, format hint).
MemoryEstimator         — algorithm-specific VRAM pressure model.
AlgorithmStrategySelector — graph-aware strategy choice per algorithm.
RuntimeProfiler         — in-memory execution history, returns
                          tolerance / max_iter recommendations.
generate_profile_report — structured report (logging / display).

Caching
-------
_HARDWARE_CONFIG          : detected once per process
_GRAPH_PROFILE_CACHE      : {fingerprint: profile}
_STRATEGY_CACHE           : {(algo, fingerprint): strategy}
RuntimeProfiler._history  : {(algo, fingerprint): [run_records...]}
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Optional pycuda
# ---------------------------------------------------------------------------

try:
    import pycuda.driver as _pycuda
    _pycuda.init()
    _PYCUDA_AVAILABLE = True
except Exception:                                       # noqa: BLE001
    _pycuda = None                                      # type: ignore[assignment]
    _PYCUDA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

_CACHED_CONFIG: dict | None = None
_HARDWARE_CONFIG: dict | None = None              # alias of _CACHED_CONFIG
_GRAPH_PROFILE_CACHE: dict[str, dict] = {}
_STRATEGY_CACHE: dict[tuple[str, str], dict] = {}
# MEMORY_FIX (Fix Cat. 5): per-graph derived-artefact caches keyed by
# GraphProfiler fingerprint.  Repeated benchmark runs against the same
# graph reuse these rather than recomputing the transpose / degrees /
# ELLPACK split every algorithm invocation.
_TRANSPOSE_CACHE: dict[str, tuple] = {}   # {fp: (indptr, indices, data)}
_DEGREE_CACHE:    dict[str, tuple] = {}   # {fp: (out_deg, in_deg)}

_VALID_ALGORITHMS = ("pagerank", "louvain", "rwr", "hits", "bfs", "mcl")


def _reset_cache() -> None:
    """Clear all module-level caches.  Used by tests."""
    global _CACHED_CONFIG, _HARDWARE_CONFIG
    _CACHED_CONFIG = None
    _HARDWARE_CONFIG = None
    _GRAPH_PROFILE_CACHE.clear()
    _STRATEGY_CACHE.clear()
    _TRANSPOSE_CACHE.clear()
    _DEGREE_CACHE.clear()


# ---------------------------------------------------------------------------
# Per-graph derived-artefact caches (MEMORY_FIX, Fix Category 5)
# ---------------------------------------------------------------------------

def get_cached_transpose(graph_csr, fingerprint: str) -> tuple:
    """Return ``(row_ptr_T, col_idx_T, values_T)`` as cached int32/float32
    arrays of the transposed CSR.  Computed once per fingerprint."""
    import numpy as _np
    import gc as _gc
    cached = _TRANSPOSE_CACHE.get(fingerprint)
    if cached is not None:
        return cached
    A_T = graph_csr.T.tocsr()
    arrays = (
        _np.ascontiguousarray(A_T.indptr,  _np.int32),
        _np.ascontiguousarray(A_T.indices, _np.int32),
        _np.ascontiguousarray(A_T.data,    _np.float32),
    )
    _TRANSPOSE_CACHE[fingerprint] = arrays
    del A_T
    _gc.collect()
    return arrays


def get_cached_degrees(graph_csr, fingerprint: str) -> tuple:
    """Return ``(out_degrees, in_degrees)`` as float32 arrays."""
    import numpy as _np
    cached = _DEGREE_CACHE.get(fingerprint)
    if cached is not None:
        return cached
    out_deg = _np.asarray(graph_csr.sum(axis=1)).flatten().astype(_np.float32)
    in_deg  = _np.asarray(graph_csr.sum(axis=0)).flatten().astype(_np.float32)
    _DEGREE_CACHE[fingerprint] = (out_deg, in_deg)
    return _DEGREE_CACHE[fingerprint]


# ---------------------------------------------------------------------------
# GPU detection backends (unchanged from previous version)
# ---------------------------------------------------------------------------

def _query_pycuda() -> dict | None:
    """Query device 0 via pycuda.driver."""
    if not _PYCUDA_AVAILABLE:
        return None
    try:
        if _pycuda.Device.count() == 0:
            return None
        device = _pycuda.Device(0)
        cc_major, cc_minor = device.compute_capability()
        attr = _pycuda.device_attribute

        # Optional attributes — wrap individually because some older drivers
        # don't expose them and we don't want one missing attr to abort
        # detection.
        def _safe_attr(name: str, default: int = 0) -> int:
            try:
                return int(device.get_attribute(getattr(attr, name)))
            except Exception:                           # noqa: BLE001
                return default

        info = {
            "device_name":            device.name(),
            "vram_mb":                int(device.total_memory() / (1024 ** 2)),
            "compute_capability":     f"{cc_major}.{cc_minor}",
            "cc_major":               int(cc_major),
            "cc_minor":               int(cc_minor),
            "warp_size":              _safe_attr("WARP_SIZE", 32),
            "multiprocessor_count":   _safe_attr("MULTIPROCESSOR_COUNT", 0),
            "max_threads_per_block":  _safe_attr("MAX_THREADS_PER_BLOCK", 1024),
            "shared_mem_per_block_bytes":
                _safe_attr("MAX_SHARED_MEMORY_PER_BLOCK", 49152),
            "l2_cache_size_bytes":    _safe_attr("L2_CACHE_SIZE", 0),
            "free_vram_mb":           0,
        }

        try:
            ctx = device.make_context()
            try:
                free, _total = _pycuda.mem_get_info()
                info["free_vram_mb"] = int(free / (1024 ** 2))
            finally:
                ctx.pop()
                ctx.detach()
        except Exception:                               # noqa: BLE001
            pass

        return info
    except Exception:                                   # noqa: BLE001
        return None


def _query_nvidia_smi() -> dict | None:
    """Fallback detection via the nvidia-smi binary."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
            "-i", "0",
        ]
        out = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=5
        ).decode("utf-8", errors="replace")
        line = out.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        cc_str = parts[3]
        try:
            cc_major, cc_minor = cc_str.split(".")
            cc_major, cc_minor = int(cc_major), int(cc_minor)
        except Exception:                               # noqa: BLE001
            cc_major, cc_minor = 0, 0
        return {
            "device_name":            parts[0],
            "vram_mb":                int(float(parts[1])),
            "free_vram_mb":           int(float(parts[2])),
            "compute_capability":     cc_str,
            "cc_major":               cc_major,
            "cc_minor":               cc_minor,
            "warp_size":              32,
            "multiprocessor_count":   0,
            "max_threads_per_block":  1024,
            "shared_mem_per_block_bytes": 49152,
            "l2_cache_size_bytes":    0,
        }
    except Exception:                                   # noqa: BLE001
        return None


def _query_free_vram_nvidia_smi() -> int | None:
    """Return free VRAM in MB via nvidia-smi only, or None on failure."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "-i", "0",
        ]
        out = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=5
        ).decode("utf-8", errors="replace")
        return int(float(out.strip().splitlines()[0]))
    except Exception:                                   # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Architecture name mapping
# ---------------------------------------------------------------------------

_ARCH_NAMES: dict[tuple[int, int], str] = {
    (5, 0): "Maxwell",  (5, 2): "Maxwell",
    (6, 0): "Pascal",   (6, 1): "Pascal",
    (7, 0): "Volta",    (7, 2): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",   (8, 6): "Ampere",  (8, 9): "Ada",
    (9, 0): "Hopper",
}


def _architecture_name(cc_str: str) -> str:
    """Map a compute capability string to a marketing arch name."""
    try:
        major_s, minor_s = cc_str.split(".")
        major, minor = int(major_s), int(minor_s)
    except Exception:                                   # noqa: BLE001
        return "Unknown"
    if (major, minor) in _ARCH_NAMES:
        return _ARCH_NAMES[(major, minor)]
    # Fallback to major version
    fam_fallback = {5: "Maxwell", 6: "Pascal", 7: "Volta/Turing",
                    8: "Ampere/Ada", 9: "Hopper"}
    return fam_fallback.get(major, f"Compute {cc_str}")


# ---------------------------------------------------------------------------
# Tier classification & base recommendations (unchanged behaviour)
# ---------------------------------------------------------------------------

def _classify_tier(vram_mb: int, cuda_available: bool) -> str:
    if not cuda_available:
        return "cpu_only"
    if vram_mb >= 8192:
        return "high"
    if vram_mb >= 6144:
        return "mid_large"
    if vram_mb >= 3072:
        return "mid_small"
    return "cpu_only"


def _base_recommendations(tier: str) -> dict:
    """Return per-algorithm base config for the given tier."""
    if tier == "cpu_only":
        return {
            algo: {"gpu_disabled": True, "block_size": 0, "use_chunking": False}
            for algo in _VALID_ALGORITHMS
        }

    pagerank_common = {"precision": "float32", "use_shared_mem": True}
    pagerank_tier = {
        "high":      {"block_size": 512, "use_chunking": False, "use_zero_copy": False},
        "mid_large": {"block_size": 256, "use_chunking": False, "use_zero_copy": False},
        "mid_small": {"block_size": 256, "use_chunking": True,  "use_zero_copy": True},
    }

    louvain_common = {"use_shared_mem_hash": True}
    louvain_tier = {
        "high":      {"block_size": 256, "use_chunking": False, "community_id_bits": 32},
        "mid_large": {"block_size": 256, "use_chunking": False, "community_id_bits": 16},
        "mid_small": {"block_size": 128, "use_chunking": True,  "community_id_bits": 16},
    }

    rwr_common = {"precision": "float32"}
    rwr_tier = {
        "high":      {"block_size": 256, "use_zero_copy": False, "batch_seeds": True},
        "mid_large": {"block_size": 256, "use_zero_copy": False, "batch_seeds": False},
        "mid_small": {"block_size": 128, "use_zero_copy": True,  "batch_seeds": False},
    }

    hits_common = {"precision": "float32", "use_shared_mem": True}
    hits_tier = {
        "high":      {"block_size": 256, "use_chunking": False},
        "mid_large": {"block_size": 256, "use_chunking": False},
        "mid_small": {"block_size": 128, "use_chunking": True},
    }

    bfs_common = {"use_bitmap_frontier": True}
    bfs_tier = {
        "high":      {"block_size": 512, "use_direction_opt": True},
        "mid_large": {"block_size": 256, "use_direction_opt": True},
        "mid_small": {"block_size": 128, "use_direction_opt": False},
    }

    mcl_common = {"precision": "float32"}
    mcl_tier = {
        "high":      {"block_size": 128, "prune_threshold": 0.001,
                      "use_chunking": False, "top_k_per_column": 50},
        "mid_large": {"block_size": 128, "prune_threshold": 0.002,
                      "use_chunking": True,  "top_k_per_column": 30},
        "mid_small": {"block_size": 64,  "prune_threshold": 0.005,
                      "use_chunking": True,  "top_k_per_column": 20},
    }

    return {
        "pagerank": {**pagerank_common, **pagerank_tier[tier]},
        "louvain":  {**louvain_common,  **louvain_tier[tier]},
        "rwr":      {**rwr_common,      **rwr_tier[tier]},
        "hits":     {**hits_common,     **hits_tier[tier]},
        "bfs":      {**bfs_common,      **bfs_tier[tier]},
        "mcl":      {**mcl_common,      **mcl_tier[tier]},
    }


def _parse_compute_capability(cc_str: str) -> float:
    try:
        return float(cc_str)
    except (TypeError, ValueError):
        return 0.0


def _apply_cc_adjustments(recommended: dict, cc_str: str, tier: str,
                          max_threads_per_block: int) -> None:
    """Mutate `recommended` in place based on compute capability."""
    cc = _parse_compute_capability(cc_str)

    if cc > 0.0 and cc < 7.0:
        for algo_cfg in recommended.values():
            if "use_shared_mem" in algo_cfg:
                algo_cfg["use_shared_mem"] = False
            if "use_shared_mem_hash" in algo_cfg:
                algo_cfg["use_shared_mem_hash"] = False
            algo_cfg["legacy_kernel_mode"] = True
        if "use_bitmap_frontier" in recommended["bfs"]:
            recommended["bfs"]["use_bitmap_frontier"] = False
        if "use_direction_opt" in recommended["bfs"]:
            recommended["bfs"]["use_direction_opt"] = False

    if abs(cc - 7.5) < 1e-6:
        if tier == "high":
            recommended["bfs"]["block_size"] = 512

    if cc >= 8.0 and tier == "high":
        cap = max_threads_per_block if max_threads_per_block > 0 else 1024
        for algo in ("pagerank", "hits"):
            cur = recommended[algo].get("block_size", 0)
            if 0 < cur < cap:
                recommended[algo]["block_size"] = min(cur * 2, cap)


def _apply_sm_adjustments(recommended: dict, sm_count: int) -> None:
    """Halve every block_size on low-SM devices."""
    if sm_count <= 0 or sm_count >= 20:
        return
    for algo_cfg in recommended.values():
        bs = algo_cfg.get("block_size", 0)
        if bs > 0:
            algo_cfg["block_size"] = max(32, bs // 2)
        algo_cfg["low_sm_mode"] = True


# ---------------------------------------------------------------------------
# CPU-only fallback
# ---------------------------------------------------------------------------

def _cpu_only_config() -> dict:
    return {
        "device_name":            "none",
        "vram_mb":                0,
        "free_vram_mb":           0,
        "compute_capability":     "0.0",
        "cc_major":               0,
        "cc_minor":               0,
        "warp_size":              32,
        "multiprocessor_count":   0,
        "max_threads_per_block":  0,
        "shared_mem_per_block_bytes": 49152,
        "l2_cache_size_bytes":    0,
        "cuda_available":         False,
        "tier":                   "cpu_only",
        "architecture":           "Unknown",
        "mode":                   "cpu_only",
        "recommended":            _base_recommendations("cpu_only"),
    }


# ---------------------------------------------------------------------------
# Public: get_gpu_config()
# ---------------------------------------------------------------------------

def get_gpu_config() -> dict:
    """Detect the GPU exactly once and return the full hardware config dict."""
    global _CACHED_CONFIG, _HARDWARE_CONFIG
    if _CACHED_CONFIG is not None:
        return _CACHED_CONFIG

    info = _query_pycuda()
    if info is None:
        info = _query_nvidia_smi()

    if info is None:
        cfg = _cpu_only_config()
        logging.info("No CUDA device found. All GPU modes will be skipped.")
        _CACHED_CONFIG = cfg
        _HARDWARE_CONFIG = cfg
        return cfg

    if info.get("free_vram_mb", 0) == 0:
        free = _query_free_vram_nvidia_smi()
        if free is not None:
            info["free_vram_mb"] = free
        else:
            info["free_vram_mb"] = info["vram_mb"]

    tier = _classify_tier(info["vram_mb"], cuda_available=True)
    recommended = _base_recommendations(tier)
    _apply_cc_adjustments(
        recommended, info["compute_capability"], tier,
        info["max_threads_per_block"],
    )
    _apply_sm_adjustments(recommended, info["multiprocessor_count"])

    cfg = {
        "device_name":            info["device_name"],
        "vram_mb":                info["vram_mb"],
        "free_vram_mb":           info["free_vram_mb"],
        "compute_capability":     info["compute_capability"],
        "cc_major":               info.get("cc_major", 0),
        "cc_minor":               info.get("cc_minor", 0),
        "warp_size":              info["warp_size"],
        "multiprocessor_count":   info["multiprocessor_count"],
        "max_threads_per_block":  info["max_threads_per_block"],
        "shared_mem_per_block_bytes":
            info.get("shared_mem_per_block_bytes", 49152),
        "l2_cache_size_bytes":    info.get("l2_cache_size_bytes", 0),
        "cuda_available":         True,
        "tier":                   tier,
        "architecture":           _architecture_name(info["compute_capability"]),
        "recommended":            recommended,
    }

    _CACHED_CONFIG = cfg
    _HARDWARE_CONFIG = cfg
    return cfg


# ===========================================================================
# GraphProfiler — structural analysis (CPU only)
# ===========================================================================

class GraphProfiler:
    """Analyse graph structure to drive optimisation decisions.

    All metrics computed on CPU directly from CSR arrays.  Designed to
    run in under a few hundred milliseconds for graphs up to ~10M edges.
    Profile results are cached per graph fingerprint in
    ``_GRAPH_PROFILE_CACHE``.
    """

    @staticmethod
    def fingerprint(graph_csr) -> str:
        """Fast structural fingerprint for cache keying.

        Uses shape, nnz, and the sum of the first/last ~100 indptr
        values.  Stable across runs but does not hash the full matrix
        (which would be O(nnz) and dominate for large graphs).
        """
        try:
            n, m_cols = graph_csr.shape
            nnz = int(graph_csr.nnz)
            indptr = np.asarray(graph_csr.indptr)
            head_n = min(100, indptr.size)
            tail_n = min(100, indptr.size)
            sample = np.concatenate([indptr[:head_n], indptr[-tail_n:]])
            sample_sum = int(sample.sum())
        except Exception:                               # noqa: BLE001
            # Non-CSR input; fall back to repr-based hash
            key = repr(graph_csr)
            return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]

        key = f"{n}_{m_cols}_{nnz}_{sample_sum}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def profile(graph_csr) -> dict:
        """Compute structural metrics for a CSR graph.

        Returns the profile dict consumed by
        :class:`MemoryEstimator` and :class:`AlgorithmStrategySelector`.
        """
        n = int(graph_csr.shape[0])
        m = int(graph_csr.nnz)
        degrees = np.diff(np.asarray(graph_csr.indptr)).astype(np.float64)

        density = (m / max(n * n, 1)) if n > 0 else 0.0
        avg_deg = (m / max(n, 1)) if n > 0 else 0.0
        if m > 0 and degrees.size > 0:
            max_deg = int(degrees.max())
            min_deg = int(degrees.min())
            std_deg = float(degrees.std())
        else:
            max_deg = 0
            min_deg = 0
            std_deg = 0.0

        if std_deg > 0:
            try:
                from scipy import stats as _scipy_stats
                deg_skew = float(_scipy_stats.skew(degrees))
            except Exception:                           # noqa: BLE001
                # Manual Pearson skewness coefficient if scipy.stats missing.
                mean = degrees.mean()
                m3 = float(np.mean((degrees - mean) ** 3))
                deg_skew = m3 / (std_deg ** 3) if std_deg > 0 else 0.0
        else:
            deg_skew = 0.0

        hub_frac      = float(np.mean(degrees > 3.0 * avg_deg)) if n > 0 else 0.0
        isolated_frac = float(np.mean(degrees == 0)) if n > 0 else 0.0

        # Symmetry: cheap for sparse matrices (set difference of patterns).
        try:
            diff = graph_csr - graph_csr.T
            # Treat as symmetric if any residual entries are vanishingly small.
            if hasattr(diff, "data") and diff.data.size > 0:
                is_symmetric = bool(np.all(np.abs(diff.data) < 1e-8))
            else:
                is_symmetric = bool(diff.nnz == 0)
        except Exception:                               # noqa: BLE001
            is_symmetric = False

        is_bipartite = GraphProfiler._check_bipartite_heuristic(graph_csr, n)

        # VRAM estimate: indptr + indices + data + 2 float32 vectors.
        csr_bytes  = (n + 1) * 4 + m * 4 + m * 4
        vec_bytes  = n * 4 * 2
        vram_est_mb = (csr_bytes + vec_bytes) / (1024.0 * 1024.0)
        nnz_per_mb  = (m / max(vram_est_mb, 1e-6))

        if density < 1e-5:
            sparsity_class = "ultra_sparse"
        elif density < 1e-3:
            sparsity_class = "sparse"
        elif density < 0.1:
            sparsity_class = "moderate"
        else:
            sparsity_class = "dense"

        abs_skew = abs(deg_skew)
        if abs_skew < 1.0:
            degree_class = "uniform"
        elif abs_skew < 3.0:
            degree_class = "skewed"
        else:
            degree_class = "power_law"

        if sparsity_class == "ultra_sparse":
            format_hint = "coo"
        elif degree_class == "power_law":
            format_hint = "sell_c"
        elif degree_class == "skewed":
            format_hint = "hyb"
        else:
            format_hint = "csr"

        return {
            "n":                 n,
            "m":                 m,
            "density":           float(density),
            "avg_degree":        float(avg_deg),
            "max_degree":        max_deg,
            "min_degree":        min_deg,
            "std_degree":        std_deg,
            "degree_skew":       deg_skew,
            "hub_fraction":      hub_frac,
            "isolated_frac":     isolated_frac,
            "is_bipartite":      bool(is_bipartite),
            "is_symmetric":      bool(is_symmetric),
            "nnz_per_mb":        float(nnz_per_mb),
            "vram_estimate_mb":  float(vram_est_mb),
            "sparsity_class":    sparsity_class,
            "degree_class":      degree_class,
            "format_hint":       format_hint,
        }

    @staticmethod
    def _check_bipartite_heuristic(csr, n: int, sample: int = 500) -> bool:
        """BFS 2-coloring on a small subgraph (cheap heuristic).

        Returns True if the sampled subgraph has no odd cycles.
        False positives are possible — the test is intentionally
        conservative.  Used as an advisory hint only.
        """
        if n <= 1 or csr.nnz == 0:
            return True
        n_sample = min(n, sample)
        indptr = np.asarray(csr.indptr)
        indices = np.asarray(csr.indices)

        colour = np.full(n_sample, -1, dtype=np.int8)
        from collections import deque

        # BFS may need multiple seeds if the sampled subgraph is disconnected.
        for start in range(n_sample):
            if colour[start] != -1:
                continue
            colour[start] = 0
            q: deque[int] = deque([start])
            while q:
                u = q.popleft()
                s = int(indptr[u])
                e = int(indptr[u + 1])
                for j in range(s, e):
                    v = int(indices[j])
                    if v >= n_sample:
                        continue
                    if colour[v] == -1:
                        colour[v] = 1 - colour[u]
                        q.append(v)
                    elif colour[v] == colour[u]:
                        return False
        return True


def get_cached_graph_profile(graph_csr) -> dict:
    """Fingerprint + cache-or-compute a graph's structural profile.

    ``GraphProfiler.profile()`` includes an ``A - A.T`` sparse
    subtraction/transpose and a BFS-sampled bipartite check — genuinely
    expensive (multi-second) on graphs with tens of millions of edges.
    ``apply_config()`` was already memoizing this correctly via
    ``_GRAPH_PROFILE_CACHE``, but ``memory_manager.estimate_algorithm_memory``
    called ``GraphProfiler.profile()`` directly, bypassing the cache and
    silently recomputing the full profile on EVERY algorithm invocation
    regardless of the resident-graph / buffer caches an algorithm module
    maintains on its own side.  All callers should go through this
    function instead of calling ``GraphProfiler.profile()`` directly.
    """
    fingerprint = GraphProfiler.fingerprint(graph_csr)
    if fingerprint not in _GRAPH_PROFILE_CACHE:
        _GRAPH_PROFILE_CACHE[fingerprint] = GraphProfiler.profile(graph_csr)
    return _GRAPH_PROFILE_CACHE[fingerprint]


# ===========================================================================
# MemoryEstimator — algorithm-aware VRAM pressure model
# ===========================================================================

class MemoryEstimator:
    """Estimate VRAM requirements per (algorithm, graph) pair.

    Decides whether chunked execution should be activated, computes a
    recommended chunk size, and flags precision downgrade for
    extreme pressure.  Memory estimates are NOT cached — they query
    ``cuda.mem_get_info()`` live so chunking decisions reflect the
    current free VRAM rather than a stale snapshot.
    """

    ALGORITHM_MULTIPLIERS: dict[str, float] = {
        # Multiplier applied to the base CSR+vectors estimate to bound
        # the algorithm's working set.
        "pagerank": 1.5,   # PR_old, PR_new, partial sums, masks
        "bfs":      1.3,   # frontier, visited bitmap, distances
        "louvain":  3.0,   # community arrays, degree sums, coarsened graph
        "rwr":      1.6,   # p, p_new, p0, partial sums
        "hits":     2.5,   # h, a, h_new, a_new, A and A^T
        "mcl":      4.0,   # M, M_new, SpGEMM intermediate fill-in
    }

    @staticmethod
    def estimate(algorithm: str, graph_profile: dict,
                 gpu_config: dict) -> dict:
        """Return a dict describing the algorithm's expected VRAM use.

        Keys: ``base_mb``, ``algorithm_mb``, ``total_mb``,
        ``available_mb``, ``pressure``, ``needs_chunking``,
        ``recommended_chunk_size``, ``precision_downgrade``.
        """
        vram_est = float(graph_profile["vram_estimate_mb"])
        multiplier = MemoryEstimator.ALGORITHM_MULTIPLIERS.get(algorithm, 2.0)

        if algorithm == "mcl":
            avg_deg = float(graph_profile["avg_degree"])
            # SpGEMM fill-in scales with average degree; cap at 20× for
            # extremely dense graphs where the prune step bounds output.
            fill_in_factor = min(max(avg_deg * 2.0, 1.0), 20.0)
            multiplier = max(multiplier, fill_in_factor)
        elif algorithm == "hits":
            # A + A^T + 4 score vectors (h, a, h_new, a_new).  Override
            # base multiplier because the CSR-T allocation dominates.
            multiplier = 2.5

        total_mb = vram_est * multiplier

        # Live VRAM query — preferred over the cached free_vram_mb because
        # other CUDA contexts may have allocated since detection.
        available_mb: float
        try:
            if _PYCUDA_AVAILABLE and _pycuda is not None:
                # Try to use an existing context first; fall back to a
                # transient one only if no context is active.
                try:
                    free_bytes, _total = _pycuda.mem_get_info()
                    available_mb = free_bytes / (1024.0 * 1024.0)
                except Exception:                       # noqa: BLE001
                    available_mb = float(gpu_config.get("free_vram_mb", 0)) or \
                                   float(gpu_config.get("vram_mb", 6144)) * 0.8
            else:
                available_mb = float(gpu_config.get("free_vram_mb", 0)) or \
                               float(gpu_config.get("vram_mb", 6144)) * 0.8
        except Exception:                               # noqa: BLE001
            available_mb = float(gpu_config.get("vram_mb", 6144)) * 0.8

        if available_mb <= 0:
            available_mb = float(gpu_config.get("vram_mb", 6144)) * 0.8

        pressure_ratio = total_mb / max(available_mb, 1.0)
        if pressure_ratio < 0.4:
            pressure = "low"
        elif pressure_ratio < 0.7:
            pressure = "medium"
        elif pressure_ratio < 0.9:
            pressure = "high"
        else:
            pressure = "critical"

        needs_chunking = pressure in ("high", "critical")

        recommended_chunk_size: int | None = None
        if needs_chunking:
            n = int(graph_profile["n"])
            m = int(graph_profile["m"])
            # bytes per node ≈ row_ptr entry + indices + values for an
            # average row.  Factor of 8 covers int32 col_idx + float32 vals.
            bytes_per_node = max(1, int((m / max(n, 1)) * 8))
            chunk_budget   = int(available_mb * 0.5 * 1024 * 1024)
            recommended_chunk_size = max(
                1, min(n, chunk_budget // max(bytes_per_node, 1))
            )

        precision_downgrade = (
            pressure == "critical"
            and algorithm in ("pagerank", "rwr", "hits")
        )

        return {
            "base_mb":               vram_est,
            "algorithm_mb":          total_mb - vram_est,
            "total_mb":              total_mb,
            "available_mb":          available_mb,
            "pressure":              pressure,
            "needs_chunking":        needs_chunking,
            "recommended_chunk_size": recommended_chunk_size,
            "precision_downgrade":   precision_downgrade,
        }


# ===========================================================================
# AlgorithmStrategySelector — graph-aware strategy choices
# ===========================================================================

class AlgorithmStrategySelector:
    """Pick algorithm-specific strategy keys based on graph + memory.

    Returns ONLY keys that the algorithm files already know how to
    consume — adding a new key here without wiring it into a kernel
    is harmless because algorithm files ignore unknown keys.
    """

    @staticmethod
    def select(algorithm: str, graph_profile: dict,
               gpu_config: dict, memory_est: dict) -> dict:
        """Dispatch to the per-algorithm selector."""
        selector = getattr(
            AlgorithmStrategySelector, f"_{algorithm}", None
        )
        if selector is None:
            return {}
        return selector(graph_profile, gpu_config, memory_est)

    @staticmethod
    def _pagerank(gp: dict, gc: dict, me: dict) -> dict:
        use_pull = (
            gp["degree_class"] == "power_law"
            and gp["hub_fraction"] > 0.05
        )
        ellpack_frac = {
            "power_law": 0.02,
            "skewed":    0.05,
            "uniform":   0.10,
        }.get(gp["degree_class"], 0.05)

        return {
            "use_pull":         bool(use_pull),
            "pull_threshold":   max(32, int(3 * gp["avg_degree"])),
            "ellpack_fraction": float(ellpack_frac),
            "use_chunking":     bool(me["needs_chunking"]),
            "chunk_size":       me["recommended_chunk_size"],
            "precision":        "fp16_storage" if me["precision_downgrade"]
                                else "fp32",
            "kernel_variant":   "fused",
        }

    @staticmethod
    def _bfs(gp: dict, gc: dict, me: dict) -> dict:
        use_push_pull = (
            gp["degree_class"] in ("power_law", "skewed")
            or gp["n"] > 50000
        )
        frontier_thresh = 0.25 if gp["degree_class"] == "power_law" else 0.50
        return {
            "traversal_mode":      "push_pull" if use_push_pull else "push_only",
            "use_bitmap_frontier": bool(gp["n"] > 10000
                                        or gp["degree_class"] == "power_law"),
            "frontier_threshold":  float(frontier_thresh),
            "use_chunking":        bool(me["needs_chunking"]),
            "degree_threshold":    max(32, int(gp["avg_degree"] * 2)),
        }

    @staticmethod
    def _hits(gp: dict, gc: dict, me: dict) -> dict:
        return {
            "use_smem_hash":    gp["degree_class"] in ("power_law", "skewed"),
            "reorder_nodes":    gp["degree_class"] == "power_law",
            "fuse_spmv_norm":   True,
            "use_chunking":     bool(me["needs_chunking"]),
            "precision_spmv":   "fp32",
            "precision_conv":   "fp64",
            "kernel_variant":   "edge_parallel_low_deg",
        }

    @staticmethod
    def _louvain(gp: dict, gc: dict, me: dict) -> dict:
        freeze_thresh = 2 if gp["degree_class"] == "power_law" else 4
        early_stop = 0.005 if gp["degree_class"] == "power_law" else 0.01
        try:
            import cupy                                 # noqa: F401
            coarsening_backend = "cupy"
        except Exception:                               # noqa: BLE001
            coarsening_backend = "hybrid_gpu"
        return {
            "use_smem_hash":          gp["degree_class"] in
                                      ("power_law", "skewed"),
            "use_community_freezing": bool(gp["n"] > 1000),
            "freeze_threshold":       int(freeze_thresh),
            "early_stop_fraction":    float(early_stop),
            "coarsening_backend":     coarsening_backend,
            "use_chunking":           bool(me["needs_chunking"]),
        }

    @staticmethod
    def _rwr(gp: dict, gc: dict, me: dict) -> dict:
        return {
            "spmv_mode":     "edge_parallel" if gp["degree_class"] == "power_law"
                             else "csr_fused",
            "use_batched":   False,
            "reorder_nodes": gp["degree_class"] == "power_law",
            "use_zero_copy": me["pressure"] == "critical",
            "use_chunking":  bool(me["needs_chunking"]),
        }

    @staticmethod
    def _mcl(gp: dict, gc: dict, me: dict) -> dict:
        density_map = {
            "ultra_sparse": 1e-5,
            "sparse":       1e-4,
            "moderate":     1e-3,
            "dense":        1e-2,
        }
        prune_thr = density_map.get(gp["sparsity_class"], 1e-3)
        density_factor = max(gp["density"] / 1e-4, 1.0)
        top_k = max(10, min(100, int(50 / density_factor)))

        # Fix 6: Density-based safety adjustment — high avg_degree means
        # M^2 fill-in grows super-linearly, requiring tighter pruning to
        # prevent SpGEMM output buffer overflow on biological networks.
        avg_deg = float(gp.get("avg_degree", 0.0))
        density = float(gp.get("density", 0.0))
        if avg_deg > 50:
            # Very dense: force prune_threshold high enough to keep nnz bounded.
            prune_thr = max(density * 10.0, 0.01, prune_thr)
            top_k     = min(top_k, 20)
        elif avg_deg > 20:
            # Moderately dense: gentle tightening.
            prune_thr = max(density * 5.0, 0.005, prune_thr)
            top_k     = min(top_k, 30)

        return {
            "spgemm_method":    "inner_product" if
                                gp["sparsity_class"] == "dense"
                                else "hash",
            "prune_threshold":  float(prune_thr),
            "top_k_per_column": int(top_k),
            "use_bitonic_topk": bool(gp["max_degree"] > 256),
            "use_chunking":     True,
            "chunk_size":       me["recommended_chunk_size"]
                                or max(1, gp["n"] // 4),
        }


# ===========================================================================
# RuntimeProfiler — in-memory execution history
# ===========================================================================

class RuntimeProfiler:
    """Record per-(algorithm, graph) execution history.

    Stores up to the last 10 runs per pair and produces simple
    config-adjustment recommendations (tolerance relaxation,
    max_iter bump, fragmentation warning).
    """

    _history: dict[tuple[str, str], list[dict]] = {}
    _MAX_RECORDS: int = 10
    _MIN_FOR_RECOMMENDATION: int = 3

    @classmethod
    def record(cls, algorithm: str, fingerprint: str,
               execution_time: float, config_used: dict,
               result_meta: dict) -> None:
        """Append a single run record."""
        key = (algorithm, fingerprint)
        if key not in cls._history:
            cls._history[key] = []
        # Strip private metadata keys (don't store the full graph profile
        # inside every record — it bloats memory and is already in cache).
        clean_config = {
            k: v for k, v in (config_used or {}).items()
            if not k.startswith("_")
        }
        cls._history[key].append({
            "time":      float(execution_time),
            "config":    clean_config,
            "meta":      dict(result_meta or {}),
            "timestamp": time.time(),
        })
        cls._history[key] = cls._history[key][-cls._MAX_RECORDS:]

    @classmethod
    def get_recommendation(cls, algorithm: str,
                           fingerprint: str) -> dict:
        """Suggest config adjustments based on the last 3 runs."""
        key = (algorithm, fingerprint)
        history = cls._history.get(key, [])
        if len(history) < cls._MIN_FOR_RECOMMENDATION:
            return {}

        recent = history[-cls._MIN_FOR_RECOMMENDATION:]
        recommendations: dict = {}

        all_converged = all(
            r["meta"].get("converged", False) for r in recent
        )
        none_converged = not any(
            r["meta"].get("converged", False) for r in recent
        )

        if all_converged:
            current_tol = recent[-1]["config"].get("tolerance", 1e-6)
            try:
                current_tol = float(current_tol)
            except (TypeError, ValueError):
                current_tol = 1e-6
            if current_tol < 1e-4:
                recommendations["tolerance"] = current_tol * 2
                recommendations["_note"] = (
                    "Relaxed tolerance (last 3 runs converged easily)"
                )

        if none_converged:
            current_max = recent[-1]["config"].get("max_iter", 100)
            try:
                current_max = int(current_max)
            except (TypeError, ValueError):
                current_max = 100
            recommendations["max_iter"] = int(current_max * 1.5)
            recommendations["_note"] = (
                "Increased max_iter (last 3 runs did not converge)"
            )

        times = [r["time"] for r in recent]
        if len(times) >= 3 and times[-1] > times[0] * 1.5:
            recommendations["_warning"] = (
                "Execution time increasing — possible GPU memory "
                "fragmentation. Consider restarting the process."
            )

        return recommendations

    @classmethod
    def get_history(cls, algorithm: str | None = None) -> dict:
        """Export the runtime history (optionally filtered by algorithm)."""
        if algorithm:
            return {k: list(v) for k, v in cls._history.items()
                    if k[0] == algorithm}
        return {k: list(v) for k, v in cls._history.items()}

    @classmethod
    def clear(cls, algorithm: str | None = None) -> None:
        """Reset the in-memory history."""
        if algorithm:
            cls._history = {k: v for k, v in cls._history.items()
                            if k[0] != algorithm}
        else:
            cls._history = {}


# ===========================================================================
# Profile report generator
# ===========================================================================

def _build_recommendations_list(
    graph_profile: dict, memory_est: dict, strategy: dict,
    runtime_rec: dict,
) -> list[str]:
    """Build human-readable recommendation strings."""
    msgs: list[str] = []

    dc = graph_profile.get("degree_class", "uniform")
    if dc == "power_law":
        msgs.append(
            "Graph has power-law degree distribution — pull-based / "
            "ELLPACK kernels activated for hub nodes."
        )
    elif dc == "skewed":
        msgs.append(
            "Graph is skewed — shared-memory hash and hub ELLPACK "
            "are enabled for moderate hubs."
        )

    if graph_profile.get("is_bipartite", False):
        msgs.append(
            "Bipartite structure detected — appropriate for "
            "miRNA-target networks."
        )

    if graph_profile.get("is_symmetric", False):
        msgs.append(
            "Symmetric adjacency detected — PPI-style undirected "
            "handling is valid."
        )

    pressure = memory_est.get("pressure", "low")
    if pressure in ("high", "critical"):
        chunk_size = memory_est.get("recommended_chunk_size")
        msgs.append(
            f"Memory pressure is {pressure.upper()} — chunked "
            f"execution enabled with chunk_size="
            f"{chunk_size if chunk_size else 'adaptive'}."
        )
    elif pressure == "medium":
        msgs.append("Memory pressure is MEDIUM — chunking on standby.")

    if memory_est.get("precision_downgrade", False):
        msgs.append(
            "Critical VRAM pressure — precision downgrade to FP16 "
            "storage suggested for score vectors."
        )

    if "ellpack_fraction" in strategy:
        msgs.append(
            f"ELLPACK fraction set to {strategy['ellpack_fraction']} "
            f"(adaptive from degree distribution)."
        )
    if "use_pull" in strategy and strategy["use_pull"]:
        msgs.append(
            f"Pull-mode active for nodes with in-degree >= "
            f"{strategy.get('pull_threshold', '?')}."
        )

    if "coarsening_backend" in strategy:
        be = strategy["coarsening_backend"]
        if be == "cupy":
            msgs.append(
                "CuPy is available — Louvain Phase 2 coarsening "
                "runs fully on GPU."
            )
        else:
            msgs.append(
                "CuPy not available — Louvain Phase 2 uses hybrid "
                "GPU+CPU coarsening."
            )

    if "spgemm_method" in strategy:
        msgs.append(
            f"MCL SpGEMM method: {strategy['spgemm_method']} "
            f"(driven by sparsity class)."
        )

    if "_note" in runtime_rec:
        msgs.append(f"Runtime feedback: {runtime_rec['_note']}")
    if "_warning" in runtime_rec:
        msgs.append(f"Runtime warning: {runtime_rec['_warning']}")

    return msgs


def generate_profile_report(
    algorithm: str,
    graph_csr,
    params: dict | None = None,
    execution_result: dict | None = None,
) -> dict:
    """Produce a structured optimisation report for an algorithm run.

    Returns a dict with the sections ``hardware``, ``graph``,
    ``memory``, ``strategy``, ``runtime_history``, ``execution``
    (if ``execution_result`` is supplied), and ``recommendations``
    (a list of human-readable strings).
    """
    params = params or {}
    cfg = get_gpu_config()
    fingerprint = GraphProfiler.fingerprint(graph_csr)
    if fingerprint not in _GRAPH_PROFILE_CACHE:
        _GRAPH_PROFILE_CACHE[fingerprint] = GraphProfiler.profile(graph_csr)
    gp = _GRAPH_PROFILE_CACHE[fingerprint]
    me = MemoryEstimator.estimate(algorithm, gp, cfg)
    strategy = AlgorithmStrategySelector.select(algorithm, gp, cfg, me)
    runtime_rec = RuntimeProfiler.get_recommendation(algorithm, fingerprint)
    history = RuntimeProfiler.get_history(algorithm).get(
        (algorithm, fingerprint), []
    )

    avg_time = (
        float(np.mean([r["time"] for r in history])) if history else None
    )

    report: dict = {
        "hardware": {
            "device_name":        cfg.get("device_name", "unknown"),
            "vram_mb":            cfg.get("vram_mb", 0),
            "compute_capability": cfg.get("compute_capability", "0.0"),
            "architecture":       cfg.get("architecture", "Unknown"),
        },
        "graph": {
            "n":               gp["n"],
            "m":               gp["m"],
            "density":         gp["density"],
            "degree_class":    gp["degree_class"],
            "sparsity_class":  gp["sparsity_class"],
            "format_hint":     gp["format_hint"],
            "hub_fraction":    gp["hub_fraction"],
            "is_symmetric":    gp["is_symmetric"],
            "is_bipartite":    gp["is_bipartite"],
        },
        "memory": {
            "estimated_total_mb": me["total_mb"],
            "available_mb":       me["available_mb"],
            "pressure":           me["pressure"],
            "chunking_needed":    me["needs_chunking"],
            "chunk_size":         me["recommended_chunk_size"],
        },
        "strategy": dict(strategy),
        "runtime_history": {
            "num_past_runs":     len(history),
            "avg_time_seconds":  avg_time,
            "recommendation":    runtime_rec,
        },
    }

    if execution_result is not None:
        report["execution"] = {
            "time_seconds":   execution_result.get("execution_time"),
            "iterations":     (execution_result.get("result") or {}).get(
                "iterations"
            ),
            "converged":      (execution_result.get("result") or {}).get(
                "converged"
            ),
            "speedup_vs_cpu": execution_result.get("speedup_vs_cpu"),
        }

    report["recommendations"] = _build_recommendations_list(
        gp, me, strategy, runtime_rec
    )
    return report


# ===========================================================================
# Public: apply_config()  — orchestrator
# ===========================================================================

def apply_config(
    algorithm_name: str,
    graph_csr,
    params: dict | None = None,
    override: bool = False,
    enable_profiling: bool = False,
) -> dict:
    """Merge hardware + graph + algorithm + runtime params for a GPU run.

    Backward compatible with the previous signature
    (``apply_config(name, csr, params, override=False)``).  The new
    ``enable_profiling`` flag attaches a ``_profile_report`` to the
    returned dict when ``True``.

    Priority (highest wins, default ``override=False``):

        user params > runtime_rec > strategy > hw_recommended

    When ``override=True``, hw recommendations beat user params on
    keys they have in common (legacy behaviour).
    """
    if params is None:
        params = {}

    # 1. Hardware config (cached forever)
    global _HARDWARE_CONFIG
    if _HARDWARE_CONFIG is None:
        get_gpu_config()  # populates both _CACHED_CONFIG and _HARDWARE_CONFIG
    cfg = _HARDWARE_CONFIG or get_gpu_config()

    # Unknown algorithm: degrade gracefully (preserves legacy behaviour).
    if algorithm_name not in _VALID_ALGORITHMS:
        logging.warning(
            "apply_config received unknown algorithm '%s' — "
            "returning user params unchanged.", algorithm_name,
        )
        return dict(params)

    # 2. Graph profile (cached per fingerprint)
    try:
        fingerprint = GraphProfiler.fingerprint(graph_csr)
    except Exception:                                   # noqa: BLE001
        # Non-CSR input: fall back to hw-only behaviour
        return _legacy_apply(cfg, algorithm_name, graph_csr, params, override)

    if fingerprint not in _GRAPH_PROFILE_CACHE:
        try:
            _GRAPH_PROFILE_CACHE[fingerprint] = GraphProfiler.profile(graph_csr)
        except Exception as exc:                        # noqa: BLE001
            logging.warning(
                "GraphProfiler.profile failed (%s) — falling back to "
                "hardware-only config.", exc,
            )
            return _legacy_apply(cfg, algorithm_name, graph_csr, params, override)
    graph_profile = _GRAPH_PROFILE_CACHE[fingerprint]

    # 3. Memory estimate (NOT cached — depends on live free_vram)
    memory_est = MemoryEstimator.estimate(algorithm_name, graph_profile, cfg)

    # 4. Strategy (cached per (algo, fingerprint))
    strategy_key = (algorithm_name, fingerprint)
    if strategy_key not in _STRATEGY_CACHE:
        _STRATEGY_CACHE[strategy_key] = AlgorithmStrategySelector.select(
            algorithm_name, graph_profile, cfg, memory_est,
        )
    strategy = _STRATEGY_CACHE[strategy_key]

    # 4b. Memory-aware execution plan (MemoryManager).  Lazy-imported to
    # avoid a circular dependency at module load.  The plan keys are
    # additive: existing strategy keys are preserved, new keys
    # (execution_mode, chunk_size, use_unified_memory, ...) are merged
    # into the returned dict alongside the metadata block.
    try:
        from src.optimization.memory_manager import MemoryManager       # noqa: PLC0415
        memory_plan = MemoryManager.select_execution_mode(
            algorithm_name, graph_csr, params,
        )
    except Exception as exc:                                            # noqa: BLE001
        logging.warning(
            "MemoryManager.select_execution_mode failed (%s) — "
            "memory plan keys will be omitted.", exc,
        )
        memory_plan = {}

    # 5. Runtime feedback
    runtime_rec = RuntimeProfiler.get_recommendation(
        algorithm_name, fingerprint,
    )

    # 6. Hardware recommendations (existing tiered defaults)
    hw_recommended = cfg.get("recommended", {}).get(algorithm_name, {})

    # 7. Merge
    merged: dict = {}
    merged.update(hw_recommended)
    merged.update(strategy)
    # Memory plan keys go in BEFORE user params so user always wins
    # (unless override=True).  Strategy may already set ``use_chunking``
    # but the memory plan refines it with a concrete ``chunk_size`` and
    # the explicit ``execution_mode`` flag.
    for k, v in memory_plan.items():
        merged[k] = v
    for k, v in runtime_rec.items():
        if not k.startswith("_"):
            merged[k] = v

    # Apply user params with the requested precedence rule.
    if override:
        # legacy override=True semantics: hw recommendations win on
        # overlapping keys; user params fill in the gaps.
        for k, v in params.items():
            if k not in merged:
                merged[k] = v
    else:
        for k, v in params.items():
            merged[k] = v

    # 8. Metadata keys — never overridden, easy for downstream tools.
    merged["_graph_fingerprint"] = fingerprint
    merged["_graph_profile"]     = graph_profile
    merged["_memory_estimate"]   = memory_est
    merged["_strategy_selected"] = strategy
    merged["_memory_plan"]       = memory_plan
    merged["_hardware_config"]   = {
        k: v for k, v in cfg.items() if k != "recommended"
    }
    if "_note" in runtime_rec:
        merged["_runtime_note"] = runtime_rec["_note"]
    if "_warning" in runtime_rec:
        logging.warning("[gpu_config] %s", runtime_rec["_warning"])

    # 9. Optional full report
    if enable_profiling:
        merged["_profile_report"] = generate_profile_report(
            algorithm_name, graph_csr, merged,
        )

    # 10. Legacy chunking/OOM hints (kept for full backward compatibility
    # with the previous apply_config behaviour).
    _apply_legacy_chunking_hints(merged, graph_csr, cfg)

    return merged


def _legacy_apply(cfg: dict, algorithm_name: str, graph_csr,
                  params: dict, override: bool) -> dict:
    """Fallback path identical to the pre-expansion apply_config()."""
    recommended = cfg.get("recommended", {}).get(algorithm_name, {})
    if override:
        merged = {**params, **recommended}
    else:
        merged = {**recommended, **params}
    _apply_legacy_chunking_hints(merged, graph_csr, cfg)
    return merged


def _apply_legacy_chunking_hints(merged: dict, graph_csr, cfg: dict) -> None:
    """Preserve the previous nnz-based chunking heuristics."""
    try:
        nnz = int(graph_csr.nnz)
        n_nodes = int(graph_csr.shape[0])
    except Exception:                                   # noqa: BLE001
        return

    edge_memory_mb = (nnz * 8) / (1024 ** 2)
    node_memory_mb = (n_nodes * 4) / (1024 ** 2)
    estimated_mb = edge_memory_mb + node_memory_mb
    free_vram_mb = cfg.get("free_vram_mb", 0)

    if nnz < 10_000:
        # Don't FORCE chunking off if the new strategy already set it on
        # for a small graph (e.g. MCL always-on).  Only soft-suggest.
        merged.setdefault("use_chunking", False)
    elif free_vram_mb > 0 and estimated_mb > free_vram_mb * 0.6:
        merged["use_chunking"] = True
        merged.setdefault("chunking_reason", "graph_exceeds_vram_threshold")
    elif nnz > 5_000_000:
        merged["use_chunking"] = True

    if free_vram_mb > 0 and estimated_mb > free_vram_mb * 1.5:
        merged["warn_oom_risk"] = True


# ---------------------------------------------------------------------------
# Public: print_gpu_summary()
# ---------------------------------------------------------------------------

def print_gpu_summary(graph_csr=None, algorithm: str | None = None) -> None:
    """Print a boxed human-readable summary.

    When ``graph_csr`` and ``algorithm`` are provided, also prints
    the graph profile and the algorithm's selected strategy.
    """
    cfg = get_gpu_config()
    arch = cfg.get("architecture", _architecture_name(cfg["compute_capability"]))

    lines = [
        ("Device",             cfg["device_name"]),
        ("VRAM",               f"{cfg['vram_mb']} MB"),
        ("Free VRAM",          f"{cfg['free_vram_mb']} MB"),
        ("Compute capability", cfg["compute_capability"]),
        ("Architecture",       arch),
        ("SM count",           str(cfg["multiprocessor_count"])),
        ("Max SMEM/block",     f"{cfg.get('shared_mem_per_block_bytes', 0)} B"),
        ("CUDA available",     str(cfg["cuda_available"])),
        ("Tier",               cfg["tier"]),
    ]

    label_w = max(len(k) for k, _ in lines)
    value_w = max(len(str(v)) for _, v in lines)
    inner_w = label_w + 3 + value_w
    border = "+" + "-" * (inner_w + 2) + "+"

    print(border)
    print("| " + "GPU Configuration".center(inner_w) + " |")
    print(border)
    for k, v in lines:
        print(f"| {k.ljust(label_w)} : {str(v).ljust(value_w)} |")
    print(border)

    if graph_csr is None or algorithm is None:
        return

    try:
        profile = GraphProfiler.profile(graph_csr)
        memory  = MemoryEstimator.estimate(algorithm, profile, cfg)
        strategy = AlgorithmStrategySelector.select(
            algorithm, profile, cfg, memory,
        )
    except Exception as exc:                            # noqa: BLE001
        print(f"  (graph profiling failed: {exc})")
        return

    print()
    print(f"Graph profile for algorithm '{algorithm}':")
    print(f"  Nodes:    {profile['n']}, edges: {profile['m']}, "
          f"density: {profile['density']:.2e}")
    print(f"  Degree:   {profile['degree_class']} "
          f"(skew={profile['degree_skew']:.2f}, "
          f"max={profile['max_degree']}, "
          f"hub_frac={profile['hub_fraction']:.2%})")
    print(f"  Sparsity: {profile['sparsity_class']}")
    print(f"  Format:   {profile['format_hint']}")
    print(f"  Symmetric={profile['is_symmetric']}, "
          f"bipartite={profile['is_bipartite']}")
    print(f"  Memory:   {memory['total_mb']:.1f} MB est, "
          f"{memory['available_mb']:.1f} MB free, "
          f"pressure={memory['pressure']}")
    print(f"  Chunking: {memory['needs_chunking']} "
          f"(chunk_size={memory['recommended_chunk_size']})")
    print(f"Strategy keys:")
    for k, v in sorted(strategy.items()):
        print(f"  {algorithm}.{k}: {v}")


# ---------------------------------------------------------------------------
# Script entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_gpu_summary()
    print()
    print(json.dumps({k: v for k, v in get_gpu_config().items()
                      if k != "recommended"}, indent=2, default=str))
