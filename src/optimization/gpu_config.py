"""
optimizer/gpu_config.py — Architecture-Aware GPU Configuration
===============================================================

Overview
--------
This module detects the host's NVIDIA GPU (if any) and produces a set of
*per-algorithm* configuration parameters that are appropriate for the
detected hardware tier and compute capability.  The goal is to give every
algorithm (pagerank, louvain, rwr, hits, bfs, mcl) sensible defaults that
match what the GPU can actually support — without the algorithm authors
having to hard-code device-specific magic numbers.

The module is intentionally **network-type agnostic**: it works equally
well for GRN, PPI, and miRNA-target adjacency matrices.  Only the *graph
size* (nnz, shape) and the *GPU specs* drive the recommendations.

Public API
----------
get_gpu_config()  -> dict
    Detect the GPU exactly once per process and return a structured config
    dict.  Result is cached at module level so repeated calls are free.

apply_config(algorithm_name, graph_csr, params=None, override=False) -> dict
    Merge the algorithm-specific GPU recommendations into the caller's
    ``params`` dict and apply dynamic graph-size adjustments (chunking,
    OOM warnings).  Safe to call before every GPU kernel launch.

print_gpu_summary()
    Print a boxed human-readable summary of the detected GPU.

Detection
---------
1. pycuda.driver  — preferred, queries the device directly without creating
                    a long-lived CUDA context (a transient context is used
                    only to read free VRAM, then immediately popped).
2. nvidia-smi     — subprocess fallback for systems where pycuda is missing
                    but the NVIDIA driver tools are installed.
3. cpu_only       — both above failed; algorithms should skip GPU mode.

Tier classification (drives base block sizes, chunking strategies, etc.)
----------------------------------------------------------------------
    high       : vram_mb >= 8192            (e.g. RTX 3070+, A100, etc.)
    mid_large  : 6144 <= vram_mb < 8192     (e.g. RTX 2060 6 GB target)
    mid_small  : 3072 <= vram_mb < 6144     (e.g. GTX 1650 / 1060 3 GB)
    cpu_only   : no CUDA device detected

Compute-capability adjustments (applied on top of the base tier config)
----------------------------------------------------------------------
    < 7.0  : disable shared-memory hash maps, bitmap frontiers, direction
             optimisation — old hardware lacks the atomic / warp primitives
             these rely on.  legacy_kernel_mode is flagged.
    == 7.5 : Turing target hardware.  BFS block_size may be 512 if tier
             already allows it.
    >= 8.0 : Ampere / Hopper — high-tier block sizes are allowed to grow.
    SM < 20: low-SM device — every block_size is halved (floor 32) and
             low_sm_mode is flagged so kernels can dispatch less work per
             block to keep occupancy.
"""

import json
import os
import subprocess

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Optional pycuda
# ---------------------------------------------------------------------------

try:
    import pycuda.driver as _pycuda
    _pycuda.init()
    _PYCUDA_AVAILABLE = True
except Exception:
    _pycuda = None
    _PYCUDA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module-level cache — detection runs only once per process
# ---------------------------------------------------------------------------

_CACHED_CONFIG: dict | None = None

_VALID_ALGORITHMS = ("pagerank", "louvain", "rwr", "hits", "bfs", "mcl")


def _reset_cache() -> None:
    """Clear the cached config.  Primarily for tests that mock GPU state."""
    global _CACHED_CONFIG
    _CACHED_CONFIG = None


# ---------------------------------------------------------------------------
# GPU detection backends
# ---------------------------------------------------------------------------

def _query_pycuda() -> dict | None:
    """
    Query device 0 via pycuda.driver.

    Most attributes (name, total memory, compute capability, SM count,
    warp size, max threads/block) do not require a CUDA context.  Free
    VRAM does — we create a transient context, query, then detach so we
    do not interfere with any other CUDA context (e.g. CuPy's).
    """
    if not _PYCUDA_AVAILABLE:
        return None
    try:
        if _pycuda.Device.count() == 0:
            return None
        device = _pycuda.Device(0)
        cc_major, cc_minor = device.compute_capability()
        attr = _pycuda.device_attribute

        info = {
            "device_name":            device.name(),
            "vram_mb":                int(device.total_memory() / (1024 ** 2)),
            "compute_capability":     f"{cc_major}.{cc_minor}",
            "warp_size":              int(device.get_attribute(attr.WARP_SIZE)),
            "multiprocessor_count":   int(device.get_attribute(attr.MULTIPROCESSOR_COUNT)),
            "max_threads_per_block":  int(device.get_attribute(attr.MAX_THREADS_PER_BLOCK)),
            "free_vram_mb":           0,   # filled below if context succeeds
        }

        # Transient context only for mem_get_info; pop immediately to avoid
        # conflicting with any other CUDA context the process may create.
        try:
            ctx = device.make_context()
            try:
                free, _total = _pycuda.mem_get_info()
                info["free_vram_mb"] = int(free / (1024 ** 2))
            finally:
                ctx.pop()
                ctx.detach()
        except Exception:
            pass  # free_vram_mb stays 0; nvidia-smi fallback may fill it

        return info
    except Exception:
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
        return {
            "device_name":            parts[0],
            "vram_mb":                int(float(parts[1])),
            "free_vram_mb":           int(float(parts[2])),
            "compute_capability":     parts[3],
            # nvidia-smi does not expose these — use safe defaults
            "warp_size":              32,
            "multiprocessor_count":   0,
            "max_threads_per_block":  1024,
        }
    except Exception:
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
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tier classification & base recommendations
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
    return "cpu_only"   # GPU too small to be useful


def _base_recommendations(tier: str) -> dict:
    """Return per-algorithm base config for the given tier."""
    if tier == "cpu_only":
        return {
            algo: {"gpu_disabled": True, "block_size": 0, "use_chunking": False}
            for algo in _VALID_ALGORITHMS
        }

    # --- pagerank ---
    pagerank_common = {"precision": "float32", "use_shared_mem": True}
    pagerank_tier = {
        "high":      {"block_size": 512, "use_chunking": False, "use_zero_copy": False},
        "mid_large": {"block_size": 256, "use_chunking": False, "use_zero_copy": False},
        "mid_small": {"block_size": 256, "use_chunking": True,  "use_zero_copy": True},
    }

    # --- louvain ---
    louvain_common = {"use_shared_mem_hash": True}
    louvain_tier = {
        "high":      {"block_size": 256, "use_chunking": False, "community_id_bits": 32},
        "mid_large": {"block_size": 256, "use_chunking": False, "community_id_bits": 16},
        "mid_small": {"block_size": 128, "use_chunking": True,  "community_id_bits": 16},
    }

    # --- rwr ---
    rwr_common = {"precision": "float32"}
    rwr_tier = {
        "high":      {"block_size": 256, "use_zero_copy": False, "batch_seeds": True},
        "mid_large": {"block_size": 256, "use_zero_copy": False, "batch_seeds": False},
        "mid_small": {"block_size": 128, "use_zero_copy": True,  "batch_seeds": False},
    }

    # --- hits ---
    hits_common = {"precision": "float32", "use_shared_mem": True}
    hits_tier = {
        "high":      {"block_size": 256, "use_chunking": False},
        "mid_large": {"block_size": 256, "use_chunking": False},
        "mid_small": {"block_size": 128, "use_chunking": True},
    }

    # --- bfs ---
    bfs_common = {"use_bitmap_frontier": True}
    bfs_tier = {
        "high":      {"block_size": 512, "use_direction_opt": True},
        "mid_large": {"block_size": 256, "use_direction_opt": True},
        "mid_small": {"block_size": 128, "use_direction_opt": False},
    }

    # --- mcl ---
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


# ---------------------------------------------------------------------------
# Compute-capability and SM-count adjustments
# ---------------------------------------------------------------------------

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
        # Pre-Volta: lacks the modern shared-memory atomics / warp shuffles
        # that the optimised kernels rely on.  Fall back to legacy kernels.
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
        # Turing — our explicit target hardware (RTX 2060).
        # BFS at high tier already uses 512; mid_large can stay at 256.
        if tier == "high":
            recommended["bfs"]["block_size"] = 512

    if cc >= 8.0 and tier == "high":
        # Ampere/Hopper at high tier: allow larger pagerank/hits blocks
        # up to the device limit (typically 1024).
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
# Architecture name helper
# ---------------------------------------------------------------------------

def _architecture_name(cc_str: str) -> str:
    try:
        major_s, minor_s = cc_str.split(".")
        major, minor = int(major_s), int(minor_s)
    except Exception:
        return "Unknown"
    if major == 5:
        return "Maxwell"
    if major == 6:
        return "Pascal"
    if major == 7:
        if minor <= 2:
            return "Volta"
        if minor == 5:
            return "Turing"
        return "Volta/Turing"
    if major == 8:
        return "Ampere"
    if major == 9:
        return "Hopper"
    return f"Compute {cc_str}"


# ---------------------------------------------------------------------------
# CPU-only fallback config
# ---------------------------------------------------------------------------

def _cpu_only_config() -> dict:
    return {
        "device_name":            "none",
        "vram_mb":                0,
        "free_vram_mb":           0,
        "compute_capability":     "0.0",
        "warp_size":              32,
        "multiprocessor_count":   0,
        "max_threads_per_block":  0,
        "cuda_available":         False,
        "tier":                   "cpu_only",
        "mode":                   "cpu_only",
        "recommended":            _base_recommendations("cpu_only"),
    }


# ---------------------------------------------------------------------------
# Public: get_gpu_config()
# ---------------------------------------------------------------------------

def get_gpu_config() -> dict:
    """
    Detect the GPU exactly once and return the full configuration dict.

    The detected config is cached at module scope; every subsequent call
    returns the same object (identity-equal).  Use :func:`_reset_cache`
    in tests to force re-detection.
    """
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None:
        return _CACHED_CONFIG

    # ---- Detection ----
    info = _query_pycuda()
    if info is None:
        info = _query_nvidia_smi()

    if info is None:
        cfg = _cpu_only_config()
        print("No CUDA device found. All GPU modes will be skipped.")
        _CACHED_CONFIG = cfg
        return cfg

    # If pycuda gave us info but free_vram_mb is 0, try nvidia-smi for it
    if info.get("free_vram_mb", 0) == 0:
        free = _query_free_vram_nvidia_smi()
        if free is not None:
            info["free_vram_mb"] = free
        else:
            # Conservative fallback: assume all VRAM is free
            info["free_vram_mb"] = info["vram_mb"]

    tier = _classify_tier(info["vram_mb"], cuda_available=True)
    recommended = _base_recommendations(tier)

    _apply_cc_adjustments(
        recommended,
        info["compute_capability"],
        tier,
        info["max_threads_per_block"],
    )
    _apply_sm_adjustments(recommended, info["multiprocessor_count"])

    cfg = {
        "device_name":            info["device_name"],
        "vram_mb":                info["vram_mb"],
        "free_vram_mb":           info["free_vram_mb"],
        "compute_capability":     info["compute_capability"],
        "warp_size":              info["warp_size"],
        "multiprocessor_count":   info["multiprocessor_count"],
        "max_threads_per_block":  info["max_threads_per_block"],
        "cuda_available":         True,
        "tier":                   tier,
        "recommended":            recommended,
    }

    _CACHED_CONFIG = cfg
    return cfg


# ---------------------------------------------------------------------------
# Public: apply_config()
# ---------------------------------------------------------------------------

def apply_config(
    algorithm_name: str,
    graph_csr,
    params: dict | None = None,
    override: bool = False,
) -> dict:
    """
    Merge the algorithm's GPU recommendations into ``params`` and apply
    dynamic graph-size adjustments (chunking, OOM warnings).

    Parameters
    ----------
    algorithm_name : str
        One of ``pagerank``, ``louvain``, ``rwr``, ``hits``, ``bfs``, ``mcl``.
    graph_csr      : scipy.sparse.csr_matrix (or anything with .nnz / .shape)
        The graph the algorithm will run on — used to size memory estimates.
    params         : dict, optional
        Caller-supplied parameters.  Defaults to ``{}``.
    override       : bool, default False
        If ``False``, user params take precedence over recommendations.
        If ``True``, recommendations overwrite matching user keys.

    Returns
    -------
    dict — the merged, possibly graph-size-adjusted parameter dict.
    """
    if params is None:
        params = {}

    cfg = get_gpu_config()

    if algorithm_name not in _VALID_ALGORITHMS:
        print(
            f"Warning: apply_config received unknown algorithm "
            f"'{algorithm_name}' — returning params unchanged."
        )
        return params

    recommended = cfg["recommended"].get(algorithm_name, {})

    # Merge: order of {**a, **b} = b wins on key collisions
    if override:
        merged = {**params, **recommended}    # recommendations win
    else:
        merged = {**recommended, **params}    # user params win

    # --- Memory estimate ---
    try:
        nnz = int(graph_csr.nnz)
        n_nodes = int(graph_csr.shape[0])
    except Exception:
        # Not a sparse-matrix-like object — skip graph-size adjustments
        return merged

    edge_memory_mb = (nnz * 8) / (1024 ** 2)
    node_memory_mb = (n_nodes * 4) / (1024 ** 2)
    estimated_mb = edge_memory_mb + node_memory_mb
    free_vram_mb = cfg.get("free_vram_mb", 0)

    # --- Dynamic graph-size adjustments ---
    if nnz < 10_000:
        # Chunking overhead exceeds the benefit on tiny graphs.
        merged["use_chunking"] = False
    elif free_vram_mb > 0 and estimated_mb > free_vram_mb * 0.6:
        merged["use_chunking"] = True
        merged["chunking_reason"] = "graph_exceeds_vram_threshold"
    elif nnz > 5_000_000:
        merged["use_chunking"] = True
        print("Info: graph has >5M edges, enabling chunking regardless of VRAM")

    if free_vram_mb > 0 and estimated_mb > free_vram_mb * 1.5:
        merged["warn_oom_risk"] = True
        print(
            "Warning: graph may exceed available VRAM even with chunking. "
            "Consider reducing dataset size or using cpu_multi mode."
        )

    return merged


# ---------------------------------------------------------------------------
# Public: print_gpu_summary()
# ---------------------------------------------------------------------------

def print_gpu_summary() -> None:
    """Print a boxed human-readable summary of the detected GPU."""
    cfg = get_gpu_config()
    arch = _architecture_name(cfg["compute_capability"])

    lines = [
        ("Device",             cfg["device_name"]),
        ("VRAM",               f"{cfg['vram_mb']} MB"),
        ("Free VRAM",          f"{cfg['free_vram_mb']} MB"),
        ("Compute capability", cfg["compute_capability"]),
        ("Architecture",       arch),
        ("SM count",           str(cfg["multiprocessor_count"])),
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


# ---------------------------------------------------------------------------
# Script entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_gpu_summary()
    print()
    print(json.dumps(get_gpu_config(), indent=2))
