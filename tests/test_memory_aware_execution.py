"""Tests for end-to-end integration of the memory-aware layer.

These tests verify that:
  1. ``apply_config()`` returns the new memory-plan keys for every algorithm.
  2. User params still win over auto-selected plan keys.
  3. The unified-memory wrappers degrade gracefully without PyCUDA.
  4. No algorithm is silently downgraded to CPU based on VRAM pressure
     (the planner emits a recommendation; algorithms either honour it
     or raise ``MemoryError`` — never silently fall back).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import scipy.sparse as sp


def _uniform_csr(n: int, avg_deg: int, seed: int = 0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    density = min(0.9, avg_deg / max(1, n))
    return sp.random(n, n, density=density, format="csr",
                     dtype=np.float32, random_state=rng)


# ---------------------------------------------------------------------------
# apply_config integration
# ---------------------------------------------------------------------------

ALGOS = ("pagerank", "bfs", "hits", "louvain", "rwr", "mcl")


@pytest.mark.parametrize("algo", ALGOS)
def test_apply_config_emits_memory_plan(algo):
    from src.optimization.gpu_config import apply_config
    csr = _uniform_csr(80, 4)
    p = apply_config(algo, csr, {})
    # Memory plan keys are now part of the merged params.
    for k in (
        "execution_mode", "estimated_memory_mb", "available_vram_mb",
        "memory_pressure", "use_chunking", "chunk_size",
        "use_unified_memory", "use_zero_copy", "use_partitioning",
        "partition_strategy", "mode_reason",
    ):
        assert k in p, f"{algo}: plan key {k!r} missing"
    # Metadata block.
    assert "_memory_plan" in p
    assert p["_memory_plan"]["execution_mode"] == p["execution_mode"]


def test_user_chunk_size_overrides_planner():
    from src.optimization.gpu_config import apply_config
    csr = _uniform_csr(80, 4)
    p = apply_config("pagerank", csr, {"chunk_size": 9999, "use_chunking": True})
    assert p["chunk_size"] == 9999
    assert p["use_chunking"] is True


def test_user_execution_mode_overrides_planner():
    from src.optimization.gpu_config import apply_config
    csr = _uniform_csr(80, 4)
    p = apply_config("pagerank", csr, {"execution_mode": "normal_gpu"})
    assert p["execution_mode"] == "normal_gpu"


def test_apply_config_override_true_lets_user_keys_fall_through():
    """Even with override=True, *new* user keys land in the dict — the
    override flag only flips precedence on overlapping keys.
    """
    from src.optimization.gpu_config import apply_config
    csr = _uniform_csr(40, 3)
    p = apply_config(
        "pagerank", csr,
        {"my_custom_flag": "keep"},
        override=True,
    )
    assert p["my_custom_flag"] == "keep"


@pytest.mark.parametrize("algo", ALGOS)
def test_no_silent_cpu_fallback_keys(algo):
    """The planner must not emit a ``mode='cpu'`` shortcut."""
    from src.optimization.gpu_config import apply_config
    csr = _uniform_csr(60, 3)
    p = apply_config(algo, csr, {})
    assert p["execution_mode"] in (
        "normal_gpu", "chunked_gpu", "unified_memory",
        "zero_copy", "partitioned_gpu",
    ), f"{algo}: unexpected execution_mode {p['execution_mode']!r}"


# ---------------------------------------------------------------------------
# Unified memory capability probe
# ---------------------------------------------------------------------------

def test_unified_memory_capabilities_shape():
    from src.optimization.unified_memory import capabilities
    caps = capabilities()
    for k in ("pycuda_available", "managed_memory",
              "pagelocked_memory", "mem_prefetch_async"):
        assert k in caps and isinstance(caps[k], bool)


def test_unified_memory_allocator_safe_without_pycuda():
    """Without PyCUDA, allocate_device_array still returns a usable
    ManagedArray (host_only mode)."""
    from src.optimization.unified_memory import allocate_device_array
    arr = allocate_device_array((100,), np.float32)
    assert arr is not None
    assert arr.mode in ("device", "host_only")
    arr.free()


def test_unified_memory_copy_to_managed_array_round_trip():
    from src.optimization.unified_memory import copy_to_managed_array
    src = np.arange(50, dtype=np.float32)
    arr = copy_to_managed_array(src)
    assert arr.mode in (
        "unified_memory", "zero_copy", "pinned", "device", "host_only",
    )
    # host_view must mirror src for the host-shared modes.
    if arr.mode in ("unified_memory", "zero_copy", "pinned", "host_only"):
        np.testing.assert_array_equal(arr.host_view, src)
    arr.free()


# ---------------------------------------------------------------------------
# Decision matrix end-to-end (synthetic VRAM)
# ---------------------------------------------------------------------------

def test_critical_pressure_avoids_cpu(monkeypatch):
    """Force the estimator to report CRITICAL memory pressure and check
    that the planner picks a GPU path (never 'cpu' / 'cpu_fallback')."""
    from src.optimization import memory_manager as mm

    # Force the estimator to look critical.
    def fake_estimate(algo, csr, params):
        return {
            "total_mb":         99999.0,
            "available_mb":     1024.0,
            "pressure":         "critical",
            "needs_chunking":   True,
            "free_mb":          1024,
            "total_mb_alloc":   99999.0,
            "graph_profile":    {"degree_class": "power_law"},
        }

    monkeypatch.setattr(mm, "estimate_algorithm_memory", fake_estimate)
    csr = _uniform_csr(80, 4)
    for algo in ALGOS:
        plan = mm.MemoryManager.select_execution_mode(algo, csr, {})
        assert plan["execution_mode"] in (
            "chunked_gpu", "unified_memory", "zero_copy", "partitioned_gpu",
        ), f"{algo}: critical pressure must not fall back to CPU; got {plan['execution_mode']!r}"


def test_low_pressure_picks_normal_gpu(monkeypatch):
    from src.optimization import memory_manager as mm

    def fake_estimate(algo, csr, params):
        return {
            "total_mb":         1.0,
            "available_mb":     8000.0,
            "pressure":         "low",
            "needs_chunking":   False,
            "free_mb":          8000,
            "total_mb_alloc":   1.0,
            "graph_profile":    {"degree_class": "uniform"},
        }

    monkeypatch.setattr(mm, "estimate_algorithm_memory", fake_estimate)
    csr = _uniform_csr(80, 4)
    plan = mm.MemoryManager.select_execution_mode("pagerank", csr, {})
    assert plan["execution_mode"] == "normal_gpu"
    assert plan["use_chunking"] is False


def test_chunked_path_for_capable_algo_when_pressure_high(monkeypatch):
    from src.optimization import memory_manager as mm

    def fake_estimate(algo, csr, params):
        return {
            "total_mb":         5000.0,
            "available_mb":     4000.0,
            "pressure":         "high",
            "needs_chunking":   True,
            "free_mb":          4000,
            "total_mb_alloc":   5000.0,
            "graph_profile":    {"degree_class": "uniform"},
        }

    monkeypatch.setattr(mm, "estimate_algorithm_memory", fake_estimate)
    csr = _uniform_csr(80, 4)
    plan = mm.MemoryManager.select_execution_mode("pagerank", csr, {})
    # PageRank IS chunk-capable, so high pressure → chunked_gpu.
    assert plan["execution_mode"] == "chunked_gpu"
    assert plan["use_chunking"] is True
    assert plan["chunk_size"] >= 1


def test_non_chunk_capable_algo_does_not_get_chunked(monkeypatch):
    """HITS / MCL / BFS aren't in _CHUNK_CAPABLE_ALGORITHMS, so under
    pressure they should land on unified_memory / zero_copy /
    partitioned_gpu — never silently on chunked_gpu (which the
    algorithm wouldn't honour)."""
    from src.optimization import memory_manager as mm

    def fake_estimate(algo, csr, params):
        return {
            "total_mb":         5000.0,
            "available_mb":     1024.0,
            "pressure":         "critical",
            "needs_chunking":   True,
            "free_mb":          1024,
            "total_mb_alloc":   5000.0,
            "graph_profile":    {"degree_class": "skewed"},
        }

    monkeypatch.setattr(mm, "estimate_algorithm_memory", fake_estimate)
    csr = _uniform_csr(80, 4)
    for algo in ("hits", "mcl", "bfs"):
        plan = mm.MemoryManager.select_execution_mode(algo, csr, {})
        assert plan["execution_mode"] != "chunked_gpu", \
            f"{algo} got chunked_gpu but doesn't implement chunking"


# ---------------------------------------------------------------------------
# Backward compatibility: prior _memory_estimate / _strategy_selected blocks
# must still be present.
# ---------------------------------------------------------------------------

def test_legacy_metadata_blocks_preserved():
    from src.optimization.gpu_config import apply_config
    csr = _uniform_csr(60, 3)
    p = apply_config("pagerank", csr, {})
    assert "_graph_fingerprint" in p
    assert "_graph_profile"     in p
    assert "_memory_estimate"   in p
    assert "_strategy_selected" in p
    assert "_hardware_config"   in p
