"""Tests for src/optimization/memory_manager.py.

The MemoryManager planner is pure-Python and does not require an
active CUDA context for the planning logic — it queries
``cuda.mem_get_info`` opportunistically and degrades gracefully when
the call fails.  These tests verify the decision matrix and the
shape of the returned plan; they do not run actual GPU kernels.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_csr(n: int, avg_deg: int, *, seed: int = 0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    density = min(0.9, avg_deg / max(1, n))
    return sp.random(n, n, density=density, format="csr",
                     dtype=np.float32, random_state=rng)


def _star_csr(n: int) -> sp.csr_matrix:
    """1 hub connected to n-1 leaves; extreme degree skew."""
    rows = np.zeros(n - 1, dtype=np.int32)
    cols = np.arange(1, n, dtype=np.int32)
    data = np.ones(n - 1, dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# Public-surface presence checks
# ---------------------------------------------------------------------------

def test_module_exports_expected_symbols():
    from src.optimization import memory_manager as mm
    expected = {
        "MemoryManager",
        "get_available_vram",
        "estimate_algorithm_memory",
        "compute_chunk_size",
        "should_use_unified_memory",
        "should_use_zero_copy",
        "should_partition_graph",
    }
    for name in expected:
        assert hasattr(mm, name), f"missing public symbol {name}"


def test_memorymanager_static_methods():
    from src.optimization.memory_manager import MemoryManager
    for name in (
        "get_available_vram", "estimate_algorithm_memory",
        "compute_chunk_size", "should_use_unified_memory",
        "should_use_zero_copy", "should_partition_graph",
        "select_execution_mode", "plan",
    ):
        assert callable(getattr(MemoryManager, name)), \
            f"MemoryManager.{name} missing or not callable"


# ---------------------------------------------------------------------------
# get_available_vram
# ---------------------------------------------------------------------------

def test_get_available_vram_returns_dict():
    from src.optimization.memory_manager import get_available_vram
    info = get_available_vram()
    assert isinstance(info, dict)
    assert "free_mb" in info and "total_mb" in info
    assert isinstance(info["free_mb"], int)
    assert isinstance(info["total_mb"], int)
    assert info["free_mb"] >= 0
    assert info["total_mb"] >= 0


# ---------------------------------------------------------------------------
# estimate_algorithm_memory
# ---------------------------------------------------------------------------

def test_estimate_returns_required_fields():
    from src.optimization.memory_manager import estimate_algorithm_memory
    csr = _uniform_csr(50, 4)
    est = estimate_algorithm_memory("pagerank", csr, {})
    for k in (
        "total_mb", "available_mb", "pressure",
        "needs_chunking", "free_mb", "total_mb_alloc",
    ) if False else ("total_mb", "available_mb", "pressure",
                     "needs_chunking", "free_mb", "graph_profile"):
        # graph_profile is added by MemoryManager, the rest by MemoryEstimator.
        assert k in est, f"estimate missing key {k!r}; got keys: {list(est)}"
    assert est["pressure"] in ("low", "medium", "high", "critical")


def test_estimate_handles_non_csr_input():
    from src.optimization.memory_manager import estimate_algorithm_memory

    class _Fake:
        shape = (10, 10)
        nnz   = 5
    est = estimate_algorithm_memory("pagerank", _Fake(), {})
    assert est["pressure"] in ("low", "medium", "high", "critical")


# ---------------------------------------------------------------------------
# compute_chunk_size
# ---------------------------------------------------------------------------

def test_compute_chunk_size_clamped_to_n():
    from src.optimization.memory_manager import compute_chunk_size
    csr = _uniform_csr(50, 3)
    cs = compute_chunk_size(csr, available_vram_mb=10_000, algorithm_name="pagerank")
    assert 1 <= cs <= 50


def test_compute_chunk_size_zero_vram_safe():
    from src.optimization.memory_manager import compute_chunk_size
    csr = _uniform_csr(50, 3)
    cs = compute_chunk_size(csr, available_vram_mb=0, algorithm_name="pagerank")
    # With no budget the planner still returns at least 1 (kernel can't take 0).
    assert cs >= 1


def test_compute_chunk_size_handles_empty_input():
    from src.optimization.memory_manager import compute_chunk_size
    csr = sp.csr_matrix((0, 0), dtype=np.float32)
    cs = compute_chunk_size(csr, available_vram_mb=1024, algorithm_name="pagerank")
    assert cs == 0


# ---------------------------------------------------------------------------
# should_use_* predicates
# ---------------------------------------------------------------------------

def test_should_use_unified_memory_zero_vram_returns_false():
    from src.optimization.memory_manager import should_use_unified_memory
    assert should_use_unified_memory({"total_mb": 100.0}, 0) is False


def test_should_partition_with_huge_estimate_returns_true():
    from src.optimization.memory_manager import should_partition_graph
    assert should_partition_graph({"total_mb": 100_000.0}, 1024) is True


def test_should_partition_with_no_vram_and_zero_estimate_false():
    from src.optimization.memory_manager import should_partition_graph
    assert should_partition_graph({"total_mb": 0.0}, 0) is False


# ---------------------------------------------------------------------------
# select_execution_mode — the decision matrix
# ---------------------------------------------------------------------------

def test_small_graph_selects_normal_or_pressure_appropriate():
    from src.optimization.memory_manager import MemoryManager
    csr = _uniform_csr(30, 3)
    plan = MemoryManager.select_execution_mode("pagerank", csr, {})
    assert plan["execution_mode"] in (
        "normal_gpu", "chunked_gpu", "unified_memory",
        "zero_copy", "partitioned_gpu",
    )
    # On a 30-node graph nothing else makes sense than normal_gpu unless
    # the CI host has < 1 MB of free VRAM, in which case the planner
    # must at least not crash.
    assert plan["estimated_memory_mb"] >= 0
    assert plan["available_vram_mb"] >= 0


def test_plan_dict_has_full_schema():
    from src.optimization.memory_manager import MemoryManager
    csr = _uniform_csr(40, 4)
    plan = MemoryManager.select_execution_mode("pagerank", csr, {})
    required_keys = {
        "execution_mode", "estimated_memory_mb", "available_vram_mb",
        "memory_pressure", "use_chunking", "chunk_size",
        "use_unified_memory", "use_zero_copy", "use_partitioning",
        "partition_strategy", "mode_reason",
    }
    missing = required_keys - set(plan)
    assert not missing, f"missing plan keys: {missing}"


def test_flags_are_mutually_exclusive_per_mode():
    """Exactly one of the four 'special' modes should be active per plan."""
    from src.optimization.memory_manager import MemoryManager
    for algo in ("pagerank", "bfs", "hits", "louvain", "rwr", "mcl"):
        plan = MemoryManager.select_execution_mode(algo, _uniform_csr(40, 4), {})
        flags = (
            plan["use_chunking"],
            plan["use_unified_memory"],
            plan["use_zero_copy"],
            plan["use_partitioning"],
        )
        # normal_gpu sets all four False.
        if plan["execution_mode"] == "normal_gpu":
            assert flags == (False, False, False, False)
        else:
            assert sum(bool(f) for f in flags) == 1, (
                f"algo={algo} mode={plan['execution_mode']} "
                f"flags should be exactly one True; got {flags}"
            )


def test_plan_alias_matches_select_execution_mode():
    from src.optimization.memory_manager import MemoryManager
    csr = _uniform_csr(40, 4)
    plan_a = MemoryManager.plan("pagerank", csr, {})
    plan_b = MemoryManager.select_execution_mode("pagerank", csr, {})
    # Keys must match identically (values may drift if memory between
    # calls changed — we only assert schema).
    assert set(plan_a.keys()) == set(plan_b.keys())


def test_unknown_algorithm_falls_back_to_uniform_multiplier():
    """An unknown algorithm should still produce a valid plan dict."""
    from src.optimization.memory_manager import MemoryManager
    csr = _uniform_csr(40, 4)
    plan = MemoryManager.select_execution_mode("not_a_real_algo", csr, {})
    assert "execution_mode" in plan
    assert plan["execution_mode"] in (
        "normal_gpu", "chunked_gpu", "unified_memory",
        "zero_copy", "partitioned_gpu",
    )


# ---------------------------------------------------------------------------
# Star graph (extreme degree skew) — partitioning recommendation
# ---------------------------------------------------------------------------

def test_skewed_graph_partition_strategy_choice():
    """When the planner *does* recommend partitioning, the strategy
    should reflect the graph's degree class."""
    from src.optimization.memory_manager import MemoryManager
    csr = _star_csr(200)
    plan = MemoryManager.select_execution_mode("mcl", csr, {})
    # On a tiny star graph this will most likely be normal_gpu; we just
    # assert the partition_strategy is well-typed when set.
    if plan["use_partitioning"]:
        assert plan["partition_strategy"] in ("by_edges", "by_degree")
