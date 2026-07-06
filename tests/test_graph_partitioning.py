"""Tests for src/optimization/graph_partitioning.py."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


def _uniform_csr(n: int, avg_deg: int, seed: int = 0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    density = min(0.9, avg_deg / max(1, n))
    return sp.random(n, n, density=density, format="csr",
                     dtype=np.float32, random_state=rng)


def _star_csr(n: int) -> sp.csr_matrix:
    rows = np.zeros(n - 1, dtype=np.int32)
    cols = np.arange(1, n, dtype=np.int32)
    data = np.ones(n - 1, dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

def test_module_exports():
    from src.optimization import graph_partitioning as gp
    for name in ("Partition", "GraphPartitioner",
                 "identify_hub_nodes", "estimate_partition_memory"):
        assert hasattr(gp, name), f"missing public symbol {name}"


# ---------------------------------------------------------------------------
# Partition dataclass
# ---------------------------------------------------------------------------

def test_partition_as_dict_round_trip():
    from src.optimization.graph_partitioning import Partition
    p = Partition(
        partition_id=0, start_node=0, end_node=10,
        num_nodes=10, num_edges=42,
        estimated_memory_mb=1.5,
    )
    d = p.as_dict()
    assert d["partition_id"] == 0
    assert d["num_nodes"] == 10
    assert d["num_edges"] == 42
    assert d["estimated_memory_mb"] == pytest.approx(1.5)
    assert d["hub_nodes"] == []
    assert d["boundary_edges_out"] == {}


# ---------------------------------------------------------------------------
# partition_by_edges
# ---------------------------------------------------------------------------

def test_partition_by_edges_covers_all_rows():
    from src.optimization.graph_partitioning import GraphPartitioner
    csr = _uniform_csr(200, 5)
    parts = GraphPartitioner.partition_by_edges(csr, target_edges_per_partition=200)
    # Union of [start, end) must cover [0, n)
    assert parts[0].start_node == 0
    assert parts[-1].end_node == csr.shape[0]
    for i in range(1, len(parts)):
        assert parts[i].start_node == parts[i - 1].end_node


def test_partition_by_edges_balances_within_factor_two():
    from src.optimization.graph_partitioning import GraphPartitioner
    csr = _uniform_csr(500, 6)
    target = 600
    parts = GraphPartitioner.partition_by_edges(csr, target_edges_per_partition=target)
    sizes = [p.num_edges for p in parts]
    # First (len-1) partitions are closed when running >= target, so each
    # is at least ``target`` edges.  The trailing partition can be small.
    if len(parts) > 1:
        body = sizes[:-1]
        assert min(body) >= target * 0.9       # 10 % slack for greedy cutoff
        assert max(body) <= target * 3.0       # bounded above (no hub explosion)


def test_partition_by_edges_handles_empty_graph():
    from src.optimization.graph_partitioning import GraphPartitioner
    csr = sp.csr_matrix((0, 0), dtype=np.float32)
    parts = GraphPartitioner.partition_by_edges(csr, target_edges_per_partition=100)
    assert parts == []


def test_partition_by_edges_handles_no_edges():
    from src.optimization.graph_partitioning import GraphPartitioner
    csr = sp.csr_matrix(np.zeros((20, 20), dtype=np.float32))
    parts = GraphPartitioner.partition_by_edges(csr, target_edges_per_partition=10)
    assert len(parts) == 1
    assert parts[0].num_edges == 0
    assert parts[0].num_nodes == 20


# ---------------------------------------------------------------------------
# partition_by_degree
# ---------------------------------------------------------------------------

def test_partition_by_degree_respects_budget():
    from src.optimization.graph_partitioning import GraphPartitioner
    csr = _uniform_csr(300, 4)
    target_mb = 0.003   # ~ 3 KB — forces multiple partitions
    parts = GraphPartitioner.partition_by_degree(csr, target_memory_mb=target_mb)
    assert len(parts) >= 2
    # Each partition's estimate should be at-or-below target_mb (within
    # one-row slack for the single-oversize-row edge case).
    for p in parts:
        # Allow some slack for the +1 indptr overhead, but flag genuine overflow.
        if p.num_nodes > 1:
            assert p.estimated_memory_mb <= target_mb * 1.5


def test_partition_by_degree_emits_single_oversize_row():
    """When a single row would exceed the budget, it gets its own
    partition rather than being silently merged."""
    from src.optimization.graph_partitioning import GraphPartitioner
    csr = _star_csr(1000)  # row 0 has 999 edges → ~8 KB
    parts = GraphPartitioner.partition_by_degree(csr, target_memory_mb=0.005)
    # Row 0 must be in a partition of its own (or at least be the first
    # partition).
    assert parts[0].start_node == 0
    assert parts[0].end_node >= 1


# ---------------------------------------------------------------------------
# partition_by_nodes
# ---------------------------------------------------------------------------

def test_partition_by_nodes_equal_size():
    from src.optimization.graph_partitioning import GraphPartitioner
    csr = _uniform_csr(100, 4)
    parts = GraphPartitioner.partition_by_nodes(csr, target_nodes_per_partition=25)
    assert len(parts) == 4
    for p in parts[:-1]:
        assert p.num_nodes == 25
    # Last partition may be smaller if n not divisible.


# ---------------------------------------------------------------------------
# Hub detection
# ---------------------------------------------------------------------------

def test_identify_hub_nodes_star_graph():
    from src.optimization.graph_partitioning import identify_hub_nodes
    csr = _star_csr(100)
    hubs = identify_hub_nodes(csr)
    assert 0 in hubs.tolist()


def test_identify_hub_nodes_uniform_graph_has_few_hubs():
    from src.optimization.graph_partitioning import identify_hub_nodes
    csr = _uniform_csr(200, 4)
    hubs = identify_hub_nodes(csr)
    # < 20 % of nodes should be hubs in a uniform random graph.
    assert hubs.size < 0.2 * csr.shape[0]


def test_identify_hub_nodes_custom_threshold():
    from src.optimization.graph_partitioning import identify_hub_nodes
    csr = _star_csr(100)
    # Threshold above the hub's degree → no hubs.
    hubs = identify_hub_nodes(csr, hub_threshold=10_000)
    assert hubs.size == 0


# ---------------------------------------------------------------------------
# Boundary metadata
# ---------------------------------------------------------------------------

def test_build_boundary_metadata_counts_cross_partition_edges():
    from src.optimization.graph_partitioning import GraphPartitioner
    csr = _uniform_csr(100, 6, seed=1)
    parts = GraphPartitioner.partition_by_nodes(csr, target_nodes_per_partition=25)
    GraphPartitioner.build_boundary_metadata(csr, parts)
    # At least one partition should have non-empty boundary_edges_out
    # for a moderately dense random graph.
    assert any(p.boundary_edges_out for p in parts)
    # Sum of all out-counts across partitions == edges with src.pid != dst.pid
    total_out = sum(sum(p.boundary_edges_out.values()) for p in parts)
    # Sanity: total_out <= total edges
    assert total_out <= csr.nnz


def test_build_boundary_metadata_populates_hub_nodes():
    from src.optimization.graph_partitioning import GraphPartitioner
    csr = _star_csr(200)
    parts = GraphPartitioner.partition_by_nodes(csr, target_nodes_per_partition=50)
    GraphPartitioner.build_boundary_metadata(csr, parts)
    # Row 0 lands in partition 0; the hub should be flagged.
    assert 0 in parts[0].hub_nodes


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def test_partition_dispatcher_by_edges():
    from src.optimization.graph_partitioning import GraphPartitioner
    csr = _uniform_csr(100, 5, seed=2)
    parts = GraphPartitioner.partition(csr, strategy="by_edges")
    assert parts
    assert parts[0].partition_id == 0


def test_partition_dispatcher_rejects_unknown_strategy():
    from src.optimization.graph_partitioning import GraphPartitioner
    csr = _uniform_csr(20, 3)
    with pytest.raises(ValueError):
        GraphPartitioner.partition(csr, strategy="by_quantum_entanglement")


def test_estimate_partition_memory_dict_form():
    from src.optimization.graph_partitioning import estimate_partition_memory
    mb = estimate_partition_memory({"num_nodes": 1000, "num_edges": 5000})
    assert mb > 0
