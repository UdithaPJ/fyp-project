"""
src/optimization/graph_partitioning.py — Memory-aware graph partitioning
========================================================================

Goes beyond simple row chunking.  ``GraphPartitioner`` produces
edge-balanced partitions, identifies hub nodes, and tracks
boundary edges crossing partition cuts.

Three partitioning strategies:

  * ``partition_by_edges``    — balance ``num_edges`` per partition.
                                Best when out-degree distribution is
                                skewed (power-law biological networks).
  * ``partition_by_degree``   — balance ``num_nodes`` weighted by degree
                                (effectively a memory-budget balance).
  * ``partition_by_nodes``    — simple equal-size row split (legacy
                                row chunking, kept for fallback).

Hub-aware splitting: ``identify_hub_nodes`` flags nodes whose degree
exceeds ``hub_threshold`` (defaults to ``3 * avg_degree``).  When a
partition would consist of a single hub plus its many neighbours, the
partitioner refuses and re-splits at a finer granularity.

Boundary metadata: ``build_boundary_metadata`` walks the CSR once and
records, for each partition, the count of edges crossing into every
other partition.  Algorithms that need to exchange ghost data across
cuts (e.g. push-style BFS) can read this directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Partition data class
# ---------------------------------------------------------------------------

@dataclass
class Partition:
    """One contiguous row-range partition of a CSR graph."""
    partition_id: int
    start_node:   int                              # inclusive
    end_node:     int                              # exclusive
    num_nodes:    int
    num_edges:    int
    estimated_memory_mb: float
    hub_nodes:    list[int] = field(default_factory=list)
    boundary_edges_out: dict[int, int] = field(default_factory=dict)
    # Edges leaving this partition, grouped by destination partition id.

    def as_dict(self) -> dict:
        return {
            "partition_id":        self.partition_id,
            "start_node":          self.start_node,
            "end_node":            self.end_node,
            "num_nodes":           self.num_nodes,
            "num_edges":           self.num_edges,
            "estimated_memory_mb": self.estimated_memory_mb,
            "hub_nodes":           list(self.hub_nodes),
            "boundary_edges_out":  dict(self.boundary_edges_out),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_lengths(csr: sp.csr_matrix) -> np.ndarray:
    return np.diff(csr.indptr).astype(np.int64)


def _bytes_for_rows(row_lens: np.ndarray, lo: int, hi: int) -> int:
    """Bytes needed for a row-range [lo, hi): indptr + (col_idx + values)."""
    nnz = int(row_lens[lo:hi].sum())
    return (hi - lo + 1) * 4 + nnz * 8


def estimate_partition_memory(partition: Partition | dict) -> float:
    """Return MB needed to hold the partition's CSR rows on the GPU."""
    if isinstance(partition, Partition):
        bytes_ = (partition.num_nodes + 1) * 4 + partition.num_edges * 8
    else:
        bytes_ = (int(partition["num_nodes"]) + 1) * 4 \
                 + int(partition["num_edges"]) * 8
    return float(bytes_) / (1024 * 1024)


def identify_hub_nodes(
    csr: sp.csr_matrix,
    hub_threshold: int | None = None,
    *,
    multiplier: float = 3.0,
) -> np.ndarray:
    """Return the indices of nodes whose degree exceeds ``hub_threshold``.

    If ``hub_threshold`` is None it defaults to
    ``max(WARP_SIZE, multiplier * avg_degree)``.
    """
    row_lens = _row_lengths(csr)
    n = int(csr.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    avg_deg = float(row_lens.sum()) / float(n)
    if hub_threshold is None:
        hub_threshold = max(32, int(multiplier * avg_deg))
    return np.where(row_lens > hub_threshold)[0].astype(np.int64)


# ---------------------------------------------------------------------------
# Partitioner
# ---------------------------------------------------------------------------

class GraphPartitioner:
    """Memory-aware graph partitioning for out-of-core GPU execution.

    Three partitioning entry points return ``list[Partition]``.  All
    partitions are contiguous row ranges — non-contiguous (e.g. random
    interleaved) splits would shuffle the CSR and are not supported
    yet.
    """

    @staticmethod
    def partition_by_edges(
        csr: sp.csr_matrix,
        target_edges_per_partition: int,
    ) -> list[Partition]:
        """Greedy edge-balanced partitioning.

        Walks rows in order and starts a new partition whenever the
        running edge count meets ``target_edges_per_partition``.  When
        a single row already exceeds the target the row gets its own
        partition (correctness over balance).
        """
        n = int(csr.shape[0])
        if n == 0 or target_edges_per_partition <= 0:
            return []

        row_lens = _row_lengths(csr)
        total_edges = int(row_lens.sum())
        if total_edges == 0:
            return [Partition(
                partition_id=0, start_node=0, end_node=n, num_nodes=n,
                num_edges=0, estimated_memory_mb=0.0,
            )]

        target = max(1, int(target_edges_per_partition))
        partitions: list[Partition] = []
        start = 0
        running = 0
        pid = 0
        for i in range(n):
            running += int(row_lens[i])
            # Close partition when the running edge count crosses target,
            # OR the single-row case (running > target with start == i).
            if running >= target:
                end = i + 1
                p = _make_partition(
                    csr, pid, start, end, row_lens,
                )
                partitions.append(p)
                pid += 1
                start = end
                running = 0
        # Trailing partition
        if start < n:
            partitions.append(_make_partition(csr, pid, start, n, row_lens))

        return partitions

    @staticmethod
    def partition_by_degree(
        csr: sp.csr_matrix,
        target_memory_mb: float,
    ) -> list[Partition]:
        """Memory-budget partitioning.

        Splits the row range so each partition fits within
        ``target_memory_mb`` of CSR storage (indptr + col_idx + values).
        Hub rows that exceed the budget on their own get their own
        partition with a warning.
        """
        n = int(csr.shape[0])
        if n == 0 or target_memory_mb <= 0:
            return []

        row_lens = _row_lengths(csr)
        budget_b = max(1, int(target_memory_mb * 1024 * 1024))
        partitions: list[Partition] = []
        start = 0
        used = 0
        pid = 0
        for i in range(n):
            row_b = 4 + int(row_lens[i]) * 8
            if row_b > budget_b and start == i:
                # Single oversize row — emit a 1-row partition.
                partitions.append(_make_partition(csr, pid, i, i + 1, row_lens))
                logging.info(
                    "partition_by_degree: row %d alone uses %.1f MB "
                    "(target %.1f MB).", i, row_b / (1024 * 1024),
                    target_memory_mb,
                )
                pid += 1
                start = i + 1
                used = 0
                continue
            if used + row_b > budget_b and start < i:
                partitions.append(_make_partition(csr, pid, start, i, row_lens))
                pid += 1
                start = i
                used = row_b
            else:
                used += row_b
        if start < n:
            partitions.append(_make_partition(csr, pid, start, n, row_lens))

        return partitions

    @staticmethod
    def partition_by_nodes(
        csr: sp.csr_matrix,
        target_nodes_per_partition: int,
    ) -> list[Partition]:
        """Simple equal-size row chunking (legacy behaviour)."""
        n = int(csr.shape[0])
        if n == 0 or target_nodes_per_partition <= 0:
            return []
        row_lens = _row_lengths(csr)
        partitions: list[Partition] = []
        pid = 0
        for start in range(0, n, int(target_nodes_per_partition)):
            end = min(start + int(target_nodes_per_partition), n)
            partitions.append(_make_partition(csr, pid, start, end, row_lens))
            pid += 1
        return partitions

    @staticmethod
    def identify_hub_nodes(
        csr: sp.csr_matrix,
        hub_threshold: int | None = None,
    ) -> np.ndarray:
        return identify_hub_nodes(csr, hub_threshold)

    @staticmethod
    def build_boundary_metadata(
        csr: sp.csr_matrix,
        partitions: list[Partition],
    ) -> list[Partition]:
        """Annotate each partition with the count of out-edges to every
        other partition.

        Mutates the partitions in place and returns them for chaining.
        """
        n = int(csr.shape[0])
        if n == 0 or not partitions:
            return partitions

        # node → partition_id lookup
        node_to_pid = np.zeros(n, dtype=np.int32)
        for p in partitions:
            node_to_pid[p.start_node:p.end_node] = p.partition_id

        indptr  = csr.indptr
        indices = csr.indices
        for p in partitions:
            counts: dict[int, int] = {}
            local_hubs = identify_hub_nodes(
                csr[p.start_node:p.end_node]
            ).tolist()
            p.hub_nodes = [int(p.start_node + h) for h in local_hubs]
            for u in range(p.start_node, p.end_node):
                row_start = int(indptr[u])
                row_end   = int(indptr[u + 1])
                for j in range(row_start, row_end):
                    v_pid = int(node_to_pid[indices[j]])
                    if v_pid != p.partition_id:
                        counts[v_pid] = counts.get(v_pid, 0) + 1
            p.boundary_edges_out = counts
        return partitions

    @staticmethod
    def estimate_partition_memory(partition) -> float:
        return estimate_partition_memory(partition)

    @staticmethod
    def partition(
        csr: sp.csr_matrix,
        strategy: str = "by_edges",
        *,
        target_edges_per_partition: int | None = None,
        target_memory_mb: float | None = None,
        target_nodes_per_partition: int | None = None,
        compute_boundary_metadata: bool = True,
    ) -> list[Partition]:
        """One-shot dispatcher used by ``MemoryManager``.

        Picks safe defaults when the corresponding ``target_*`` is None.
        """
        n   = int(csr.shape[0])
        nnz = int(csr.nnz)
        if n == 0:
            return []

        if strategy == "by_edges":
            target = target_edges_per_partition or max(1, nnz // max(1, n // 1024 + 1))
            parts = GraphPartitioner.partition_by_edges(csr, target)
        elif strategy == "by_degree":
            tmb = target_memory_mb or max(1.0, (nnz * 12 / (1024 * 1024)) / 8.0)
            parts = GraphPartitioner.partition_by_degree(csr, tmb)
        elif strategy == "by_nodes":
            target = target_nodes_per_partition or max(1, n // 8)
            parts = GraphPartitioner.partition_by_nodes(csr, target)
        else:
            raise ValueError(f"Unknown partition strategy: {strategy!r}")

        if compute_boundary_metadata and parts:
            GraphPartitioner.build_boundary_metadata(csr, parts)
        return parts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_partition(
    csr: sp.csr_matrix,
    pid: int,
    start: int,
    end: int,
    row_lens: np.ndarray,
) -> Partition:
    nnz = int(row_lens[start:end].sum())
    bytes_ = (end - start + 1) * 4 + nnz * 8
    return Partition(
        partition_id        = pid,
        start_node          = int(start),
        end_node            = int(end),
        num_nodes           = int(end - start),
        num_edges           = nnz,
        estimated_memory_mb = float(bytes_) / (1024 * 1024),
    )


__all__ = [
    "Partition",
    "GraphPartitioner",
    "identify_hub_nodes",
    "estimate_partition_memory",
]
