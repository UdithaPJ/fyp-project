"""
src/algorithms/cpu/multi_threaded/louvain.py
============================================

Louvain community detection — multi-process CPU implementation only.

Phase 1 is parallelised by distributing nodes across workers.  Each worker
computes the best community move for its node batch using a *snapshot* of
the community degree sums taken at the start of each batch round.  All
proposed moves are applied simultaneously (bulk-synchronous update).

.. warning::
    Parallel Louvain non-determinism — this mode may produce different
    community assignments and modularity values than cpu_single because
    simultaneous moves can conflict (e.g. two nodes propose to swap
    communities).  This is a documented property of bulk-synchronous Louvain
    and does not indicate an error.

Phase 2 (community collapse) runs on the main process (sequential).

The ProcessPoolExecutor pool is created once per ``louvain_cpu_multi`` call
and reused across all Phase 1 passes and Louvain levels, amortising the
expensive process-spawn cost on Windows (spawn start method).

Shared helpers imported from src.algorithms.common.helpers:
    _symmetrize, _build_result, _run_louvain

Exclusive to this file:
    _louvain_batch_worker — bulk-synchronous batch Phase 1 worker

For the deterministic sequential variant see:
    src.algorithms.cpu.single_threaded.louvain
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import (
    _build_result,
    _run_louvain,
    _symmetrize,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "min_delta_q":       1e-4,
    "max_levels":        10,
    "resolution":        1.0,
    "max_phase1_passes": 100,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# Module-level worker — must be at module scope for ProcessPoolExecutor pickle
# ---------------------------------------------------------------------------

def _louvain_batch_worker(args: tuple) -> list[tuple[int, int]]:
    """
    ProcessPoolExecutor worker: compute the best move for a batch of nodes.

    Uses a *snapshot* of ``comm_degsums`` taken before the batch starts.
    All proposals are returned as (node_index, best_community) pairs.
    The caller applies all moves together (bulk-synchronous update).
    """
    (adj_data, adj_indices, adj_indptr,
     communities, degrees, comm_degsums,
     m, resolution, min_delta_q, node_batch) = args

    proposals: list[tuple[int, int]] = []
    n_comms = len(comm_degsums)

    for i in node_batch:
        ki    = float(degrees[i])
        c_old = int(communities[i])

        sigma_old = float(comm_degsums[c_old]) - ki

        s = int(adj_indptr[i])
        e = int(adj_indptr[i + 1])
        if s == e:
            proposals.append((i, c_old))
            continue

        nb_comms   = communities[adj_indices[s:e]]
        nb_weights = adj_data[s:e].astype(np.float64)

        unique_comms, inverse = np.unique(nb_comms, return_inverse=True)
        k_i_c = np.zeros(len(unique_comms), dtype=np.float64)
        np.add.at(k_i_c, inverse, nb_weights)

        k_i_c_old = k_i_c[unique_comms == c_old].sum()

        best_dq   = min_delta_q
        best_comm = c_old

        for idx, c_new in enumerate(unique_comms):
            if c_new == c_old or c_new >= n_comms:
                continue
            sigma_new = float(comm_degsums[c_new])
            dq = (
                (k_i_c[idx] - k_i_c_old) / m
                - resolution * ki * (sigma_new - sigma_old) / (2.0 * m * m)
            )
            if dq > best_dq:
                best_dq   = dq
                best_comm = int(c_new)

        proposals.append((i, best_comm))

    return proposals


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def louvain_cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict,
    n_workers: int = 4,
) -> dict:
    """
    Louvain — multi-process CPU with bulk-synchronous batch Phase 1.

    The directed GRN is symmetrized internally (A_sym = A + A^T).
    A single ProcessPoolExecutor is created for the entire Louvain run
    and reused across all Phase 1 passes and levels to amortise the
    process-spawn cost.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
    params    : dict
        min_delta_q       (float, default 1e-4)
        max_levels        (int,   default 10)
        resolution        (float, default 1.0)
        max_phase1_passes (int,   default 100)
    n_workers : int

    Returns
    -------
    dict with keys: community_assignments, num_communities, modularity,
                    hierarchy, top_communities
    """
    warnings.warn(
        "louvain_cpu_multi uses bulk-synchronous parallel Phase 1.  "
        "Community assignments may differ from cpu_single due to batched "
        "move application — this is expected behaviour, not a bug.",
        UserWarning,
        stacklevel=2,
    )

    p                 = _merge_params(params)
    resolution        = float(p["resolution"])
    min_delta_q       = float(p["min_delta_q"])
    max_levels        = int(p["max_levels"])
    max_phase1_passes = int(p["max_phase1_passes"])

    A_sym = _symmetrize(graph_csr)
    m     = float(A_sym.sum()) / 2.0

    with ProcessPoolExecutor(max_workers=n_workers) as ex:

        def _phase1_batch(
            adj_csr:     sp.csr_matrix,
            communities: np.ndarray,
            degrees:     np.ndarray,
            m_inner:     float,
            res:         float,
            mdq:         float,
        ) -> tuple[np.ndarray, bool]:
            """Bulk-synchronous batch Phase 1 — reuses the outer pool."""
            N_inner   = adj_csr.shape[0]
            n_slots   = int(communities.max()) + 1
            comm_degs = np.zeros(n_slots, dtype=np.float64)
            np.add.at(comm_degs, communities, degrees)

            chunk_size = max(1, (N_inner + n_workers - 1) // n_workers)
            batches = [
                list(range(i, min(i + chunk_size, N_inner)))
                for i in range(0, N_inner, chunk_size)
            ]
            args_list = [
                (adj_csr.data, adj_csr.indices, adj_csr.indptr,
                 communities, degrees, comm_degs,
                 m_inner, res, mdq, batch)
                for batch in batches
            ]

            all_proposals = list(ex.map(_louvain_batch_worker, args_list))

            improved = False
            for batch_props in all_proposals:
                for node_i, new_comm in batch_props:
                    if communities[node_i] != new_comm:
                        communities[node_i] = new_comm
                        improved = True

            return communities, improved

        final_labels, hierarchy = _run_louvain(
            A_sym, m,
            resolution        = resolution,
            min_delta_q       = min_delta_q,
            max_levels        = max_levels,
            phase1_fn         = _phase1_batch,
            max_phase1_passes = max_phase1_passes,
        )

    return _build_result(A_sym, final_labels, hierarchy, m, resolution)


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict | None = None,
    n_workers: int = 4,
    **_,
) -> dict:
    """Benchmark runner entry-point for cpu_multi mode."""
    p = _merge_params(params)
    return {
        "output":       louvain_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "n_workers": n_workers},
    }
