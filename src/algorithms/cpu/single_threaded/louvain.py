"""
src/algorithms/cpu/single_threaded/louvain.py
=============================================

Louvain community detection — single-threaded CPU implementation only.

Phase 1 is a deterministic sequential node scan: for each node, the best
modularity-improving community move is computed and applied immediately.
Phase 2 collapses communities into weighted super-nodes via scipy sparse.

The directed GRN is symmetrized internally (A_sym = A + A^T) before
running Louvain — see module docstring of the original cpu/louvain.py for
the biological rationale.

Shared helpers imported from src.algorithms.common.helpers:
    _symmetrize, _compute_modularity, _phase2_collapse,
    _build_result, _run_louvain

Exclusive to this file:
    _phase1_single — deterministic sequential Phase 1

For the bulk-synchronous parallel variant see:
    src.algorithms.cpu.multi_threaded.louvain
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import (
    _build_result,
    _phase2_collapse,
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
# Phase 1 — sequential (exclusive to single-threaded mode)
# ---------------------------------------------------------------------------

def _phase1_single(
    adj_csr:     sp.csr_matrix,
    communities: np.ndarray,
    degrees:     np.ndarray,
    m:           float,
    resolution:  float,
    min_delta_q: float,
) -> tuple[np.ndarray, bool]:
    """
    One complete sequential pass of Louvain Phase 1.

    For each node, computes ΔQ for moving to each neighbour community and
    performs the best greedy move if ΔQ > min_delta_q.

    ΔQ (move i from c_old to c_new) =
        (k_{i,c_new} - k_{i,c_old}) / m
        - resolution · k_i · (Σ_tot_c_new - Σ_tot_c_old) / (2m²)

    where Σ_tot_c is the sum of degrees of nodes in c (after removing i).

    Returns (communities, improved).
    """
    N      = len(communities)
    n_slots = int(communities.max()) + 1
    comm_degsums = np.zeros(n_slots, dtype=np.float64)
    np.add.at(comm_degsums, communities, degrees)

    improved = False

    for i in range(N):
        ki    = degrees[i]
        c_old = int(communities[i])

        comm_degsums[c_old] -= ki

        s = int(adj_csr.indptr[i])
        e = int(adj_csr.indptr[i + 1])
        if s == e:
            comm_degsums[c_old] += ki
            continue

        nb_indices = adj_csr.indices[s:e]
        nb_weights = adj_csr.data[s:e].astype(np.float64)
        nb_comms   = communities[nb_indices].astype(np.int32)

        unique_comms, inverse = np.unique(nb_comms, return_inverse=True)
        k_i_c = np.zeros(len(unique_comms), dtype=np.float64)
        np.add.at(k_i_c, inverse, nb_weights)

        k_i_c_old = k_i_c[unique_comms == c_old].sum()

        best_dq   = 0.0
        best_comm = c_old

        for idx, c_new in enumerate(unique_comms):
            if c_new == c_old:
                continue
            dq = (
                (k_i_c[idx] - k_i_c_old) / m
                - resolution * ki * (comm_degsums[c_new] - comm_degsums[c_old])
                / (2.0 * m * m)
            )
            if dq > best_dq:
                best_dq   = dq
                best_comm = int(c_new)

        communities[i] = best_comm
        comm_degsums[best_comm] += ki
        if best_comm != c_old:
            improved = True

    return communities, improved


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def louvain_cpu_single(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """
    Louvain — single-threaded CPU with sequential Phase 1.

    The directed GRN is symmetrized internally (A_sym = A + A^T) before
    running Louvain.  Phase 1 is a deterministic sequential node scan.
    Phase 2 collapses communities into weighted super-nodes via scipy sparse.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix — directed GRN adjacency (CSR).
    params    : dict
        min_delta_q       (float, default 1e-4)
        max_levels        (int,   default 10)
        resolution        (float, default 1.0)
        max_phase1_passes (int,   default 100)

    Returns
    -------
    dict with keys: community_assignments, num_communities, modularity,
                    hierarchy, top_communities
    """
    p     = _merge_params(params)
    A_sym = _symmetrize(graph_csr)
    m     = float(A_sym.sum()) / 2.0

    final_labels, hierarchy = _run_louvain(
        A_sym, m,
        resolution        = float(p["resolution"]),
        min_delta_q       = float(p["min_delta_q"]),
        max_levels        = int(p["max_levels"]),
        phase1_fn         = _phase1_single,
        max_phase1_passes = int(p["max_phase1_passes"]),
    )
    return _build_result(A_sym, final_labels, hierarchy, m, float(p["resolution"]))


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _cpu_single(
    graph_csr: sp.csr_matrix,
    params: dict | None = None,
    **_,
) -> dict:
    """Benchmark runner entry-point for cpu_single mode."""
    p = _merge_params(params)
    return {"output": louvain_cpu_single(graph_csr, p), "extra_params": p}
