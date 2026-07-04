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
    performs the best greedy move if ΔQ > 0.

    ΔQ (move i from c_old to c_new) =
        (k_{i,c_new} - k_{i,c_old}) / m
        - resolution · k_i · (Σ_tot_c_new - Σ_tot_c_old) / (2m²)

    where Σ_tot_c is the sum of degrees of nodes in c (after removing i).

    Performance
    -----------
    The neighbour-community weights ``k_{i,c}`` are accumulated with a
    persistent scratch array (``wtc``) that is cleared per node via a
    ``touched`` list.  This replaces the previous per-node
    ``np.unique`` + ``np.add.at`` + boolean-mask ``.sum()`` — three numpy
    calls whose ~38 µs fixed overhead on ~12-element neighbour arrays
    dominated the runtime (profiled: ``np.unique`` alone was ~48 % of
    total).  Hot arrays are materialised as Python lists so the inner
    scan is pure list indexing with no numpy scalar-access overhead.

    Convergence
    -----------
    ``improved`` reports whether any node changed community — the true
    Phase-1 fixed point.  The *practical* early stop is the
    modularity-delta check in :func:`_run_louvain` (stop when a pass raises
    Q by less than ``min_delta_q``); that is where ``min_delta_q`` is
    consumed.  It is deliberately NOT used as a per-move acceptance
    threshold here: the natural per-move gain magnitude is ~1/(2m), so any
    fixed absolute value would reject every move once the graph has more
    than a few thousand edges.  The per-move criterion is simply ΔQ > 0.

    Returns (communities, improved).
    """
    N       = len(communities)
    n_slots = int(communities.max()) + 1

    comm_degsums_np = np.zeros(n_slots, dtype=np.float64)
    np.add.at(comm_degsums_np, communities, degrees)

    # Python lists for the hot read/update paths — list indexing is ~5× the
    # throughput of numpy scalar indexing inside this Python-level loop.
    comm_degsums = comm_degsums_np.tolist()
    deg_list     = degrees.tolist()
    wtc          = [0.0] * n_slots          # scratch: weight from i to comm c

    indptr  = adj_csr.indptr
    indices = adj_csr.indices
    data    = adj_csr.data

    inv_m   = 1.0 / m
    inv_2m2 = 1.0 / (2.0 * m * m)

    moves = 0

    for i in range(N):
        ki    = deg_list[i]
        c_old = int(communities[i])

        comm_degsums[c_old] -= ki

        s = int(indptr[i])
        e = int(indptr[i + 1])
        if s == e:
            comm_degsums[c_old] += ki
            continue

        # One vectorised gather each, then a pure-Python accumulation scan.
        nbr_comms = communities[indices[s:e]].tolist()
        nbr_wts   = data[s:e].tolist()

        touched = []
        for c, w in zip(nbr_comms, nbr_wts):
            prev = wtc[c]
            if prev == 0.0:
                touched.append(c)
            wtc[c] = prev + w

        k_i_c_old = wtc[c_old]          # 0.0 if no neighbour in c_old
        cds_old   = comm_degsums[c_old]
        res_ki    = resolution * ki

        best_gain = 0.0
        best_comm = c_old
        for c_new in touched:
            if c_new == c_old:
                continue
            gain = (
                (wtc[c_new] - k_i_c_old) * inv_m
                - res_ki * (comm_degsums[c_new] - cds_old) * inv_2m2
            )
            if gain > best_gain:
                best_gain = gain
                best_comm = c_new

        # Clear only the touched slots so the scratch array stays reusable.
        for c in touched:
            wtc[c] = 0.0

        communities[i] = best_comm
        comm_degsums[best_comm] += ki
        if best_comm != c_old:
            moves += 1

    improved = moves > 0
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
