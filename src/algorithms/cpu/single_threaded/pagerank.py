"""
src/algorithms/cpu/single_threaded/pagerank.py
===============================================

PageRank — single-threaded CPU implementation only.

Propagates regulatory influence through the directed GRN via power iteration.
Dangling mass (from pure target genes with out_degree == 0) is redistributed
only to nodes that have at least one outgoing edge (TFs / intermediate
regulators), preserving the biological asymmetry between regulators and targets.

Shared helpers imported from src.algorithms.common.helpers:
    _build_transition_matrix, _get_top_nodes, _split_top_nodes

For the multi-process variant see:
    src.algorithms.cpu.multi_threaded.pagerank
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import (
    _build_transition_matrix,
    _get_top_nodes,
    _split_top_nodes,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "damping":   0.85,
    "max_iter":  100,
    "tolerance": 1e-6,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def pagerank_cpu_single(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """
    PageRank — single-threaded CPU power iteration via scipy SpMV.

    Propagates regulatory influence through the directed GRN.  Dangling mass
    (from pure target genes with out_degree == 0) is redistributed only to
    nodes that have at least one outgoing edge (TFs / intermediate regulators),
    preserving the biological asymmetry between regulators and targets.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
        Pre-processed adjacency matrix (CSR, directed).
    params    : dict
        damping   (float, default 0.85) — edge-follow probability
        max_iter  (int,   default 100)  — hard iteration cap
        tolerance (float, default 1e-6) — L1-norm convergence threshold

    Returns
    -------
    dict with keys:
        scores         : list[float]  — PageRank score per node
        iterations     : int
        converged      : bool
        top_regulators : list[int]    — top-15 nodes with out_degree > 0 (TFs)
        top_targets    : list[int]    — top-15 nodes with out_degree == 0
    """
    p        = _merge_params(params)
    d        = float(p["damping"])
    max_iter = int(p["max_iter"])
    tol      = float(p["tolerance"])

    N = graph_csr.shape[0]
    M, dangling_mask = _build_transition_matrix(graph_csr)
    teleport_per_node = (1.0 - d) / N

    out_degrees = np.asarray(graph_csr.sum(axis=1)).flatten()
    active_mask = out_degrees > 0
    n_active    = int(active_mask.sum())
    if n_active == 0:
        warnings.warn(
            "Warning: no outgoing-edge nodes found, using uniform dangling redistribution",
            UserWarning,
            stacklevel=2,
        )

    PR        = np.full(N, 1.0 / N, dtype=np.float64)
    converged = False

    for iteration in range(1, max_iter + 1):
        PR_old = PR

        dangling_mass = d * float(PR_old[dangling_mask].sum())
        dangling_contrib = np.zeros(N, dtype=np.float64)
        if n_active > 0:
            dangling_contrib[active_mask] = dangling_mass / n_active
        else:
            dangling_contrib[:] = dangling_mass / N

        PR = d * (M @ PR_old) + dangling_contrib + teleport_per_node

        if np.abs(PR - PR_old).sum() < tol:
            converged = True
            break

    top_reg, top_tgt = _split_top_nodes(PR, out_degrees)
    return {
        "scores":         PR.tolist(),
        "iterations":     iteration,
        "converged":      converged,
        "top_regulators": top_reg,
        "top_targets":    top_tgt,
    }


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
    return {"output": pagerank_cpu_single(graph_csr, p), "extra_params": p}
