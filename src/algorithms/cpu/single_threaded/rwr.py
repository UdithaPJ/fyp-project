"""
src/algorithms/cpu/single_threaded/rwr.py
==========================================

Random Walk with Restart (RWR) — single-threaded CPU implementation only.

Iterates  p_new = (1-r)·W·p + r·p₀  until the L1 change drops below
tolerance or max_iter is reached.  A single flat seed set is supported;
for parallel multi-seed-set execution see the multi-threaded variant.

If ``params["seed_nodes"]`` is a list-of-lists, only the first inner list
is used as the seed set.

Shared helpers imported from src.algorithms.common.helpers:
    _build_transition_matrix, _make_p0, _top_tfs, _top_nodes

For the multi-seed-set parallel variant see:
    src.algorithms.cpu.multi_threaded.rwr
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import (
    _build_transition_matrix,
    _make_p0,
    _top_nodes,
    _top_tfs,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "restart_prob": 0.3,
    "max_iter":     100,
    "tolerance":    1e-6,
    "seed_nodes":   [0],
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def rwr_cpu_single(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """
    RWR — single-threaded CPU power iteration via scipy sparse SpMV.

    Models the propagation of regulatory influence from a curated set of
    seed TFs across the entire GRN.  The steady-state score vector ranks
    every gene by how strongly it is influenced (directly or indirectly)
    by the seed TFs.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix — directed GRN adjacency (CSR).
    params    : dict
        restart_prob (float, default 0.3)  — teleportation probability back to seeds
        max_iter     (int,   default 100)  — hard iteration cap
        tolerance    (float, default 1e-6) — L1-norm convergence threshold
        seed_nodes   (list[int] or list[list[int]]) — seed TF indices; if
                      list-of-lists, only the first inner list is used

    Returns
    -------
    dict with keys: scores, iterations, top_nodes, top_tfs
    """
    p        = _merge_params(params)
    r        = float(p["restart_prob"])
    max_iter = int(p["max_iter"])
    tol      = float(p["tolerance"])
    seeds    = p["seed_nodes"]

    # Normalise: accept flat list or first set of a list-of-lists
    if seeds and isinstance(seeds[0], list):
        seeds = seeds[0]

    N = graph_csr.shape[0]
    # dangling_self_loops=True keeps the walk mass-conserving on directed
    # graphs (GRN/miRNA dangling target genes); matches the GPU RWR spec.
    W, _ = _build_transition_matrix(graph_csr, dangling_self_loops=True)
    p0   = _make_p0(seeds, N)
    pr   = p0.copy()

    # MEMORY_FIX (M-3): keep scalars in FP32 to match the FP32 pr / W.
    one_minus_r = np.float32(1.0 - r)
    r_fp32      = np.float32(r)
    for iteration in range(1, max_iter + 1):
        pr_old = pr
        pr = one_minus_r * (W @ pr_old) + r_fp32 * p0
        if np.abs(pr - pr_old).sum() < tol:
            break

    return {
        "scores":     pr.tolist(),
        "iterations": iteration,
        "top_nodes":  _top_nodes(pr),
        "top_tfs":    _top_tfs(pr, seeds),
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
    return {"output": rwr_cpu_single(graph_csr, p), "extra_params": p}
