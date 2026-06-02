"""
src/algorithms/cpu/multi_threaded/rwr.py
=========================================

Random Walk with Restart (RWR) — multi-process CPU implementation only.

If ``params["seed_nodes"]`` is a flat list (single seed set), falls back
to ``rwr_cpu_single`` — there is nothing to parallelise.

If it is a list-of-lists, each inner list is treated as an independent
seed set.  One RWR run is launched per seed set via ProcessPoolExecutor;
the per-node scores are averaged across all runs to produce a consensus
regulatory influence vector.

Averaging across seed sets is biologically useful when multiple disease-
associated TF panels (e.g. TP53 + BRCA1 vs. MYC + EGFR) are compared:
the mean score highlights genes that are convergently regulated.

Shared helpers imported from src.algorithms.common.helpers:
    _build_transition_matrix, _top_tfs, _top_nodes

Cross-file import:
    rwr_cpu_single — fallback when seed_nodes is a flat list

Exclusive to this file:
    _rwr_seed_set_worker — per-seed-set RWR subprocess worker

For the single-seed-set variant see:
    src.algorithms.cpu.single_threaded.rwr
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import (
    _build_transition_matrix,
    _top_nodes,
    _top_tfs,
    _worker_init_no_blas,
)
from src.algorithms.cpu.single_threaded.rwr import rwr_cpu_single

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
# Module-level worker — must be at module scope for ProcessPoolExecutor pickle
# ---------------------------------------------------------------------------

def _rwr_seed_set_worker(args: tuple) -> tuple[list[float], int]:
    """
    Run a single RWR from one seed set.

    Receives the full transition matrix as raw CSR arrays (avoids pickling a
    sparse object) plus the seed list and hyperparameters.  Returns the final
    score vector as a list and the iteration count.
    """
    W_data, W_indices, W_indptr, W_shape, seed_nodes, restart_prob, max_iter, tol = args
    W = sp.csr_matrix((W_data, W_indices, W_indptr), shape=W_shape)
    N = W_shape[0]

    p0 = np.zeros(N, dtype=np.float64)
    valid = [s for s in seed_nodes if 0 <= s < N]
    if valid:
        p0[valid] = 1.0 / len(valid)
    else:
        p0[:] = 1.0 / N

    r = float(restart_prob)
    p = p0.copy()

    for iteration in range(1, max_iter + 1):
        p_old = p
        p = (1.0 - r) * (W @ p_old) + r * p0
        if np.abs(p - p_old).sum() < tol:
            break

    return p.tolist(), iteration


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def rwr_cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict,
    n_workers: int = 4,
) -> dict:
    """
    RWR — multi-process CPU implementation for multiple seed sets.

    When ``params["seed_nodes"]`` is a flat list (single seed set), this
    function delegates to ``rwr_cpu_single`` — there is nothing to
    parallelise.

    When it is a list-of-lists, each inner list is an independent seed set.
    One RWR run is launched per seed set via ProcessPoolExecutor; the
    per-node scores are then averaged to produce a consensus influence vector.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
    params    : dict
        restart_prob (float, default 0.3)
        max_iter     (int,   default 100)
        tolerance    (float, default 1e-6)
        seed_nodes   (list[list[int]]) — list of seed-TF index lists
    n_workers : int

    Returns
    -------
    dict with keys: scores, iterations, top_nodes, top_tfs
    """
    p     = _merge_params(params)
    seeds = p["seed_nodes"]

    # Single seed set — no parallelism available
    if not seeds or not isinstance(seeds[0], list):
        return rwr_cpu_single(graph_csr, params)

    r        = float(p["restart_prob"])
    max_iter = int(p["max_iter"])
    tol      = float(p["tolerance"])

    N = graph_csr.shape[0]
    W, _ = _build_transition_matrix(graph_csr)

    args_list = [
        (W.data, W.indices, W.indptr, W.shape, seed_set, r, max_iter, tol)
        for seed_set in seeds
    ]

    # MEMORY_FIX (H-3): worker BLAS pinned to 1 thread to avoid oversubscription.
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_worker_init_no_blas
    ) as executor:
        results = list(executor.map(_rwr_seed_set_worker, args_list))

    score_matrix = np.array([res[0] for res in results], dtype=np.float64)
    avg_scores   = score_matrix.mean(axis=0)
    max_iter_out = max(res[1] for res in results)

    all_seeds: list[int] = []
    for seed_set in seeds:
        all_seeds.extend(seed_set)
    all_seeds = list(dict.fromkeys(all_seeds))

    return {
        "scores":     avg_scores.tolist(),
        "iterations": max_iter_out,
        "top_nodes":  _top_nodes(avg_scores),
        "top_tfs":    _top_tfs(avg_scores, all_seeds),
    }


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
        "output":       rwr_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "n_workers": n_workers},
    }
