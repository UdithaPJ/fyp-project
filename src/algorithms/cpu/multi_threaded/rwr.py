"""
src/algorithms/cpu/multi_threaded/rwr.py
=========================================

Random Walk with Restart (RWR) — GraphBLAS-backed cpu_multi.

Power iteration with restart vector.  One SpMV per iteration via
SuiteSparse:GraphBLAS ``plus_times``.  Multiple seed sets (list-of-lists
input) are iterated sequentially and the per-node scores are averaged
across seed sets — matching the documented multi-seed-set behaviour of
the previous ProcessPoolExecutor implementation.

For the single-seed-set variant see
``src.algorithms.cpu.single_threaded.rwr``.
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
from src.algorithms.cpu.multi_threaded._graphblas_utils import (
    _configure_threads,
    _from_scipy,
    _require_graphblas,
    _vec_from_np,
    _vec_to_np,
    gb,
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


def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def rwr_cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict,
    n_workers: int | None = None,
) -> dict:
    """RWR using SuiteSparse:GraphBLAS for the SpMV step.

    When ``params["seed_nodes"]`` is a flat list this runs one RWR;
    when it is a list-of-lists, each inner list is treated as an
    independent seed set and the per-node scores are averaged.
    """
    _require_graphblas()
    n_threads = _configure_threads(n_workers)

    p        = _merge_params(params)
    r        = float(p["restart_prob"])
    max_iter = int(p["max_iter"])
    tol      = float(p["tolerance"])
    seeds    = p["seed_nodes"]

    n = int(graph_csr.shape[0])
    if n == 0:
        return {
            "scores": [], "iterations": 0,
            "top_nodes": [], "top_tfs": [],
            "note": "graphblas SuiteSparse (empty graph)",
        }

    # Normalize seed input into a list of seed sets.
    # Empty seeds → a single empty seed set, so ``_make_p0`` produces the
    # uniform 1/N restart vector (global, PageRank-like RWR) — matching
    # cpu_single / gpu / gpu_baseline.  Do NOT fall back to [0]; seeding at
    # node 0 makes cpu_multi disagree with every other implementation.
    if seeds and isinstance(seeds[0], (list, tuple)):
        seed_sets = [list(s) for s in seeds]
    else:
        seed_sets = [list(seeds)]

    # Build column-stochastic transition matrix W once.
    W_scipy, _ = _build_transition_matrix(graph_csr)
    W_gb = _from_scipy(W_scipy.astype(np.float32, copy=False), dtype=np.float32)

    one_minus_r = np.float32(1.0 - r)
    r_fp32      = np.float32(r)

    score_list: list[np.ndarray] = []
    max_iter_used = 0

    for sset in seed_sets:
        p0 = _make_p0(sset, n).astype(np.float32)
        pr = p0.copy()
        iteration = 0
        for iteration in range(1, max_iter + 1):
            pr_old = pr
            pr_gb = _vec_from_np(pr_old, dtype=gb.dtypes.FP32)
            result_gb = W_gb.mxv(pr_gb, gb.semiring.plus_times).new()
            result = _vec_to_np(result_gb, n, dtype=np.float32, fill=0.0)
            pr = one_minus_r * result + r_fp32 * p0
            if float(np.abs(pr - pr_old).sum()) < tol:
                break
        score_list.append(pr)
        max_iter_used = max(max_iter_used, iteration)

    avg_scores = np.mean(np.stack(score_list, axis=0), axis=0).astype(np.float32)
    all_seeds: list[int] = []
    for sset in seed_sets:
        all_seeds.extend(int(s) for s in sset)
    all_seeds = list(dict.fromkeys(all_seeds))

    return {
        "scores":     avg_scores.tolist(),
        "iterations": max_iter_used,
        "top_nodes":  _top_nodes(avg_scores),
        "top_tfs":    _top_tfs(avg_scores, all_seeds),
        "note":       f"graphblas SuiteSparse nthreads={n_threads}"
                      f" seed_sets={len(seed_sets)}",
    }


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict | None = None,
    n_workers: int | None = None,
    **_,
) -> dict:
    """Benchmark runner entry-point for cpu_multi mode (now GraphBLAS-backed)."""
    p = _merge_params(params)
    return {
        "output":       rwr_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "backend": "graphblas"},
    }
