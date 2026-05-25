"""
algorithms/rwr.py — Random Walk with Restart (RWR) for GRN Target Prioritisation
==================================================================================

Biological Context
------------------
Gene Regulatory Networks (GRNs) encode directed regulatory relationships between
Transcription Factors (TFs) and their downstream target genes.  RWR models the
propagation of *regulatory influence* from a curated set of seed TFs across the
entire network.

Seeding from disease-associated TFs (e.g. TP53 in cancer, NF-κB in inflammation)
allows the algorithm to rank every other gene by how strongly it is influenced by
those TFs — either directly (one hop) or indirectly (multi-hop chains).  This
produces a biologically motivated priority list of candidate downstream regulatory
targets, even in noisy or incomplete GRN data.

The *restart probability* r is the key parameter governing locality:
  * At each step the walker follows an out-edge with probability (1 − r).
  * With probability r it teleports back to a seed node (drawn uniformly).
  * A high r (e.g. 0.7) keeps scores tightly local to seeds and their direct
    neighbours.  A low r (e.g. 0.1) allows influence to diffuse far across the
    network, potentially surfacing distant but co-regulated gene modules.
  * r = 0.3 (default) is a standard compromise used in most GRN studies.

The steady-state score vector is the solution to:
    p* = (1 − r) · W · p* + r · p₀
where W is the column-stochastic transition matrix of the directed GRN and p₀
is the uniform distribution over the seed set.

Returns
-------
scores      : regulatory influence per node (sums to ≈ 1)
top_nodes   : the 20 most-influenced genes (non-seed, high-score nodes)
top_tfs     : top 10 seed TFs ordered by their own post-diffusion score
              (identifies which seeds are most "connected" to the rest)

Parameter Guide
---------------
restart_prob  (float, default 0.3)   Teleportation probability back to seeds.
max_iter      (int,   default 100)   Hard iteration cap.
tolerance     (float, default 1e-6)  L1-norm convergence threshold.
seed_nodes    (list[int] or
               list[list[int]])      Seed TF indices.  A flat list runs one RWR;
                                     a list-of-lists runs one RWR per seed set in
                                     parallel (cpu_multi) and averages the scores.
"""

# ── GPU / CUDA-optimised implementation ──────────────────────────────────
# Source:    biological_network_framework/algorithms/rwr.py
# Requires:  cupy-cuda11x (or matching CUDA version), pycuda
# Used by:   webapp routes, GPU benchmarking (src.benchmarking.benchmark)
# Modes:     _cpu_single, _cpu_multi, _gpu  (all three)
# CPU-only counterparts (benchmarking only — never import in webapp):
#   src.algorithms.cpu.single_threaded.rwr
#   src.algorithms.cpu.multi_threaded.rwr
# ──────────────────────────────────────────────────────────────────────────

import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Optional CuPy
# ---------------------------------------------------------------------------

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    cpsp = None
    _CUPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "restart_prob": 0.3,
    "max_iter": 100,
    "tolerance": 1e-6,
    "seed_nodes": [0],
}

_TOP_N: int = 20   # most-influenced genes to return
_TOP_TF: int = 10  # top seed TFs to return


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


def _build_transition_matrix(
    graph_csr: sp.csr_matrix,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Build the column-stochastic transition matrix W for a directed GRN.

    W[j, i] = A[i, j] / out_degree(i)

    Dangling TFs (out_degree == 0, i.e. TFs with no annotated target genes)
    are identified and their column in W is left as all-zero.  During the RWR
    iteration the walker restarts from seeds when it reaches a dangling node,
    which is handled by the restart term r·p₀ naturally.

    Returns
    -------
    W            : (N, N) CSR column-stochastic transition matrix
    dangling_mask: bool array, True for nodes with no out-edges
    """
    out_degrees = np.asarray(graph_csr.sum(axis=1)).flatten()
    dangling_mask = out_degrees == 0
    safe_degrees = np.where(dangling_mask, 1.0, out_degrees)
    D_inv = sp.diags(1.0 / safe_degrees, format="csr")
    W = (D_inv @ graph_csr).T.tocsr().astype(np.float64)
    return W, dangling_mask


def _make_p0(seed_nodes: list[int], N: int) -> np.ndarray:
    """Uniform distribution over seed nodes; zeros elsewhere."""
    p0 = np.zeros(N, dtype=np.float64)
    valid = [s for s in seed_nodes if 0 <= s < N]
    if not valid:
        # Fallback: uniform over all nodes
        return np.full(N, 1.0 / N, dtype=np.float64)
    p0[valid] = 1.0 / len(valid)
    return p0


def _top_tfs(scores: np.ndarray, seed_nodes: list[int], k: int = _TOP_TF) -> list[int]:
    """Return the top-k seed nodes ordered by their post-diffusion score."""
    seed_arr = np.array([s for s in seed_nodes if 0 <= s < len(scores)])
    if len(seed_arr) == 0:
        return []
    order = np.argsort(scores[seed_arr])[::-1]
    return seed_arr[order[:k]].tolist()


def _top_nodes(scores: np.ndarray, k: int = _TOP_N) -> list[int]:
    return np.argsort(scores)[::-1][:k].tolist()


# ---------------------------------------------------------------------------
# Module-level worker for ProcessPoolExecutor (must be picklable)
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
# Public implementations
# ---------------------------------------------------------------------------

def rwr_cpu_single(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """
    RWR — single-threaded CPU power iteration via scipy sparse SpMV.

    Iterates  p_new = (1−r)·W·p + r·p₀  until the L1 change drops below
    *tolerance* or *max_iter* is reached.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix — directed GRN adjacency (CSR).
    params    : dict — see module docstring.

    Returns
    -------
    dict with keys: scores, iterations, top_nodes, top_tfs
    """
    p = _merge_params(params)
    r        = float(p["restart_prob"])
    max_iter = int(p["max_iter"])
    tol      = float(p["tolerance"])
    seeds    = p["seed_nodes"]

    # Normalise seed_nodes: accept flat list or first set of a list-of-lists
    if seeds and isinstance(seeds[0], list):
        seeds = seeds[0]

    N = graph_csr.shape[0]
    W, _ = _build_transition_matrix(graph_csr)
    p0   = _make_p0(seeds, N)
    pr   = p0.copy()

    for iteration in range(1, max_iter + 1):
        pr_old = pr
        pr = (1.0 - r) * (W @ pr_old) + r * p0
        if np.abs(pr - pr_old).sum() < tol:
            break

    return {
        "scores":     pr.tolist(),
        "iterations": iteration,
        "top_nodes":  _top_nodes(pr),
        "top_tfs":    _top_tfs(pr, seeds),
    }


def rwr_cpu_multi(
    graph_csr: sp.csr_matrix,
    params: dict,
    n_workers: int = 4,
) -> dict:
    """
    RWR — multi-process CPU implementation for multiple seed sets.

    If ``params["seed_nodes"]`` is a flat list (single seed set), falls back
    to ``rwr_cpu_single`` — there is nothing to parallelise.

    If it is a list-of-lists, each inner list is treated as an independent
    seed set.  One RWR run is launched per seed set via
    ``ProcessPoolExecutor``; the per-node scores are averaged across all
    runs to produce a consensus regulatory influence vector.

    Averaging across seed sets is biologically useful when multiple disease-
    associated TF panels (e.g. TP53 + BRCA1 vs. MYC + EGFR) are compared:
    the mean score highlights genes that are convergently regulated.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
    params    : dict
    n_workers : int

    Returns
    -------
    Same structure as rwr_cpu_single.
    """
    p = _merge_params(params)
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

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_rwr_seed_set_worker, args_list))

    score_matrix = np.array([r[0] for r in results], dtype=np.float64)  # (K, N)
    avg_scores   = score_matrix.mean(axis=0)
    max_iter_out = max(r[1] for r in results)

    # Aggregate seed set for top_tfs: union of all seeds
    all_seeds: list[int] = []
    for seed_set in seeds:
        all_seeds.extend(seed_set)
    all_seeds = list(dict.fromkeys(all_seeds))   # deduplicate, preserve order

    return {
        "scores":     avg_scores.tolist(),
        "iterations": max_iter_out,
        "top_nodes":  _top_nodes(avg_scores),
        "top_tfs":    _top_tfs(avg_scores, all_seeds),
    }


def rwr_gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """
    RWR — GPU-accelerated implementation via CuPy sparse.

    Both the score vector ``p`` and the seed distribution ``p0`` live on the
    GPU throughout all iterations.  Only the final converged vector is
    transferred back to CPU.  The SpMV  W_gpu @ p_gpu  is dispatched to
    cuSPARSE via cupyx.scipy.sparse.

    Falls back to ``rwr_cpu_single`` with a warning if CuPy is unavailable.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
    params    : dict

    Returns
    -------
    Same structure as rwr_cpu_single.
    """
    try:
        from src.benchmarking.benchmark import _ensure_cuda_context
        if not _ensure_cuda_context():
            raise RuntimeError("no CUDA context")
    except Exception:
        pass

    if not _CUPY_AVAILABLE:
        warnings.warn(
            "CuPy unavailable — falling back to rwr_cpu_single.",
            RuntimeWarning,
            stacklevel=2,
        )
        return rwr_cpu_single(graph_csr, params)

    p = _merge_params(params)
    r        = float(p["restart_prob"])
    max_iter = int(p["max_iter"])
    tol      = float(p["tolerance"])
    seeds    = p["seed_nodes"]

    if seeds and isinstance(seeds[0], list):
        seeds = seeds[0]

    N = graph_csr.shape[0]
    W_cpu, _ = _build_transition_matrix(graph_csr)
    p0_cpu   = _make_p0(seeds, N)

    # Upload sparse transition matrix and seed vector to GPU
    W_coo = W_cpu.tocoo()
    W_gpu = cpsp.csr_matrix(
        (
            cp.asarray(W_coo.data,   dtype=cp.float64),
            (cp.asarray(W_coo.row),  cp.asarray(W_coo.col)),
        ),
        shape=W_coo.shape,
    )
    p0_gpu = cp.asarray(p0_cpu, dtype=cp.float64)
    p_gpu  = p0_gpu.copy()

    for iteration in range(1, max_iter + 1):
        p_old = p_gpu
        p_gpu = (1.0 - r) * (W_gpu @ p_old) + r * p0_gpu
        if float(cp.abs(p_gpu - p_old).sum()) < tol:
            break

    scores_cpu = cp.asnumpy(p_gpu)

    del p_gpu, p_old, p0_gpu, W_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return {
        "scores":     scores_cpu.tolist(),
        "iterations": iteration,
        "top_nodes":  _top_nodes(scores_cpu),
        "top_tfs":    _top_tfs(scores_cpu, seeds),
    }


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _cpu_single(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    p = _merge_params(params)
    return {"output": rwr_cpu_single(graph_csr, p), "extra_params": p}


def _cpu_multi(
    graph_csr: sp.csr_matrix,
    params: dict | None = None,
    n_workers: int = 4,
    **_,
) -> dict:
    p = _merge_params(params)
    return {
        "output":      rwr_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "n_workers": n_workers},
    }


def _gpu(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    p = _merge_params(params)
    return {"output": rwr_gpu(graph_csr, p), "extra_params": p}
