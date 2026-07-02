"""
src/algorithms/cpu/single_threaded/hits.py
==========================================

HITS (Hyperlink-Induced Topic Search) — single-threaded CPU implementation only.

Uses the directed adjacency matrix directly (no symmetrisation).
Two scipy SpMV operations per iteration:
    authority update : a ← A^T · h   then L2-normalise
    hub      update  : h ← A   · a   then L2-normalise

Shared helpers imported from src.algorithms.common.helpers:
    _l2_normalize, _top_k, hits_pack_result  (imported as _pack_result)

For the parallel SpMV variant see:
    src.algorithms.cpu.multi_threaded.hits
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import (
    _l2_normalize,
    _top_k,
    hits_pack_result as _pack_result,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
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

def hits_cpu_single(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """
    HITS — single-threaded CPU using scipy sparse SpMV on the directed graph.

    Uses the directed adjacency matrix directly (no symmetrisation).
    High hub score → master-regulator TF; high authority score → key
    effector / convergence-point gene.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix — directed GRN adjacency.
    params    : dict
        max_iter  (int,   default 100)   — hard iteration cap
        tolerance (float, default 1e-6)  — combined L2-norm convergence threshold

    Returns
    -------
    dict with keys: hub_scores, authority_scores, iterations, converged,
                    top_hubs, top_authorities, hub_authority_overlap
    """
    p        = _merge_params(params)
    max_iter = int(p["max_iter"])
    tol      = float(p["tolerance"])

    N   = graph_csr.shape[0]
    # Scale-invariant (per-node RMS) convergence threshold — the L2-norm score
    # change grows like sqrt(N) for a fixed per-node change, so scaling the
    # threshold by sqrt(N) keeps `tolerance` meaning the same thing regardless
    # of graph size.  Matches the GPU HITS and cuGraph's n-scaled definition.
    conv_tol = tol * float(np.sqrt(N))
    # MEMORY_FIX (M-8): FP32 throughout.  HITS uses L2-norm convergence,
    # which is well-behaved at FP32 for biological networks.  Saves
    # ~240 MB on 15 M-edge graphs (A + A^T).
    A   = graph_csr.astype(np.float32)
    A_T = A.T.tocsr()

    h         = np.ones(N, dtype=np.float32)
    a         = np.ones(N, dtype=np.float32)
    converged = False

    for iteration in range(1, max_iter + 1):
        h_old, a_old = h, a

        a_new = _l2_normalize(A_T @ h_old)
        h_new = _l2_normalize(A   @ a_new)

        if np.linalg.norm(h_new - h_old) + np.linalg.norm(a_new - a_old) < conv_tol:
            converged = True
            a, h = a_new, h_new
            break

        a, h = a_new, h_new

    return _pack_result(h, a, iteration, converged)


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
    return {"output": hits_cpu_single(graph_csr, p), "extra_params": p}
