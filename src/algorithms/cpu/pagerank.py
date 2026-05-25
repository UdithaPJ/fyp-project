"""
algorithms/pagerank.py — PageRank for GRN Hub Gene / TF Identification
=======================================================================

Biological Context
------------------
In a Gene Regulatory Network (GRN), PageRank models *regulatory influence* as
a flow propagated forward along directed TF → gene edges.  At each step, a
fraction d of the influence follows an outgoing regulatory edge and fraction
(1 − d) teleports uniformly — reflecting the biological reality that
regulatory cascades can be interrupted or reset at any point.

Damping factor d = 0.85 means each regulatory hop has an 85% chance of
continuing to propagate and a 15% chance of dissipating.  This limits
effective influence depth in a biologically meaningful way (empirically
matching average GRN cascade lengths of 4–6 hops).

Two distinct node classes are ranked and returned separately:

  top_regulators — nodes with out_degree > 0 (TFs, intermediate regulators)
      A TF with high PageRank AND high out-degree is a *master regulator*:
      it receives influence from other important TFs AND drives many targets.
      These are primary candidates for therapeutic intervention or knock-out
      validation experiments.

  top_targets — nodes with out_degree == 0 (pure target genes)
      A target with high PageRank is a *convergence point* of regulatory
      input — many influential TFs funnel their influence to it.  These are
      key effector genes and likely members of core biological pathways.

Dangling node handling (GRN-specific)
--------------------------------------
Dangling nodes (out_degree == 0) are pure target genes with no annotated
outgoing regulatory edges.  Standard PageRank redistributes their probability
mass uniformly across ALL nodes, which artificially inflates scores of
unrelated TFs.  This implementation redirects dangling mass exclusively to
nodes with at least one outgoing edge (i.e. actual regulators), preserving
the biological asymmetry between TFs and targets.

If no regulator nodes exist in the graph, a UserWarning is emitted and the
fallback uniform redistribution is used.

Algorithm (power iteration)
----------------------------
1. Build a column-stochastic transition matrix M from the directed adjacency.
2. Identify dangling nodes (out_degree == 0) and active nodes (out_degree > 0).
3. Iterate until convergence (or max_iter):
      PR_new = d·M·PR + d·(Σ PR[dangling] / N_active)·1_{active} + (1−d)/N
   where N_active = number of nodes with out_degree > 0.
4. Return separate ranked lists for regulators and target genes.

Parameter guide
---------------
damping   (float, default 0.85)  Probability of following an edge vs. teleporting.
                                  0.85 reflects typical regulatory cascade depth.
max_iter  (int,   default 100)   Hard iteration cap.
tolerance (float, default 1e-6)  L1-norm convergence threshold.
"""

# ── CPU-only implementation ───────────────────────────────────────────────
# Source: biological_network_framework/algorithms/pagerank.py
# GPU functions removed; cupy/pycuda imports not required.
# ──────────────────────────────────────────────────────────────────────────

import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "damping": 0.85,
    "max_iter": 100,
    "tolerance": 1e-6,
}

_TOP_REG: int = 15         # top regulators (TFs / out-degree > 0) to return
_TOP_TGT: int = 15         # top targets   (out-degree == 0) to return
_TOP_N:   int = 20         # kept for internal helpers; not exposed in return dict
_HIGH_DEG_PERCENTILE = 90  # top-N% out-degree nodes treated as "high-degree" on GPU


# ---------------------------------------------------------------------------
# Shared CPU helpers
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


def _build_transition_matrix(
    graph_csr: sp.csr_matrix,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Build the column-stochastic transition matrix M and a dangling-node mask.

    M[j, i] = A[i, j] / out_degree(i)
    Dangling nodes (out_degree == 0) have no outgoing edges; their rows in
    graph_csr are all-zero, so they contribute nothing to M.  They are tracked
    separately so the caller can redistribute their probability mass uniformly.

    Returns
    -------
    M            : (N, N) CSR, column-stochastic over non-dangling nodes
    dangling_mask: boolean array of length N, True for dangling nodes
    """
    out_degrees = np.asarray(graph_csr.sum(axis=1)).flatten()
    dangling_mask = out_degrees == 0
    safe_degrees = np.where(dangling_mask, 1.0, out_degrees)
    D_inv = sp.diags(1.0 / safe_degrees, format="csr")
    M = (D_inv @ graph_csr).T.tocsr().astype(np.float64)
    return M, dangling_mask


def _get_top_nodes(scores: np.ndarray, n: int = _TOP_N) -> list[int]:
    """Return indices of the top-n highest-scoring nodes (internal helper)."""
    k = min(n, len(scores))
    return np.argsort(scores)[::-1][:k].tolist()


def _split_top_nodes(
    scores: np.ndarray,
    out_degrees: np.ndarray,
    n_reg: int = _TOP_REG,
    n_tgt: int = _TOP_TGT,
) -> tuple[list[int], list[int]]:
    """
    Partition top-scoring nodes into two biologically distinct lists.

    Parameters
    ----------
    scores      : PageRank score per node.
    out_degrees : Out-degree per node (from the original directed graph).
    n_reg       : Number of top regulators to return.
    n_tgt       : Number of top targets to return.

    Returns
    -------
    top_regulators : top-n_reg node indices with out_degree > 0
                     (TFs and intermediate regulators / master regulators)
    top_targets    : top-n_tgt node indices with out_degree == 0
                     (pure target genes / convergence points)
    """
    reg_idx = np.where(out_degrees > 0)[0]
    tgt_idx = np.where(out_degrees == 0)[0]

    if len(reg_idx) > 0:
        top_reg = reg_idx[np.argsort(scores[reg_idx])[::-1][:n_reg]].tolist()
    else:
        top_reg = []

    if len(tgt_idx) > 0:
        top_tgt = tgt_idx[np.argsort(scores[tgt_idx])[::-1][:n_tgt]].tolist()
    else:
        top_tgt = []

    return top_reg, top_tgt


# ---------------------------------------------------------------------------
# Module-level worker — must be at module scope for ProcessPoolExecutor pickle
# ---------------------------------------------------------------------------

def _spmv_row_chunk(args: tuple) -> np.ndarray:
    """
    ProcessPoolExecutor worker: compute CSR SpMV for a contiguous row range.

    Receives (data, indices, indptr, pr, n_cols) where indptr is re-zeroed
    to the chunk's local start.  Returns the partial result vector of length
    equal to the number of rows in this chunk.
    """
    data, indices, indptr, pr, n_cols = args
    n_rows_chunk = len(indptr) - 1
    M_chunk = sp.csr_matrix(
        (data, indices, indptr),
        shape=(n_rows_chunk, n_cols),
    )
    return (M_chunk @ pr).astype(np.float64)


# ---------------------------------------------------------------------------
# Public implementations
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
    params    : dict — see module docstring for key descriptions.

    Returns
    -------
    dict with keys:
        scores         : list[float]  — PageRank score per node
        iterations     : int
        converged      : bool
        top_regulators : list[int]    — top-15 nodes with out_degree > 0 (TFs)
        top_targets    : list[int]    — top-15 nodes with out_degree == 0
    """
    p = _merge_params(params)
    d        = float(p["damping"])
    max_iter = int(p["max_iter"])
    tol      = float(p["tolerance"])

    N = graph_csr.shape[0]
    M, dangling_mask = _build_transition_matrix(graph_csr)
    teleport_per_node = (1.0 - d) / N

    # Nodes with out-degree > 0 are TFs / regulators.
    # Dangling mass is sent ONLY to these nodes, not to pure target genes.
    out_degrees = np.asarray(graph_csr.sum(axis=1)).flatten()
    active_mask = out_degrees > 0
    n_active    = int(active_mask.sum())
    if n_active == 0:
        warnings.warn(
            "Warning: no outgoing-edge nodes found, using uniform dangling redistribution",
            UserWarning,
            stacklevel=2,
        )

    PR = np.full(N, 1.0 / N, dtype=np.float64)
    converged = False

    for iteration in range(1, max_iter + 1):
        PR_old = PR

        # Dangling mass: collected from pure-target nodes, returned only to TFs
        dangling_mass = d * float(PR_old[dangling_mask].sum())
        dangling_contrib = np.zeros(N, dtype=np.float64)
        if n_active > 0:
            dangling_contrib[active_mask] = dangling_mass / n_active
        else:
            dangling_contrib[:] = dangling_mass / N   # fallback: uniform

        # Edge contribution via SpMV
        PR = d * (M @ PR_old) + dangling_contrib + teleport_per_node

        # L1 convergence check
        if np.abs(PR - PR_old).sum() < tol:
            converged = True
            break

    top_reg, top_tgt = _split_top_nodes(PR, out_degrees)
    return {
        "scores":          PR.tolist(),
        "iterations":      iteration,
        "converged":       converged,
        "top_regulators":  top_reg,
        "top_targets":     top_tgt,
    }


def pagerank_cpu_multi(
    graph_csr: sp.csr_matrix,
    params: dict,
    n_workers: int = 4,
) -> dict:
    """
    PageRank — multi-process CPU implementation.

    The SpMV step (M @ PR) — the bottleneck in each iteration — is split into
    row-range chunks and dispatched to a ProcessPoolExecutor.  Each worker
    reconstructs a CSR sub-matrix for its rows and performs a local SpMV.
    The partial results are concatenated to form the full output vector.

    Dangling redistribution and convergence checks remain serial (trivial cost).

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
    params    : dict
    n_workers : int — number of parallel worker processes

    Returns
    -------
    Same structure as pagerank_cpu_single.
    """
    p = _merge_params(params)
    d        = float(p["damping"])
    max_iter = int(p["max_iter"])
    tol      = float(p["tolerance"])

    N = graph_csr.shape[0]
    M, dangling_mask = _build_transition_matrix(graph_csr)
    teleport_per_node = (1.0 - d) / N

    # Nodes with out-degree > 0 are TFs / regulators.
    # Dangling mass is sent ONLY to these nodes, not to pure target genes.
    out_degrees = np.asarray(graph_csr.sum(axis=1)).flatten()
    active_mask = out_degrees > 0
    n_active    = int(active_mask.sum())
    if n_active == 0:
        warnings.warn(
            "Warning: no outgoing-edge nodes found, using uniform dangling redistribution",
            UserWarning,
            stacklevel=2,
        )

    # Pre-build chunk argument templates (data/indices/indptr slices of M)
    chunk_size = max(1, (N + n_workers - 1) // n_workers)
    chunk_specs: list[tuple] = []
    row_ranges: list[tuple[int, int]] = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        ptr_s = int(M.indptr[start])
        ptr_e = int(M.indptr[end])
        local_indptr = (M.indptr[start:end + 1] - M.indptr[start]).copy()
        chunk_specs.append((
            M.data[ptr_s:ptr_e].copy(),
            M.indices[ptr_s:ptr_e].copy(),
            local_indptr,
            None,   # placeholder — PR is filled per-iteration below
            N,
        ))
        row_ranges.append((start, end))

    PR = np.full(N, 1.0 / N, dtype=np.float64)
    converged = False

    for iteration in range(1, max_iter + 1):
        PR_old = PR

        # Dangling mass: collected from pure-target nodes, returned only to TFs
        dangling_mass = d * float(PR_old[dangling_mask].sum())
        dangling_contrib = np.zeros(N, dtype=np.float64)
        if n_active > 0:
            dangling_contrib[active_mask] = dangling_mass / n_active
        else:
            dangling_contrib[:] = dangling_mass / N   # fallback: uniform

        # Inject current PR into each chunk spec
        args_list = [
            (spec[0], spec[1], spec[2], PR_old, spec[4])
            for spec in chunk_specs
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            partial_results = list(executor.map(_spmv_row_chunk, args_list))

        # Reassemble full SpMV result
        spmv_result = np.concatenate(partial_results)
        PR = d * spmv_result + dangling_contrib + teleport_per_node

        if np.abs(PR - PR_old).sum() < tol:
            converged = True
            break

    top_reg, top_tgt = _split_top_nodes(PR, out_degrees)
    return {
        "scores":          PR.tolist(),
        "iterations":      iteration,
        "converged":       converged,
        "top_regulators":  top_reg,
        "top_targets":     top_tgt,
    }


# ---------------------------------------------------------------------------
# Runner interface — called by benchmark/runner.py as _cpu_single, etc.
# ---------------------------------------------------------------------------

def _cpu_single(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    """Benchmark runner entry-point for cpu_single mode."""
    p = _merge_params(params)
    return {"output": pagerank_cpu_single(graph_csr, p), "extra_params": p}


def _cpu_multi(
    graph_csr: sp.csr_matrix,
    params: dict | None = None,
    n_workers: int = 4,
    **_,
) -> dict:
    """Benchmark runner entry-point for cpu_multi mode."""
    p = _merge_params(params)
    return {
        "output":      pagerank_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "n_workers": n_workers},
    }
