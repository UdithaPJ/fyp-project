"""
src/algorithms/common/helpers.py — Shared helpers for CPU algorithm files
==========================================================================

Contains:
- Functions shared between *both* the single-threaded and multi-threaded
  variants of the same algorithm (used by both single_threaded/<name>.py
  and multi_threaded/<name>.py).
- The ``_build_transition_matrix`` function that is **cross-algorithm**
  (identical implementation used by both pagerank and rwr).

Naming conventions
------------------
Where two algorithms define a function with the same name but different
signatures, the helpers here are prefixed with the algorithm name:
    bfs_pack_result  — packs a BFS distance/cascade result dict
    hits_pack_result — packs a HITS hub/authority result dict

Importers alias them back to the local name they expect, e.g.:
    from src.algorithms.common.helpers import bfs_pack_result as _pack_result
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph


# ===========================================================================
# Worker-process initializer (used by every cpu_multi algorithm)
# ===========================================================================

def _worker_init_no_blas() -> None:
    """Multiprocessing pool initializer — pin worker BLAS threads to 1.

    MEMORY_FIX (H-3): without this, every worker process inherits the
    parent's OMP_NUM_THREADS=physical_cores and each Pool with 4 workers
    spawns 4 × physical_cores BLAS threads on the same CPU.  The
    resulting oversubscription collapses throughput on biological graphs.

    The env-var changes must happen BEFORE numpy/scipy import inside the
    worker — `import` order is preserved by `spawn` start method as long
    as we set the variables before any heavy import the worker may use.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = "1"
    # Some BLAS libs expose runtime knobs via threadpoolctl; set them too
    # when the module is already loaded.
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:                                              # noqa: BLE001
        pass

# ---------------------------------------------------------------------------
# Module-level constants (used as default parameters in helper signatures)
# ---------------------------------------------------------------------------

_TOP_N:   int = 20   # top nodes (pagerank, rwr)
_TOP_REG: int = 15   # top regulators (pagerank)
_TOP_TGT: int = 15   # top target genes (pagerank)
_TOP_TF:  int = 10   # top seed TFs (rwr)
_TOP_K:   int = 15   # top hubs / authorities (hits)


# ===========================================================================
# Shared across pagerank AND rwr
# ===========================================================================

def _build_transition_matrix(
    graph_csr: sp.csr_matrix,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Build the column-stochastic transition matrix M and a dangling-node mask.

    M[j, i] = A[i, j] / out_degree(i)

    Dangling nodes (out_degree == 0) have no outgoing edges; their rows in
    graph_csr are all-zero.  They are tracked separately so the caller can
    redistribute their probability mass (PageRank) or treat the restart term
    as the sole source of probability (RWR).

    Used by
    -------
    src.algorithms.cpu.single_threaded.pagerank
    src.algorithms.cpu.multi_threaded.pagerank
    src.algorithms.cpu.single_threaded.rwr
    src.algorithms.cpu.multi_threaded.rwr

    Returns
    -------
    M            : (N, N) CSR, column-stochastic over non-dangling nodes
    dangling_mask: boolean array of length N, True for dangling nodes
    """
    # MEMORY_FIX (M-3): build the transition matrix in float32.  Score
    # vectors converge under FP32 tolerance ≥ 1e-7; FP64 halves SIMD
    # throughput and doubles RAM for no biological-precision benefit.
    out_degrees   = np.asarray(graph_csr.sum(axis=1)).flatten().astype(np.float32)
    dangling_mask = out_degrees == 0
    safe_degrees  = np.where(dangling_mask, np.float32(1.0), out_degrees)
    D_inv         = sp.diags(1.0 / safe_degrees, format="csr", dtype=np.float32)
    M             = (D_inv @ graph_csr).T.tocsr().astype(np.float32)
    return M, dangling_mask


# ===========================================================================
# PageRank helpers
# ===========================================================================

def _get_top_nodes(scores: np.ndarray, n: int = _TOP_N) -> list[int]:
    """Return indices of the top-n highest-scoring nodes."""
    k = min(n, len(scores))
    return np.argsort(scores)[::-1][:k].tolist()


def _split_top_nodes(
    scores:      np.ndarray,
    out_degrees: np.ndarray,
    n_reg: int = _TOP_REG,
    n_tgt: int = _TOP_TGT,
) -> tuple[list[int], list[int]]:
    """
    Partition top-scoring nodes into regulators (out_degree > 0) and
    target genes (out_degree == 0).

    Returns
    -------
    top_regulators : top-n_reg node indices with out_degree > 0
    top_targets    : top-n_tgt node indices with out_degree == 0
    """
    reg_idx = np.where(out_degrees > 0)[0]
    tgt_idx = np.where(out_degrees == 0)[0]

    top_reg = (
        reg_idx[np.argsort(scores[reg_idx])[::-1][:n_reg]].tolist()
        if len(reg_idx) > 0 else []
    )
    top_tgt = (
        tgt_idx[np.argsort(scores[tgt_idx])[::-1][:n_tgt]].tolist()
        if len(tgt_idx) > 0 else []
    )
    return top_reg, top_tgt


# ===========================================================================
# BFS helpers
# ===========================================================================

def bfs_pack_result(
    distances:        np.ndarray,
    visited_order:    list[int],
    cascade_by_depth: dict[int, list[int]],
) -> dict:
    """Pack BFS output arrays into the standard result dict."""
    return {
        "distances":        distances.tolist(),
        "visited_order":    visited_order,
        "num_reachable":    int((distances >= 0).sum()),
        "cascade_by_depth": {str(k): v for k, v in cascade_by_depth.items()},
    }


# ===========================================================================
# HITS helpers
# ===========================================================================

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalise a vector; return unchanged if norm is zero."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0.0 else v


def _top_k(scores: np.ndarray, k: int = _TOP_K) -> list[int]:
    """Return indices of the top-k highest-scoring entries."""
    return np.argsort(scores)[::-1][:k].tolist()


def hits_pack_result(
    hub:       np.ndarray,
    auth:      np.ndarray,
    iteration: int,
    converged: bool,
) -> dict:
    """Pack HITS hub/authority arrays into the standard result dict."""
    top_h   = _top_k(hub)
    top_a   = _top_k(auth)
    overlap = sorted(set(top_h) & set(top_a))
    return {
        "hub_scores":            hub.tolist(),
        "authority_scores":      auth.tolist(),
        "iterations":            iteration,
        "converged":             converged,
        "top_hubs":              top_h,
        "top_authorities":       top_a,
        "hub_authority_overlap": overlap,
    }


# ===========================================================================
# RWR helpers
# ===========================================================================

def _make_p0(seed_nodes: list[int], N: int) -> np.ndarray:
    """Uniform distribution over seed nodes; falls back to 1/N if none valid."""
    # MEMORY_FIX (M-3): FP32 — matches the FP32 transition matrix.
    p0    = np.zeros(N, dtype=np.float32)
    valid = [s for s in seed_nodes if 0 <= s < N]
    if not valid:
        return np.full(N, np.float32(1.0 / N), dtype=np.float32)
    p0[valid] = np.float32(1.0 / len(valid))
    return p0


def _top_tfs(
    scores:     np.ndarray,
    seed_nodes: list[int],
    k:          int = _TOP_TF,
) -> list[int]:
    """Return the top-k seed nodes ordered by their post-diffusion score."""
    seed_arr = np.array([s for s in seed_nodes if 0 <= s < len(scores)])
    if len(seed_arr) == 0:
        return []
    order = np.argsort(scores[seed_arr])[::-1]
    return seed_arr[order[:k]].tolist()


def _top_nodes(scores: np.ndarray, k: int = _TOP_N) -> list[int]:
    """Return the top-k highest-scoring node indices."""
    return np.argsort(scores)[::-1][:k].tolist()


# ===========================================================================
# Louvain helpers
# ===========================================================================

def _symmetrize(graph_csr: sp.csr_matrix) -> sp.csr_matrix:
    """
    Symmetrize a directed GRN adjacency: A_sym = A + A^T.

    Mutual TF↔gene edges get weight 2× (tighter co-regulation);
    one-way edges keep their original weight.
    """
    A_sym = (graph_csr + graph_csr.T).tocsr()
    A_sym.sum_duplicates()
    # MEMORY_FIX (M-6): float32 keeps the symmetric view cheap; modularity
    # under FP32 differs from FP64 only in the 7th significant digit.
    return A_sym.astype(np.float32)


def _compute_modularity(
    adj_sym:    sp.csr_matrix,
    labels:     np.ndarray,
    m:          float,
    resolution: float,
) -> float:
    """Compute modularity Q for a labelled partition in O(nnz)."""
    degrees = np.asarray(adj_sym.sum(axis=1)).flatten()
    coo     = adj_sym.tocoo()
    same    = labels[coo.row] == labels[coo.col]
    expected = degrees[coo.row] * degrees[coo.col] / (2.0 * m)
    Q = float(np.sum((coo.data - resolution * expected) * same) / (2.0 * m))
    return Q


def _phase2_collapse(
    adj_csr:     sp.csr_matrix,
    communities: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Collapse a community assignment into a new weighted super-node graph.

    Returns
    -------
    new_adj   : (K × K) CSR — weighted super-node adjacency (with self-loops)
    new_labels: (n,) array  — old node index → new super-node index
    """
    _, new_labels = np.unique(communities, return_inverse=True)
    K   = int(new_labels.max()) + 1
    coo = adj_csr.tocoo()
    row_new = new_labels[coo.row]
    col_new = new_labels[coo.col]
    new_adj = sp.coo_matrix(
        (coo.data.astype(np.float64), (row_new, col_new)),
        shape=(K, K),
    ).tocsr()
    new_adj.sum_duplicates()
    return new_adj, new_labels.astype(np.int32)


def _build_result(
    A_sym:        sp.csr_matrix,
    final_labels: np.ndarray,
    hierarchy:    list[list[int]],
    m:            float,
    resolution:   float,
) -> dict:
    """Build the standard Louvain result dict from final community labels."""
    K    = int(final_labels.max()) + 1
    Q    = _compute_modularity(A_sym, final_labels, m, resolution)
    sizes = np.bincount(final_labels, minlength=K)
    top5  = np.argsort(sizes)[::-1][:5]
    top_communities = [
        {
            "community_id": int(c),
            "size":         int(sizes[c]),
            "member_nodes": np.where(final_labels == c)[0].tolist(),
        }
        for c in top5 if sizes[c] > 0
    ]
    return {
        "community_assignments": final_labels.tolist(),
        "num_communities":       K,
        "modularity":            Q,
        "hierarchy":             hierarchy,
        "top_communities":       top_communities,
    }


def _run_louvain(
    A_sym:             sp.csr_matrix,
    m:                 float,
    resolution:        float,
    min_delta_q:       float,
    max_levels:        int,
    phase1_fn:         Callable,
    max_phase1_passes: int = 100,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Execute the two-phase Louvain loop using a pluggable Phase 1 function.

    ``phase1_fn`` must accept
    ``(adj_csr, communities, degrees, m, resolution, min_delta_q)`` and
    return ``(communities, improved: bool)``.

    ``max_phase1_passes`` caps the inner while-loop.  Sequential Phase 1
    (cpu_single) converges monotonically and typically exits before the cap.
    Bulk-synchronous Phase 1 (cpu_multi) can oscillate, so the cap is its
    primary termination condition.

    Returns ``(final_labels, hierarchy)``.
    """
    N             = A_sym.shape[0]
    node_to_super = np.arange(N, dtype=np.int32)
    current_adj   = A_sym
    current_m     = m
    hierarchy: list[list[int]] = []

    for _level in range(max_levels):
        n_cur    = current_adj.shape[0]
        degrees  = np.asarray(current_adj.sum(axis=1)).flatten().astype(np.float64)
        communities = np.arange(n_cur, dtype=np.int32)

        changed  = True
        pass_num = 0
        prev_Q   = None
        while changed and pass_num < max_phase1_passes:
            communities, changed = phase1_fn(
                current_adj, communities, degrees, current_m, resolution, min_delta_q
            )
            pass_num += 1

            # Modularity-delta convergence.  Q is bounded in ~[-0.5, 1], so an
            # absolute ``min_delta_q`` threshold on the realized ΔQ is
            # scale-robust — unlike a per-move or summed-gain threshold, whose
            # fixed value is swamped by the many tiny per-node gains a large
            # graph accumulates.  Phase 1's move-count flag has a very long,
            # low-yield tail (a 100k graph keeps moving 1-3 % of nodes for 100+
            # passes while Q barely changes); stopping when a pass improves Q
            # by less than min_delta_q cuts dozens of those passes with no
            # meaningful loss of partition quality.
            Q = _compute_modularity(current_adj, communities, current_m, resolution)
            if prev_Q is not None and (Q - prev_Q) < min_delta_q:
                break
            prev_Q = Q

        level_comms = communities[node_to_super]
        _, level_renumbered = np.unique(level_comms, return_inverse=True)
        hierarchy.append(level_renumbered.tolist())

        new_adj, new_labels = _phase2_collapse(current_adj, communities)
        K = new_adj.shape[0]

        if K >= n_cur:
            break

        node_to_super = new_labels[communities[node_to_super]]
        current_adj   = new_adj
        current_m     = float(new_adj.sum()) / 2.0

    final_labels = node_to_super
    _, final_labels = np.unique(final_labels, return_inverse=True)
    return final_labels.astype(np.int32), hierarchy


# ===========================================================================
# MCL helpers
# ===========================================================================

def _add_self_loops(M: sp.csr_matrix) -> sp.csr_matrix:
    """Add identity to ensure every node has at least one self-transition."""
    n   = M.shape[0]
    eye = sp.eye(n, format="csr", dtype=M.dtype)
    result = M + eye
    result.sum_duplicates()
    return result


def _col_normalize(M: sp.csr_matrix) -> sp.csr_matrix:
    """Column-normalise M so each column sums to 1 (column-stochastic matrix)."""
    M_csc    = M.tocsc().astype(np.float64)
    col_sums = np.asarray(M_csc.sum(axis=0)).flatten()
    col_sums[col_sums == 0.0] = 1.0
    inv      = sp.diags(1.0 / col_sums, format="csr")
    return (M_csc @ inv).tocsr()


def _expand(M: sp.csr_matrix, e: int) -> sp.csr_matrix:
    """Expansion step: raise M to the integer power e via repeated SpGEMM."""
    result = M
    for _ in range(e - 1):
        result = result @ M
    return result


def _prune(M: sp.csr_matrix, threshold: float) -> sp.csr_matrix:
    """Zero out entries below threshold and remove structural zeros."""
    M = M.copy()
    M.data[M.data < threshold] = 0.0
    M.eliminate_zeros()
    return M


def _frobenius_diff(A: sp.csr_matrix, B: sp.csr_matrix) -> float:
    """Frobenius norm of (A - B) for sparse matrices."""
    diff = A - B
    return float(np.sqrt(diff.data @ diff.data))


def _extract_clusters(M: sp.csr_matrix) -> np.ndarray:
    """
    Extract cluster labels from a converged MCL matrix.

    Strategy
    --------
    1. Identify attractor nodes: columns j where M[j,j] > 0.
    2. For each node i, assign it to the attractor j = argmax M[i, attractors].
    3. Fallback: weakly-connected components if no diagonal entry is non-zero.
    """
    n     = M.shape[0]
    M_csr = M.tocsr()
    diag  = np.asarray(M_csr.diagonal()).flatten()
    attractors = np.where(diag > 0)[0]

    if len(attractors) == 0:
        _, labels = csgraph.connected_components(
            M_csr, directed=False, connection="weak"
        )
        return labels.astype(np.int32)

    # MEMORY_FIX (C-4): the previous `np.asarray(M_csr[:, attractors].todense())`
    # allocated an (n × K) dense FP64 matrix.  For n=500k, K=1000 that is
    # 4 GB and reliably OOMs.  We now walk the CSR row-wise and pick the
    # heaviest attractor for each node, allocating only an (n,) int32
    # assignment vector and an int32 column→attractor-index lookup.
    attractor_rank = np.full(n, -1, dtype=np.int32)
    attractor_rank[attractors] = np.arange(len(attractors), dtype=np.int32)

    assignments = np.zeros(n, dtype=np.int32)
    indptr  = M_csr.indptr
    indices = M_csr.indices
    data    = M_csr.data
    for i in range(n):
        s, e = int(indptr[i]), int(indptr[i + 1])
        if s == e:
            continue
        # Mask this row's columns down to those that are attractors.
        cols = indices[s:e]
        ranks = attractor_rank[cols]
        keep = ranks >= 0
        if not keep.any():
            continue
        vals = data[s:e][keep]
        sub_ranks = ranks[keep]
        # Index of max value within the kept subset → attractor index.
        assignments[i] = int(sub_ranks[int(np.argmax(vals))])
    return assignments
