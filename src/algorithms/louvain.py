"""
src/algorithms/louvain.py — Louvain community detection for GRN modules
========================================================================

Biological context
------------------
Communities detected here correspond to co-regulated gene modules — sets
of genes more densely connected (via shared TF inputs or TF↔gene feedback)
to each other than to the rest of the network.  The directed GRN is
symmetrised internally (A_sym = A + A^T) because modularity Q is defined
for undirected graphs; mutual TF↔gene feedback edges get weight 2× as a
result, reflecting tighter functional coupling.

Parallel non-determinism
------------------------
``cpu_multi`` and ``gpu`` use bulk-synchronous batch Phase 1 — all node
move proposals are computed from a snapshot, then applied at once.  Two
adjacent nodes can therefore swap communities then swap back next pass,
which would loop forever without the ``max_phase1_passes`` cap.  The cap
defaults to 100; sequential ``cpu_single`` exits naturally well before it.
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

from .base import AlgorithmBase

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    cpsp = None
    _CUPY_AVAILABLE = False


_SYMMETRIZE_NOTE = (
    "Graph was symmetrized (A + A^T) for Louvain. "
    "Original edge directions not preserved; mutual edges weighted 2×."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _symmetrize(graph_csr: sp.csr_matrix) -> sp.csr_matrix:
    A_sym = (graph_csr + graph_csr.T).tocsr()
    A_sym.sum_duplicates()
    return A_sym.astype(np.float64)


def _compute_modularity(adj_sym, labels, m, resolution):
    degrees = np.asarray(adj_sym.sum(axis=1)).flatten()
    coo = adj_sym.tocoo()
    same = labels[coo.row] == labels[coo.col]
    expected = degrees[coo.row] * degrees[coo.col] / (2.0 * m)
    return float(np.sum((coo.data - resolution * expected) * same) / (2.0 * m))


def _phase2_collapse(adj_csr, communities):
    _, new_labels = np.unique(communities, return_inverse=True)
    K = int(new_labels.max()) + 1
    coo = adj_csr.tocoo()
    row_new = new_labels[coo.row]
    col_new = new_labels[coo.col]
    new_adj = sp.coo_matrix(
        (coo.data.astype(np.float64), (row_new, col_new)),
        shape=(K, K),
    ).tocsr()
    new_adj.sum_duplicates()
    return new_adj, new_labels.astype(np.int32)


def _build_result_data(A_sym, final_labels, hierarchy, m, resolution) -> dict:
    K = int(final_labels.max()) + 1
    Q = _compute_modularity(A_sym, final_labels, m, resolution)
    sizes = np.bincount(final_labels, minlength=K)
    top5 = np.argsort(sizes)[::-1][:5]
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
        "top_communities":       top_communities,
        "note":                  _SYMMETRIZE_NOTE,
    }


# ---------------------------------------------------------------------------
# Phase 1 — sequential
# ---------------------------------------------------------------------------

def _phase1_single(adj_csr, communities, degrees, m, resolution, min_delta_q):
    N = len(communities)
    n_slots = int(communities.max()) + 1
    comm_degsums = np.zeros(n_slots, dtype=np.float64)
    np.add.at(comm_degsums, communities, degrees)
    improved = False

    for i in range(N):
        ki = degrees[i]
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

        unique_c, inverse = np.unique(nb_comms, return_inverse=True)
        k_i_c = np.zeros(len(unique_c), dtype=np.float64)
        np.add.at(k_i_c, inverse, nb_weights)
        k_i_c_old = k_i_c[unique_c == c_old].sum()

        best_dq, best_comm = 0.0, c_old
        for idx, c_new in enumerate(unique_c):
            if c_new == c_old: continue
            dq = ((k_i_c[idx] - k_i_c_old) / m
                  - resolution * ki * (comm_degsums[c_new] - comm_degsums[c_old])
                  / (2.0 * m * m))
            if dq > best_dq:
                best_dq, best_comm = dq, int(c_new)

        communities[i] = best_comm
        comm_degsums[best_comm] += ki
        if best_comm != c_old:
            improved = True

    return communities, improved


def _louvain_batch_worker(args: tuple) -> list[tuple[int, int]]:
    """Bulk-synchronous batch Phase 1 worker (pickle-safe, module-level)."""
    (data, indices, indptr, communities, degrees, comm_degsums,
     m, resolution, min_delta_q, node_batch) = args
    proposals: list[tuple[int, int]] = []
    n_comms = len(comm_degsums)

    for i in node_batch:
        ki = float(degrees[i])
        c_old = int(communities[i])
        sigma_old = float(comm_degsums[c_old]) - ki

        s, e = int(indptr[i]), int(indptr[i + 1])
        if s == e:
            proposals.append((i, c_old))
            continue

        nb_comms   = communities[indices[s:e]]
        nb_weights = data[s:e].astype(np.float64)
        unique_c, inverse = np.unique(nb_comms, return_inverse=True)
        k_i_c = np.zeros(len(unique_c), dtype=np.float64)
        np.add.at(k_i_c, inverse, nb_weights)
        k_i_c_old = k_i_c[unique_c == c_old].sum()

        best_dq, best_comm = min_delta_q, c_old
        for idx, c_new in enumerate(unique_c):
            if c_new == c_old or c_new >= n_comms: continue
            sigma_new = float(comm_degsums[c_new])
            dq = ((k_i_c[idx] - k_i_c_old) / m
                  - resolution * ki * (sigma_new - sigma_old) / (2.0 * m * m))
            if dq > best_dq:
                best_dq, best_comm = dq, int(c_new)
        proposals.append((i, best_comm))

    return proposals


def _phase1_gpu(adj_gpu, comms_gpu, degrees_gpu, m, resolution, min_delta_q):
    N = int(adj_gpu.shape[0])
    n_comms = int(comms_gpu.max()) + 1
    comm_degsums = cp.zeros(n_comms, dtype=cp.float64)
    cp.add.at(comm_degsums, comms_gpu, degrees_gpu)

    coo = adj_gpu.tocoo()
    e_rows = coo.row.astype(cp.int64)
    e_cols = coo.col.astype(cp.int64)
    e_wts  = coo.data.astype(cp.float64)

    target_comms = comms_gpu[e_cols]
    join_gain = (e_wts / m
                 - resolution * comm_degsums[target_comms] * degrees_gpu[e_rows]
                 / (2.0 * m * m))

    same_comm = (target_comms == comms_gpu[e_rows]).astype(cp.float64)
    k_self = cp.zeros(N, dtype=cp.float64)
    cp.add.at(k_self, e_rows, e_wts * same_comm)

    leave_gain = (k_self / m
                  - resolution * (comm_degsums[comms_gpu] - degrees_gpu)
                  * degrees_gpu / (2.0 * m * m))
    delta_q = join_gain - leave_gain[e_rows]

    best_dq = cp.full(N, min_delta_q, dtype=cp.float64)
    cp.maximum.at(best_dq, e_rows, delta_q)

    is_best = ((cp.abs(delta_q - best_dq[e_rows]) < 1e-12)
               & (delta_q > min_delta_q))

    new_comms = comms_gpu.copy()
    if bool(is_best.any()):
        new_comms[e_rows[is_best]] = comms_gpu[e_cols[is_best]]

    improved = bool(cp.any(new_comms != comms_gpu))
    return new_comms, improved


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _run_louvain(A_sym, m, resolution, min_delta_q, max_levels,
                 phase1_fn, max_phase1_passes=100):
    N = A_sym.shape[0]
    node_to_super = np.arange(N, dtype=np.int32)
    current_adj, current_m = A_sym, m
    hierarchy: list[list[int]] = []

    for _level in range(max_levels):
        n_cur = current_adj.shape[0]
        degrees = np.asarray(current_adj.sum(axis=1)).flatten().astype(np.float64)
        communities = np.arange(n_cur, dtype=np.int32)

        changed, pass_num = True, 0
        while changed and pass_num < max_phase1_passes:
            communities, changed = phase1_fn(
                current_adj, communities, degrees, current_m, resolution, min_delta_q
            )
            pass_num += 1

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

    _, final_labels = np.unique(node_to_super, return_inverse=True)
    return final_labels.astype(np.int32), hierarchy


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class Louvain(AlgorithmBase):
    """Louvain community detection with bulk-synchronous parallel modes."""

    NAME = "louvain"
    PARAM_SCHEMA = {
        "min_delta_q":       1e-4,
        "max_levels":        10,
        "resolution":        1.0,
        "max_phase1_passes": 100,
    }

    @staticmethod
    def cpu_single(graph_csr: sp.csr_matrix, params: dict) -> dict:
        t0 = time.perf_counter()
        A_sym = _symmetrize(graph_csr)
        m = float(A_sym.sum()) / 2.0

        final, hier = _run_louvain(
            A_sym, m,
            resolution        = float(params.get("resolution",  1.0)),
            min_delta_q       = float(params.get("min_delta_q", 1e-4)),
            max_levels        = int(params.get("max_levels",    10)),
            phase1_fn         = _phase1_single,
            max_phase1_passes = int(params.get("max_phase1_passes", 100)),
        )
        elapsed = time.perf_counter() - t0
        data = _build_result_data(A_sym, final, hier, m,
                                  float(params.get("resolution", 1.0)))
        return Louvain.build_result(
            mode="cpu_single", execution_time=elapsed,
            graph_csr=graph_csr, result_data=data,
        )

    @staticmethod
    def cpu_multi(graph_csr: sp.csr_matrix, params: dict) -> dict:
        warnings.warn(
            "louvain cpu_multi uses bulk-synchronous parallel Phase 1.  "
            "Community assignments may differ from cpu_single — this is expected.",
            UserWarning, stacklevel=2,
        )
        t0 = time.perf_counter()
        resolution        = float(params.get("resolution",  1.0))
        min_delta_q       = float(params.get("min_delta_q", 1e-4))
        max_levels        = int(params.get("max_levels",    10))
        max_phase1_passes = int(params.get("max_phase1_passes", 100))
        n_workers         = int(params.get("n_workers", 4))

        A_sym = _symmetrize(graph_csr)
        m = float(A_sym.sum()) / 2.0

        # ONE pool reused for the full run (avoids spawn-per-pass on Windows)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:

            def _phase1_batch(adj_csr, communities, degrees, m_inner, res, mdq):
                N_inner = adj_csr.shape[0]
                n_slots = int(communities.max()) + 1
                comm_degs = np.zeros(n_slots, dtype=np.float64)
                np.add.at(comm_degs, communities, degrees)

                chunk_size = max(1, (N_inner + n_workers - 1) // n_workers)
                batches = [
                    list(range(i, min(i + chunk_size, N_inner)))
                    for i in range(0, N_inner, chunk_size)
                ]
                args_list = [
                    (adj_csr.data, adj_csr.indices, adj_csr.indptr,
                     communities, degrees, comm_degs, m_inner, res, mdq, b)
                    for b in batches
                ]
                all_proposals = list(ex.map(_louvain_batch_worker, args_list))

                improved = False
                for batch_props in all_proposals:
                    for node_i, new_comm in batch_props:
                        if communities[node_i] != new_comm:
                            communities[node_i] = new_comm
                            improved = True
                return communities, improved

            final, hier = _run_louvain(
                A_sym, m,
                resolution        = resolution,
                min_delta_q       = min_delta_q,
                max_levels        = max_levels,
                phase1_fn         = _phase1_batch,
                max_phase1_passes = max_phase1_passes,
            )

        elapsed = time.perf_counter() - t0
        data = _build_result_data(A_sym, final, hier, m, resolution)
        return Louvain.build_result(
            mode="cpu_multi", execution_time=elapsed,
            graph_csr=graph_csr, result_data=data,
        )

    @staticmethod
    def gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
        if not _CUPY_AVAILABLE:
            warnings.warn(
                "CuPy unavailable — falling back to louvain cpu_single.",
                RuntimeWarning, stacklevel=2,
            )
            r = Louvain.cpu_single(graph_csr, params)
            r["mode"] = "gpu"
            return r

        warnings.warn(
            "louvain gpu uses bulk-synchronous parallel Phase 1.  "
            "Community assignments may differ from cpu_single — this is expected.",
            UserWarning, stacklevel=2,
        )

        t0 = time.perf_counter()
        resolution        = float(params.get("resolution",  1.0))
        min_delta_q       = float(params.get("min_delta_q", 1e-4))
        max_levels        = int(params.get("max_levels",    10))
        max_phase1_passes = int(params.get("max_phase1_passes", 100))

        A_sym = _symmetrize(graph_csr)
        m = float(A_sym.sum()) / 2.0

        def _phase1_gpu_wrap(adj_csr, communities, degrees, m_inner, res, mdq):
            coo = adj_csr.tocoo()
            adj_gpu = cpsp.csr_matrix(
                (cp.asarray(coo.data, dtype=cp.float64),
                 (cp.asarray(coo.row), cp.asarray(coo.col))),
                shape=coo.shape,
            )
            comms_gpu   = cp.asarray(communities, dtype=cp.int32)
            degrees_gpu = cp.asarray(degrees,     dtype=cp.float64)
            new_comms_gpu, improved = _phase1_gpu(
                adj_gpu, comms_gpu, degrees_gpu, m_inner, res, mdq
            )
            new_comms_cpu = cp.asnumpy(new_comms_gpu).astype(np.int32)
            del adj_gpu, comms_gpu, degrees_gpu, new_comms_gpu
            cp.get_default_memory_pool().free_all_blocks()
            return new_comms_cpu, improved

        final, hier = _run_louvain(
            A_sym, m,
            resolution        = resolution,
            min_delta_q       = min_delta_q,
            max_levels        = max_levels,
            phase1_fn         = _phase1_gpu_wrap,
            max_phase1_passes = max_phase1_passes,
        )

        elapsed = time.perf_counter() - t0
        data = _build_result_data(A_sym, final, hier, m, resolution)
        return Louvain.build_result(
            mode="gpu", execution_time=elapsed,
            graph_csr=graph_csr, result_data=data,
        )
