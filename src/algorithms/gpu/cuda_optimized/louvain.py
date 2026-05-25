"""
algorithms/louvain.py — Louvain Community Detection for GRN Module Discovery
=============================================================================

Biological Context — Co-regulated Gene Modules
------------------------------------------------
Louvain community detection on a GRN identifies groups of genes that share
common transcription factor regulators or participate in the same regulatory
pathway.  Within a detected community, genes are more densely connected to
each other (via shared TF inputs or TF–target–TF feedback arcs) than to the
rest of the network.  These communities often correspond to:

  * Functionally coherent gene modules (e.g. cell-cycle genes all regulated
    by E2F factors)
  * Co-expressed gene clusters that respond to the same upstream signal
  * Biological pathway membership (MAPK cascade, NF-κB regulon, etc.)

The ``top_communities`` output (largest K communities with member node lists)
can be submitted directly to GO-term enrichment tools (e.g. Enrichr, g:Profiler)
to assess whether each detected module is biologically coherent.

Why the GRN must be symmetrized
---------------------------------
Louvain optimises modularity Q, which is defined for *undirected* graphs:

    Q = (1/2m) Σ_{i,j} (A_{ij} − k_i·k_j/2m) · δ(c_i, c_j)

Applying Q to an asymmetric matrix produces mathematically undefined results
because the null-model k_i·k_j/2m assumes undirected degree.

The symmetrisation used here is:

    A_sym = A + A^T

Biological meaning of each edge weight after symmetrisation:

  * One-way TF→gene edge:    weight = original weight (one regulatory event)
  * Mutual TF↔gene feedback: weight = 2 × original weight (stronger co-regulation)

This weighting is intentional: mutual regulation reflects tighter functional
coupling and should group nodes into the same community more strongly.

This symmetrisation step is performed *inside* each Louvain function, NOT
during preprocessing, because all other algorithms (HITS, RWR, BFS, PageRank)
require the original directed graph.

Parallel non-determinism
--------------------------
The ``cpu_multi`` and ``gpu`` modes batch all community moves in Phase 1 and
apply them simultaneously.  This eliminates the sequential dependency of
``cpu_single`` Phase 1 and enables parallelism, but means that two nodes
that would beneficially swap communities may BOTH propose the swap
simultaneously, and the outcome depends on which write lands last.

This is a known and expected property of parallel Louvain (documented in
Traag et al. 2019, "From Louvain to Leiden").  A runtime warning is printed
whenever these modes are used.  Results remain valid (modularity is
monotonically non-decreasing in practice) but may differ between runs.

Algorithm Outline
-----------------
Phase 1 — Move nodes to improve modularity:
    For each node i, compute ΔQ for moving i to each neighbour community.
    ΔQ = (k_{i,C}/m) − γ·k_i·Σ_tot_C/(2m²)
    Move i to the community with the highest positive ΔQ.
    Repeat until no node moves.

Phase 2 — Collapse communities into super-nodes:
    Each community becomes a single super-node.
    Edge weights between super-nodes = sum of weights of inter-community edges.
    Self-loops on super-nodes = sum of intra-community edge weights.
    Recursively run Phase 1 + Phase 2 until no improvement.

Parameter Guide
---------------
min_delta_q  (float, default 1e-4)  Minimum ΔQ to accept a node move.
max_levels   (int,   default 10)    Maximum Phase 1 + Phase 2 recursion depth.
resolution   (float, default 1.0)   γ in the modularity formula.
                                     > 1.0 → more, smaller communities.
                                     < 1.0 → fewer, larger communities.
"""

# ── GPU / CUDA-optimised implementation ──────────────────────────────────
# Source:    biological_network_framework/algorithms/louvain.py
# Requires:  cupy-cuda11x (or matching CUDA version), pycuda
# Used by:   webapp routes, GPU benchmarking (src.benchmarking.benchmark)
# Modes:     _cpu_single, _cpu_multi, _gpu  (all three)
# CPU-only counterparts (benchmarking only — never import in webapp):
#   src.algorithms.cpu.single_threaded.louvain
#   src.algorithms.cpu.multi_threaded.louvain
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
    "min_delta_q":        1e-4,
    "max_levels":         10,
    "resolution":         1.0,
    # Maximum Phase 1 passes per Louvain level.
    # Sequential cpu_single converges naturally; parallel modes (cpu_multi,
    # gpu) use bulk-synchronous moves that can oscillate indefinitely without
    # this cap.  100 passes is well above the typical convergence point for
    # GRN-sized graphs while guaranteeing termination.
    "max_phase1_passes":  100,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


def _symmetrize(graph_csr: sp.csr_matrix) -> sp.csr_matrix:
    """
    Symmetrize the directed GRN adjacency: A_sym = A + A^T.

    Mutual TF↔gene edges get weight 2× (tighter co-regulation);
    one-way edges keep their original weight.
    """
    A_sym = (graph_csr + graph_csr.T).tocsr()
    A_sym.sum_duplicates()
    return A_sym.astype(np.float64)


def _compute_modularity(
    adj_sym: sp.csr_matrix,
    labels: np.ndarray,
    m: float,
    resolution: float,
) -> float:
    """Compute modularity Q for a labelled partition in O(nnz)."""
    degrees = np.asarray(adj_sym.sum(axis=1)).flatten()
    coo = adj_sym.tocoo()
    same = labels[coo.row] == labels[coo.col]
    expected = degrees[coo.row] * degrees[coo.col] / (2.0 * m)
    Q = float(np.sum((coo.data - resolution * expected) * same) / (2.0 * m))
    return Q


def _phase2_collapse(
    adj_csr: sp.csr_matrix,
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


def _build_result(
    A_sym: sp.csr_matrix,
    final_labels: np.ndarray,
    hierarchy: list[list[int]],
    m: float,
    resolution: float,
) -> dict:
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
        "hierarchy":             hierarchy,
        "top_communities":       top_communities,
    }


# ---------------------------------------------------------------------------
# Phase 1 — sequential (cpu_single)
# ---------------------------------------------------------------------------

def _phase1_single(
    adj_csr: sp.csr_matrix,
    communities: np.ndarray,
    degrees: np.ndarray,
    m: float,
    resolution: float,
    min_delta_q: float,
) -> tuple[np.ndarray, bool]:
    """
    One complete sequential pass of Louvain Phase 1.

    For each node, computes ΔQ for moving to each neighbour community and
    performs the best greedy move if ΔQ > min_delta_q.

    ΔQ (move i from c_old to c_new) =
        (k_{i,c_new} − k_{i,c_old}) / m
        − resolution · k_i · (Σ_tot_c_new − Σ_tot_c_old) / (2m²)

    where Σ_tot_c is the sum of degrees of nodes in c (after removing i).

    Returns (communities, improved).
    """
    N = len(communities)
    n_slots = int(communities.max()) + 1
    comm_degsums = np.zeros(n_slots, dtype=np.float64)
    np.add.at(comm_degsums, communities, degrees)

    improved = False

    for i in range(N):
        ki    = degrees[i]
        c_old = int(communities[i])

        # Temporarily remove node i from its current community
        comm_degsums[c_old] -= ki

        s = int(adj_csr.indptr[i])
        e = int(adj_csr.indptr[i + 1])
        if s == e:                              # isolated node
            comm_degsums[c_old] += ki
            continue

        nb_indices = adj_csr.indices[s:e]
        nb_weights = adj_csr.data[s:e].astype(np.float64)
        nb_comms   = communities[nb_indices].astype(np.int32)

        # Aggregate: k_{i,c} = sum of weights from i to community c
        unique_comms, inverse = np.unique(nb_comms, return_inverse=True)
        k_i_c = np.zeros(len(unique_comms), dtype=np.float64)
        np.add.at(k_i_c, inverse, nb_weights)

        k_i_c_old = k_i_c[unique_comms == c_old].sum()   # 0 if c_old not neighbour

        best_dq   = 0.0    # must beat this threshold
        best_comm = c_old

        for idx, c_new in enumerate(unique_comms):
            if c_new == c_old:
                continue
            dq = ((k_i_c[idx] - k_i_c_old) / m
                  - resolution * ki * (comm_degsums[c_new] - comm_degsums[c_old]) / (2.0 * m * m))
            if dq > best_dq:
                best_dq   = dq
                best_comm = int(c_new)

        communities[i] = best_comm
        comm_degsums[best_comm] += ki
        if best_comm != c_old:
            improved = True

    return communities, improved


# ---------------------------------------------------------------------------
# Phase 1 — batch parallel worker (cpu_multi)
# ---------------------------------------------------------------------------

def _louvain_batch_worker(args: tuple) -> list[tuple[int, int]]:
    """
    ProcessPoolExecutor worker: compute the best move for a batch of nodes.

    Uses a *snapshot* of ``comm_degsums`` taken before the batch starts.
    All proposals are returned as (node_index, best_community) pairs.
    The caller applies all moves together (bulk-synchronous update).
    """
    (adj_data, adj_indices, adj_indptr,
     communities, degrees, comm_degsums,
     m, resolution, min_delta_q, node_batch) = args

    proposals: list[tuple[int, int]] = []
    n_comms = len(comm_degsums)

    for i in node_batch:
        ki    = float(degrees[i])
        c_old = int(communities[i])

        # Snapshot: Σ_tot after hypothetically removing i
        sigma_old = float(comm_degsums[c_old]) - ki

        s = int(adj_indptr[i])
        e = int(adj_indptr[i + 1])
        if s == e:
            proposals.append((i, c_old))
            continue

        nb_comms   = communities[adj_indices[s:e]]
        nb_weights = adj_data[s:e].astype(np.float64)

        unique_comms, inverse = np.unique(nb_comms, return_inverse=True)
        k_i_c = np.zeros(len(unique_comms), dtype=np.float64)
        np.add.at(k_i_c, inverse, nb_weights)

        k_i_c_old = k_i_c[unique_comms == c_old].sum()

        best_dq   = min_delta_q
        best_comm = c_old

        for idx, c_new in enumerate(unique_comms):
            if c_new == c_old or c_new >= n_comms:
                continue
            sigma_new = float(comm_degsums[c_new])
            dq = ((k_i_c[idx] - k_i_c_old) / m
                  - resolution * ki * (sigma_new - sigma_old) / (2.0 * m * m))
            if dq > best_dq:
                best_dq   = dq
                best_comm = int(c_new)

        proposals.append((i, best_comm))

    return proposals


# ---------------------------------------------------------------------------
# Phase 1 — vectorized GPU (gpu)
# ---------------------------------------------------------------------------

def _phase1_gpu(
    adj_csr_gpu,           # CuPy CSR sparse
    communities_gpu,       # (N,) CuPy int32
    degrees_gpu,           # (N,) CuPy float64
    m: float,
    resolution: float,
    min_delta_q: float,
) -> tuple:                # (new_communities_gpu, improved: bool)
    """
    One vectorized batch-parallel Phase 1 pass on the GPU.

    For every edge (i, j, w) simultaneously:
        join_gain[e] = w/m − γ·Σ_tot[comm[j]]·k_i / (2m²)
    For every node i:
        leave_gain[i] = k_{i,c_i}/m − γ·(Σ_tot[c_i] − k_i)·k_i / (2m²)
        delta_q[e]    = join_gain[e] − leave_gain[edge_rows[e]]

    For each node the edge achieving the maximum delta_q (subject to
    delta_q > min_delta_q) is selected using cp.maximum.at (GPU atomics).
    All moves are applied simultaneously (bulk-synchronous Louvain).
    """
    N = int(adj_csr_gpu.shape[0])

    # Community degree sums
    n_comms = int(communities_gpu.max()) + 1
    comm_degsums = cp.zeros(n_comms, dtype=cp.float64)
    cp.add.at(comm_degsums, communities_gpu, degrees_gpu)

    # Edge arrays from COO view
    coo = adj_csr_gpu.tocoo()
    e_rows = coo.row.astype(cp.int64)
    e_cols = coo.col.astype(cp.int64)
    e_wts  = coo.data.astype(cp.float64)
    nnz    = int(coo.nnz)

    target_comms = communities_gpu[e_cols]   # community of each edge's target

    # Join gain per edge
    join_gain = (e_wts / m
                 - resolution * comm_degsums[target_comms] * degrees_gpu[e_rows]
                 / (2.0 * m * m))

    # Leave gain per node: k_{i,c_i} via scatter-add over same-community edges
    same_comm = (target_comms == communities_gpu[e_rows]).astype(cp.float64)
    k_self = cp.zeros(N, dtype=cp.float64)
    cp.add.at(k_self, e_rows, e_wts * same_comm)

    leave_gain = (k_self / m
                  - resolution * (comm_degsums[communities_gpu] - degrees_gpu)
                  * degrees_gpu / (2.0 * m * m))

    # ΔQ per edge
    delta_q = join_gain - leave_gain[e_rows]   # (nnz,)

    # Per-node maximum ΔQ via GPU atomic scatter-max
    # Baseline = min_delta_q so only improving moves are flagged
    best_dq = cp.full(N, min_delta_q, dtype=cp.float64)
    cp.maximum.at(best_dq, e_rows, delta_q)

    # Mark edges that achieve the best ΔQ for their source node AND improve Q
    is_best = (
        (cp.abs(delta_q - best_dq[e_rows]) < 1e-12)
        & (delta_q > min_delta_q)
    )

    # Apply batch moves: scatter target communities of best edges back to sources
    # For nodes with ties (multiple edges sharing best_dq), last write wins —
    # this is the expected non-determinism of parallel Louvain.
    new_communities = communities_gpu.copy()
    if bool(is_best.any()):
        best_src   = e_rows[is_best]
        best_tgt_c = communities_gpu[e_cols[is_best]]
        new_communities[best_src] = best_tgt_c

    improved = bool(cp.any(new_communities != communities_gpu))
    return new_communities, improved


# ---------------------------------------------------------------------------
# Louvain main loop (shared across modes)
# ---------------------------------------------------------------------------

def _run_louvain(
    A_sym: sp.csr_matrix,
    m: float,
    resolution: float,
    min_delta_q: float,
    max_levels: int,
    phase1_fn,            # callable: (adj, comms, degs, m, res, mdq) → (comms, improved)
    max_phase1_passes: int = 100,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Execute the two-phase Louvain loop using a pluggable Phase 1 function.

    ``max_phase1_passes`` caps the inner while-loop for each Louvain level.
    Sequential (cpu_single) Phase 1 converges monotonically and typically
    exits long before the cap.  Parallel (cpu_multi / gpu) Phase 1 uses
    bulk-synchronous moves that can oscillate — two adjacent nodes swapping
    communities in alternating passes — so the cap is the primary termination
    condition for those modes.

    Returns (final_labels, hierarchy).
    """
    N = A_sym.shape[0]
    node_to_super = np.arange(N, dtype=np.int32)
    current_adj   = A_sym
    current_m     = m
    hierarchy: list[list[int]] = []

    for _level in range(max_levels):
        n_cur = current_adj.shape[0]
        degrees = np.asarray(current_adj.sum(axis=1)).flatten().astype(np.float64)
        communities = np.arange(n_cur, dtype=np.int32)

        # ---- Phase 1 ----
        # The pass counter prevents infinite oscillation in bulk-synchronous
        # parallel modes (cpu_multi, gpu).  Sequential cpu_single will exit
        # naturally via changed=False well before max_phase1_passes.
        changed   = True
        pass_num  = 0
        while changed and pass_num < max_phase1_passes:
            communities, changed = phase1_fn(
                current_adj, communities, degrees, current_m, resolution, min_delta_q
            )
            pass_num += 1

        # Record community assignments for original nodes at this level
        level_comms = communities[node_to_super]
        _, level_renumbered = np.unique(level_comms, return_inverse=True)
        hierarchy.append(level_renumbered.tolist())

        # ---- Phase 2: collapse ----
        new_adj, new_labels = _phase2_collapse(current_adj, communities)
        K = new_adj.shape[0]

        if K >= n_cur:          # no merging — converged
            break

        # Update node → super-node mapping
        node_to_super = new_labels[communities[node_to_super]]
        current_adj   = new_adj
        current_m     = float(new_adj.sum()) / 2.0

    # Final assignments: each original node maps to its super-node at the last level
    final_labels = node_to_super
    _, final_labels = np.unique(final_labels, return_inverse=True)
    return final_labels.astype(np.int32), hierarchy


# ---------------------------------------------------------------------------
# Public implementations
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
    params    : dict — see module docstring.

    Returns
    -------
    dict with keys: community_assignments, num_communities, modularity,
                    hierarchy, top_communities
    """
    p = _merge_params(params)
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


def louvain_cpu_multi(
    graph_csr: sp.csr_matrix,
    params: dict,
    n_workers: int = 4,
) -> dict:
    """
    Louvain — multi-process CPU with bulk-synchronous batch Phase 1.

    Phase 1 is parallelised by distributing nodes across workers.  Each
    worker computes the best community move for its node batch using a
    *snapshot* of the community degree sums taken at the start of each
    batch round.  All proposed moves are then applied simultaneously.

    .. warning::
        Parallel Louvain non-determinism — this mode may produce different
        community assignments and modularity values than ``cpu_single``
        because simultaneous moves can conflict (e.g. two nodes propose to
        swap communities).  This is a documented property of bulk-synchronous
        Louvain and does not indicate an error.

    Phase 2 (community collapse) runs on the main process (sequential) as
    it involves graph restructuring that is not trivially parallelisable.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
    params    : dict
    n_workers : int

    Returns
    -------
    Same structure as louvain_cpu_single.
    """
    warnings.warn(
        "louvain_cpu_multi uses bulk-synchronous parallel Phase 1.  "
        "Community assignments may differ from cpu_single due to batched "
        "move application — this is expected behaviour, not a bug.",
        UserWarning,
        stacklevel=2,
    )

    p                 = _merge_params(params)
    resolution        = float(p["resolution"])
    min_delta_q       = float(p["min_delta_q"])
    max_levels        = int(p["max_levels"])
    max_phase1_passes = int(p["max_phase1_passes"])

    A_sym = _symmetrize(graph_csr)
    m     = float(A_sym.sum()) / 2.0

    # Create ONE worker pool for the entire Louvain run.
    # Previously the pool was spawned and torn down inside _phase1_batch on
    # every single Phase 1 pass.  On Windows (spawn start method) each pool
    # creation costs ~0.3–1 s of process start-up per worker, so hundreds of
    # passes × 4 workers meant minutes of pure overhead before any work.
    # Hoisting the pool here amortises that cost across all passes and levels.
    with ProcessPoolExecutor(max_workers=n_workers) as ex:

        def _phase1_batch(
            adj_csr: sp.csr_matrix,
            communities: np.ndarray,
            degrees: np.ndarray,
            m_inner: float,
            res: float,
            mdq: float,
        ) -> tuple[np.ndarray, bool]:
            """Bulk-synchronous batch Phase 1 — reuses the outer pool."""
            N_inner   = adj_csr.shape[0]
            n_slots   = int(communities.max()) + 1
            comm_degs = np.zeros(n_slots, dtype=np.float64)
            np.add.at(comm_degs, communities, degrees)

            chunk_size = max(1, (N_inner + n_workers - 1) // n_workers)
            batches = [
                list(range(i, min(i + chunk_size, N_inner)))
                for i in range(0, N_inner, chunk_size)
            ]
            args_list = [
                (adj_csr.data, adj_csr.indices, adj_csr.indptr,
                 communities, degrees, comm_degs,
                 m_inner, res, mdq, batch)
                for batch in batches
            ]

            all_proposals = list(ex.map(_louvain_batch_worker, args_list))

            # Apply all proposed moves simultaneously (bulk-synchronous update)
            improved = False
            for batch_props in all_proposals:
                for node_i, new_comm in batch_props:
                    if communities[node_i] != new_comm:
                        communities[node_i] = new_comm
                        improved = True

            return communities, improved

        final_labels, hierarchy = _run_louvain(
            A_sym, m,
            resolution        = resolution,
            min_delta_q       = min_delta_q,
            max_levels        = max_levels,
            phase1_fn         = _phase1_batch,
            max_phase1_passes = max_phase1_passes,
        )

    return _build_result(A_sym, final_labels, hierarchy, m, resolution)


def louvain_gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
    """
    Louvain — GPU-accelerated bulk-synchronous Phase 1 via CuPy.

    The symmetrized adjacency is uploaded to GPU once per Louvain level.
    Phase 1 uses fully vectorized CuPy operations (see ``_phase1_gpu``):
    join/leave gains are computed for all edges in parallel; per-node
    best moves are found via GPU atomic scatter-max (``cp.maximum.at``);
    all moves are applied simultaneously as a batch index-scatter.

    Phase 2 (community collapse) runs on CPU since it involves graph
    restructuring (scatter-add on a new COO matrix), which scipy handles
    efficiently for the reduced super-node graph.

    .. warning::
        Same non-determinism caveat as ``cpu_multi`` — bulk-synchronous
        move application may produce different partitions across runs.

    Falls back to ``louvain_cpu_single`` with a warning if CuPy is
    unavailable.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
    params    : dict

    Returns
    -------
    Same structure as louvain_cpu_single.
    """
    try:
        from src.benchmarking.benchmark import _ensure_cuda_context
        if not _ensure_cuda_context():
            raise RuntimeError("no CUDA context")
    except Exception:
        pass

    if not _CUPY_AVAILABLE:
        warnings.warn(
            "CuPy unavailable — falling back to louvain_cpu_single.",
            RuntimeWarning,
            stacklevel=2,
        )
        return louvain_cpu_single(graph_csr, params)

    warnings.warn(
        "louvain_gpu uses bulk-synchronous parallel Phase 1.  "
        "Community assignments may differ from cpu_single — this is expected.",
        UserWarning,
        stacklevel=2,
    )

    p                 = _merge_params(params)
    resolution        = float(p["resolution"])
    min_delta_q       = float(p["min_delta_q"])
    max_levels        = int(p["max_levels"])
    max_phase1_passes = int(p["max_phase1_passes"])

    A_sym = _symmetrize(graph_csr)
    m     = float(A_sym.sum()) / 2.0

    def _phase1_gpu_wrapper(
        adj_csr: sp.csr_matrix,
        communities: np.ndarray,
        degrees: np.ndarray,
        m_inner: float,
        res: float,
        mdq: float,
    ) -> tuple[np.ndarray, bool]:
        """Upload current level to GPU, run vectorized Phase 1, download."""
        coo = adj_csr.tocoo()
        adj_gpu = cpsp.csr_matrix(
            (
                cp.asarray(coo.data,  dtype=cp.float64),
                (cp.asarray(coo.row), cp.asarray(coo.col)),
            ),
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

    final_labels, hierarchy = _run_louvain(
        A_sym, m,
        resolution        = resolution,
        min_delta_q       = min_delta_q,
        max_levels        = max_levels,
        phase1_fn         = _phase1_gpu_wrapper,
        max_phase1_passes = max_phase1_passes,
    )
    return _build_result(A_sym, final_labels, hierarchy, m, resolution)


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _cpu_single(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    p = _merge_params(params)
    return {"output": louvain_cpu_single(graph_csr, p), "extra_params": p}


def _cpu_multi(
    graph_csr: sp.csr_matrix,
    params: dict | None = None,
    n_workers: int = 4,
    **_,
) -> dict:
    p = _merge_params(params)
    return {
        "output":      louvain_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "n_workers": n_workers},
    }


def _gpu(graph_csr: sp.csr_matrix, params: dict | None = None, **_) -> dict:
    p = _merge_params(params)
    return {"output": louvain_gpu(graph_csr, p), "extra_params": p}
