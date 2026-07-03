"""
src/algorithms/cpu/multi_threaded/louvain.py
============================================

Louvain — cpu_multi mode, NetworKit Parallel Louvain Method (PLM / PLMR).

.. note::
   SuiteSparse:GraphBLAS has **no** native Louvain primitive, and Louvain
   Phase 1 (sequential greedy node-by-node modularity maximization) does
   not vectorize cleanly under GraphBLAS semirings.  Every other
   ``cpu_multi`` algorithm in this project is GraphBLAS-backed; Louvain is
   the single, intentional exception:  it uses **NetworKit** instead.

   NetworKit's ``community.PLM`` is the *Parallel Louvain Method*
   (Staudt & Meyerhenke) — a genuine OpenMP-multithreaded modularity
   optimiser implemented in C++.  ``PLMR`` is the same algorithm with an
   extra refinement sweep after each coarsening step (``refine=True``),
   which usually yields slightly higher modularity.  Thread count is set
   via ``nk.setNumberOfThreads`` so this mode actually exercises multiple
   cores, unlike the previous scipy ``cpu_single`` delegation.

   ``resolution`` maps to NetworKit's ``gamma``.  Multi-level coarsening
   (``recurse=True``) is handled internally by PLM, so ``max_levels`` /
   ``max_phase1_passes`` are not forwarded — NetworKit runs to its own
   convergence.  The returned community assignments are renumbered to a
   contiguous ``0..K-1`` range and passed through the shared
   :func:`_build_result` so ``num_communities`` / ``modularity`` /
   ``top_communities`` are computed identically to ``cpu_single`` — this
   keeps benchmark comparisons apples-to-apples.

   If NetworKit is **not** installed the call degrades gracefully to the
   deterministic scipy ``cpu_single`` implementation (with a warning), so
   the framework keeps working on machines without NetworKit.

Install
-------
``pip install networkit``  (prebuilt Linux/macOS wheels; MIT licensed)

For the deterministic single-thread variant see
``src.algorithms.cpu.single_threaded.louvain``.
"""

from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import _build_result, _symmetrize

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import of NetworKit
# ---------------------------------------------------------------------------

_NETWORKIT_AVAILABLE = False
_NK_IMPORT_ERROR: str | None = None

try:
    import networkit as nk            # type: ignore
    _NETWORKIT_AVAILABLE = True
except ImportError as _exc:           # pragma: no cover
    nk = None                          # type: ignore[assignment]
    _NK_IMPORT_ERROR = str(_exc)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "min_delta_q":       1e-4,
    "max_levels":        10,
    "resolution":        1.0,
    "max_phase1_passes": 100,
    # NetworKit-specific: refine=True selects PLMR (PLM + refinement pass),
    # which trades a little runtime for slightly higher modularity.
    "refine":            True,
}


def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# Thread control
# ---------------------------------------------------------------------------

def _configure_nk_threads(n_workers: int | None = None) -> int:
    """Set NetworKit's OpenMP thread count and return the effective value.

    When ``n_workers`` is None / 0 / negative, uses ``OMP_NUM_THREADS`` if
    set, otherwise ``os.cpu_count()``.
    """
    if not _NETWORKIT_AVAILABLE:
        return 0
    if n_workers and int(n_workers) > 0:
        n = int(n_workers)
    else:
        n = int(os.environ.get("OMP_NUM_THREADS", "0")) or os.cpu_count() or 4
    try:
        nk.setNumberOfThreads(int(n))
        return int(nk.getMaxNumberOfThreads())
    except Exception as exc:                                       # noqa: BLE001
        _LOG.debug("could not set networkit threads=%d: %s", n, exc)
        return int(n)


# ---------------------------------------------------------------------------
# scipy -> NetworKit conversion
# ---------------------------------------------------------------------------

def _scipy_to_networkit(A_sym: sp.csr_matrix):
    """Build an undirected weighted NetworKit graph from a symmetric CSR.

    Only the upper triangle (``row <= col``, self-loops included) is used
    so each undirected edge is added exactly once — feeding the full
    symmetric matrix would double every off-diagonal edge.  Prefers the
    vectorised ``nk.GraphFromCoo`` bulk constructor and falls back to a
    per-edge ``addEdge`` loop only if that entry point is unavailable in
    the installed NetworKit release.
    """
    n   = int(A_sym.shape[0])
    coo = A_sym.tocoo()
    upper = coo.row <= coo.col
    rows  = coo.row[upper].astype(np.int64, copy=False)
    cols  = coo.col[upper].astype(np.int64, copy=False)
    data  = coo.data[upper].astype(np.float64, copy=False)

    graph_from_coo = getattr(nk, "GraphFromCoo", None)
    if graph_from_coo is not None:
        try:
            upper_coo = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
            return graph_from_coo(upper_coo, weighted=True, directed=False)
        except Exception as exc:                                  # noqa: BLE001
            _LOG.debug("nk.GraphFromCoo failed (%s); using addEdge fallback", exc)

    G = nk.Graph(n, weighted=True, directed=False)
    add_edge = G.addEdge
    for a, b, w in zip(rows.tolist(), cols.tolist(), data.tolist()):
        add_edge(a, b, w)
    return G


# ---------------------------------------------------------------------------
# NetworKit PLM / PLMR path
# ---------------------------------------------------------------------------

def _louvain_networkit(
    graph_csr: sp.csr_matrix,
    params:    dict,
    n_threads: int,
) -> dict:
    """Run NetworKit PLM / PLMR and pack the standard Louvain result dict."""
    A_sym = _symmetrize(graph_csr)
    m     = float(A_sym.sum()) / 2.0

    G       = _scipy_to_networkit(A_sym)
    refine  = bool(params.get("refine", True))
    gamma   = float(params.get("resolution", 1.0))

    plm = nk.community.PLM(G, refine=refine, gamma=gamma)
    plm.run()
    partition = plm.getPartition()

    # NetworKit subset ids are not necessarily contiguous — renumber to 0..K-1.
    raw_labels = np.asarray(partition.getVector(), dtype=np.int64)
    _, final_labels = np.unique(raw_labels, return_inverse=True)
    final_labels = final_labels.astype(np.int32)

    # Multi-level hierarchy is internal to PLM and not exposed as a list of
    # per-level label vectors, so hierarchy is left empty (mirrors the
    # gpu_baseline result shape).
    result = _build_result(A_sym, final_labels, [], m, gamma)

    algo = "PLMR" if refine else "PLM"
    result["note"] = (
        f"cpu_multi backend: NetworKit {algo} (parallel Louvain), "
        f"threads={n_threads}, gamma={gamma}."
    )
    return result


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def louvain_cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict,
    n_workers: int | None = None,
) -> dict:
    """Run Louvain via NetworKit's parallel PLM / PLMR.

    Falls back to the deterministic scipy ``cpu_single`` implementation
    (with a warning) if NetworKit is not installed, so the framework keeps
    functioning on machines without it.
    """
    if not _NETWORKIT_AVAILABLE:
        warnings.warn(
            "louvain_cpu_multi: NetworKit is not installed; falling back to "
            "scipy cpu_single.  Install parallel Louvain with "
            f"'pip install networkit'.  (ImportError: {_NK_IMPORT_ERROR})",
            UserWarning, stacklevel=2,
        )
        from src.algorithms.cpu.single_threaded.louvain import louvain_cpu_single
        result = louvain_cpu_single(graph_csr, params)
        existing_note = result.get("note") or ""
        extra = (
            "cpu_multi backend: NetworKit unavailable — "
            "delegated to scipy cpu_single."
        )
        result["note"] = (existing_note + " | " + extra).strip(" |")
        return result

    n_threads = _configure_nk_threads(n_workers)
    return _louvain_networkit(graph_csr, params, n_threads)


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict | None = None,
    n_workers: int | None = None,
    **_,
) -> dict:
    """Benchmark runner entry-point for cpu_multi mode."""
    p = _merge_params(params)
    if _NETWORKIT_AVAILABLE:
        backend = "networkit_plmr" if bool(p.get("refine", True)) else "networkit_plm"
    else:
        backend = "networkit_unavailable_fallback_scipy"
    return {
        "output":       louvain_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "backend": backend},
    }
