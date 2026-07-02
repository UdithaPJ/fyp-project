"""
src/algorithms/gpu/basic/hits.py
================================

HITS — GPU baseline implementation.

Backend
-------
cuGraph ``cugraph.hits`` only — no CuPy fallback.  Matches the hard-fail
policy shared by every cuGraph-only baseline in this package (pagerank,
bfs, louvain, rwr): raise ``ImportError`` at module import time if
cuGraph/cuDF are absent, and raise on any cuGraph runtime failure rather
than silently switching backends.

Known RAPIDS regression (informational — no longer worked around here)
------------------------------------------------------------------------
In RAPIDS 24.02–26.x, ``cugraph.hits`` can be silently re-routed through
the ``pylibcugraph`` multi-GPU handle-based API.  On a single-GPU machine
without Dask / NCCL setup the C layer returns ``CUGRAPH_UNKNOWN_E``,
surfaced as::

    RuntimeError: non-success value returned from cugraph_mg_hits: CUGRAPH_UNKNOWN_E

If you hit this, it is a RAPIDS packaging defect on this install, not a
bug in this file — per the project's baseline hard-fail policy, HITS
raises rather than falling back to CuPy.  Fix at the environment level
(RAPIDS version pin / rebuild), not by adding a fallback here.

Fixes retained
---------------
FIX-1  Always symmetrize the graph before passing to cugraph.hits.
       HITS is defined on undirected graphs (Kleinberg 1999).  Passing a
       directed graph for GRN/miRNA triggered a second, separate C
       dispatch path that also routes to cugraph_mg_hits in RAPIDS 24.x.
       Always directed=False, always symmetrize, regardless of network_type.

FIX-3  Score scatter via numpy indexing instead of np.put.
       np.put(hub_arr, vertex, hubs) is correct for 1-D arrays, but
       if cuGraph renumbers vertices internally the returned vertex IDs
       may not cover 0..n-1.  Explicit advanced indexing plus a bounds
       check is safer and equally fast.

Result keys (mirror src/algorithms/gpu/cuda_optimized/hits.py)
-------------------------------------------------------------
    ppi         : hub_scores, authority_scores, top_nodes, iterations, converged
    grn / mirna : hub_scores, authority_scores, top_hubs, top_authorities,
                  hub_authority_overlap, iterations, converged
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

# Hard-fail: cuGraph + cuDF are required.  No CuPy fallback — matches
# pagerank.py / bfs.py / louvain.py / rwr.py.
try:
    import cugraph   # noqa: F401
    import cudf      # noqa: F401
    _CUGRAPH_AVAILABLE = True
except ImportError as _e:
    _CUGRAPH_AVAILABLE = False
    raise ImportError(
        "src/algorithms/gpu/basic/hits.py requires cuGraph and cuDF "
        "(RAPIDS).  Install via:\n"
        "  conda install -c rapidsai -c nvidia -c conda-forge "
        "rapids=24.02 python=3.10 cudatoolkit=11.8\n"
        f"Original error: {_e}"
    ) from _e

from ._utils import (
    BASELINE_MODE_CUGRAPH,
    TOP_HITS,
    build_envelope, cugraph_build_graph, cugraph_extract_column,
    cugraph_function, symmetrize_for,
    top_k_global,
)

_LOG = logging.getLogger(__name__)

_DEFAULT_PARAMS: dict = {
    "max_iter":     100,
    "tolerance":    1e-6,
    "network_type": "grn",
}


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------

def _log_environment() -> str:
    """Return a one-line RAPIDS/CUDA environment string and log it."""
    try:
        import cudf as _cudf
        cudf_v = getattr(_cudf, "__version__", "?")
    except Exception:
        cudf_v = "unavailable"
    try:
        import cugraph as _cg
        cg_v = getattr(_cg, "__version__", "?")
    except Exception:
        cg_v = "unavailable"
    try:
        import cupy as _cp
        cuda_v = _cp.cuda.runtime.runtimeGetVersion()
        gpu_name = _cp.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        free_b, total_b = _cp.cuda.runtime.memGetInfo()
        mem_info = f"free={free_b//1024**2} MB / total={total_b//1024**2} MB"
    except Exception:
        cuda_v = "?"
        gpu_name = "?"
        mem_info = "?"
    env = (f"cugraph={cg_v} cudf={cudf_v} "
           f"cuda={cuda_v} gpu={gpu_name!r} vram={mem_info}")
    _LOG.info("[HITS baseline env] %s", env)
    return env


def _log_graph_properties(graph_csr: sp.csr_matrix) -> str:
    """Return a graph property string and log it."""
    n    = int(graph_csr.shape[0])
    nnz  = int(graph_csr.nnz)
    degs = np.diff(graph_csr.indptr)
    msg  = (f"n={n} nnz={nnz} avg_deg={nnz/max(n,1):.1f} "
            f"max_deg={int(degs.max()) if degs.size else 0} "
            f"dtype={graph_csr.dtype}")
    _LOG.info("[HITS baseline graph] %s", msg)
    return msg


# ---------------------------------------------------------------------------
# Result packing
# ---------------------------------------------------------------------------

def _pack_result(
    hub: np.ndarray, auth: np.ndarray,
    iterations: int, converged: bool, network_type: str,
) -> dict:
    nt = str(network_type).lower()
    if nt == "ppi":
        combined = hub + auth
        return {
            "hub_scores":       hub.tolist(),
            "authority_scores": auth.tolist(),
            "top_nodes":        top_k_global(combined, TOP_HITS),
            "iterations":       iterations,
            "converged":        converged,
        }
    top_h   = top_k_global(hub,  TOP_HITS)
    top_a   = top_k_global(auth, TOP_HITS)
    overlap = sorted(set(top_h) & set(top_a))
    return {
        "hub_scores":            hub.tolist(),
        "authority_scores":      auth.tolist(),
        "iterations":            iterations,
        "converged":             converged,
        "top_hubs":              top_h,
        "top_authorities":       top_a,
        "hub_authority_overlap": overlap,
    }


# ---------------------------------------------------------------------------
# FIX-1 + FIX-3: corrected cuGraph path
# ---------------------------------------------------------------------------

def _hits_cugraph(
    G, graph_csr: sp.csr_matrix, params: dict,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Run cugraph.hits against an already-constructed single-GPU Graph.

    FIX-3: Use numpy advanced indexing instead of np.put so that
    renumbered vertex IDs from cuGraph map correctly.  np.put(a, ind, v)
    and a[ind] = v are equivalent for 1-D arrays but the latter is clearer
    and handles non-contiguous vertex ranges without silent data corruption.
    """
    hits_fn = cugraph_function("hits")
    if hits_fn is None:
        raise RuntimeError("cugraph.hits is not available in this RAPIDS version.")

    # Detect and warn if we somehow got the MG variant.
    fn_module = getattr(hits_fn, "__module__", "")
    if "dask" in fn_module.lower() or "mg" in fn_module.lower():
        raise RuntimeError(
            f"cugraph.hits resolved to MG module ({fn_module!r}). "
            "This is a RAPIDS packaging defect — the single-GPU path is "
            "not reachable in this installation."
        )

    import inspect
    try:
        accepted = set(inspect.signature(hits_fn).parameters.keys())
    except (TypeError, ValueError):
        accepted = set()

    kwargs: dict[str, Any] = {}
    if "max_iter"  in accepted: kwargs["max_iter"]  = int(params["max_iter"])
    if "tol"       in accepted: kwargs["tol"]       = float(params["tolerance"])
    if "tolerance" in accepted: kwargs["tolerance"] = float(params["tolerance"])
    # nstart: leave as default (None = uniform init).  Passing a cuDF
    # Series here triggers a separate code path in some RAPIDS versions.

    df = hits_fn(G, **kwargs)
    vertex = cugraph_extract_column(df, ["vertex", "node", "id"]).astype(np.int64)
    hubs   = cugraph_extract_column(df, ["hubs", "hub", "hub_score"]).astype(np.float32)
    auths  = cugraph_extract_column(df, ["authorities", "authority", "authority_score"]).astype(np.float32)

    n = int(graph_csr.shape[0])
    hub_arr  = np.zeros(n, dtype=np.float32)
    auth_arr = np.zeros(n, dtype=np.float32)

    # FIX-3: bounds-checked advanced indexing.
    valid = (vertex >= 0) & (vertex < n)
    hub_arr [vertex[valid]] = hubs [valid]
    auth_arr[vertex[valid]] = auths[valid]

    return hub_arr, auth_arr, int(params["max_iter"]), True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def hits_gpu_baseline(
    graph_csr: sp.csr_matrix, params: dict | None = None,
) -> dict:
    """HITS — GPU baseline (cuGraph only, no fallback).

    Returns
    -------
    dict
        7-key standard result envelope; ``result["mode"]`` is
        ``"gpu_baseline_cugraph"``.  Raises on any cuGraph failure —
        see the module docstring for the known RAPIDS regression this
        may surface as.
    """
    p            = {**_DEFAULT_PARAMS, **(params or {})}
    network_type = str(p.get("network_type", "grn")).lower()

    # FIX-1: HITS is defined on undirected graphs (Kleinberg 1999).
    # Always symmetrize regardless of network type — the old code only
    # symmetrized for PPI and passed a directed graph for GRN/miRNA, which
    # triggered a second, separate MG dispatch path in RAPIDS 24.x.
    csr_sym  = symmetrize_for(graph_csr, "ppi")   # "ppi" always returns A+A^T binarised

    # Build graph BEFORE timing (mirrors the H-4 fix in the sibling files).
    # FIX-1 continued: always directed=False — HITS has no concept of edge direction.
    G = cugraph_build_graph(csr_sym, directed=False, weighted=False)

    _log_graph_properties(graph_csr)
    _log_environment()

    t0 = time.perf_counter()
    hub, auth, iters, converged = _hits_cugraph(G, graph_csr, p)
    elapsed = time.perf_counter() - t0

    _LOG.info(
        "[HITS baseline] cuGraph path OK  n=%d nnz=%d iters=%d elapsed=%.3fs",
        graph_csr.shape[0], graph_csr.nnz, iters, elapsed,
    )

    inner = _pack_result(hub, auth, iters, converged, network_type)
    inner["backend"] = "cugraph"

    return build_envelope(
        algorithm="hits",
        network_type=network_type,
        execution_time=elapsed,
        graph_csr=graph_csr,
        inner=inner,
        mode=BASELINE_MODE_CUGRAPH,
    )
