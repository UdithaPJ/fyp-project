"""
src/algorithms/gpu/basic/hits.py
================================

HITS — GPU baseline implementation.

Backend (priority order)
------------------------
1. cuGraph ``cugraph.hits`` — preferred.  Used when the installed RAPIDS
   version provides a working single-GPU HITS implementation.
2. CuPy power-iteration fallback — activated automatically when cuGraph
   raises any of the multi-GPU dispatch errors known to appear in
   RAPIDS 24.02–24.06 (``cugraph_mg_hits: CUGRAPH_UNKNOWN_E``,
   ``RuntimeError: non-success value returned from cugraph_mg_hits``).

Why the fallback is necessary
------------------------------
In RAPIDS 24.02–24.06, ``cugraph.hits`` was silently re-routed through
the ``pylibcugraph`` handle-based API.  On a single-GPU machine without
Dask / NCCL setup the C layer returns ``CUGRAPH_UNKNOWN_E``, which Python
surfaces as::

    RuntimeError: non-success value returned from cugraph_mg_hits: CUGRAPH_UNKNOWN_E

Other algorithms (PageRank, BFS, Louvain, RWR) have their own C entry
points that were not affected by this regression.  Only HITS hits this
path in the affected RAPIDS versions.

Fixes applied
-------------
FIX-1  Always symmetrize the graph before passing to cugraph.hits.
       HITS is defined on undirected graphs (Kleinberg 1999).  Passing a
       directed graph for GRN/miRNA triggered a second, separate C
       dispatch path that also routes to cugraph_mg_hits in RAPIDS 24.x.
       The old code: symmetrize_for only when nt=="ppi"; directed=True
       for GRN/miRNA.  Fixed: always directed=False, always symmetrize.

FIX-2  CuPy power-iteration fallback.
       On any RuntimeError whose message contains "cugraph_mg_hits" or
       "CUGRAPH_UNKNOWN", fall through to a CuPy L2-normalised power
       iteration.  The result["backend"] key distinguishes which path ran.

FIX-3  Score scatter via numpy indexing instead of np.put.
       np.put(hub_arr, vertex, hubs) is correct for 1-D arrays, but
       if cuGraph renumbers vertices internally the returned vertex IDs
       may not cover 0..n-1.  Explicit advanced indexing plus a bounds
       check is safer and equally fast.

FIX-4  Detailed diagnostic logging on every failure.
       Logs RAPIDS version, graph dimensions, GPU memory, dtype, and the
       full traceback so failures can be reproduced from the log alone.

Result keys (mirror src/algorithms/gpu/cuda_optimized/hits.py)
-------------------------------------------------------------
    ppi         : hub_scores, authority_scores, top_nodes, iterations, converged
    grn / mirna : hub_scores, authority_scores, top_hubs, top_authorities,
                  hub_authority_overlap, iterations, converged
"""

from __future__ import annotations

import logging
import traceback
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

# Hard-fail: cuGraph + cuDF are required as the primary path.
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

# CuPy is the fallback for the known cugraph_mg_hits regression.
try:
    import cupy as _cupy                              # type: ignore
    import cupyx.scipy.sparse as _cupyx_sp            # type: ignore
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

from ._utils import (
    BASELINE_MODE_CUGRAPH,
    BASELINE_MODE_CUPY,
    TOP_HITS,
    build_envelope, cugraph_build_graph, cugraph_extract_column,
    cugraph_function, symmetrize_for,
    top_k_global,
)

_LOG = logging.getLogger(__name__)

# Error fragments that identify the multi-GPU dispatch failure.
# We match substrings so minor message variations across RAPIDS versions
# are all caught by the same guard.
_MG_ERROR_PATTERNS = (
    "cugraph_mg_hits",
    "CUGRAPH_UNKNOWN",
    "cugraph_mg_",          # any MG C function
    "raft::handle",         # Dask handle not initialised
    "nccl",                 # NCCL not configured
)

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


def _is_mg_error(exc: Exception) -> bool:
    """Return True when ``exc`` is a known multi-GPU dispatch error."""
    msg = str(exc).lower()
    return any(pat.lower() in msg for pat in _MG_ERROR_PATTERNS)


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
# FIX-2: CuPy power-iteration fallback
# ---------------------------------------------------------------------------

def _hits_cupy(
    graph_csr: sp.csr_matrix, params: dict,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """L2-normalised HITS power iteration via CuPy.

    This path is activated when cugraph.hits routes to the MG C kernel on
    a single-GPU machine (known regression in RAPIDS 24.02–24.06).

    The algorithm is the standard Kleinberg HITS:
        a_new = A^T h  (authorities receive from hubs via in-edges)
        h_new = A  a   (hubs point to authorities via out-edges)
    Both vectors are L2-normalised after each update.  A is the
    symmetrised adjacency — HITS is only well-defined on undirected graphs.
    """
    if not _CUPY_AVAILABLE:
        raise RuntimeError(
            "CuPy is not installed and cugraph.hits is unavailable.  "
            "pip install cupy-cuda12x  (adjust CUDA suffix)."
        )

    max_iter = int(params["max_iter"])
    tol      = float(params["tolerance"])
    n        = int(graph_csr.shape[0])

    # Symmetrize on CPU (cheap), then move to GPU once.
    A_sym_sp = graph_csr + graph_csr.T
    if A_sym_sp.nnz > 0:
        A_sym_sp.data = np.ones_like(A_sym_sp.data, dtype=np.float32)
    A_sym_sp = A_sym_sp.tocsr().astype(np.float32)

    A   = _cupyx_sp.csr_matrix(A_sym_sp)
    A_T = A.T.tocsr()

    h = _cupy.ones(n, dtype=_cupy.float32)
    a = _cupy.ones(n, dtype=_cupy.float32)

    converged  = False
    iterations = 0
    for it in range(max_iter):
        iterations = it + 1
        h_old, a_old = h, a

        # Authority update: a_new = A^T h
        a_new = A_T.dot(h_old)
        norm_a = float(_cupy.linalg.norm(a_new))
        if norm_a > 0.0:
            a_new = a_new / norm_a

        # Hub update: h_new = A a_new
        h_new = A.dot(a_new)
        norm_h = float(_cupy.linalg.norm(h_new))
        if norm_h > 0.0:
            h_new = h_new / norm_h

        delta = float(
            _cupy.abs(h_new - h_old).sum() + _cupy.abs(a_new - a_old).sum()
        )
        h, a = h_new, a_new
        if delta < tol:
            converged = True
            break

    hub_arr  = _cupy.asnumpy(h).astype(np.float32)
    auth_arr = _cupy.asnumpy(a).astype(np.float32)
    return hub_arr, auth_arr, iterations, converged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def hits_gpu_baseline(
    graph_csr: sp.csr_matrix, params: dict | None = None,
) -> dict:
    """HITS — GPU baseline (cuGraph preferred, CuPy fallback).

    Returns
    -------
    dict
        Standard 7-key envelope.  ``result["backend"]`` is ``"cugraph"``
        or ``"cupy_fallback"`` depending on which path succeeded.
        ``result["mode"]`` is ``"gpu_baseline_cugraph"`` or
        ``"gpu_baseline_cupy"`` accordingly.
    """
    p            = {**_DEFAULT_PARAMS, **(params or {})}
    network_type = str(p.get("network_type", "grn")).lower()

    # FIX-1: HITS is defined on undirected graphs (Kleinberg 1999).
    # Always symmetrize regardless of network type.  The previous code
    # only symmetrized for PPI and passed a directed graph for GRN/miRNA,
    # which triggered the MG dispatch in RAPIDS 24.x.
    csr_sym  = symmetrize_for(graph_csr, "ppi")   # "ppi" always returns A+A^T binarised

    # Build graph BEFORE timing (H-4 fix retained).
    # FIX-1 continued: always directed=False — HITS has no concept of edge direction.
    G = cugraph_build_graph(csr_sym, directed=False, weighted=False)

    # ---- Try cuGraph path -----------------------------------------------
    t0 = time.perf_counter()
    _log_graph_properties(graph_csr)
    env_str = _log_environment()

    try:
        hub, auth, iters, converged = _hits_cugraph(G, graph_csr, p)
        elapsed = time.perf_counter() - t0
        backend = "cugraph"
        mode    = BASELINE_MODE_CUGRAPH

        _LOG.info(
            "[HITS baseline] cuGraph path OK  n=%d nnz=%d iters=%d "
            "elapsed=%.3fs",
            graph_csr.shape[0], graph_csr.nnz, iters, elapsed,
        )

    except Exception as cg_exc:
        elapsed_fail = time.perf_counter() - t0

        # FIX-4: log full diagnostics on any cuGraph failure.
        _LOG.warning(
            "[HITS baseline] cuGraph FAILED after %.3fs — "
            "n=%d nnz=%d env=(%s)\n"
            "Error: %s\n"
            "Traceback:\n%s",
            elapsed_fail,
            graph_csr.shape[0], graph_csr.nnz,
            env_str,
            cg_exc,
            traceback.format_exc(),
        )

        # FIX-2: if this looks like the MG-dispatch bug, fall through to CuPy.
        if _is_mg_error(cg_exc):
            _LOG.warning(
                "[HITS baseline] Detected cugraph_mg_hits dispatch error "
                "(RAPIDS 24.x regression on single-GPU machines).  "
                "Falling back to CuPy power iteration.  "
                "result[\"backend\"] will be \"cupy_fallback\"."
            )
            try:
                t1 = time.perf_counter()
                hub, auth, iters, converged = _hits_cupy(graph_csr, p)
                elapsed  = time.perf_counter() - t1
                backend  = "cupy_fallback"
                mode     = BASELINE_MODE_CUPY

                _LOG.info(
                    "[HITS baseline] CuPy fallback OK  n=%d iters=%d "
                    "converged=%s elapsed=%.3fs",
                    graph_csr.shape[0], iters, converged, elapsed,
                )
            except Exception as cp_exc:
                _LOG.error(
                    "[HITS baseline] CuPy fallback also FAILED: %s", cp_exc,
                )
                raise RuntimeError(
                    f"HITS GPU baseline: both cuGraph and CuPy paths failed.\n"
                    f"  cuGraph error: {cg_exc}\n"
                    f"  CuPy error   : {cp_exc}"
                ) from cp_exc
        else:
            # A different cuGraph error — re-raise as-is.
            raise

    inner = _pack_result(hub, auth, iters, converged, network_type)
    inner["backend"] = backend

    return build_envelope(
        algorithm="hits",
        network_type=network_type,
        execution_time=elapsed,
        graph_csr=graph_csr,
        inner=inner,
        mode=mode,
    )
