"""
src/algorithms/gpu/basic/hits.py
================================

HITS — GPU baseline implementation.

Backends
--------
    1. cuGraph ``cugraph.hits``  (preferred)
    2. CuPy power iteration on A and A.T  (fallback)
    3. RuntimeError                       (neither installed)

Result keys (mirror src/algorithms/gpu/cuda_optimized/hits.py)
-------------------------------------------------------------
    ppi          : hub_scores, authority_scores, top_nodes, iterations, converged
    grn / mirna  : hub_scores, authority_scores, iterations, converged,
                   top_hubs, top_authorities, hub_authority_overlap
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

from ._utils import (
    CUGRAPH_AVAILABLE, CUPY_AVAILABLE,
    TOP_HITS,
    build_envelope, cugraph_build_graph, cugraph_extract_column,
    cugraph_function, require_any_backend, symmetrize_for,
    top_k_global, to_cupy_csr, cupy_get,
)


_DEFAULT_PARAMS: dict = {
    "max_iter":     100,
    "tolerance":    1e-6,
    "network_type": "grn",
}


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
    top_h = top_k_global(hub, TOP_HITS)
    top_a = top_k_global(auth, TOP_HITS)
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
# cuGraph path
# ---------------------------------------------------------------------------

def _hits_cugraph(
    graph_csr: sp.csr_matrix, params: dict,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    hits = cugraph_function("hits")
    if hits is None:
        raise RuntimeError("cugraph.hits is not available.")

    nt = str(params.get("network_type", "grn")).lower()
    # PPI: symmetrize before handing to cuGraph (matches optimized version).
    csr_in = symmetrize_for(graph_csr, nt) if nt == "ppi" else graph_csr
    directed = nt != "ppi"
    G = cugraph_build_graph(csr_in, directed=directed, weighted=False)

    import inspect
    try:
        accepted = set(inspect.signature(hits).parameters.keys())
    except (TypeError, ValueError):
        accepted = set()

    kwargs: dict[str, Any] = {}
    if "max_iter" in accepted: kwargs["max_iter"] = int(params["max_iter"])
    if "tol"      in accepted: kwargs["tol"]      = float(params["tolerance"])
    if "tolerance"in accepted: kwargs["tolerance"]= float(params["tolerance"])

    df = hits(G, **kwargs)
    vertex = cugraph_extract_column(df, ["vertex", "node", "id"]).astype(np.int64)
    hubs   = cugraph_extract_column(df, ["hubs", "hub", "hub_score"]).astype(np.float32)
    auths  = cugraph_extract_column(df, ["authorities", "authority", "authority_score"]).astype(np.float32)

    n = int(graph_csr.shape[0])
    hub_arr  = np.zeros(n, dtype=np.float32)
    auth_arr = np.zeros(n, dtype=np.float32)
    np.put(hub_arr,  vertex, hubs)
    np.put(auth_arr, vertex, auths)
    return hub_arr, auth_arr, int(params["max_iter"]), True


# ---------------------------------------------------------------------------
# CuPy fallback — power iteration on A and A.T
# ---------------------------------------------------------------------------

def _hits_cupy(
    graph_csr: sp.csr_matrix, params: dict,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    import cupy as cp

    n = int(graph_csr.shape[0])
    if n == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32), 0, True

    max_iter  = int(params["max_iter"])
    tolerance = float(params["tolerance"])
    nt = str(params.get("network_type", "grn")).lower()

    A_cpu = symmetrize_for(graph_csr, nt) if nt == "ppi" else graph_csr.astype(np.float32)
    A_gpu  = to_cupy_csr(A_cpu)
    AT_gpu = to_cupy_csr(A_cpu.T.tocsr())

    hub  = cp.full((n,), 1.0 / max(1, n) ** 0.5, dtype=cp.float32)
    auth = cp.full((n,), 1.0 / max(1, n) ** 0.5, dtype=cp.float32)

    converged = False
    iterations = 0
    for it in range(max_iter):
        iterations = it + 1
        # authority = A^T @ hub  (normalise by L2)
        new_auth = AT_gpu.dot(hub)
        nrm = float(cp.sqrt(cp.sum(new_auth * new_auth)).get())
        if nrm > 0:
            new_auth = new_auth / cp.float32(nrm)

        # hub = A @ authority  (normalise by L2)
        new_hub = A_gpu.dot(new_auth)
        nrm = float(cp.sqrt(cp.sum(new_hub * new_hub)).get())
        if nrm > 0:
            new_hub = new_hub / cp.float32(nrm)

        delta_h = float(cp.sum(cp.abs(new_hub  - hub )).get())
        delta_a = float(cp.sum(cp.abs(new_auth - auth)).get())
        hub, auth = new_hub, new_auth
        if (delta_h + delta_a) < tolerance:
            converged = True
            break

    return cupy_get(hub).astype(np.float32), cupy_get(auth).astype(np.float32), iterations, converged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def hits_gpu_baseline(
    graph_csr: sp.csr_matrix, params: dict | None = None,
) -> dict:
    """HITS — GPU baseline.  cuGraph preferred, CuPy power iteration fallback."""
    require_any_backend("hits")
    p = {**_DEFAULT_PARAMS, **(params or {})}
    network_type = str(p.get("network_type", "grn")).lower()

    t0 = time.perf_counter()
    backend: str
    if CUGRAPH_AVAILABLE and cugraph_function("hits") is not None:
        try:
            hub, auth, iters, converged = _hits_cugraph(graph_csr, p)
            backend = "cugraph"
        except Exception as exc:                            # noqa: BLE001
            logging.warning(
                "hits_gpu_baseline: cuGraph path failed (%s) — falling back "
                "to CuPy.", exc,
            )
            if not CUPY_AVAILABLE:
                raise
            hub, auth, iters, converged = _hits_cupy(graph_csr, p)
            backend = "cupy"
    elif CUPY_AVAILABLE:
        hub, auth, iters, converged = _hits_cupy(graph_csr, p)
        backend = "cupy"
    else:
        raise RuntimeError("Unreachable — require_any_backend should have raised.")
    elapsed = time.perf_counter() - t0

    inner = _pack_result(hub, auth, iters, converged, network_type)
    inner["backend"] = backend

    return build_envelope(
        algorithm="hits",
        network_type=network_type,
        execution_time=elapsed,
        graph_csr=graph_csr,
        inner=inner,
    )
