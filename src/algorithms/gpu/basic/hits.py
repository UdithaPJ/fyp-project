"""
src/algorithms/gpu/basic/hits.py
================================

HITS — GPU baseline implementation.

Backend
-------
cuGraph ``cugraph.hits`` only.  Raises ``ImportError`` immediately on
module import if cuGraph / cuDF (RAPIDS) are not installed.  There is
no CuPy fallback — the benchmarking baseline must use the cuGraph path.

Result keys (mirror src/algorithms/gpu/cuda_optimized/hits.py)
-------------------------------------------------------------
    ppi          : hub_scores, authority_scores, top_nodes, iterations, converged
    grn / mirna  : hub_scores, authority_scores, iterations, converged,
                   top_hubs, top_authorities, hub_authority_overlap
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.sparse as sp

# Hard-fail: cuGraph + cuDF are required.
try:
    import cugraph   # noqa: F401
    import cudf      # noqa: F401
except ImportError as _e:
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
        raise RuntimeError("cugraph.hits is not available in this RAPIDS version.")

    nt = str(params.get("network_type", "grn")).lower()
    csr_in = symmetrize_for(graph_csr, nt) if nt == "ppi" else graph_csr
    directed = nt != "ppi"
    G = cugraph_build_graph(csr_in, directed=directed, weighted=False)

    import inspect
    try:
        accepted = set(inspect.signature(hits).parameters.keys())
    except (TypeError, ValueError):
        accepted = set()

    kwargs: dict[str, Any] = {}
    if "max_iter"  in accepted: kwargs["max_iter"]  = int(params["max_iter"])
    if "tol"       in accepted: kwargs["tol"]       = float(params["tolerance"])
    if "tolerance" in accepted: kwargs["tolerance"] = float(params["tolerance"])

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
# Public entry point
# ---------------------------------------------------------------------------

def hits_gpu_baseline(
    graph_csr: sp.csr_matrix, params: dict | None = None,
) -> dict:
    """HITS — GPU baseline (cuGraph only).

    Returns
    -------
    dict
        7-key standard result envelope; ``result["mode"]`` is
        ``"gpu_baseline_cugraph"``.
    """
    p = {**_DEFAULT_PARAMS, **(params or {})}
    network_type = str(p.get("network_type", "grn")).lower()

    t0 = time.perf_counter()
    hub, auth, iters, converged = _hits_cugraph(graph_csr, p)
    elapsed = time.perf_counter() - t0

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
