"""
src/algorithms/gpu/basic/pagerank.py
====================================

PageRank — GPU baseline implementation.

Backend
-------
cuGraph ``cugraph.pagerank`` only.  Raises ``ImportError`` immediately
on module import if cuGraph / cuDF (RAPIDS) are not installed.  There is
no CuPy fallback — use the CuPy-backed implementations in
``src/algorithms/cpu/`` if RAPIDS is unavailable, or the custom-kernel
version in ``src/algorithms/gpu/cuda_optimized/``.

Result keys match the optimized implementation exactly so the two are
schema-comparable:

    grn   : scores, top_regulators, top_targets, iterations, converged
    ppi   : scores, top_nodes,                   iterations, converged
    mirna : scores, top_mirnas, top_target_genes, iterations, converged
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.sparse as sp

# Hard-fail: cuGraph + cuDF are required.  This intentionally raises
# ImportError at import time (not at call time) so benchmarking code
# fails early with a clear message rather than silently using a different
# backend.
try:
    import cugraph   # noqa: F401
    import cudf      # noqa: F401
except ImportError as _e:
    raise ImportError(
        "src/algorithms/gpu/basic/pagerank.py requires cuGraph and cuDF "
        "(RAPIDS).  Install via:\n"
        "  conda install -c rapidsai -c nvidia -c conda-forge "
        "rapids=24.02 python=3.10 cudatoolkit=11.8\n"
        f"Original error: {_e}"
    ) from _e

from ._utils import (
    BASELINE_MODE_CUGRAPH,
    TOP_NODES_PPI, TOP_REG, TOP_TGT,
    build_envelope, cugraph_build_graph, cugraph_extract_column,
    cugraph_function, out_in_degrees,
    top_k_among, top_k_global,
)


_DEFAULT_PARAMS: dict = {
    "damping":      0.85,
    "max_iter":     100,
    "tolerance":    1e-6,
    "network_type": "grn",
}


# ---------------------------------------------------------------------------
# Result packing — exactly mirrors src/algorithms/gpu/cuda_optimized/pagerank.py
# ---------------------------------------------------------------------------

def _pack_result(
    scores: np.ndarray,
    out_degrees: np.ndarray,
    iterations: int,
    converged: bool,
    network_type: str,
) -> dict:
    """Build the inner result dict in the network-type-specific shape."""
    nt = str(network_type).lower()
    scores_list = scores.astype(float).tolist()

    if nt == "ppi":
        return {
            "scores":     scores_list,
            "top_nodes":  top_k_global(scores, TOP_NODES_PPI),
            "iterations": iterations,
            "converged":  converged,
        }

    regulators = np.where(out_degrees > 0.0)[0]
    targets    = np.where(out_degrees == 0.0)[0]

    if nt == "mirna":
        return {
            "scores":           scores_list,
            "top_mirnas":       top_k_among(scores, regulators, TOP_REG),
            "top_target_genes": top_k_among(scores, targets,    TOP_TGT),
            "iterations":       iterations,
            "converged":        converged,
        }

    return {
        "scores":         scores_list,
        "top_regulators": top_k_among(scores, regulators, TOP_REG),
        "top_targets":    top_k_among(scores, targets,    TOP_TGT),
        "iterations":     iterations,
        "converged":      converged,
    }


# ---------------------------------------------------------------------------
# cuGraph path
# ---------------------------------------------------------------------------

def _pagerank_cugraph(
    graph_csr: sp.csr_matrix, params: dict,
) -> tuple[np.ndarray, int, bool]:
    """Run cuGraph PageRank.  Returns ``(scores, iterations, converged)``.

    RAPIDS 26.04 compatibility notes
    ---------------------------------
    * ``fail_on_nonconvergence=False`` is passed when accepted so that partial
      results are returned instead of ``FailedToConvergeError`` being raised.
    * Tolerance is floored at 1e-5 (``max(user_tol, 1e-5)``) to reduce
      spurious non-convergence on small benchmark graphs.
    * Some cuGraph versions return a ``(df, converged_bool)`` 2-tuple; this
      is unwrapped automatically.
    """
    pagerank = cugraph_function("pagerank")
    if pagerank is None:
        raise RuntimeError("cugraph.pagerank is not available in this RAPIDS version.")

    nt = str(params.get("network_type", "grn")).lower()
    directed = nt != "ppi"

    G = cugraph_build_graph(graph_csr, directed=directed, weighted=True)

    import inspect
    try:
        sig = inspect.signature(pagerank)
        accepted = set(sig.parameters.keys())
    except (TypeError, ValueError):
        accepted = set()

    # Floor tolerance at 1e-5 so strict user values don't cause FailedToConvergeError.
    tol_val = max(float(params.get("tolerance", 1e-6)), 1e-5)

    kwargs: dict[str, Any] = {}
    if "alpha"                  in accepted: kwargs["alpha"]                  = float(params["damping"])
    if "damping_factor"         in accepted: kwargs["damping_factor"]         = float(params["damping"])
    if "max_iter"               in accepted: kwargs["max_iter"]               = int(params["max_iter"])
    if "tol"                    in accepted: kwargs["tol"]                    = tol_val
    if "tolerance"              in accepted: kwargs["tolerance"]              = tol_val
    # RAPIDS 26.04+: return partial result instead of raising on non-convergence.
    if "fail_on_nonconvergence" in accepted: kwargs["fail_on_nonconvergence"] = False

    converged = True
    try:
        raw = pagerank(G, **kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        if "converge" in msg or "failed" in msg:
            # Retry without fail_on_nonconvergence in case the installed RAPIDS
            # version ignores the flag but still raises (older releases).
            kwargs_retry = {k: v for k, v in kwargs.items() if k != "fail_on_nonconvergence"}
            try:
                raw = pagerank(G, **kwargs_retry)
                converged = False
            except Exception:
                raise exc   # re-raise original if retry also fails
        else:
            raise

    # Some cuGraph releases return (df, converged_bool) instead of just df.
    if isinstance(raw, tuple) and len(raw) == 2 and not hasattr(raw, "columns"):
        df, _conv_flag = raw
        if isinstance(_conv_flag, bool):
            converged = bool(_conv_flag)
    else:
        df = raw

    score_col  = cugraph_extract_column(df, ["pagerank", "score", "scores"])
    vertex_col = cugraph_extract_column(df, ["vertex", "node", "id"])

    n = int(graph_csr.shape[0])
    scores = np.zeros(n, dtype=np.float32)
    np.put(scores, vertex_col.astype(np.int64), score_col.astype(np.float32))

    return scores, int(params["max_iter"]), converged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def pagerank_gpu_baseline(
    graph_csr: sp.csr_matrix, params: dict | None = None,
) -> dict:
    """PageRank — GPU baseline (cuGraph only).

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
        Adjacency matrix (directed for GRN/miRNA, undirected for PPI).
    params : dict, optional
        ``damping``, ``max_iter``, ``tolerance``, ``network_type``.

    Returns
    -------
    dict
        7-key standard result envelope; ``result["mode"]`` is
        ``"gpu_baseline_cugraph"``.
    """
    p = {**_DEFAULT_PARAMS, **(params or {})}
    network_type = str(p.get("network_type", "grn")).lower()

    out_deg, _ = out_in_degrees(graph_csr)

    t0 = time.perf_counter()
    scores, iterations, converged = _pagerank_cugraph(graph_csr, p)
    elapsed = time.perf_counter() - t0

    inner = _pack_result(
        scores=scores,
        out_degrees=out_deg,
        iterations=iterations,
        converged=converged,
        network_type=network_type,
    )
    inner["backend"] = "cugraph"

    return build_envelope(
        algorithm="pagerank",
        network_type=network_type,
        execution_time=elapsed,
        graph_csr=graph_csr,
        inner=inner,
        mode=BASELINE_MODE_CUGRAPH,
    )
