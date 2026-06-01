"""
src/algorithms/gpu/basic/rwr.py
===============================

Random Walk with Restart — GPU baseline implementation.

Backend
-------
cuGraph ``cugraph.personalized_pagerank`` only.  Raises ``ImportError``
immediately on module import if cuGraph / cuDF (RAPIDS) are not
installed.  There is no CuPy fallback — the benchmarking baseline must
use the cuGraph path.

The "restart" formulation of RWR is mathematically equivalent to
personalized PageRank with the seed nodes as the personalization
distribution and ``alpha = 1 - restart_prob``.

Result keys (mirror src/algorithms/gpu/cuda_optimized/rwr.py)
-------------------------------------------------------------
    scores, top_nodes, top_seeds, iterations, converged, note
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
        "src/algorithms/gpu/basic/rwr.py requires cuGraph and cuDF "
        "(RAPIDS).  Install via:\n"
        "  conda install -c rapidsai -c nvidia -c conda-forge "
        "rapids=24.02 python=3.10 cudatoolkit=11.8\n"
        f"Original error: {_e}"
    ) from _e

from ._utils import (
    BASELINE_MODE_CUGRAPH,
    TOP_RWR_NODES, TOP_RWR_SEEDS,
    build_envelope, cugraph_build_graph, cugraph_extract_column,
    cugraph_function, out_in_degrees,
    symmetrize_for, top_k_among, top_k_global,
)


_DEFAULT_PARAMS: dict = {
    "restart_prob": 0.3,
    "max_iter":     100,
    "tolerance":    1e-6,
    "seed_nodes":   [],
    "network_type": "grn",
}


# ---------------------------------------------------------------------------
# Seed normalisation (accepts ``[1,2,3]`` or ``[[1,2],[3,4]]``)
# ---------------------------------------------------------------------------

def _flatten_seeds(seed_nodes) -> list[int]:
    """Return a flat list of seed indices; supports single-set or multi-set."""
    if seed_nodes is None:
        return []
    if isinstance(seed_nodes, (list, tuple, np.ndarray)) and len(seed_nodes) > 0:
        first = seed_nodes[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            flat: list[int] = []
            for sub in seed_nodes:
                flat.extend(int(x) for x in sub)
            return flat
    return [int(x) for x in seed_nodes]


# ---------------------------------------------------------------------------
# cuGraph path — personalized PageRank
# ---------------------------------------------------------------------------

def _rwr_cugraph(
    graph_csr: sp.csr_matrix, params: dict, seeds: list[int],
) -> tuple[np.ndarray, int, bool, str]:
    import cudf as _cudf

    ppr = cugraph_function("personalized_pagerank")
    if ppr is None:
        # Some RAPIDS releases route through a single ``pagerank`` with a
        # ``personalization`` argument instead.
        pr = cugraph_function("pagerank")
        if pr is None:
            raise RuntimeError(
                "Neither cugraph.personalized_pagerank nor cugraph.pagerank "
                "is available in this RAPIDS version."
            )
        ppr = pr

    nt = str(params.get("network_type", "grn")).lower()
    directed = nt != "ppi"
    csr_in = symmetrize_for(graph_csr, nt) if nt == "ppi" else graph_csr
    G = cugraph_build_graph(csr_in, directed=directed, weighted=True)

    n = int(graph_csr.shape[0])
    restart_prob = float(params["restart_prob"])
    alpha = 1.0 - restart_prob

    if not seeds:
        personalization = None
    else:
        weight = 1.0 / float(len(seeds))
        personalization = _cudf.DataFrame({
            "vertex": np.asarray(seeds, dtype=np.int32),
            "values": np.full(len(seeds), weight, dtype=np.float32),
        })

    import inspect
    try:
        accepted = set(inspect.signature(ppr).parameters.keys())
    except (TypeError, ValueError):
        accepted = set()

    kwargs: dict[str, Any] = {}
    if "alpha"          in accepted: kwargs["alpha"]          = alpha
    if "damping_factor" in accepted: kwargs["damping_factor"] = alpha
    if "max_iter"       in accepted: kwargs["max_iter"]       = int(params["max_iter"])
    if "tol"            in accepted: kwargs["tol"]            = float(params["tolerance"])
    if "tolerance"      in accepted: kwargs["tolerance"]      = float(params["tolerance"])
    if personalization is not None and "personalization" in accepted:
        kwargs["personalization"] = personalization

    df = ppr(G, **kwargs)
    vertex = cugraph_extract_column(df, ["vertex", "node", "id"]).astype(np.int64)
    score  = cugraph_extract_column(df, ["pagerank", "score", "scores"]).astype(np.float32)

    scores = np.zeros(n, dtype=np.float32)
    np.put(scores, vertex, score)

    note = (
        f"cuGraph personalized_pagerank (alpha={alpha:.3f}, "
        f"seeds={len(seeds)})."
    )
    return scores, int(params["max_iter"]), True, note


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def rwr_gpu_baseline(
    graph_csr: sp.csr_matrix, params: dict | None = None,
) -> dict:
    """Random Walk with Restart — GPU baseline (cuGraph only).

    Returns
    -------
    dict
        7-key standard result envelope; ``result["mode"]`` is
        ``"gpu_baseline_cugraph"``.
    """
    p = {**_DEFAULT_PARAMS, **(params or {})}
    network_type = str(p.get("network_type", "grn")).lower()
    seeds = _flatten_seeds(p.get("seed_nodes", []))

    t0 = time.perf_counter()
    scores, iters, converged, note = _rwr_cugraph(graph_csr, p, seeds)
    elapsed = time.perf_counter() - t0

    top_nodes = top_k_global(scores, TOP_RWR_NODES)
    if seeds:
        seed_arr = np.asarray(sorted(set(seeds)), dtype=np.int64)
        top_seeds = top_k_among(scores, seed_arr, TOP_RWR_SEEDS)
    else:
        top_seeds = top_nodes[:TOP_RWR_SEEDS]

    inner = {
        "scores":     scores.astype(float).tolist(),
        "top_nodes":  top_nodes,
        "top_seeds":  top_seeds,
        "iterations": iters,
        "converged":  converged,
        "note":       f"{note}  Backend=cugraph.",
        "backend":    "cugraph",
    }

    return build_envelope(
        algorithm="rwr",
        network_type=network_type,
        execution_time=elapsed,
        graph_csr=graph_csr,
        inner=inner,
        mode=BASELINE_MODE_CUGRAPH,
    )
