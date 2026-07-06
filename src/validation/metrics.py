"""
src/validation/metrics.py
=========================

Pure metric functions for comparing two algorithm result dicts.

Every function takes two result-inner dicts ``a`` and ``b`` (the
``result["result"]`` payload returned by ``run_algorithm`` / direct CPU
calls / GPU baselines) and returns a flat ``{metric_name: value}`` dict
of float / int metrics.

These functions are intentionally pure — they NEVER mutate or re-run
algorithms.  All algorithm orchestration happens in
``cross_implementation_validation.py``.

Per-algorithm metric sets
-------------------------
    PageRank : Spearman, Pearson, MAE, RMSE, top-10 overlap
    BFS      : Exact distance match %, reachable-node agreement
    HITS     : Hub / authority Spearman + Pearson
    Louvain  : Modularity difference, NMI, ARI
    RWR      : Spearman, Pearson, MAE, RMSE, top-10 overlap
    MCL      : NMI, ARI, cluster-count difference

The dispatcher ``compute_metrics(algorithm, a, b)`` selects the right
function based on the algorithm name.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from scipy import stats as _stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unwrap(result: Any) -> dict:
    """Accept either the full envelope or the inner result dict."""
    if isinstance(result, dict) and isinstance(result.get("result"), dict):
        return result["result"]
    if isinstance(result, dict) and isinstance(result.get("output"), dict):
        return _unwrap(result["output"])
    if not isinstance(result, dict):
        raise TypeError(
            f"metrics: expected a dict result, got {type(result).__name__}"
        )
    return result


def _as_indices(value: Any) -> list[int]:
    """Coerce a list that may contain ints OR {'index': int, 'label': str}
    dicts (the runner attaches labels) into a flat list of ints.
    """
    out: list[int] = []
    if value is None:
        return out
    for v in value:
        if isinstance(v, dict) and "index" in v:
            out.append(int(v["index"]))
        else:
            out.append(int(v))
    return out


def _as_float_array(value: Iterable) -> np.ndarray:
    arr = np.asarray(list(value), dtype=np.float64)
    return arr


def _top_overlap(a: Iterable[int], b: Iterable[int], k: int) -> float:
    """Jaccard-style overlap on the top-K node-index sets.

    Returns the fraction of common elements over the size of the larger
    set, clipped to ``k`` if both lists are longer.  Matches the formula
    used by ``experiments/validation/compare_cpu_gpu_raw.py::top_overlap``
    but with an explicit ``k`` cap.
    """
    sa = set(list(a)[:k])
    sb = set(list(b)[:k])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), len(sb))


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    """Return scipy stats Spearman / Pearson correlation, robust to
    constant inputs (returns 1.0 if both vectors are identically
    constant, 0.0 if only one is).
    """
    if x.size != y.size or x.size < 2:
        return float("nan")
    if np.all(x == x[0]) and np.all(y == y[0]):
        return 1.0 if math.isclose(float(x[0]), float(y[0])) else 0.0
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    if kind == "spearman":
        r, _ = _stats.spearmanr(x, y)
    elif kind == "pearson":
        r, _ = _stats.pearsonr(x, y)
    else:
        raise ValueError(f"unknown correlation kind: {kind!r}")
    return float(r) if np.isfinite(r) else float("nan")


def _mae(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x - y)))


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((x - y) ** 2)))


def _align_scores(a: dict, b: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    xa = _as_float_array(a.get(key, []))
    xb = _as_float_array(b.get(key, []))
    n = min(xa.size, xb.size)
    return xa[:n], xb[:n]


# ---------------------------------------------------------------------------
# Clustering metrics — NMI / ARI implemented in pure numpy
# ---------------------------------------------------------------------------

def _contingency(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    """Dense contingency table (rows = labels_a, cols = labels_b)."""
    ua, ia = np.unique(labels_a, return_inverse=True)
    ub, ib = np.unique(labels_b, return_inverse=True)
    n_a, n_b = ua.size, ub.size
    table = np.zeros((n_a, n_b), dtype=np.int64)
    np.add.at(table, (ia, ib), 1)
    return table


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def normalized_mutual_info(a: list[int] | np.ndarray,
                           b: list[int] | np.ndarray) -> float:
    """Symmetric NMI (arithmetic-mean normaliser).

    Returns 1.0 for identical partitions, ~0.0 for independent ones.
    Matches sklearn's ``normalized_mutual_info_score(..., average='arithmetic')``.
    """
    a_arr = np.asarray(list(a), dtype=np.int64)
    b_arr = np.asarray(list(b), dtype=np.int64)
    if a_arr.size != b_arr.size or a_arr.size == 0:
        return float("nan")

    table = _contingency(a_arr, b_arr)
    total = float(table.sum())
    if total <= 0:
        return float("nan")

    pij = table / total
    pi = table.sum(axis=1, keepdims=True) / total
    pj = table.sum(axis=0, keepdims=True) / total

    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.log(pij / (pi * pj))
    log_term[~np.isfinite(log_term)] = 0.0
    mi = float((pij * log_term).sum())

    h_a = _entropy(table.sum(axis=1))
    h_b = _entropy(table.sum(axis=0))
    denom = 0.5 * (h_a + h_b)
    if denom <= 0:
        return 1.0 if mi == 0 else 0.0
    return float(max(0.0, min(1.0, mi / denom)))


def adjusted_rand_index(a: list[int] | np.ndarray,
                        b: list[int] | np.ndarray) -> float:
    """Adjusted Rand Index — chance-corrected partition agreement.

    Returns 1.0 for identical partitions, ~0.0 for random partitions,
    and can be slightly negative for worse-than-random agreement.
    Matches sklearn's ``adjusted_rand_score``.
    """
    a_arr = np.asarray(list(a), dtype=np.int64)
    b_arr = np.asarray(list(b), dtype=np.int64)
    if a_arr.size != b_arr.size or a_arr.size == 0:
        return float("nan")

    table = _contingency(a_arr, b_arr).astype(np.float64)
    n = float(table.sum())
    if n <= 1:
        return float("nan")

    def _comb2(x: np.ndarray) -> np.ndarray:
        return x * (x - 1.0) / 2.0

    sum_comb = _comb2(table).sum()
    sum_a = _comb2(table.sum(axis=1)).sum()
    sum_b = _comb2(table.sum(axis=0)).sum()
    comb_n = _comb2(np.array([n]))[0]

    expected = sum_a * sum_b / comb_n if comb_n > 0 else 0.0
    maximum = 0.5 * (sum_a + sum_b)
    denom = maximum - expected
    if denom == 0:
        # Both partitions are trivial (one cluster) — define ARI = 1
        return 1.0
    return float((sum_comb - expected) / denom)


# ---------------------------------------------------------------------------
# Per-algorithm metric functions
# ---------------------------------------------------------------------------

def pagerank_metrics(a: dict, b: dict, k: int = 10) -> dict:
    """Score-vector comparison metrics for PageRank result dicts."""
    a, b = _unwrap(a), _unwrap(b)
    xa, xb = _align_scores(a, b, "scores")

    spearman = _safe_corr(xa, xb, "spearman")
    pearson  = _safe_corr(xa, xb, "pearson")
    mae      = _mae(xa, xb)
    rmse     = _rmse(xa, xb)

    # top-K overlap: prefer top_nodes; fall back to top_regulators (GRN/mirna)
    top_a = (_as_indices(a.get("top_nodes"))
             or _as_indices(a.get("top_regulators")))
    top_b = (_as_indices(b.get("top_nodes"))
             or _as_indices(b.get("top_regulators")))
    if not top_a or not top_b:
        # synthesise top-K from scores when missing
        if xa.size:
            top_a = list(np.argsort(xa)[::-1][:k])
        if xb.size:
            top_b = list(np.argsort(xb)[::-1][:k])

    return {
        "spearman":      spearman,
        "pearson":       pearson,
        "mae":           mae,
        "rmse":          rmse,
        "top10_overlap": _top_overlap(top_a, top_b, k),
    }


def rwr_metrics(a: dict, b: dict, k: int = 10) -> dict:
    """Score-vector comparison metrics for RWR result dicts."""
    a, b = _unwrap(a), _unwrap(b)
    xa, xb = _align_scores(a, b, "scores")

    spearman = _safe_corr(xa, xb, "spearman")
    pearson  = _safe_corr(xa, xb, "pearson")
    mae      = _mae(xa, xb)
    rmse     = _rmse(xa, xb)

    top_a = _as_indices(a.get("top_nodes"))
    top_b = _as_indices(b.get("top_nodes"))
    if not top_a or not top_b:
        if xa.size:
            top_a = list(np.argsort(xa)[::-1][:k])
        if xb.size:
            top_b = list(np.argsort(xb)[::-1][:k])

    return {
        "spearman":      spearman,
        "pearson":       pearson,
        "mae":           mae,
        "rmse":          rmse,
        "top10_overlap": _top_overlap(top_a, top_b, k),
    }


def hits_metrics(a: dict, b: dict) -> dict:
    """Hub & authority correlation metrics for HITS result dicts."""
    a, b = _unwrap(a), _unwrap(b)
    hub_a, hub_b = _align_scores(a, b, "hub_scores")
    auth_a, auth_b = _align_scores(a, b, "authority_scores")
    return {
        "hub_spearman":  _safe_corr(hub_a, hub_b, "spearman"),
        "hub_pearson":   _safe_corr(hub_a, hub_b, "pearson"),
        "hub_mae":       _mae(hub_a, hub_b),
        "auth_spearman": _safe_corr(auth_a, auth_b, "spearman"),
        "auth_pearson":  _safe_corr(auth_a, auth_b, "pearson"),
        "auth_mae":      _mae(auth_a, auth_b),
    }


def bfs_metrics(a: dict, b: dict) -> dict:
    """Exact-match and reachability agreement for BFS distance arrays.

    BFS is deterministic so an optimised implementation should match the
    CPU baseline to the integer; any disagreement is a correctness bug.
    """
    a, b = _unwrap(a), _unwrap(b)
    da = np.asarray(a.get("distances", []), dtype=np.int64)
    db = np.asarray(b.get("distances", []), dtype=np.int64)
    n = min(da.size, db.size)
    da, db = da[:n], db[:n]

    if n == 0:
        return {
            "exact_match_pct":        float("nan"),
            "reachable_agreement":    float("nan"),
            "reachable_diff":         0,
        }

    UNREACHED = -1   # convention used by every CPU/GPU BFS in this repo
    reach_a = da != UNREACHED
    reach_b = db != UNREACHED

    # Exact match: distance equal at every node
    exact_pct = float(np.mean(da == db))
    # Reachable-set agreement: fraction of nodes where both implementations
    # agree on reachability (regardless of the exact distance)
    reach_agree = float(np.mean(reach_a == reach_b))

    n_reach_a = int(a.get("num_reachable", int(reach_a.sum())))
    n_reach_b = int(b.get("num_reachable", int(reach_b.sum())))

    return {
        "exact_match_pct":        exact_pct,
        "reachable_agreement":    reach_agree,
        "reachable_diff":         int(abs(n_reach_a - n_reach_b)),
    }


def louvain_metrics(a: dict, b: dict) -> dict:
    """Partition agreement metrics for Louvain result dicts."""
    a, b = _unwrap(a), _unwrap(b)
    la = np.asarray(a.get("community_assignments", []), dtype=np.int64)
    lb = np.asarray(b.get("community_assignments", []), dtype=np.int64)
    n = min(la.size, lb.size)
    la, lb = la[:n], lb[:n]

    mod_a = float(a.get("modularity", float("nan")))
    mod_b = float(b.get("modularity", float("nan")))
    mod_diff = (abs(mod_a - mod_b)
                if math.isfinite(mod_a) and math.isfinite(mod_b)
                else float("nan"))

    return {
        "nmi":                normalized_mutual_info(la, lb),
        "ari":                adjusted_rand_index(la, lb),
        "modularity_diff":    mod_diff,
        "num_communities_a":  int(a.get("num_communities", len(np.unique(la)) if la.size else 0)),
        "num_communities_b":  int(b.get("num_communities", len(np.unique(lb)) if lb.size else 0)),
    }


def mcl_metrics(a: dict, b: dict) -> dict:
    """Partition agreement metrics for MCL result dicts."""
    a, b = _unwrap(a), _unwrap(b)
    la = np.asarray(a.get("cluster_assignments", []), dtype=np.int64)
    lb = np.asarray(b.get("cluster_assignments", []), dtype=np.int64)
    n = min(la.size, lb.size)
    la, lb = la[:n], lb[:n]

    na = int(a.get("num_clusters", len(np.unique(la)) if la.size else 0))
    nb = int(b.get("num_clusters", len(np.unique(lb)) if lb.size else 0))

    return {
        "nmi":                  normalized_mutual_info(la, lb),
        "ari":                  adjusted_rand_index(la, lb),
        "cluster_count_diff":   int(abs(na - nb)),
        "num_clusters_a":       na,
        "num_clusters_b":       nb,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_METRIC_DISPATCH = {
    "pagerank": pagerank_metrics,
    "rwr":      rwr_metrics,
    "hits":     hits_metrics,
    "bfs":      bfs_metrics,
    "louvain":  louvain_metrics,
    "mcl":      mcl_metrics,
}


def compute_metrics(algorithm: str, a: dict, b: dict) -> dict:
    """Dispatch to the right metric function based on algorithm name.

    Accepts either a full envelope dict (``{"algorithm", "mode",
    "result", ...}``) or the inner ``result`` dict directly.
    """
    algo = str(algorithm).lower()
    fn = _METRIC_DISPATCH.get(algo)
    if fn is None:
        raise ValueError(
            f"compute_metrics: unknown algorithm {algorithm!r}.  "
            f"Known: {sorted(_METRIC_DISPATCH.keys())}"
        )
    return fn(a, b)


__all__ = [
    "compute_metrics",
    "pagerank_metrics",
    "rwr_metrics",
    "hits_metrics",
    "bfs_metrics",
    "louvain_metrics",
    "mcl_metrics",
    "normalized_mutual_info",
    "adjusted_rand_index",
]
