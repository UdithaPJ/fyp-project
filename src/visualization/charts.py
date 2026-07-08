"""
src/visualization/charts.py — Chart data builders for algorithm results
=========================================================================

Pure data transformations.  No matplotlib — every function returns a
plain JSON-safe dict that the frontend renders with its chart library
(e.g. Recharts, Chart.js).

Output schema
-------------
    {
      "chart_type": "bar",
      "labels":     list[str],
      "values":     list[float],
      "title":      str,
      "x_label":    str,
      "y_label":    str,
      ...optional extras (series, units, etc.)
    }
"""

from __future__ import annotations

from typing import Any

import numpy as np

_SCORE_ALGOS   = {"pagerank", "hits", "rwr"}
_CLUSTER_ALGOS = {"louvain", "mcl"}


def _reverse_map(node_index_map: dict | None) -> dict[int, str]:
    if not node_index_map:
        return {}
    return {int(v): str(k) for k, v in node_index_map.items()}


def _out_degrees(graph_csr: Any, n: int) -> np.ndarray | None:
    """Return a length-``n`` out-degree vector, or ``None`` if unavailable.

    Used to split a directed GRN / miRNA graph into regulators
    (out-degree > 0) and pure targets (out-degree == 0) so PageRank
    visualisations can honour the network-type semantics.  Weighted sum so
    weighted graphs are handled; a positive value means the node has at
    least one out-edge.
    """
    if graph_csr is None:
        return None
    try:
        if graph_csr.shape[0] != n:
            return None
        return np.asarray(graph_csr.sum(axis=1)).ravel()
    except Exception:  # noqa: BLE001 - defensive: never break viz on a bad csr
        return None


def _unsupported(reason: str) -> dict:
    return {
        "chart_type": "bar",
        "labels":     [],
        "values":     [],
        "title":      "",
        "x_label":    "",
        "y_label":    "",
        "unsupported": True,
        "reason":     reason,
    }


# ---------------------------------------------------------------------------
# make_score_chart_data — top-k score bar chart
# ---------------------------------------------------------------------------

def make_score_chart_data(
    result: dict,
    node_index_map: dict | None = None,
    top_k: int = 20,
    graph_csr: Any = None,
) -> dict:
    """
    Build a top-K bar chart from a score-producing algorithm result.

    Works for pagerank, hits (hub scores), rwr.  HITS additionally
    exposes an authority-score series under ``series_auth``.

    Network-type aware: for a directed GRN / miRNA PageRank result the bars
    are ranked among **regulators** (out-degree > 0) — the top TFs / miRNAs —
    and the title reflects that, rather than presenting the top-scoring nodes
    overall (which in a GRN are dominated by heavily-regulated *target*
    genes).  ``graph_csr`` supplies the out-degrees needed for that split;
    when it is absent the ranking falls back to all nodes with an honest
    "nodes" title (never the misleading "regulators" label).
    """
    algo = result.get("algorithm")
    if algo not in _SCORE_ALGOS:
        return _unsupported(
            f"score chart is not defined for algorithm '{algo}'"
        )

    inner = result.get("result", {})
    reverse = _reverse_map(node_index_map)
    network_type = str(result.get("network_type", "grn")).lower()

    if algo == "hits":
        hub  = inner.get("hub_scores", []) or []
        auth = inner.get("authority_scores", []) or []
        if not hub:
            return _unsupported("hits result has no hub_scores")
        order = sorted(range(len(hub)), key=lambda i: hub[i], reverse=True)[:top_k]
        labels = [reverse.get(int(i), f"node_{i}") for i in order]
        # PPI symmetrises the graph, so hub_scores == authority_scores; a
        # second identical series would just be visual noise.  Only GRN /
        # miRNA (directed) carry a meaningful hub-vs-authority distinction.
        if network_type == "ppi":
            return {
                "chart_type": "bar",
                "labels":     labels,
                "values":     [float(hub[i]) for i in order],
                "title":      f"Top {len(order)} nodes (HITS)",
                "x_label":    "Node",
                "y_label":    "Centrality score",
            }
        return {
            "chart_type": "bar",
            "labels":     labels,
            "values":     [float(hub[i]) for i in order],
            "series_auth": [float(auth[i]) if i < len(auth) else 0.0 for i in order],
            "title":      f"Top {len(order)} hubs (HITS)",
            "x_label":    "Node",
            "y_label":    "Hub score",
        }

    scores = inner.get("scores", []) or []
    if not scores:
        return _unsupported(f"{algo} result has no scores")

    if algo == "pagerank":
        return _pagerank_chart(scores, reverse, top_k, network_type, graph_csr)

    # rwr — proximity to the seed set; no regulator/target semantics.
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return {
        "chart_type": "bar",
        "labels":     [reverse.get(int(i), f"node_{i}") for i in order],
        "values":     [float(scores[i]) for i in order],
        "title":      f"Top {len(order)} nodes by RWR proximity",
        "x_label":    "Node",
        "y_label":    "Score",
    }


def _pagerank_chart(
    scores: list[float],
    reverse: dict[int, str],
    top_k: int,
    network_type: str,
    graph_csr: Any,
) -> dict:
    """Network-type-aware PageRank bar chart (see ``make_score_chart_data``)."""
    n = len(scores)

    # PPI: undirected, all nodes are peers — rank everything.
    if network_type == "ppi":
        candidates: list[int] = list(range(n))
        noun = "nodes"
    else:
        # GRN / miRNA: rank among regulators (out-degree > 0) so the chart
        # actually shows master TFs / miRNAs, matching its title.
        deg = _out_degrees(graph_csr, n)
        if deg is None:
            candidates = list(range(n))
            noun = "nodes"
        else:
            candidates = [i for i in range(n) if deg[i] > 0]
            noun = "miRNAs" if network_type == "mirna" else "regulators"
            if not candidates:  # degenerate graph — don't return an empty chart
                candidates = list(range(n))
                noun = "nodes"

    order = sorted(candidates, key=lambda i: scores[i], reverse=True)[:top_k]
    return {
        "chart_type": "bar",
        "labels":     [reverse.get(int(i), f"node_{i}") for i in order],
        "values":     [float(scores[i]) for i in order],
        "title":      f"Top {len(order)} {noun} by PageRank",
        "x_label":    "Node",
        "y_label":    "Score",
    }


# ---------------------------------------------------------------------------
# make_cluster_size_chart_data — community/cluster size bar chart
# ---------------------------------------------------------------------------

def make_cluster_size_chart_data(result: dict) -> dict:
    """
    Build a bar chart of cluster sizes for community-detection algorithms.

    For Louvain, uses the precomputed ``top_communities`` list when present
    (already capped at 5).  For MCL, builds sizes from
    ``cluster_assignments``.  Returns all cluster sizes — the frontend can
    further filter / paginate if needed.
    """
    algo = result.get("algorithm")
    if algo not in _CLUSTER_ALGOS:
        return _unsupported(
            f"cluster-size chart is not defined for algorithm '{algo}'"
        )

    inner = result.get("result", {})

    if algo == "louvain":
        comms = inner.get("top_communities", []) or []
        if not comms:
            # Fall back to computing from community_assignments
            assignments = inner.get("community_assignments", []) or []
            if not assignments:
                return _unsupported("louvain result has no communities")
            counts: dict[int, int] = {}
            for c in assignments:
                counts[int(c)] = counts.get(int(c), 0) + 1
            sorted_pairs = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            return {
                "chart_type": "bar",
                "labels":     [f"Community {c}" for c, _ in sorted_pairs],
                "values":     [int(s) for _, s in sorted_pairs],
                "title":      "Community sizes (Louvain)",
                "x_label":    "Community",
                "y_label":    "Member count",
            }
        return {
            "chart_type": "bar",
            "labels":     [f"Community {int(c.get('community_id', -1))}" for c in comms],
            "values":     [int(c.get("size", 0)) for c in comms],
            "title":      "Top communities (Louvain)",
            "x_label":    "Community",
            "y_label":    "Member count",
        }

    # MCL
    assignments = inner.get("cluster_assignments", []) or []
    if not assignments:
        return _unsupported("mcl result has no cluster_assignments")
    counts: dict[int, int] = {}
    for c in assignments:
        counts[int(c)] = counts.get(int(c), 0) + 1
    sorted_pairs = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "chart_type": "bar",
        "labels":     [f"Cluster {c}" for c, _ in sorted_pairs],
        "values":     [int(s) for _, s in sorted_pairs],
        "title":      "Cluster sizes (MCL)",
        "x_label":    "Cluster",
        "y_label":    "Member count",
    }
