"""
src/visualization/graph_viz.py — Highlighted-graph data builder
=================================================================

Builds the data structure the frontend ``GraphView`` component needs to
render an algorithm-aware graph: nodes carry a score / community /
highlight flag, edges carry their weight.  Pure data — no layout
computation, no SVG, no rendering.

Output schema
-------------
    {
      "nodes": [
        {"id": str, "label": str, "score": float,
         "highlight": bool, "community": int or None},
        ...
      ],
      "edges": [
        {"source": str, "target": str, "weight": float},
        ...
      ],
      "node_count_capped": bool,   # True if max_nodes truncated the view
      "total_nodes_in_graph": int,
    }

The view is capped at ``max_nodes`` (default 200) so the frontend can
render the result of a 100k-node analysis without melting the browser.
Selection strategy:

  * Score-style algorithms (pagerank/hits/rwr): highest-scoring nodes
    survive.  Top 10 of those are flagged ``highlight=True``.
  * Cluster-style algorithms (louvain/mcl): largest communities first,
    distributing the budget proportionally to community size.
  * Cascade-style (bfs): every reachable node + the source, capped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp


_SCORE_ALGOS   = {"pagerank", "hits", "rwr"}
_CLUSTER_ALGOS = {"louvain", "mcl"}
_CASCADE_ALGOS = {"bfs"}


def _reverse_map(node_index_map: dict | None) -> dict[int, str]:
    if not node_index_map:
        return {}
    return {int(v): str(k) for k, v in node_index_map.items()}


def _empty_payload(reason: str, total_nodes: int = 0) -> dict:
    return {
        "nodes": [],
        "edges": [],
        "node_count_capped": False,
        "total_nodes_in_graph": int(total_nodes),
        "unsupported": True,
        "reason":      reason,
    }


# ---------------------------------------------------------------------------
# Per-algorithm node selection
# ---------------------------------------------------------------------------

def _select_nodes_score(
    inner: dict, algo: str, n_total: int, max_nodes: int
) -> tuple[list[int], dict[int, float], set[int]]:
    """Return (selected_indices, score_map, highlight_set) for score-style algorithms."""
    if algo == "hits":
        scores = inner.get("hub_scores", []) or []
    else:
        scores = inner.get("scores", []) or []

    if not scores or len(scores) != n_total:
        # Algorithm result is malformed for this graph — return all nodes
        # without scores rather than crashing.
        selected = list(range(min(n_total, max_nodes)))
        return selected, {i: 0.0 for i in selected}, set()

    scores_np = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores_np)[::-1]
    selected = order[:max_nodes].tolist()
    score_map = {int(i): float(scores_np[i]) for i in selected}
    highlight = set(int(i) for i in order[:10])
    return selected, score_map, highlight


def _select_nodes_cluster(
    inner: dict, algo: str, n_total: int, max_nodes: int
) -> tuple[list[int], dict[int, int], set[int]]:
    """Return (selected_indices, community_map, highlight_set) for clustering algorithms."""
    key = "community_assignments" if algo == "louvain" else "cluster_assignments"
    assignments = inner.get(key, []) or []
    if not assignments or len(assignments) != n_total:
        selected = list(range(min(n_total, max_nodes)))
        return selected, {}, set()

    assignments_np = np.asarray(assignments, dtype=np.int64)

    # Cluster sizes, sorted desc
    unique, counts = np.unique(assignments_np, return_counts=True)
    order = np.argsort(counts)[::-1]
    unique_sorted = unique[order]
    counts_sorted = counts[order]
    total = int(counts_sorted.sum())

    selected: list[int] = []
    community_map: dict[int, int] = {}
    for cid, size in zip(unique_sorted.tolist(), counts_sorted.tolist()):
        if len(selected) >= max_nodes:
            break
        # Budget for this community = ceil(max_nodes * size / total)
        budget = max(1, int(round(max_nodes * size / max(total, 1))))
        # Indices in this community
        members = np.where(assignments_np == cid)[0]
        take = members[:budget].tolist()
        for node_idx in take:
            if len(selected) >= max_nodes:
                break
            selected.append(int(node_idx))
            community_map[int(node_idx)] = int(cid)

    # Highlight nodes from the single largest community
    largest_cid = int(unique_sorted[0]) if len(unique_sorted) > 0 else -1
    highlight = {i for i, c in community_map.items() if c == largest_cid}
    return selected, community_map, highlight


def _select_nodes_cascade(
    inner: dict, params_source: int | None, n_total: int, max_nodes: int
) -> tuple[list[int], dict[int, float], set[int]]:
    """Return (selected_indices, depth_as_score_map, highlight_set) for BFS."""
    distances = inner.get("distances", []) or []
    if not distances or len(distances) != n_total:
        selected = list(range(min(n_total, max_nodes)))
        return selected, {}, set()

    distances_np = np.asarray(distances, dtype=np.int64)
    reachable = np.where(distances_np >= 0)[0]
    # Sort by depth ascending so the source and near layers come first
    reachable_sorted = reachable[np.argsort(distances_np[reachable])]
    selected = reachable_sorted[:max_nodes].tolist()
    # Use depth (negated, so smaller depth → higher 'score') for layered colouring
    score_map = {int(i): float(distances_np[i]) for i in selected}
    # Highlight the source node
    source_candidates = np.where(distances_np == 0)[0]
    highlight = {int(i) for i in source_candidates}
    return selected, score_map, highlight


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def make_highlight_data(
    result: dict,
    graph_csr: sp.csr_matrix,
    node_index_map: dict | None = None,
    max_nodes: int = 200,
) -> dict:
    """
    Build a node/edge payload for the frontend graph visualisation.

    Caps at ``max_nodes`` for performance.  Edges are restricted to those
    where BOTH endpoints survive the node cap.
    """
    n_total = int(graph_csr.shape[0]) if graph_csr is not None else 0
    if n_total == 0:
        return _empty_payload("graph has no nodes")

    algo = result.get("algorithm")
    inner = result.get("result", {}) or {}
    reverse = _reverse_map(node_index_map)

    score_map: dict[int, float] = {}
    community_map: dict[int, int] = {}
    highlight: set[int] = set()

    if algo in _SCORE_ALGOS:
        selected, score_map, highlight = _select_nodes_score(
            inner, algo, n_total, max_nodes
        )
    elif algo in _CLUSTER_ALGOS:
        selected, community_map, highlight = _select_nodes_cluster(
            inner, algo, n_total, max_nodes
        )
    elif algo in _CASCADE_ALGOS:
        selected, score_map, highlight = _select_nodes_cascade(
            inner, None, n_total, max_nodes
        )
    else:
        return _empty_payload(
            f"graph-viz not defined for algorithm '{algo}'", n_total
        )

    selected_set = set(int(i) for i in selected)

    nodes_payload = []
    for idx in selected:
        idx_int = int(idx)
        nodes_payload.append({
            "id":        reverse.get(idx_int, f"node_{idx_int}"),
            "label":     reverse.get(idx_int, f"node_{idx_int}"),
            "score":     float(score_map.get(idx_int, 0.0)),
            "highlight": bool(idx_int in highlight),
            "community": community_map.get(idx_int),  # None if not clustering
        })

    # ---- Edges restricted to surviving endpoints ----
    # Iterate the CSR rows for selected sources only — avoids touching the
    # full nnz of a huge graph.
    edges_payload: list[dict] = []
    indptr  = graph_csr.indptr
    indices = graph_csr.indices
    data    = graph_csr.data
    for src_idx in selected:
        s = int(indptr[src_idx])
        e = int(indptr[src_idx + 1])
        for k in range(s, e):
            tgt_idx = int(indices[k])
            if tgt_idx in selected_set:
                edges_payload.append({
                    "source": reverse.get(int(src_idx), f"node_{src_idx}"),
                    "target": reverse.get(tgt_idx,      f"node_{tgt_idx}"),
                    "weight": float(data[k]),
                })

    return {
        "nodes": nodes_payload,
        "edges": edges_payload,
        "node_count_capped":     bool(len(selected) < n_total),
        "total_nodes_in_graph":  int(n_total),
    }
