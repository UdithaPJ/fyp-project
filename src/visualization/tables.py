"""
src/visualization/tables.py — Table builders for algorithm results
====================================================================

Pure data transformations.  No matplotlib, no file I/O — every function
returns a list of plain dicts that the frontend can render in a table.

After the runner has attached node labels (see
``src/runner/algorithm_runner.py``) the relevant fields look like
``[{"index": int, "label": str}, ...]``.  These functions also accept
the raw integer form (when called outside the runner) by checking each
entry's type at runtime — that way the visualization layer remains
useful in tests and notebooks too.
"""

from __future__ import annotations

from typing import Any

# Algorithms that support each table type
_SCORE_ALGOS   = {"pagerank", "hits", "rwr"}
_CLUSTER_ALGOS = {"louvain", "mcl"}
_CASCADE_ALGOS = {"bfs"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _is_label_dict(x: Any) -> bool:
    return isinstance(x, dict) and "index" in x and "label" in x


def _label_of(x: Any, reverse_map: dict | None = None) -> str:
    if _is_label_dict(x):
        return str(x["label"])
    if reverse_map is not None:
        return reverse_map.get(int(x), f"node_{x}")
    return f"node_{x}"


def _index_of(x: Any) -> int:
    if _is_label_dict(x):
        return int(x["index"])
    return int(x)


def _reverse_map(node_index_map: dict | None) -> dict[int, str]:
    if not node_index_map:
        return {}
    return {int(v): str(k) for k, v in node_index_map.items()}


def _unsupported(reason: str) -> list[dict]:
    """Return a single-row sentinel for unsupported algorithms."""
    return [{"unsupported": True, "reason": reason}]


# ---------------------------------------------------------------------------
# make_top_nodes_table — score-style algorithms
# ---------------------------------------------------------------------------

def make_top_nodes_table(
    result: dict,
    node_index_map: dict | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Build a ranked-node table from a score-producing algorithm result.

    Works for pagerank (``scores``), hits (``hub_scores`` + ``authority_scores``),
    and rwr (``scores``).  Returns a list of ``{"rank", "node_label", "score", ...}``
    dicts.  For HITS, hub and authority scores are both included on each row.

    Returns an "unsupported" sentinel for cluster/cascade algorithms.
    """
    algo = result.get("algorithm")
    if algo not in _SCORE_ALGOS:
        return _unsupported(
            f"top-nodes table is not defined for algorithm '{algo}'"
        )

    inner = result.get("result", {})
    reverse = _reverse_map(node_index_map)

    if algo == "hits":
        hub = inner.get("hub_scores", []) or []
        auth = inner.get("authority_scores", []) or []
        # Rank by hub score; include authority score on each row
        order = sorted(range(len(hub)), key=lambda i: hub[i], reverse=True)[:top_k]
        return [
            {
                "rank":            rank + 1,
                "node_index":      int(i),
                "node_label":      reverse.get(int(i), f"node_{i}"),
                "hub_score":       float(hub[i])  if i < len(hub)  else 0.0,
                "authority_score": float(auth[i]) if i < len(auth) else 0.0,
            }
            for rank, i in enumerate(order)
        ]

    # pagerank, rwr — single ``scores`` list
    scores = inner.get("scores", []) or []
    if not scores:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {
            "rank":       rank + 1,
            "node_index": int(i),
            "node_label": reverse.get(int(i), f"node_{i}"),
            "score":      float(scores[i]),
        }
        for rank, i in enumerate(order)
    ]


# ---------------------------------------------------------------------------
# make_cluster_table — community-detection algorithms
# ---------------------------------------------------------------------------

def make_cluster_table(
    result: dict,
    node_index_map: dict | None = None,
    top_member_count: int = 10,
) -> list[dict]:
    """
    Build a cluster summary table.

    Works for louvain (``top_communities`` + ``community_assignments``) and
    mcl (``cluster_assignments``).  Returns rows of
    ``{"cluster_id", "size", "top_members"}`` sorted by size desc.
    """
    algo = result.get("algorithm")
    if algo not in _CLUSTER_ALGOS:
        return _unsupported(
            f"cluster table is not defined for algorithm '{algo}'"
        )

    inner = result.get("result", {})
    reverse = _reverse_map(node_index_map)

    # Louvain: top_communities[] is the canonical structure
    if algo == "louvain":
        comms = inner.get("top_communities", []) or []
        return [
            {
                "cluster_id":  int(c.get("community_id", -1)),
                "size":        int(c.get("size", 0)),
                "top_members": [
                    _label_of(m, reverse)
                    for m in (c.get("member_nodes", []) or [])[:top_member_count]
                ],
            }
            for c in comms
        ]

    # MCL: build from cluster_assignments
    assignments = inner.get("cluster_assignments", []) or []
    if not assignments:
        return []

    cluster_to_members: dict[int, list[int]] = {}
    for node_idx, cluster_id in enumerate(assignments):
        cluster_to_members.setdefault(int(cluster_id), []).append(int(node_idx))

    # Sort clusters by size descending; cap top members
    sorted_clusters = sorted(
        cluster_to_members.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    return [
        {
            "cluster_id":  int(cid),
            "size":        len(members),
            "top_members": [
                reverse.get(int(i), f"node_{i}")
                for i in members[:top_member_count]
            ],
        }
        for cid, members in sorted_clusters
    ]


# ---------------------------------------------------------------------------
# make_cascade_table — BFS
# ---------------------------------------------------------------------------

def make_cascade_table(
    result: dict,
    node_index_map: dict | None = None,
) -> list[dict]:
    """
    Build a per-depth cascade table for BFS.

    Returns rows of ``{"depth", "num_nodes", "nodes"}`` sorted by depth asc.
    """
    algo = result.get("algorithm")
    if algo not in _CASCADE_ALGOS:
        return _unsupported(
            f"cascade table is not defined for algorithm '{algo}'"
        )

    inner = result.get("result", {})
    cascade = inner.get("cascade_by_depth", {}) or {}
    reverse = _reverse_map(node_index_map)

    rows = []
    for depth_key in sorted(cascade.keys(), key=lambda d: int(d)):
        nodes = cascade.get(depth_key, []) or []
        rows.append({
            "depth":     int(depth_key),
            "num_nodes": len(nodes),
            "nodes":     [reverse.get(int(i), f"node_{i}") for i in nodes],
        })
    return rows
