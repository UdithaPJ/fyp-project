from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.visualization.graph_viz import make_highlight_data


def _pagerank_result(scores: list[float]) -> dict:
    return {
        "algorithm": "pagerank",
        "mode": "gpu",
        "network_type": "grn",
        "execution_time": 0.0,
        "num_nodes": len(scores),
        "num_edges": 0,
        "result": {
            "scores": scores,
            "top_regulators": [],
            "top_targets": [],
            "iterations": 1,
            "converged": True,
        },
    }


def test_make_highlight_data_caps_edges_and_reports_totals():
    graph = sp.csr_matrix(np.ones((6, 6), dtype=np.float32))
    labels = {f"gene_{i}": i for i in range(6)}

    payload = make_highlight_data(
        _pagerank_result([0.1, 0.6, 0.2, 0.5, 0.4, 0.3]),
        graph,
        labels,
        max_nodes=4,
        max_edges=3,
    )

    assert len(payload["nodes"]) == 4
    assert len(payload["edges"]) == 3
    assert payload["node_count_capped"] is True
    assert payload["edge_count_capped"] is True
    assert payload["total_nodes_in_graph"] == 6
    assert payload["total_edges_in_graph"] == 36
    assert payload["rendered_edge_limit"] == 3
    assert {edge["source"] for edge in payload["edges"]} <= {
        node["id"] for node in payload["nodes"]
    }
    assert {edge["target"] for edge in payload["edges"]} <= {
        node["id"] for node in payload["nodes"]
    }


def test_make_highlight_data_marks_top_score_nodes():
    graph = sp.eye(4, dtype=np.float32, format="csr")
    labels = {f"node_{i}": i for i in range(4)}

    payload = make_highlight_data(
        _pagerank_result([0.1, 0.9, 0.4, 0.7]),
        graph,
        labels,
        max_nodes=4,
        max_edges=10,
    )

    assert [node["id"] for node in payload["nodes"]] == [
        "node_1",
        "node_3",
        "node_2",
        "node_0",
    ]
    assert all(node["highlight"] for node in payload["nodes"])
    assert payload["edge_count_capped"] is False
