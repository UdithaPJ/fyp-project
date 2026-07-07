"""
tests/test_visualization_network_type.py
========================================

Regression tests for network-type-aware visualisation of score-style
algorithm results (the PageRank GRN / miRNA regulator-vs-target split and
the HITS PPI hub/authority redundancy).

The motivating bug: a directed GRN PageRank result was rendered in a chart
titled "Top N regulators by PageRank" whose bars were actually the
top-scoring nodes overall — which in a GRN are dominated by heavily
regulated *target* genes (e.g. CDKN1A in TRRUST).  The viz layer discarded
the regulator/target distinction the algorithm computed.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.visualization.charts import make_score_chart_data
from src.visualization.graph_viz import make_highlight_data
from src.visualization.tables import make_top_nodes_table


def _grn_result(network_type: str = "grn") -> tuple[dict, sp.csr_matrix, dict]:
    """Build a tiny directed graph where a pure *target* outscores every TF.

    Nodes: 0=TF_A, 1=TF_B, 2=TF_C (regulators, out-degree > 0),
           3=TARGET (out-degree 0, regulated by all three TFs).
    TARGET is given the highest PageRank score to reproduce the CDKN1A case.
    """
    # Edges: 0->3, 1->3, 2->3  (all TFs regulate the target)
    rows = np.array([0, 1, 2])
    cols = np.array([3, 3, 3])
    data = np.ones(3, dtype=float)
    csr = sp.csr_matrix((data, (rows, cols)), shape=(4, 4))

    node_index_map = {"TF_A": 0, "TF_B": 1, "TF_C": 2, "TARGET": 3}
    scores = [0.10, 0.12, 0.15, 0.40]  # TARGET (idx 3) is highest
    reg_key = "top_mirnas" if network_type == "mirna" else "top_regulators"
    tgt_key = "top_target_genes" if network_type == "mirna" else "top_targets"
    result = {
        "algorithm": "pagerank",
        "network_type": network_type,
        "result": {
            "scores": scores,
            reg_key: [2, 1, 0],
            tgt_key: [3],
            "iterations": 10,
            "converged": True,
        },
    }
    return result, csr, node_index_map


def test_grn_pagerank_chart_ranks_regulators_not_targets():
    result, csr, nim = _grn_result("grn")
    chart = make_score_chart_data(result, nim, top_k=10, graph_csr=csr)

    # Title must not claim "regulators" while showing a target on top.
    assert "regulator" in chart["title"].lower()
    # The highest-scoring node is TARGET, but it must NOT appear — the chart
    # ranks among regulators only.
    assert "TARGET" not in chart["labels"]
    # The top bar is the highest-scoring regulator (TF_C, score 0.15).
    assert chart["labels"][0] == "TF_C"


def test_mirna_pagerank_chart_titles_mirnas():
    result, csr, nim = _grn_result("mirna")
    chart = make_score_chart_data(result, nim, top_k=10, graph_csr=csr)
    assert "mirna" in chart["title"].lower()
    assert "TARGET" not in chart["labels"]


def test_pagerank_chart_falls_back_to_honest_title_without_csr():
    """Without out-degrees the ranking can't split — must not lie."""
    result, _csr, nim = _grn_result("grn")
    chart = make_score_chart_data(result, nim, top_k=10, graph_csr=None)
    # Falls back to all-node ranking; title says "nodes", never "regulators".
    assert "regulator" not in chart["title"].lower()
    assert chart["labels"][0] == "TARGET"  # honest: top score shown as-is


def test_grn_pagerank_table_tags_target_role():
    result, csr, nim = _grn_result("grn")
    table = make_top_nodes_table(result, nim, top_k=10, graph_csr=csr)
    by_label = {row["node_label"]: row for row in table}
    # The #1 ranked row is the target, correctly labelled as such.
    assert table[0]["node_label"] == "TARGET"
    assert by_label["TARGET"]["role"] == "Target"
    assert by_label["TF_A"]["role"] == "Regulator"


def test_mirna_pagerank_table_uses_mirna_role_labels():
    result, csr, nim = _grn_result("mirna")
    table = make_top_nodes_table(result, nim, top_k=10, graph_csr=csr)
    by_label = {row["node_label"]: row for row in table}
    assert by_label["TARGET"]["role"] == "Target gene"
    assert by_label["TF_A"]["role"] == "miRNA"


def test_grn_pagerank_graph_highlights_regulators():
    result, csr, nim = _grn_result("grn")
    viz = make_highlight_data(result, csr, nim, max_nodes=200)
    highlighted = {n["id"] for n in viz["nodes"] if n["highlight"]}
    # The target must not be highlighted as an important driver.
    assert "TARGET" not in highlighted
    assert highlighted  # at least one regulator highlighted
    assert highlighted <= {"TF_A", "TF_B", "TF_C"}


def test_ppi_pagerank_chart_ranks_all_nodes():
    """PPI is undirected — all nodes are peers, top score legitimately wins."""
    result, csr, nim = _grn_result("grn")
    result["network_type"] = "ppi"
    result["result"] = {
        "scores": [0.10, 0.12, 0.15, 0.40],
        "top_nodes": [3, 2, 1, 0],
        "iterations": 10,
        "converged": True,
    }
    chart = make_score_chart_data(result, nim, top_k=10, graph_csr=csr)
    assert "regulator" not in chart["title"].lower()
    assert chart["labels"][0] == "TARGET"  # highest score, and that's correct
    # No role column on PPI tables.
    table = make_top_nodes_table(result, nim, top_k=10, graph_csr=csr)
    assert all("role" not in row for row in table)


def test_ppi_hits_chart_drops_redundant_authority_series():
    """PPI HITS symmetrises → hub == authority; the duplicate series is noise."""
    result = {
        "algorithm": "hits",
        "network_type": "ppi",
        "result": {
            "hub_scores": [0.1, 0.2, 0.3, 0.4],
            "authority_scores": [0.1, 0.2, 0.3, 0.4],
            "top_nodes": [3, 2, 1, 0],
        },
    }
    nim = {"A": 0, "B": 1, "C": 2, "D": 3}
    chart = make_score_chart_data(result, nim, top_k=10)
    assert "series_auth" not in chart
    assert "node" in chart["title"].lower()


def test_grn_hits_chart_keeps_hub_and_authority_series():
    result = {
        "algorithm": "hits",
        "network_type": "grn",
        "result": {
            "hub_scores": [0.4, 0.3, 0.2, 0.0],
            "authority_scores": [0.0, 0.1, 0.2, 0.4],
            "top_hubs": [0, 1, 2],
            "top_authorities": [3, 2, 1],
            "hub_authority_overlap": [],
        },
    }
    nim = {"A": 0, "B": 1, "C": 2, "D": 3}
    chart = make_score_chart_data(result, nim, top_k=10)
    assert "series_auth" in chart
    assert "hub" in chart["title"].lower()
