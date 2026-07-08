"""Regression tests for BFS/RWR node-name typeahead data."""

from __future__ import annotations

import pandas as pd

from webapp.backend.routes.graph import graph_nodes
from webapp.backend.services.dataset_store import dataset_store
from webapp.backend.services.preprocessing_service import preprocessing_service


def _payload(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def test_graph_nodes_uses_cached_node_index_after_dataframe_cleanup():
    """Node search must keep working after preprocessing drops the raw dataframe."""

    dataframe = pd.DataFrame(
        {
            "source": ["TP53", "TP53", "MYC"],
            "target": ["BRCA1", "EGFR", "EGFR"],
            "weight": [1.0, 2.0, 1.0],
        }
    )
    record = dataset_store.save("tiny_network.csv", dataframe)

    preprocessing_service.preprocess(
        upload_id=record.upload_id,
        mapping={"source": "source", "target": "target", "weight": "weight"},
        duplicate_method="mean",
    )

    stored = dataset_store.get(record.upload_id)
    assert stored.dataframe is None
    assert stored.node_index_map

    response = graph_nodes(record.upload_id, search="", limit=50)
    nodes = _payload(response)["nodes"]

    assert nodes == [
        {"index": 1, "label": "BRCA1"},
        {"index": 2, "label": "EGFR"},
        {"index": 3, "label": "MYC"},
        {"index": 0, "label": "TP53"},
    ]


def test_graph_nodes_filters_by_label_substring():
    dataframe = pd.DataFrame(
        {
            "source": ["TP53", "MYC"],
            "target": ["BRCA1", "EGFR"],
            "weight": [1.0, 1.0],
        }
    )
    record = dataset_store.save("tiny_network_search.csv", dataframe)

    preprocessing_service.preprocess(
        upload_id=record.upload_id,
        mapping={"source": "source", "target": "target", "weight": "weight"},
        duplicate_method="mean",
    )

    response = graph_nodes(record.upload_id, search="tp", limit=50)

    assert _payload(response)["nodes"] == [{"index": 0, "label": "TP53"}]
