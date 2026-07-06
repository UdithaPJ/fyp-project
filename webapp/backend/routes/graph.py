"""
backend/routes/graph.py — Graph inspection endpoints
=====================================================

Provides lightweight graph statistics and a node/edge preview for a
dataset that has already been uploaded and preprocessed.  The frontend
uses these endpoints to display the "Graph Summary" panel before the user
chooses which algorithm to run.

Endpoints
---------
    GET  /graph/stats/{upload_id}    — node count, edge count, density, …
    GET  /graph/preview/{upload_id}  — first N nodes and edges (for the
                                       mini graph visualisation in Step 4)
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

CURRENT_DIR  = Path(__file__).resolve().parent   # webapp/backend/routes
BACKEND_DIR  = CURRENT_DIR.parent                # webapp/backend
WEBAPP_DIR   = BACKEND_DIR.parent                # webapp
PROJECT_ROOT = WEBAPP_DIR.parent                 # fyp-project (top of repo)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ..models.responses import GraphStatsResponse, NodeListResponse, NodeEntry
    from ..services.dataset_store import dataset_store
except ImportError:  # pragma: no cover - fallback for running from backend directory
    from models.responses import GraphStatsResponse, NodeListResponse, NodeEntry
    from services.dataset_store import dataset_store

from src.graph.converter import get_graph_stats


router = APIRouter(prefix="/graph", tags=["graph"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_csr_for_upload(upload_id: str):
    """Return the cached ``(graph_csr, node_index_map)`` for an upload.

    FIX (upload-time / correctness): this used to call
    ``PreprocessingPipeline.run_dataframe(record.dataframe, ...)`` on every
    request — re-running schema detection/cleaning/dedup/graph-building
    from scratch, with a hardcoded ``user_override=None,
    duplicate_strategy="mean"`` that silently ignored whatever mapping or
    duplicate strategy the user actually confirmed in ``/preprocess``. It
    also crashed outright once ``/preprocess`` had run, because
    ``preprocessing_service._store_graph_artefacts`` nulls
    ``record.dataframe`` right after building the CSR (see
    ``dataset_store.py`` MEMORY_FIX C-2) — so every call here after that
    point raised ``AttributeError: 'NoneType' object has no attribute
    'empty'``, breaking the BFS/RWR node-search typeahead
    (``GET /graph/nodes``) in practice.

    The pipeline already runs exactly once, in ``/preprocess`` — this now
    reads the ``graph_csr``/``node_index_map`` it already cached, exactly
    like ``algorithm_service.py`` does, instead of re-deriving (and
    potentially mismatching) the graph.
    """
    try:
        record = dataset_store.get(upload_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    graph_csr = getattr(record, "graph_csr", None)
    node_index_map = getattr(record, "node_index_map", None)
    if graph_csr is None or node_index_map is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Graph has not been preprocessed yet. "
                "Call /preprocess (or /preprocess/stream) for this upload_id first."
            ),
        )
    return graph_csr, node_index_map


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/stats/{upload_id}", response_model=GraphStatsResponse)
def graph_stats(upload_id: str) -> GraphStatsResponse:
    """
    Return structural statistics for a preprocessed graph.

    The response includes node count, edge count, density, average degree,
    maximum degree, whether the graph is directed, and the number of
    connected components.  All values are computed from the CSR matrix so
    they reflect the deduplicated graph — not the raw CSV row count.
    """
    graph_csr, node_index_map = _get_csr_for_upload(upload_id)
    stats = get_graph_stats(graph_csr, node_index_map)
    return GraphStatsResponse(**stats)


@router.get("/preview/{upload_id}")
def graph_preview(
    upload_id: str,
    max_nodes: int = 50,
    max_edges: int = 100,
) -> dict:
    """
    Return a lightweight node/edge preview for the frontend graph widget.

    The preview is capped at ``max_nodes`` / ``max_edges`` so the JSON
    payload remains small even for large graphs.  Node labels are resolved
    from the node_index_map so the frontend can display gene/protein names
    immediately without a separate lookup.

    Query parameters
    ----------------
    max_nodes : int, optional
        Maximum number of nodes to return (default 50, capped at 500).
    max_edges : int, optional
        Maximum number of edges to return (default 100, capped at 1000).
    """
    max_nodes = min(int(max_nodes), 500)
    max_edges = min(int(max_edges), 1000)

    graph_csr, node_index_map = _get_csr_for_upload(upload_id)

    # Build reverse map: index → label
    reverse: dict[int, str] = {int(v): str(k) for k, v in node_index_map.items()}
    n = graph_csr.shape[0]

    nodes = [
        {"id": reverse.get(i, f"node_{i}"), "label": reverse.get(i, f"node_{i}")}
        for i in range(min(n, max_nodes))
    ]

    edges: list[dict] = []
    selected_indices = set(range(min(n, max_nodes)))
    indptr  = graph_csr.indptr
    indices = graph_csr.indices
    data    = graph_csr.data

    for src in range(min(n, max_nodes)):
        if len(edges) >= max_edges:
            break
        s, e = int(indptr[src]), int(indptr[src + 1])
        for k in range(s, e):
            if len(edges) >= max_edges:
                break
            tgt = int(indices[k])
            if tgt in selected_indices:
                edges.append({
                    "source": reverse.get(src, f"node_{src}"),
                    "target": reverse.get(tgt, f"node_{tgt}"),
                    "weight": float(data[k]),
                })

    return {
        "upload_id":   upload_id,
        "nodes":       nodes,
        "edges":       edges,
        "total_nodes": int(n),
        "total_edges": int(graph_csr.nnz),
        "preview_capped": bool(n > max_nodes or graph_csr.nnz > max_edges),
    }


# ---------------------------------------------------------------------------
# GET /graph/nodes — node-typeahead endpoint for the parameter form
# ---------------------------------------------------------------------------

_NODES_MAX_LIMIT = 200
_NODES_DEFAULT_LIMIT = 50


@router.get("/nodes", response_model=NodeListResponse)
def graph_nodes(
    upload_id: str,
    search: str = "",
    limit: int = _NODES_DEFAULT_LIMIT,
) -> NodeListResponse:
    """Search-and-paginate the node_index_map of a preprocessed graph.

    Used by the BFS source selector and the RWR seed-nodes multi-selector
    in the AlgorithmSelector form.  Returns ``{index, label}`` pairs in
    label-sorted order so the frontend renders them without further
    sorting.

    Query parameters
    ----------------
    upload_id : str (required)
    search    : str
        Substring filter applied to node labels (case-insensitive).
        Empty string returns the first ``limit`` labels in alphabetical
        order.
    limit     : int
        Maximum number of nodes to return.  Default 50, capped at 200.
    """
    if limit <= 0:
        limit = _NODES_DEFAULT_LIMIT
    limit = min(int(limit), _NODES_MAX_LIMIT)

    _, node_index_map = _get_csr_for_upload(upload_id)
    needle = (search or "").strip().lower()

    # node_index_map: {label: index}
    pairs = sorted(node_index_map.items(), key=lambda kv: str(kv[0]).lower())

    if needle:
        filtered = [
            (label, idx) for (label, idx) in pairs
            if needle in str(label).lower()
        ]
    else:
        filtered = pairs

    total = len(filtered)
    truncated = total > limit
    capped = filtered[:limit]

    return NodeListResponse(
        nodes=[
            NodeEntry(index=int(idx), label=str(label))
            for (label, idx) in capped
        ],
        total=total,
        truncated=truncated,
    )
