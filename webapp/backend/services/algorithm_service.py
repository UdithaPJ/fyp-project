"""
backend/services/algorithm_service.py — Bridge between FastAPI and src/runner
==============================================================================

This service is the ONLY place where the FastAPI layer touches the
``src/`` core Python layer.  Route handlers call the two public functions
below; they never import from ``src/`` directly.

Design decisions
----------------
* ``run_algorithm_job`` is a blocking function intended to be called inside
  a ``BackgroundTasks`` callback (FastAPI) or a ``Thread``.  It is **not**
  an async function so it can call synchronous scipy / CuPy code without
  ``asyncio.run_in_executor`` complexity.

* Visualization payloads (chart, table, graph-viz) are computed inside the
  job runner and attached directly to the job record.  The route layer can
  therefore return all three in a single GET /results/{job_id} call without
  any re-computation.

* ``get_algorithm_catalog`` is a thin wrapper around ``list_algorithms``
  from the runner — it exists so routes never import ``src`` themselves.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

CURRENT_DIR  = Path(__file__).resolve().parent   # webapp/backend/services
BACKEND_DIR  = CURRENT_DIR.parent                # webapp/backend
WEBAPP_DIR   = BACKEND_DIR.parent                # webapp
PROJECT_ROOT = WEBAPP_DIR.parent                 # fyp-project (top of repo)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Core src/ imports — all guarded so the module can be imported even if
# optional GPU dependencies (cupy, pycuda) are absent at import time.
from src.runner.algorithm_runner import list_algorithms, run_algorithm
from src.runner.progress import ProgressReporter
from src.visualization.charts import make_cluster_size_chart_data, make_score_chart_data
from src.visualization.graph_viz import make_highlight_data
from src.visualization.tables import (
    make_cascade_table,
    make_cluster_table,
    make_top_nodes_table,
)

try:
    from ..services.dataset_store import dataset_store
    from ..services.result_store import (
        STATUS_COMPLETED,
        STATUS_FAILED,
        STATUS_RUNNING,
        result_store,
    )
except ImportError:  # pragma: no cover - fallback for running from backend directory
    from services.dataset_store import dataset_store
    from services.result_store import (
        STATUS_COMPLETED,
        STATUS_FAILED,
        STATUS_RUNNING,
        result_store,
    )


def get_algorithm_catalog() -> list[Dict[str, Any]]:
    """
    Return algorithm metadata for every registered algorithm.

    Wraps ``src.runner.algorithm_runner.list_algorithms`` — route handlers
    should call this function rather than importing the runner directly.

    Returns
    -------
    list[dict]
        Each entry has ``{"name", "param_schema", "description"}``.
    """
    return list_algorithms()


def run_algorithm_job(
    job_id: str,
    upload_id: str,
    algorithm_name: str,
    mode: str,
    params: Dict[str, Any],
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> None:
    """
    Execute an algorithm job end-to-end and persist results in the job store.

    This function is **blocking** and should be called from a background
    thread or a FastAPI ``BackgroundTasks`` handler.

    Pipeline
    --------
    1. Mark job as ``running``.
    2. Retrieve the dataset record from ``dataset_store``.
    3. Read the pre-computed ``graph_csr`` and ``node_index_map`` directly
       from the record — the preprocessing pipeline already ran once
       during the /preprocess step and is NEVER re-run here.
    4. Run the algorithm via ``src.runner.algorithm_runner.run_algorithm``.
    5. Build visualization payloads (chart, table, graph-viz).
    6. Mark job as ``completed`` with all payloads attached.

    Any exception at any step marks the job as ``failed`` and stores the
    error message — it does NOT re-raise (the caller is a daemon thread).

    Parameters
    ----------
    job_id : str
        The pre-created job record id in the result_store.
    upload_id : str
        Id of a previously uploaded and preprocessed dataset.
    algorithm_name : str
        One of the six registered algorithm names.
    mode : str
        Always ``"gpu"`` — CPU modes are for benchmarking only.
    params : dict
        Algorithm parameters supplied by the user.
    progress_callback : callable, optional
        Receives progress event dicts ``{type, stage, percent, message}``
        produced by the ``ProgressReporter``.  Used by the SSE stream route.
    """
    # The webapp exclusively uses GPU mode; CPU variants are benchmarking-only.
    mode = "gpu"
    _mark_running(job_id)

    try:
        # ---- Step 2: retrieve dataset record ----
        try:
            record = dataset_store.get(upload_id)
        except KeyError as exc:
            raise ValueError(f"No dataset found for upload_id: {upload_id}") from exc

        # ---- Step 3: read pre-computed graph artefacts ----
        # The /preprocess (or /preprocess/stream) endpoint ran the pipeline
        # ONCE and attached the CSR matrix + node index map to the record.
        # We never re-run the pipeline here — that would duplicate work and
        # silently re-detect a column mapping that may not match the one
        # the user actually confirmed.
        graph_csr      = getattr(record, "graph_csr", None)
        node_index_map = getattr(record, "node_index_map", None)
        if graph_csr is None or node_index_map is None:
            raise ValueError(
                "Graph has not been preprocessed yet. "
                "Complete the preprocessing step before running algorithms. "
                f"upload_id={upload_id}"
            )

        # ---- Step 4: run algorithm ----
        reporter = ProgressReporter(callback=progress_callback)
        result = run_algorithm(
            algorithm_name=algorithm_name,
            graph_csr=graph_csr,
            node_index_map=node_index_map,
            mode=mode,
            params=params,
            progress_reporter=reporter,
        )

        # ---- Step 5: visualization payloads ----
        # graph_csr is passed to the score builders so PageRank charts/tables
        # can split regulators (out-degree > 0) from pure targets for directed
        # GRN / miRNA networks.
        chart_data  = _build_chart(result, node_index_map, params, graph_csr)
        table_data  = _build_table(result, node_index_map, params, graph_csr)
        graph_viz   = make_highlight_data(result, graph_csr, node_index_map)

        # ---- Step 6: mark completed ----
        result_store.update_job(
            job_id,
            status=STATUS_COMPLETED,
            finished_at=_now(),
            result=result,
            chart_data=chart_data,
            table_data=table_data,
            graph_viz=graph_viz,
        )

    except Exception as exc:  # noqa: BLE001
        result_store.update_job(
            job_id,
            status=STATUS_FAILED,
            finished_at=_now(),
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _mark_running(job_id: str) -> None:
    result_store.update_job(
        job_id,
        status=STATUS_RUNNING,
        started_at=_now(),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_top_k(params: Dict[str, Any], default: int, cap: int = 100) -> int:
    """Best-effort parse for a UI-provided top_k parameter (visualization only)."""
    raw = (params or {}).get("top_k", None)
    try:
        k = int(raw)
    except Exception:
        return int(default)
    if k <= 0:
        return int(default)
    return int(min(k, cap))


def _build_chart(
    result: dict,
    node_index_map: dict,
    params: Dict[str, Any],
    graph_csr: Any = None,
) -> dict:
    """Return the most appropriate chart dict for the algorithm result."""
    algo = result.get("algorithm", "")
    if algo in {"pagerank", "hits", "rwr"}:
        # Always compute a sufficiently large top-K payload; the UI can slice.
        return make_score_chart_data(result, node_index_map, top_k=200,
                                     graph_csr=graph_csr)
    if algo in {"louvain", "mcl"}:
        return make_cluster_size_chart_data(result)
    # BFS — no standard chart; return an empty marker
    return {"chart_type": "none", "labels": [], "values": [],
            "title": "", "x_label": "", "y_label": ""}


def _build_table(
    result: dict,
    node_index_map: dict,
    params: Dict[str, Any],
    graph_csr: Any = None,
) -> list:
    """Return the most appropriate table list for the algorithm result."""
    algo = result.get("algorithm", "")
    if algo in {"pagerank", "hits", "rwr"}:
        # Always compute a sufficiently large top-K payload; the UI can slice.
        return make_top_nodes_table(result, node_index_map, top_k=200,
                                    graph_csr=graph_csr)
    if algo in {"louvain", "mcl"}:
        return make_cluster_table(result, node_index_map)
    if algo == "bfs":
        return make_cascade_table(result, node_index_map)
    return []
