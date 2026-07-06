"""
backend/routes/results.py — Algorithm result retrieval and export endpoints
============================================================================

All endpoints are read-only — they never mutate job state.  The full
result payload (including visualization data) is returned by the GET
endpoint; separate export endpoints convert it to CSV or JSON files that
the user can download locally.

Endpoints
---------
    GET  /results/{job_id}            — full result + visualization payloads
    GET  /results/{job_id}/export/csv — top-nodes / cluster table as CSV
    GET  /results/{job_id}/export/json — complete result dict as JSON file
    GET  /results/                    — list all jobs (newest first)
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

CURRENT_DIR  = Path(__file__).resolve().parent
BACKEND_DIR  = CURRENT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ..services.result_store import result_store
except ImportError:  # pragma: no cover - fallback for running from backend directory
    from services.result_store import result_store


router = APIRouter(prefix="/results", tags=["results"])


# ---------------------------------------------------------------------------
# GET /results/  — job listing
# ---------------------------------------------------------------------------

@router.get("/")
def list_jobs() -> list[dict]:
    """
    Return a summary list of all algorithm jobs, newest first.

    Each entry includes ``job_id``, ``algorithm``, ``mode``, ``status``,
    ``started_at``, and ``finished_at`` — the heavy ``result``,
    ``chart_data``, ``table_data``, and ``graph_viz`` fields are omitted
    to keep the listing payload small.
    """
    jobs = result_store.list_jobs()
    return [
        {
            "job_id":      j["job_id"],
            "algorithm":   j["algorithm"],
            "mode":        j["mode"],
            "status":      j["status"],
            "upload_id":   j["upload_id"],
            "started_at":  j["started_at"],
            "finished_at": j["finished_at"],
            "error":       j.get("error"),
        }
        for j in jobs
    ]


# ---------------------------------------------------------------------------
# GET /results/{job_id}  — full result payload
# ---------------------------------------------------------------------------

@router.get("/{job_id}")
def get_result(job_id: str) -> dict:
    """
    Return the complete result for a finished job.

    Returns
    -------
    dict
        {
          "job_id":     str,
          "algorithm":  str,
          "mode":       str,
          "status":     str,
          "started_at": str | null,
          "finished_at": str | null,
          "result":     dict | null,        ← raw algorithm output
          "chart_data": dict | null,        ← bar/line chart payload
          "table_data": list | null,        ← ranked-node / cluster table
          "graph_viz":  dict | null,        ← highlighted graph nodes+edges
          "error":      str | null,
        }

    Raises
    ------
    404 if job_id is unknown.
    409 if the job is not yet finished (still pending or running).
    """
    try:
        job = result_store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if job["status"] not in ("completed", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is not finished yet (status: {job['status']}).",
        )

    return job


# ---------------------------------------------------------------------------
# GET /results/{job_id}/export/csv
# ---------------------------------------------------------------------------

@router.get("/{job_id}/export/csv")
def export_csv(job_id: str) -> Response:
    """
    Download the table_data for a completed job as a CSV file.

    The ``table_data`` field is a list of dicts (rows).  The CSV header is
    inferred from the keys of the first row.  If ``table_data`` is empty
    or absent, a CSV with a single ``unsupported`` column is returned.

    The filename in the ``Content-Disposition`` header is automatically
    formatted as ``{algorithm}_{job_id[:8]}.csv``.
    """
    try:
        job = result_store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is not completed (status: {job['status']}).",
        )

    table: list[dict] = job.get("table_data") or []

    output = io.StringIO()
    if not table:
        writer = csv.DictWriter(output, fieldnames=["note"])
        writer.writeheader()
        writer.writerow({"note": "no table data available for this algorithm"})
    else:
        # Union of all keys across all rows (handles sparse dicts gracefully).
        all_keys: list[str] = list(
            dict.fromkeys(k for row in table for k in row.keys())
        )
        writer = csv.DictWriter(output, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in table:
            writer.writerow(row)

    algo     = job.get("algorithm", "result")
    filename = f"{algo}_{job_id[:8]}.csv"
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# GET /results/{job_id}/export/json
# ---------------------------------------------------------------------------

@router.get("/{job_id}/export/json")
def export_json(job_id: str) -> Response:
    """
    Download the complete result dict for a completed job as a JSON file.

    Includes the raw algorithm ``result``, ``chart_data``, ``table_data``,
    and ``graph_viz`` payloads.  Node labels are already resolved inside the
    result dict (attached by the runner's ``_attach_labels`` step).

    The filename in the ``Content-Disposition`` header is formatted as
    ``{algorithm}_{job_id[:8]}_result.json``.
    """
    try:
        job = result_store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is not completed (status: {job['status']}).",
        )

    export_payload = {
        "job_id":      job["job_id"],
        "algorithm":   job["algorithm"],
        "mode":        job["mode"],
        "started_at":  job["started_at"],
        "finished_at": job["finished_at"],
        "result":      job["result"],
        "chart_data":  job["chart_data"],
        "table_data":  job["table_data"],
        "graph_viz":   job["graph_viz"],
    }

    algo     = job.get("algorithm", "result")
    filename = f"{algo}_{job_id[:8]}_result.json"
    return Response(
        content=json.dumps(export_payload, indent=2, default=str),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
