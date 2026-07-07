"""
backend/routes/algorithms.py — Algorithm catalog, run, and progress endpoints
==============================================================================

Endpoints
---------
    GET  /algorithms/catalog
        Return the full list of available algorithms with their parameter
        schemas — used by the frontend to build dynamic parameter forms.

    POST /algorithms/run
        Submit an algorithm job.  The job is created immediately (returns
        job_id) and execution happens in a background thread.

    GET  /algorithms/run/stream/{job_id}
        Server-Sent Events stream — emits progress events while the job is
        running, then a terminal ``result`` or ``error`` event.

    GET  /algorithms/status/{job_id}
        Lightweight status poll for clients that prefer polling over SSE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from time import monotonic

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

CURRENT_DIR  = Path(__file__).resolve().parent
BACKEND_DIR  = CURRENT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ..models.requests import RunAlgorithmRequest
    from ..models.responses import JobStatusResponse
    from ..services.algorithm_service import get_algorithm_catalog, run_algorithm_job
    from ..services.dataset_store import dataset_store
    from ..services.result_store import result_store
except ImportError:  # pragma: no cover - fallback for running from backend directory
    from models.requests import RunAlgorithmRequest
    from models.responses import JobStatusResponse
    from services.algorithm_service import get_algorithm_catalog, run_algorithm_job
    from services.dataset_store import dataset_store
    from services.result_store import result_store


router = APIRouter(prefix="/algorithms", tags=["algorithms"])


# ---------------------------------------------------------------------------
# GET /algorithms/catalog
# ---------------------------------------------------------------------------

@router.get("/catalog")
def algorithm_catalog() -> list[dict]:
    """
    Return metadata for every registered algorithm.

    Each entry contains ``{"name", "param_schema", "description"}``
    so the frontend can dynamically render algorithm-specific parameter
    forms without hardcoding field lists.
    """
    try:
        return get_algorithm_catalog()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load algorithm catalog: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# POST /algorithms/run  — submit a job
# ---------------------------------------------------------------------------

@router.post("/run", response_model=JobStatusResponse, status_code=202)
def run_algorithm(
    payload: RunAlgorithmRequest,
    background_tasks: BackgroundTasks,
) -> JobStatusResponse:
    """
    Submit an algorithm job for background execution.

    Returns immediately with ``status=pending`` and a ``job_id``.  Clients
    can then poll ``GET /algorithms/status/{job_id}`` or subscribe to the
    SSE stream at ``GET /algorithms/run/stream/{job_id}``.

    HTTP 202 (Accepted) signals that the request was received but the
    computation has not yet completed.

    Guard: the dataset referenced by ``upload_id`` must already have been
    preprocessed (i.e. ``graph_csr`` is attached to its record).  We
    short-circuit with HTTP 400 here so the user gets an immediate, clear
    error instead of a cryptic background-job failure.
    """
    # ---- Guard: dataset must exist and must already be preprocessed ----
    try:
        record = dataset_store.get(payload.upload_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"No uploaded dataset found for upload_id '{payload.upload_id}'.",
        ) from exc
    if getattr(record, "graph_csr", None) is None:
        raise HTTPException(
            status_code=400,
            detail="Dataset not preprocessed. Complete preprocessing first.",
        )

    job_id = result_store.create_job(
        algorithm=payload.algorithm,
        mode=payload.mode,
        upload_id=payload.upload_id,
        params=payload.params or {},
    )

    background_tasks.add_task(
        run_algorithm_job,
        job_id=job_id,
        upload_id=payload.upload_id,
        algorithm_name=payload.algorithm,
        mode=payload.mode,
        params=payload.params or {},
        progress_callback=None,  # no SSE stream for plain /run
    )

    job = result_store.get_job(job_id)
    return JobStatusResponse(**job)


# ---------------------------------------------------------------------------
# GET /algorithms/run/stream/{job_id}  — SSE progress stream
# ---------------------------------------------------------------------------

@router.get("/run/stream/{job_id}")
def stream_algorithm(job_id: str) -> StreamingResponse:
    """
    Subscribe to live progress events for a running (or pending) job.

    Emits NDJSON lines — one JSON object per line — until the job finishes:

        {"type": "progress", "stage": "running", "percent": 42, "message": "..."}
        {"type": "result",   "data": { … full result dict … }}

    or on failure:

        {"type": "error", "message": "RuntimeError: …"}

    If the job does not exist a 404 is returned before the stream opens.

    Implementation note
    -------------------
    A background thread calls ``run_algorithm_job`` with a queue-based
    progress callback.  The generator function drains the queue and yields
    NDJSON lines until the sentinel ``_DONE`` is received.

    If the job is already completed / failed when the client connects, the
    stream immediately emits a single terminal event and closes.
    """
    # Guard: job must exist
    try:
        job = result_store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Fast path: job already finished
    if job["status"] in ("completed", "failed"):
        def _already_done():
            if job["status"] == "completed":
                yield json.dumps({"type": "result", "data": job["result"]}).encode() + b"\n"
            else:
                yield json.dumps({"type": "error", "message": job.get("error", "unknown error")}).encode() + b"\n"
        return StreamingResponse(_already_done(), media_type="application/x-ndjson")

    _DONE = object()

    def _event_stream():
        event_queue: Queue = Queue()

        # Track last known progress so keepalive events don't reset the UI.
        last_progress: dict = {
            "type": "progress",
            "stage": "running",
            "percent": 0,
            "message": "",
        }
        last_emit = monotonic()

        def _progress_cb(event: dict) -> None:
            event_queue.put(event)

        def _worker():
            try:
                run_algorithm_job(
                    job_id=job_id,
                    upload_id=job["upload_id"],
                    algorithm_name=job["algorithm"],
                    mode=job["mode"],
                    params=job["params"],
                    progress_callback=_progress_cb,
                )
                # Fetch the now-completed job and emit the terminal event.
                finished = result_store.get_job(job_id)
                if finished["status"] == "completed":
                    event_queue.put({"type": "result", "data": finished["result"]})
                else:
                    event_queue.put({"type": "error", "message": finished.get("error", "unknown")})
            except Exception as exc:  # noqa: BLE001
                event_queue.put({"type": "error", "message": str(exc)})
            finally:
                event_queue.put(_DONE)

        worker_thread = Thread(target=_worker, daemon=True)
        worker_thread.start()

        while True:
            try:
                # Use a short timeout and emit keepalive progress if the
                # algorithm produces no events for a while (common for GPU
                # kernels). This prevents intermediate proxies/browsers from
                # closing an idle connection.
                event = event_queue.get(timeout=15)
            except Empty:
                now = monotonic()
                if now - last_emit >= 15:
                    # Keepalive event — treated as normal progress by the frontend.
                    keepalive = dict(last_progress)
                    if not keepalive.get("message"):
                        keepalive["message"] = "still running"
                    yield json.dumps(keepalive).encode() + b"\n"
                    last_emit = now
                # If the worker thread unexpectedly died without signaling _DONE,
                # fail fast so the client isn't stuck forever.
                if not worker_thread.is_alive() and event_queue.empty():
                    yield json.dumps({"type": "error", "message": "stream terminated unexpectedly"}).encode() + b"\n"
                    break
                continue
            if event is _DONE:
                break

            if isinstance(event, dict) and event.get("type") == "progress":
                # Update keepalive baseline.
                last_progress = {
                    "type": "progress",
                    "stage": event.get("stage", last_progress.get("stage", "running")),
                    "percent": event.get("percent", last_progress.get("percent", 0)),
                    "message": event.get("message", last_progress.get("message", "")),
                }

            yield json.dumps(event).encode() + b"\n"
            last_emit = monotonic()

    return StreamingResponse(_event_stream(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# GET /algorithms/status/{job_id}  — poll-style status check
# ---------------------------------------------------------------------------

@router.get("/status/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    """
    Return the current status and metadata for a job.

    The ``result``, ``chart_data``, ``table_data``, and ``graph_viz``
    fields are ``null`` while the job is pending or running.  Once the job
    reaches ``completed`` all four are populated.

    Use ``GET /results/{job_id}`` if you only need the full result payload.
    """
    try:
        job = result_store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JobStatusResponse(**job)
