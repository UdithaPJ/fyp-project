"""
backend/services/result_store.py — In-memory job store for algorithm runs
==========================================================================

Tracks the lifecycle of every algorithm job submitted through the
/algorithms/run endpoint.  Mirrors the InMemoryDatasetStore pattern so
the rest of the backend stays consistent.

Job lifecycle
-------------
    pending  → running → completed
                       → failed

Each entry is a plain dict so it can be JSON-serialised directly by the
route handlers without any additional conversion step.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4


# Canonical status values — the frontend progress bar uses these strings.
STATUS_PENDING   = "pending"
STATUS_RUNNING   = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED    = "failed"


class InMemoryJobStore:
    """Thread-safe in-memory store for algorithm job records."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def create_job(
        self,
        algorithm: str,
        mode: str,
        upload_id: str,
        params: Dict[str, Any],
    ) -> str:
        """
        Create a new job record in *pending* state and return its job_id.

        Parameters
        ----------
        algorithm : str
            Algorithm name (e.g. ``"pagerank"``).
        mode : str
            Execution mode: ``"cpu_single"``, ``"cpu_multi"``, or ``"gpu"``.
        upload_id : str
            The dataset this job will run against.
        params : dict
            User-supplied algorithm parameters (already merged / validated
            upstream, but stored here for auditability).
        """
        job_id = uuid4().hex
        record: Dict[str, Any] = {
            "job_id":      job_id,
            "status":      STATUS_PENDING,
            "algorithm":   algorithm,
            "mode":        mode,
            "upload_id":   upload_id,
            "params":      dict(params),
            "started_at":  None,
            "finished_at": None,
            "result":      None,
            "error":       None,
            # Visualization payloads attached after run completes
            "chart_data":   None,
            "table_data":   None,
            "graph_viz":    None,
        }
        with self._lock:
            self._jobs[job_id] = record
        return job_id

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        """
        Merge keyword-argument updates into the job record.

        Common call patterns::

            store.update_job(job_id, status=STATUS_RUNNING,
                             started_at=datetime.now(timezone.utc).isoformat())
            store.update_job(job_id, status=STATUS_COMPLETED, result=result_dict)
            store.update_job(job_id, status=STATUS_FAILED, error="OOM")
        """
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"No job found for id '{job_id}'.")
            self._jobs[job_id].update(kwargs)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Return a shallow copy of the job record (thread-safe snapshot)."""
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"No job found for id '{job_id}'.")
            return dict(self._jobs[job_id])

    def list_jobs(self) -> list[Dict[str, Any]]:
        """Return shallow copies of all job records, newest first."""
        with self._lock:
            records = [dict(v) for v in self._jobs.values()]
        # Sort by started_at descending; jobs not yet started go last.
        records.sort(
            key=lambda r: r.get("started_at") or "",
            reverse=True,
        )
        return records


# Module-level singleton — imported by route handlers.
result_store = InMemoryJobStore()
