"""
src/runner/progress.py — Progress event reporter for streaming runs
====================================================================

Wraps a callback-receiving event stream that mirrors the NDJSON format
already established by ``backend/routes/preprocessing.py``'s
``/preprocess/stream`` endpoint.  The web layer hands a callback that
puts events into its streaming Queue; the algorithm runner calls
:meth:`update` at well-defined stages and :meth:`done` / :meth:`error`
at the end.

Event schema
------------
    {"type": "progress", "stage": str, "percent": int, "message": str}
    {"type": "result",   "data":  dict}
    {"type": "error",    "message": str}

The frontend already knows how to consume this shape (it currently uses
``event`` for the preprocessing stream — the algorithm stream uses
``type`` so the two channels are distinguishable client-side).
"""

from __future__ import annotations

from typing import Callable, Optional


class ProgressReporter:
    """Callback-based progress channel — pure data, no I/O."""

    # Canonical stage names — kept here so the frontend can build a
    # progress bar UI from a fixed taxonomy.
    STAGE_STARTING = "starting"
    STAGE_CONFIG   = "configuring"
    STAGE_VALIDATE = "validating"
    STAGE_RUNNING  = "running"
    STAGE_PACKING  = "packing_results"
    STAGE_DONE     = "done"

    def __init__(self, callback: Optional[Callable[[dict], None]] = None):
        """
        Parameters
        ----------
        callback : callable or None
            Invoked once per event dict.  When ``None`` the reporter is a
            no-op — algorithms can be called outside of a streaming
            context without changes.
        """
        self.callback = callback

    # ---- Event helpers ----

    def update(self, stage: str, pct: int, message: str = "") -> None:
        """Emit a progress event.  ``pct`` is clamped to ``[0, 100]``."""
        if self.callback is None:
            return
        pct = max(0, min(100, int(pct)))
        self.callback({
            "type":    "progress",
            "stage":   str(stage),
            "percent": pct,
            "message": str(message),
        })

    def done(self, result: dict) -> None:
        """Emit a terminal result event carrying the full algorithm result."""
        if self.callback is None:
            return
        self.callback({
            "type": "result",
            "data": result,
        })

    def error(self, message: str) -> None:
        """Emit a terminal error event."""
        if self.callback is None:
            return
        self.callback({
            "type":    "error",
            "message": str(message),
        })
