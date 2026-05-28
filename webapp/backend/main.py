"""FastAPI entry point for the local preprocessing web application."""

from __future__ import annotations

import io
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def _force_utf8_stdio() -> None:
    """Avoid Windows console UnicodeEncodeError for logs/progress output."""
    class _SafeTextStream(io.TextIOBase):
        def __init__(self, stream):
            self._stream = stream

        def write(self, s: str) -> int:  # type: ignore[override]
            try:
                return self._stream.write(s)
            except UnicodeEncodeError:
                data = s.encode("utf-8", errors="replace")
                buf = getattr(self._stream, "buffer", None)
                if buf is not None:
                    buf.write(data)
                    return len(s)

                enc = getattr(self._stream, "encoding", None) or "utf-8"
                safe = s.encode(enc, errors="replace").decode(enc, errors="replace")
                return self._stream.write(safe)

        def flush(self) -> None:  # type: ignore[override]
            if hasattr(self._stream, "flush"):
                self._stream.flush()

        def isatty(self) -> bool:  # type: ignore[override]
            return bool(getattr(self._stream, "isatty", lambda: False)())

        def fileno(self) -> int:  # type: ignore[override]
            return int(getattr(self._stream, "fileno", lambda: -1)())

        def writable(self) -> bool:  # type: ignore[override]
            return True

        def __getattr__(self, name: str):
            return getattr(self._stream, name)

    for stream in (sys.stdout, sys.stderr):
        try:
            # Python 3.7+: TextIOWrapper supports reconfigure.
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Best-effort only — never block API startup.
            pass

    # Final guard: prevent any later UnicodeEncodeError from crashing the process.
    if not isinstance(sys.stdout, _SafeTextStream):
        sys.stdout = _SafeTextStream(sys.stdout)
    if not isinstance(sys.stderr, _SafeTextStream):
        sys.stderr = _SafeTextStream(sys.stderr)


_force_utf8_stdio()

CURRENT_DIR  = Path(__file__).resolve().parent   # webapp/backend
WEBAPP_DIR   = CURRENT_DIR.parent                # webapp
PROJECT_ROOT = WEBAPP_DIR.parent                 # fyp-project (top of repo — src/ lives here)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .routes.preprocessing import router as preprocessing_router
    from .routes.graph import router as graph_router
    from .routes.algorithms import router as algorithms_router
    from .routes.results import router as results_router
    from .routes.system import router as system_router
except ImportError:  # pragma: no cover - fallback for running from backend directory
    from routes.preprocessing import router as preprocessing_router
    from routes.graph import router as graph_router
    from routes.algorithms import router as algorithms_router
    from routes.results import router as results_router
    from routes.system import router as system_router


app = FastAPI(
    title="Local Graph Preprocessing API",
    description="Offline-capable API for tabular-to-graph preprocessing.",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(preprocessing_router)
app.include_router(graph_router)
app.include_router(algorithms_router)
app.include_router(results_router)
app.include_router(system_router)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a lightweight health message."""

    return {"message": "Local graph preprocessing API is running."}
