"""FastAPI entry point for the local preprocessing web application."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
except ImportError:  # pragma: no cover - fallback for running from backend directory
    from routes.preprocessing import router as preprocessing_router
    from routes.graph import router as graph_router
    from routes.algorithms import router as algorithms_router
    from routes.results import router as results_router


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


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a lightweight health message."""

    return {"message": "Local graph preprocessing API is running."}
