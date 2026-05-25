"""Response models for preprocessing endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .graph import GraphData


class UploadResponse(BaseModel):
    """Upload response with preview data."""

    upload_id: str
    filename: str
    columns: List[str] = Field(default_factory=list)
    preview: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int


class DetectResponse(BaseModel):
    """Detected graph columns plus confidence."""

    source: Optional[str] = None
    target: Optional[str] = None
    weight: Optional[str] = None
    confidence: float = 0.0
    field_confidence: Dict[str, float] = Field(default_factory=dict)


class PreprocessResponse(BaseModel):
    """Final preprocessing summary and lightweight graph preview."""

    nodes: int
    edges: int
    validation: Dict[str, Any] = Field(default_factory=dict)
    graph: GraphData


class JobStatusResponse(BaseModel):
    """Algorithm job status — returned immediately after POST /algorithms/run."""

    job_id: str
    status: str = Field(
        description="One of: pending, running, completed, failed."
    )
    algorithm: str
    mode: str
    upload_id: str
    params: Dict[str, Any] = Field(default_factory=dict)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    # Heavy payloads — null while the job is pending/running.
    result: Optional[Dict[str, Any]] = None
    chart_data: Optional[Dict[str, Any]] = None
    table_data: Optional[Any] = None      # list[dict] once complete
    graph_viz: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class GraphStatsResponse(BaseModel):
    """Structural statistics for a preprocessed graph."""

    num_nodes: int
    num_edges: int
    density: float
    avg_degree: float
    max_degree: int
    is_directed: bool
    num_components: int
