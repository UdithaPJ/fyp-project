"""Request models for preprocessing endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional  # Literal kept for mode field

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    """Payload for schema detection."""

    columns: List[str] = Field(default_factory=list)
    sample: List[Dict[str, Any]] = Field(default_factory=list)


class ColumnMapping(BaseModel):
    """User-provided graph column mapping."""

    source: str
    target: str
    weight: Optional[str] = None


class PreprocessRequest(BaseModel):
    """Payload for graph preprocessing."""

    upload_id: str
    mapping: ColumnMapping
    duplicate_method: Literal["count", "mean", "max", "none"] = "mean"


class RunAlgorithmRequest(BaseModel):
    """Payload for submitting an algorithm run job."""

    upload_id: str = Field(
        ...,
        description="Id of a previously uploaded (and preprocessed) dataset.",
    )
    algorithm: str = Field(
        ...,
        description=(
            "Algorithm name.  One of: pagerank, hits, rwr, louvain, mcl, bfs."
        ),
    )
    mode: Literal["gpu"] = Field(
        default="gpu",
        description="Execution mode.  Always 'gpu' — CPU modes are reserved for benchmarking only.",
    )
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Algorithm-specific parameters.  Merged on top of GPU-tuned defaults.",
    )
