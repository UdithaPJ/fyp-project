"""Graph API models."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class EdgeData(BaseModel):
    """Serializable graph edge model."""

    source: str
    target: str
    attributes: Dict[str, object] = Field(default_factory=dict)


class GraphData(BaseModel):
    """Serializable graph contract for API responses."""

    nodes: Dict[str, dict] = Field(default_factory=dict)
    edges: List[EdgeData] = Field(default_factory=list)
