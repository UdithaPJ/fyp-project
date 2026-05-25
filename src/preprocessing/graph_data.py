"""Core graph data structures for the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class GraphData:
    """Container for graph nodes and directed edges."""

    nodes: Dict[str, dict] = field(default_factory=dict)
    edges: List[Tuple[str, str, dict]] = field(default_factory=list)
