"""
Algorithm implementations.

Each algorithm is a class subclassing :class:`AlgorithmBase` and exposing
``cpu_single``, ``cpu_multi``, and ``gpu`` static methods that return a
standardised result dict.

The algorithm classes are exposed both individually and via the
``ALGORITHM_REGISTRY`` mapping that the runner uses for dynamic dispatch.
"""

from .base    import AlgorithmBase
from .bfs     import BFS
from .hits    import HITS
from .louvain import Louvain
from .mcl     import MCL
from .pagerank import PageRank
from .rwr     import RWR

ALGORITHM_REGISTRY: dict[str, type[AlgorithmBase]] = {
    PageRank.NAME: PageRank,
    BFS.NAME:      BFS,
    Louvain.NAME:  Louvain,
    RWR.NAME:      RWR,
    HITS.NAME:     HITS,
    MCL.NAME:      MCL,
}

__all__ = [
    "AlgorithmBase",
    "PageRank",
    "BFS",
    "Louvain",
    "RWR",
    "HITS",
    "MCL",
    "ALGORITHM_REGISTRY",
]
