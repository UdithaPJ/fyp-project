"""
src/graph/converter.py — GraphData ↔ CSR matrix conversion utilities
====================================================================

Provides the two functions used by the webapp and benchmarking layers
to convert between the preprocessing-layer ``GraphData`` dataclass and
the scipy CSR sparse matrix format that all algorithms consume.

Public API
----------
graphdata_to_csr(graph_data)
    Convert a ``GraphData`` object to a (csr_matrix, node_index_map) pair.
    The node_index_map maps each string node label → integer matrix index.

get_graph_stats(graph_csr, node_index_map)
    Compute structural statistics from a CSR matrix.  Returns a plain dict
    whose keys match the ``GraphStatsResponse`` Pydantic model.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp

# GraphData is imported lazily inside the function bodies so this module
# can be imported even when the preprocessing package is not on sys.path
# (e.g. in unit tests that mock the graph).


# ---------------------------------------------------------------------------
# graphdata_to_csr
# ---------------------------------------------------------------------------

def graphdata_to_csr(
    graph_data,
) -> Tuple[sp.csr_matrix, Dict[str, int]]:
    """
    Convert a preprocessing-layer ``GraphData`` object to a CSR matrix.

    Parameters
    ----------
    graph_data : GraphData
        Object with:
          .nodes  — ``Dict[str, dict]``  (node_label → attribute dict)
          .edges  — ``List[Tuple[str, str, dict]]``  (src, tgt, attrs)

    Returns
    -------
    csr_matrix : scipy.sparse.csr_matrix
        Square adjacency / weight matrix of shape (N, N) where N is the
        number of unique nodes.  Entry [i, j] holds the edge weight
        (defaults to 1.0 when no ``"weight"`` attribute is present).
        If duplicate (i, j) pairs exist the weights are **summed** (COO
        duplicate handling).

    node_index_map : Dict[str, int]
        Maps every node label (string) to its row/column integer index in
        the CSR matrix.  Preserves the order in which nodes appear in
        ``graph_data.nodes`` (insertion order, Python 3.7+).

    Biological use
    --------------
    This conversion is the boundary between the tabular-preprocessing layer
    and the algorithm layer.  All six algorithms (PageRank, BFS, Louvain,
    RWR, HITS, MCL) consume a CSR matrix so they can operate uniformly on
    GRN, PPI, and miRNA-target networks.
    """
    # ---- Build node index map ----
    # Use the order in graph_data.nodes; fall back to collecting from edges
    # if nodes dict is empty (e.g. algorithm was given a raw edge list).
    if graph_data.nodes:
        node_labels = list(graph_data.nodes.keys())
    else:
        seen: dict[str, int] = {}
        for src, tgt, _ in graph_data.edges:
            if src not in seen:
                seen[src] = len(seen)
            if tgt not in seen:
                seen[tgt] = len(seen)
        node_labels = list(seen.keys())

    node_index_map: Dict[str, int] = {label: idx for idx, label in enumerate(node_labels)}
    n = len(node_labels)

    if n == 0:
        empty = sp.csr_matrix((0, 0), dtype=np.float32)
        return empty, {}

    # ---- Build COO triplets ----
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for src, tgt, attrs in graph_data.edges:
        i = node_index_map.get(src)
        j = node_index_map.get(tgt)
        if i is None or j is None:
            # Edge references a node not in the node map — skip silently.
            continue
        weight = float(attrs.get("weight", 1.0)) if attrs else 1.0
        rows.append(i)
        cols.append(j)
        vals.append(weight)

    if not rows:
        empty = sp.csr_matrix((n, n), dtype=np.float32)
        return empty, node_index_map

    # scipy COO → CSR: duplicate (i, j) entries are summed automatically.
    coo = sp.coo_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int32),
          np.array(cols, dtype=np.int32))),
        shape=(n, n),
    )
    csr = coo.tocsr()
    return csr, node_index_map


# ---------------------------------------------------------------------------
# get_graph_stats
# ---------------------------------------------------------------------------

def get_graph_stats(
    graph_csr: sp.csr_matrix,
    node_index_map: Dict[str, int],
) -> dict:
    """
    Compute structural statistics from a CSR adjacency matrix.

    Parameters
    ----------
    graph_csr : scipy.sparse.csr_matrix
        Adjacency / weight matrix produced by ``graphdata_to_csr``.
    node_index_map : Dict[str, int]
        Mapping from node label → matrix index (used only for ``num_nodes``
        cross-check; not required for computation).

    Returns
    -------
    dict with keys:
        num_nodes   : int   — number of nodes (matrix dimension)
        num_edges   : int   — number of non-zero entries (directed edge count)
        density     : float — edges / (nodes * (nodes - 1)), or 0 if < 2 nodes
        avg_degree  : float — average out-degree (sum of row nnz / num_nodes)
        max_degree  : int   — maximum out-degree across all nodes
        is_directed : bool  — True when the matrix is not symmetric
        num_components : int — number of weakly connected components

    Biological use
    --------------
    These statistics are shown in the "Graph Summary" panel (Step 3 / 4) so
    the user can sanity-check the network before choosing an algorithm.
    Density and component count are especially informative for sparse
    biological networks where many genes may be isolated.
    """
    n = int(graph_csr.shape[0])
    nnz = int(graph_csr.nnz)

    # ---- Density ----
    max_possible = n * (n - 1)
    density = float(nnz) / float(max_possible) if max_possible > 0 else 0.0

    # ---- Degree (out-degree from row sums) ----
    if n > 0:
        row_nnz = np.diff(graph_csr.indptr)          # number of non-zeros per row
        avg_degree = float(row_nnz.mean())
        max_degree = int(row_nnz.max())
    else:
        avg_degree = 0.0
        max_degree = 0

    # ---- Directed check (compare with transpose) ----
    is_directed = True
    try:
        diff = graph_csr - graph_csr.T
        is_directed = bool(diff.nnz > 0)
    except Exception:
        pass  # keep is_directed = True on any error

    # ---- Connected components (weakly, via undirected adjacency) ----
    num_components = 1
    try:
        # Use undirected adjacency for component counting regardless of
        # directed/undirected status (weakly connected components).
        undirected = (graph_csr + graph_csr.T)
        undirected.data[:] = 1          # binarize
        n_comp, _ = sp.csgraph.connected_components(
            undirected, directed=False, return_labels=True
        )
        num_components = int(n_comp)
    except Exception:
        pass  # keep num_components = 1 on any error

    return {
        "num_nodes":      n,
        "num_edges":      nnz,
        "density":        round(density, 8),
        "avg_degree":     round(avg_degree, 4),
        "max_degree":     max_degree,
        "is_directed":    is_directed,
        "num_components": num_components,
    }
