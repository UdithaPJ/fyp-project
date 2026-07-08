"""
src/algorithms/gpu/basic/_utils.py
==================================

Shared helpers for the GPU baseline algorithms.

These baselines are intentionally simple — they exist as a benchmarking
reference point against the highly tuned implementations in
``src/algorithms/gpu/cuda_optimized/``.

Backend policy
--------------
The five cuGraph-backed algorithms (pagerank, bfs, hits, louvain, rwr)
**require** cuGraph + cuDF.  Importing this module on a machine without
RAPIDS raises ``ImportError`` immediately with installation instructions.

MCL has no cuGraph equivalent and uses CuPy directly from its own module
(``src/algorithms/gpu/basic/mcl.py``).  MCL **does not** import from
this file so that it remains independently importable on a CuPy-only
installation.

Hard-fail import
----------------
The following block raises ``ImportError`` (not a silent flag) when
cuGraph or cuDF is missing.  This guarantees that benchmarking runs
always use the cuGraph backend — there is no silent CuPy fallback.

Install RAPIDS:
    conda install -c rapidsai -c nvidia -c conda-forge \\
        rapids=24.02 python=3.10 cudatoolkit=11.8

Reusable contracts
------------------
* ``BASELINE_MODE_CUGRAPH``  — mode string for cuGraph-backed results.
* ``BASELINE_MODE_CUPY``     — mode string for CuPy-backed results (MCL).
* ``BASELINE_MODE``          — backward-compat alias for BASELINE_MODE_CUGRAPH.
* ``build_envelope(...)``    — assemble the 7-key outer result dict.
* ``top_k_global(...)``      — top-K indices over a score vector.
* ``top_k_among(...)``       — top-K indices over a subset of a score vector.
* ``symmetrize_for(...)``    — network-type-aware symmetrization.
* ``out_in_degrees(...)``    — out / in degree arrays from a CSR.
* ``cugraph_function(...)``  — runtime API probe (no hard-coding signatures).
* ``cugraph_build_graph(...)``— version-adaptive cuGraph.Graph construction.
* ``cugraph_extract_column(...)`` — flexible column extraction from cuGraph DFs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Hard-fail: cuGraph + cuDF are required for this module.
# MCL (mcl.py) does NOT import from here to avoid this dependency.
# ---------------------------------------------------------------------------

try:
    import cugraph          # noqa: F401
    import cudf             # noqa: F401
except ImportError as _e:
    raise ImportError(
        "src/algorithms/gpu/basic/ requires cuGraph and cuDF (RAPIDS). "
        "Install via:\n"
        "  conda install -c rapidsai -c nvidia -c conda-forge "
        "rapids=24.02 python=3.10 cudatoolkit=11.8\n"
        f"Original error: {_e}"
    ) from _e


# ---------------------------------------------------------------------------
# Mode identifiers (single source of truth)
# ---------------------------------------------------------------------------

BASELINE_MODE_CUGRAPH: str = "gpu_baseline_cugraph"
BASELINE_MODE_CUPY: str    = "gpu_baseline_cupy"
BASELINE_MODE: str         = BASELINE_MODE_CUGRAPH   # backward-compat alias


# ---------------------------------------------------------------------------
# Top-K display limits — mirror the optimized package's constants.
# ---------------------------------------------------------------------------

TOP_NODES_PPI: int = 20            # PageRank PPI top_nodes
TOP_REG: int       = 15            # PageRank GRN top_regulators, miRNA top_mirnas
TOP_TGT: int       = 15            # PageRank GRN top_targets,    miRNA top_target_genes
TOP_HITS: int      = 15            # HITS  top_hubs / top_authorities / top_nodes
TOP_RWR_NODES: int = 20            # RWR   top_nodes
TOP_RWR_SEEDS: int = 10            # RWR   top_seeds
LOUVAIN_TOP_COMMUNITIES: int = 5   # Louvain top_communities


# ---------------------------------------------------------------------------
# cuGraph version probe
# ---------------------------------------------------------------------------

def cugraph_version() -> str:
    """Return the installed cuGraph version string."""
    try:
        import cugraph as _cg
        return str(getattr(_cg, "__version__", "?"))
    except Exception:                                        # noqa: BLE001
        return "?"


def cugraph_function(name: str):
    """Return ``cugraph.<name>`` if it exists, else ``None``.

    We never hardcode the existence of a cuGraph API — we look it up at
    runtime so version drift between RAPIDS releases never silently
    breaks the baseline.
    """
    try:
        import cugraph as _cg
        return getattr(_cg, name, None)
    except Exception:                                        # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Outer result envelope
# ---------------------------------------------------------------------------

def build_envelope(
    *,
    algorithm: str,
    network_type: str,
    execution_time: float,
    graph_csr: sp.csr_matrix,
    inner: dict,
    mode: str = BASELINE_MODE_CUGRAPH,
) -> dict:
    """Assemble the standard 7-key result dict for a baseline run.

    Parameters
    ----------
    mode:
        ``"gpu_baseline_cugraph"`` (default) or ``"gpu_baseline_cupy"``
        (MCL).  The runner may overwrite this for gpu / cpu modes but
        preserves it for gpu_baseline runs.
    """
    return {
        "algorithm":      algorithm,
        "mode":           mode,
        "network_type":   network_type,
        "execution_time": float(execution_time),
        "num_nodes":      int(graph_csr.shape[0]),
        "num_edges":      int(graph_csr.nnz),
        "result":         inner,
    }


# ---------------------------------------------------------------------------
# Generic numeric helpers
# ---------------------------------------------------------------------------

def top_k_global(scores: np.ndarray, k: int) -> list[int]:
    """Return the top-K node indices ranked by descending ``scores``."""
    if scores.size == 0 or k <= 0:
        return []
    k = min(int(k), int(scores.size))
    order = np.argsort(scores)[::-1][:k]
    return [int(x) for x in order]


def top_k_among(
    scores: np.ndarray, candidates: np.ndarray, k: int,
) -> list[int]:
    """Top-K indices restricted to a ``candidates`` subset of the score space."""
    if candidates.size == 0 or k <= 0:
        return []
    sub = scores[candidates]
    k = min(int(k), int(candidates.size))
    order = np.argsort(sub)[::-1][:k]
    return [int(candidates[i]) for i in order]


def out_in_degrees(
    graph_csr: sp.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(out_degrees, in_degrees)`` as float32 arrays.

    Weighted graphs are summed; binary graphs collapse to int-valued floats.
    """
    out_deg = np.asarray(graph_csr.sum(axis=1)).flatten().astype(np.float32)
    in_deg  = np.asarray(graph_csr.sum(axis=0)).flatten().astype(np.float32)
    return out_deg, in_deg


def symmetrize_for(
    graph_csr: sp.csr_matrix, network_type: str,
) -> sp.csr_matrix:
    """Apply network-type-aware undirected conversion.

    * PPI                  → graph is treated as already undirected; returned as-is.
    * GRN / miRNA / other  → ``binarise(A + A.T)``.

    Mirrors the rule documented in CLAUDE.md.
    """
    nt = str(network_type).lower()
    if nt == "ppi":
        return graph_csr.astype(np.float32).tocsr()
    A = (graph_csr + graph_csr.T).astype(np.float32)
    if A.nnz > 0:
        A.data = np.ones_like(A.data, dtype=np.float32)
    return A.tocsr()


# ---------------------------------------------------------------------------
# cuGraph helpers — build / extract that adapt to the installed version.
# ---------------------------------------------------------------------------

def cugraph_build_graph(
    graph_csr: sp.csr_matrix,
    *,
    directed: bool,
    weighted: bool = True,
):
    """Construct a ``cugraph.Graph`` from a scipy CSR.

    This routine never hardcodes one specific cuGraph API: it tries the
    modern ``from_cudf_edgelist`` first and falls back to the older
    ``from_pandas_edgelist`` if that is what the installed RAPIDS version
    exposes.  Always uses an explicit edge list rather than CSR-direct
    construction so weight / directedness semantics are reproducible
    across RAPIDS versions.
    """
    import cugraph as _cg
    import cudf as _cudf

    coo = graph_csr.tocoo()
    src = coo.row.astype(np.int32)
    dst = coo.col.astype(np.int32)
    wts = coo.data.astype(np.float32) if weighted else None

    # Try cuDF first (the modern path).
    try:
        edge_df = _cudf.DataFrame({"src": src, "dst": dst})
        if weighted and wts is not None:
            edge_df["weight"] = wts

        G = _cg.Graph(directed=directed)
        kwargs: dict[str, Any] = {"source": "src", "destination": "dst"}
        if weighted and wts is not None:
            kwargs["edge_attr"] = "weight"
        G.from_cudf_edgelist(edge_df, **kwargs)
        return G
    except Exception as exc:                                 # noqa: BLE001
        logging.debug(
            "cugraph_build_graph: cuDF edge list path failed (%s) — "
            "trying pandas fallback.", exc,
        )

    # Pandas fallback (older RAPIDS).
    try:
        import pandas as pd
        edge_df = pd.DataFrame({"src": src, "dst": dst})
        if weighted and wts is not None:
            edge_df["weight"] = wts
        G = _cg.Graph(directed=directed)
        kwargs = {"source": "src", "destination": "dst"}
        if weighted and wts is not None:
            kwargs["edge_attr"] = "weight"
        G.from_pandas_edgelist(edge_df, **kwargs)
        return G
    except Exception as exc:                                 # noqa: BLE001
        raise RuntimeError(
            f"Failed to construct cuGraph.Graph from scipy CSR — neither "
            f"cuDF nor pandas edge-list construction succeeded ({exc})."
        ) from exc


def cugraph_extract_column(df, candidates: list[str]) -> np.ndarray:
    """Return the first column from ``df`` whose name appears in ``candidates``.

    cuGraph has renamed return-column names across releases (e.g. ``score``
    vs. ``pagerank`` vs. ``hubs``).  This helper checks the dataframe at
    runtime instead of assuming any one name.
    """
    columns = list(getattr(df, "columns", []))
    for name in candidates:
        if name in columns:
            col = df[name]
            if hasattr(col, "to_numpy"):
                return np.asarray(col.to_numpy())
            return np.asarray(col)
    raise KeyError(
        f"None of {candidates} found in cuGraph output columns ({columns})."
    )


# ---------------------------------------------------------------------------
# Diagnostic logging — runs once on first import.
# ---------------------------------------------------------------------------

def log_backend_selection() -> None:
    """Emit a single info-level log line documenting the cuGraph backend."""
    logging.info(
        "gpu_baseline: cuGraph backend active (version=%s). "
        "MCL uses CuPy independently.",
        cugraph_version(),
    )


log_backend_selection()
