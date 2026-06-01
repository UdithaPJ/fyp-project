"""
src/algorithms/gpu/basic/_utils.py
==================================

Shared helpers for the GPU baseline algorithms.

These baselines are intentionally simple — they exist as a benchmarking
reference point against the highly tuned implementations in
``src/algorithms/gpu/cuda_optimized/``.

Backend selection
-----------------
Every baseline algorithm uses the same fallback chain:

    1. cuGraph (RAPIDS) — preferred when available.
    2. CuPy sparse — fallback that still runs entirely on the GPU.
    3. RuntimeError — neither backend installed → propagate to the runner.

We never silently fall back to CPU here — that path belongs to
``src/algorithms/cpu/`` and is selected explicitly by the runner via
``mode="cpu_single"`` / ``"cpu_multi"``.

Reusable contracts
------------------
* ``detect_backends()``      — returns a tuple of availability flags.
* ``BASELINE_MODE``          — the literal string mode identifier.
* ``build_envelope(...)``    — assemble the 7-key outer result dict.
* ``top_k_global(...)``      — top-K indices over a score vector.
* ``top_k_among(...)``       — top-K indices over a subset of a score vector.
* ``symmetrize_for(...)``    — network-type-aware symmetrization.
* ``out_in_degrees(...)``    — out / in degree arrays from a CSR.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Mode identifier (single source of truth — runner / base.py imports this)
# ---------------------------------------------------------------------------

BASELINE_MODE: str = "gpu_baseline"


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
# Backend detection
# ---------------------------------------------------------------------------

def detect_backends() -> tuple[bool, bool, bool]:
    """Probe for cuGraph + cuDF + CuPy availability.

    Returns
    -------
    (cugraph_available, cudf_available, cupy_available) : tuple[bool, bool, bool]
    """
    try:
        import cugraph        # noqa: F401  (probe only)
        cugraph_available = True
    except Exception:                                       # noqa: BLE001
        cugraph_available = False

    try:
        import cudf           # noqa: F401
        cudf_available = True
    except Exception:                                       # noqa: BLE001
        cudf_available = False

    try:
        import cupy           # noqa: F401
        import cupyx.scipy.sparse  # noqa: F401
        cupy_available = True
    except Exception:                                       # noqa: BLE001
        cupy_available = False

    return cugraph_available, cudf_available, cupy_available


CUGRAPH_AVAILABLE, CUDF_AVAILABLE, CUPY_AVAILABLE = detect_backends()


def require_any_backend(algorithm_name: str) -> None:
    """Raise a clear ``RuntimeError`` when neither cuGraph nor CuPy is present.

    Baselines never fall back to CPU silently.
    """
    if not CUGRAPH_AVAILABLE and not CUPY_AVAILABLE:
        raise RuntimeError(
            f"{algorithm_name}_gpu_baseline requires either cuGraph or CuPy "
            f"on the GPU.  Install one of:\n"
            f"  - cugraph (preferred — full RAPIDS stack)\n"
            f"  - cupy-cuda12x or cupy-cuda13x (CuPy fallback path)\n"
            f"This implementation never falls back to CPU."
        )


# ---------------------------------------------------------------------------
# cuGraph version probe — used by every algorithm before invoking an API.
# ---------------------------------------------------------------------------

def cugraph_function(name: str):
    """Return ``cugraph.<name>`` if it exists, else None.

    Per the user's instruction, we never hardcode the existence of a cuGraph
    API — we look it up at runtime so version drift between RAPIDS releases
    never silently breaks the baseline.
    """
    if not CUGRAPH_AVAILABLE:
        return None
    try:
        import cugraph
        return getattr(cugraph, name, None)
    except Exception:                                       # noqa: BLE001
        return None


def cugraph_version() -> str:
    """Return the installed cuGraph version, or ``"?"`` when unavailable."""
    if not CUGRAPH_AVAILABLE:
        return "unavailable"
    try:
        import cugraph
        return str(getattr(cugraph, "__version__", "?"))
    except Exception:                                       # noqa: BLE001
        return "?"


def cupy_version() -> str:
    """Return the installed CuPy version, or ``"unavailable"`` otherwise."""
    if not CUPY_AVAILABLE:
        return "unavailable"
    try:
        import cupy
        return str(getattr(cupy, "__version__", "?"))
    except Exception:                                       # noqa: BLE001
        return "?"


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
) -> dict:
    """Assemble the standard 7-key result dict for a baseline run.

    The runner overwrites ``execution_time`` with its CUDA-event timer, but
    we still populate the internal value for direct (non-runner) callers.
    """
    return {
        "algorithm":      algorithm,
        "mode":           BASELINE_MODE,
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
# scipy CSR <-> CuPy sparse helpers (only loaded if CuPy is present)
# ---------------------------------------------------------------------------

def to_cupy_csr(graph_csr: sp.csr_matrix):
    """Convert a scipy CSR to a cupyx.scipy.sparse CSR (float32)."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available — cannot upload CSR.")
    import cupy as cp
    import cupyx.scipy.sparse as cpsp

    if graph_csr.dtype != np.float32:
        graph_csr = graph_csr.astype(np.float32)
    return cpsp.csr_matrix(
        (
            cp.asarray(graph_csr.data,    dtype=cp.float32),
            cp.asarray(graph_csr.indices, dtype=cp.int32),
            cp.asarray(graph_csr.indptr,  dtype=cp.int32),
        ),
        shape=graph_csr.shape,
    )


def to_cupy_array(arr: np.ndarray):
    """Move a numpy array to the GPU.  Type-preserving."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available — cannot upload array.")
    import cupy as cp
    return cp.asarray(arr)


def cupy_get(cp_arr) -> np.ndarray:
    """Move a CuPy array (or array-like) back to host as numpy."""
    if hasattr(cp_arr, "get"):
        return cp_arr.get()
    return np.asarray(cp_arr)


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
    if not CUGRAPH_AVAILABLE:
        raise RuntimeError("cuGraph is not available.")
    import cugraph
    import numpy as np

    coo = graph_csr.tocoo()
    src = coo.row.astype(np.int32)
    dst = coo.col.astype(np.int32)
    wts = coo.data.astype(np.float32) if weighted else None

    # Try cuDF first (the modern path).
    if CUDF_AVAILABLE:
        try:
            import cudf
            edge_df = cudf.DataFrame({"src": src, "dst": dst})
            if weighted:
                edge_df["weight"] = wts

            G = cugraph.Graph(directed=directed)
            kwargs: dict[str, Any] = {"source": "src", "destination": "dst"}
            if weighted:
                kwargs["edge_attr"] = "weight"
            G.from_cudf_edgelist(edge_df, **kwargs)
            return G
        except Exception as exc:                            # noqa: BLE001
            logging.debug(
                "cugraph_build_graph: cuDF edge list path failed (%s) — "
                "trying pandas fallback.", exc,
            )

    # Pandas fallback (older RAPIDS or cuDF missing).
    try:
        import pandas as pd
        edge_df = pd.DataFrame({"src": src, "dst": dst})
        if weighted:
            edge_df["weight"] = wts
        G = cugraph.Graph(directed=directed)
        kwargs = {"source": "src", "destination": "dst"}
        if weighted:
            kwargs["edge_attr"] = "weight"
        G.from_pandas_edgelist(edge_df, **kwargs)
        return G
    except Exception as exc:                                # noqa: BLE001
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
# Diagnostic logging — runs once on first import so it shows up in benchmarks.
# ---------------------------------------------------------------------------

def log_backend_selection() -> None:
    """Emit a single info-level log line documenting backend availability."""
    backends = []
    if CUGRAPH_AVAILABLE:
        backends.append(f"cuGraph={cugraph_version()}")
    if CUPY_AVAILABLE:
        backends.append(f"CuPy={cupy_version()}")
    if not backends:
        logging.warning(
            "gpu_baseline: neither cuGraph nor CuPy is available — every "
            "<algorithm>_gpu_baseline() call will raise RuntimeError."
        )
    else:
        logging.info("gpu_baseline backends detected: %s", ", ".join(backends))


log_backend_selection()
