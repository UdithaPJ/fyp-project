"""
src/algorithms/cpu/multi_threaded/_graphblas_utils.py
======================================================

Shared helpers for the GraphBLAS-backed ``cpu_multi`` implementations.

Background
----------
The 2026-06 memory/performance audit (see ``experiments/outputs/reports/
memory_audit_report.md``) showed that the original ``cpu_multi`` mode —
``multiprocessing.Pool`` / ``ProcessPoolExecutor`` over scipy SpMV — was
structurally slower than ``cpu_single`` on every SpMV-bound algorithm
because:

  * pickling the CSR slices to each worker per iteration (~50–150 ms)
  * process-spawn cost on Windows (~1 s per worker)
  * BLAS oversubscription (4 workers × MKL threads each)
  * Amdahl ceiling at ~3.1× even with zero overhead

All six ``cpu_multi`` algorithm files now use SuiteSparse:GraphBLAS via
``python-graphblas``.  This is a single-process implementation that uses
**OpenMP threads inside SuiteSparse C kernels** — no pickle, no spawn,
no GIL contention, no BLAS competition.

The mode string remains ``"cpu_multi"`` so benchmark CSVs, plot wiring,
and the runner adapter do not change.  Only the implementation under the
existing public function names changes.

Install
-------
``pip install python-graphblas`` (Linux/macOS) or
``conda install -c conda-forge python-graphblas`` (Windows recommended).

If python-graphblas is not installed, every cpu_multi call raises a
clear ``RuntimeError`` at function-entry with install instructions.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import scipy.sparse as sp

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import of python-graphblas
# ---------------------------------------------------------------------------

_GRAPHBLAS_AVAILABLE = False
_GB_IMPORT_ERROR: Optional[str] = None

try:
    import graphblas as gb              # type: ignore
    _GRAPHBLAS_AVAILABLE = True
except ImportError as _exc:             # pragma: no cover
    gb = None                            # type: ignore[assignment]
    _GB_IMPORT_ERROR = str(_exc)


def _require_graphblas() -> None:
    """Raise a descriptive error message when python-graphblas is missing."""
    if not _GRAPHBLAS_AVAILABLE:
        raise RuntimeError(
            "cpu_multi mode requires python-graphblas (SuiteSparse:GraphBLAS).\n"
            "Install with one of:\n"
            "  pip install python-graphblas\n"
            "  conda install -c conda-forge python-graphblas\n"
            f"Original ImportError: {_GB_IMPORT_ERROR}"
        )


# ---------------------------------------------------------------------------
# Thread control
# ---------------------------------------------------------------------------

def _configure_threads(n_threads: Optional[int] = None) -> int:
    """Configure SuiteSparse:GraphBLAS OpenMP thread count.

    Parameters
    ----------
    n_threads : int | None
        Desired thread count.  When None / 0 / negative, uses
        ``OMP_NUM_THREADS`` if set, otherwise ``os.cpu_count()``.

    Returns
    -------
    int
        The effective thread count (0 if GraphBLAS is unavailable).
    """
    if not _GRAPHBLAS_AVAILABLE:
        return 0
    if not n_threads or int(n_threads) <= 0:
        n_threads = int(os.environ.get("OMP_NUM_THREADS", "0")) or os.cpu_count() or 4
    try:
        gb.ss.config["nthreads"] = int(n_threads)
    except Exception as exc:                                       # noqa: BLE001
        _LOG.debug("could not set graphblas nthreads=%d: %s", n_threads, exc)
    return int(n_threads)


# ---------------------------------------------------------------------------
# Conversion helpers (scipy <-> graphblas)
# ---------------------------------------------------------------------------

def _from_scipy(csr: sp.spmatrix, dtype: object = None):
    """Convert scipy sparse matrix to a graphblas Matrix.

    The scipy-to-graphblas entry point lives at ``gb.io.from_scipy_sparse``
    in python-graphblas 2024+ (NOT ``gb.Matrix.from_scipy_sparse``).
    """
    if dtype is not None and csr.dtype != dtype:
        csr = csr.astype(dtype)
    return gb.io.from_scipy_sparse(csr)


def _to_scipy(M, fmt: str = "csr") -> sp.spmatrix:
    """Convert a graphblas Matrix to a scipy sparse matrix."""
    return gb.io.to_scipy_sparse(M, fmt)


def _vec_from_np(arr: np.ndarray, dtype: object = None):
    """Build a dense graphblas Vector from a numpy array."""
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(arr.dtype)  # graphblas auto-casts via .from_dense
    return gb.Vector.from_dense(arr)


def _vec_to_np(
    v,
    n: int,
    dtype: np.dtype = np.float32,
    fill: float = 0.0,
) -> np.ndarray:
    """Convert a graphblas Vector to a dense numpy array of length n."""
    arr = v.to_dense(fill_value=fill)
    arr = np.asarray(arr)
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def _vec_sparse_bool(indices: np.ndarray, n: int):
    """Build a sparse bool Vector with True at the given indices."""
    idx = np.asarray(indices, dtype=np.int64)
    vals = np.ones(idx.size, dtype=bool)
    return gb.Vector.from_coo(idx, vals, size=n, dtype=gb.dtypes.BOOL)


__all__ = [
    "_GRAPHBLAS_AVAILABLE",
    "_require_graphblas",
    "_configure_threads",
    "_from_scipy",
    "_to_scipy",
    "_vec_from_np",
    "_vec_to_np",
    "_vec_sparse_bool",
    "gb",
]
