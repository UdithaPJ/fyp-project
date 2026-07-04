"""
src/algorithms/cpu/multi_threaded/louvain.py
============================================

Louvain — cpu_multi mode, NetworKit Parallel Louvain Method (PLM / PLMR),
run in a CUDA-isolated subprocess.

.. note::
   SuiteSparse:GraphBLAS has **no** native Louvain primitive, and Louvain
   Phase 1 (sequential greedy node-by-node modularity maximization) does
   not vectorize cleanly under GraphBLAS semirings.  Every other
   ``cpu_multi`` algorithm in this project is GraphBLAS-backed; Louvain is
   the single, intentional exception:  it uses **NetworKit** instead.

   NetworKit's ``community.PLM`` is the *Parallel Louvain Method*
   (Staudt & Meyerhenke) — a genuine OpenMP-multithreaded modularity
   optimiser implemented in C++.  ``PLMR`` is the same algorithm with an
   extra refinement sweep (``refine=True``), which usually yields slightly
   higher modularity.

Why a subprocess
----------------
Importing ``src.algorithms`` builds the algorithm registry at import time,
which forces a live CUDA context (``cupy.zeros(1)`` inside
``src.benchmarking.benchmark``).  NetworKit's bundled OpenMP/TBB runtime
cannot share a process with an already-initialized CUDA context: running
``PLM.run()`` then aborts the whole process with
``malloc(): mismatching next->prev_size`` (glibc heap corruption from two
native threading/allocator runtimes colliding).  ``MALLOC_ARENA_MAX=1``
does not help.

The fix is process isolation:  the NetworKit computation runs in
``_networkit_worker.py``, launched **by absolute file path** so the child
interpreter never imports the ``src`` package chain and therefore never
creates a CUDA context.  The symmetric adjacency is handed over via a
temporary ``.npz`` and the community labels come back via a ``.npy``.

Result shape
------------
``resolution`` maps to NetworKit's ``gamma``.  Community labels are
renumbered to a contiguous ``0..K-1`` range and passed through the shared
:func:`_build_result`, so ``num_communities`` / ``modularity`` /
``top_communities`` are computed identically to ``cpu_single`` — keeping
benchmark comparisons apples-to-apples.  Multi-level coarsening
(``recurse=True``) is internal to PLM, so ``max_levels`` /
``max_phase1_passes`` are not forwarded.

If NetworKit is **not** installed the call degrades gracefully to the
deterministic scipy ``cpu_single`` implementation (with a warning).

Install
-------
``pip install networkit``  (prebuilt Linux/macOS wheels; MIT licensed)

For the deterministic single-thread variant see
``src.algorithms.cpu.single_threaded.louvain``.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import scipy.sparse as sp

from src.algorithms.common.helpers import _build_result, _symmetrize

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NetworKit availability — checked WITHOUT importing networkit.
#
# The parent process already holds a CUDA context (see module docstring);
# even a bare ``import networkit`` loads its native threading runtime into
# this poisoned process.  ``find_spec`` tells us whether the package is
# installed without importing it — the real import happens only in the
# isolated worker subprocess.
# ---------------------------------------------------------------------------

_NETWORKIT_AVAILABLE = importlib.util.find_spec("networkit") is not None

_WORKER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_networkit_worker.py"
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "min_delta_q":       1e-4,
    "max_levels":        10,
    "resolution":        1.0,
    "max_phase1_passes": 100,
    # NetworKit-specific: refine=True selects PLMR (PLM + refinement pass),
    # which trades a little runtime for slightly higher modularity.
    "refine":            True,
}


def _merge_params(user_params: dict | None) -> dict:
    return {**_DEFAULT_PARAMS, **(user_params or {})}


# ---------------------------------------------------------------------------
# Thread control
# ---------------------------------------------------------------------------

def _resolve_thread_count(n_workers: int | None = None) -> int:
    """Resolve the OpenMP thread count to request from the worker.

    When ``n_workers`` is None / 0 / negative, uses ``OMP_NUM_THREADS`` if
    set, otherwise ``os.cpu_count()``.  This does NOT touch networkit — the
    worker subprocess applies it via ``nk.setNumberOfThreads``.
    """
    if n_workers and int(n_workers) > 0:
        return int(n_workers)
    return int(os.environ.get("OMP_NUM_THREADS", "0")) or os.cpu_count() or 4


# ---------------------------------------------------------------------------
# Isolated NetworKit subprocess
# ---------------------------------------------------------------------------

def _run_networkit_subprocess(
    A_sym:     sp.csr_matrix,
    refine:    bool,
    gamma:     float,
    n_threads: int,
) -> np.ndarray:
    """Run PLM / PLMR in a CUDA-free child process; return raw label vector.

    Hands ``A_sym`` to the worker via a temp ``.npz`` and reads the int64
    community-label vector back from a temp ``.npy``.  Raises
    ``RuntimeError`` (with the child's stderr) if the worker fails, rather
    than silently masking a NetworKit problem behind a slow CPU fallback.
    """
    tmpdir  = tempfile.mkdtemp(prefix="nk_louvain_")
    in_npz  = os.path.join(tmpdir, "A_sym.npz")
    out_npy = os.path.join(tmpdir, "labels.npy")
    try:
        sp.save_npz(in_npz, A_sym.tocsr())
        cmd = [
            sys.executable, _WORKER_PATH, in_npz, out_npy,
            "1" if refine else "0", repr(float(gamma)), str(int(n_threads)),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(out_npy):
            raise RuntimeError(
                "NetworKit Louvain worker subprocess failed "
                f"(exit={proc.returncode}).\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stderr:\n{proc.stderr.strip()}"
            )
        return np.load(out_npy)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# NetworKit PLM / PLMR path
# ---------------------------------------------------------------------------

def _louvain_networkit(
    graph_csr: sp.csr_matrix,
    params:    dict,
    n_threads: int,
) -> dict:
    """Run NetworKit PLM / PLMR (subprocess) and pack the result dict."""
    A_sym  = _symmetrize(graph_csr)
    m      = float(A_sym.sum()) / 2.0
    refine = bool(params.get("refine", True))
    gamma  = float(params.get("resolution", 1.0))

    raw_labels = _run_networkit_subprocess(A_sym, refine, gamma, n_threads)

    # NetworKit subset ids are not necessarily contiguous — renumber 0..K-1.
    _, final_labels = np.unique(raw_labels, return_inverse=True)
    final_labels = final_labels.astype(np.int32)

    # Multi-level hierarchy is internal to PLM and not exposed as a list of
    # per-level label vectors, so hierarchy is left empty (mirrors the
    # gpu_baseline result shape).
    result = _build_result(A_sym, final_labels, [], m, gamma)

    algo = "PLMR" if refine else "PLM"
    result["note"] = (
        f"cpu_multi backend: NetworKit {algo} (parallel Louvain, "
        f"subprocess-isolated from CUDA), threads={n_threads}, gamma={gamma}."
    )
    return result


# ---------------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------------

def louvain_cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict,
    n_workers: int | None = None,
) -> dict:
    """Run Louvain via NetworKit's parallel PLM / PLMR in an isolated process.

    Falls back to the deterministic scipy ``cpu_single`` implementation
    (with a warning) if NetworKit is not installed, so the framework keeps
    functioning on machines without it.
    """
    if not _NETWORKIT_AVAILABLE:
        warnings.warn(
            "louvain_cpu_multi: NetworKit is not installed; falling back to "
            "scipy cpu_single.  Install parallel Louvain with "
            "'pip install networkit'.",
            UserWarning, stacklevel=2,
        )
        from src.algorithms.cpu.single_threaded.louvain import louvain_cpu_single
        result = louvain_cpu_single(graph_csr, params)
        existing_note = result.get("note") or ""
        extra = (
            "cpu_multi backend: NetworKit unavailable — "
            "delegated to scipy cpu_single."
        )
        result["note"] = (existing_note + " | " + extra).strip(" |")
        return result

    n_threads = _resolve_thread_count(n_workers)
    return _louvain_networkit(graph_csr, params, n_threads)


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

def _cpu_multi(
    graph_csr: sp.csr_matrix,
    params:    dict | None = None,
    n_workers: int | None = None,
    **_,
) -> dict:
    """Benchmark runner entry-point for cpu_multi mode."""
    p = _merge_params(params)
    if _NETWORKIT_AVAILABLE:
        backend = "networkit_plmr" if bool(p.get("refine", True)) else "networkit_plm"
    else:
        backend = "networkit_unavailable_fallback_scipy"
    return {
        "output":       louvain_cpu_multi(graph_csr, p, n_workers=n_workers),
        "extra_params": {**p, "backend": backend},
    }
