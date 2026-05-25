"""
src/runner/algorithm_runner.py — THE algorithm orchestration entrypoint
========================================================================

This module is the *only* place where the GPU configuration layer, the
parameter validator, the timer, and the algorithm implementations meet.
Algorithms remain pure computation — they receive a fully-merged
``params`` dict and return a standardised result dict.

Pipeline
--------
    1. Resolve the algorithm class from :data:`ALGORITHM_REGISTRY`.
    2. Apply GPU recommendations via :func:`apply_config` (always
       called — for CPU modes it still merges, just with the cpu_only
       recommended block which is harmless).
    3. Validate / merge user params via :meth:`AlgorithmBase.validate_params`.
    4. Wrap the mode call in :class:`BenchmarkTimer` (CUDA-event accurate
       for GPU mode, ``perf_counter`` for CPU modes).
    5. Override ``execution_time`` in the result with the runner's
       measurement (more accurate than the algorithm's internal one).
    6. Attach node *labels* to integer indices in well-known top-K fields.
    7. Validate the result dict structure via :meth:`AlgorithmBase.validate_result`.
    8. Emit progress events at every stage via :class:`ProgressReporter`.

Public API
----------
    run_algorithm(...)            → standardised result dict
    get_algorithm_info(name)      → schema for one algorithm
    list_algorithms()             → schemas for all 6 algorithms

The runner intentionally does NOT do any file I/O — it returns the
result dict for the caller (web service, CLI, notebook) to persist.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import scipy.sparse as sp

from ..algorithms import ALGORITHM_REGISTRY, AlgorithmBase
from ..optimization.gpu_config import apply_config, get_gpu_config
from .progress import ProgressReporter


# ---------------------------------------------------------------------------
# GPU backend detection (mirrors benchmark/runner.py from the FYP project)
# ---------------------------------------------------------------------------

_GPU_TIMER_BACKEND: Optional[str] = None
_CUDA_AVAILABLE = False
_cp = None
_pycuda = None

try:
    import cupy as _cp_mod
    # Force context init so the first CuPy SpMV in the algorithm does not
    # collide with timer events created here.
    _ = _cp_mod.zeros(1)
    _cp = _cp_mod
    _GPU_TIMER_BACKEND = "cupy"
    _CUDA_AVAILABLE = True
except Exception:
    pass

if not _CUDA_AVAILABLE:
    try:
        import pycuda.driver as _pc_mod
        _pc_mod.init()
        if _pc_mod.Device.count() > 0:
            import pycuda.autoinit  # noqa: F401
            _pycuda = _pc_mod
            _GPU_TIMER_BACKEND = "pycuda"
            _CUDA_AVAILABLE = True
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class BenchmarkTimer:
    """Context manager — CUDA events for GPU mode, perf_counter for CPU."""

    def __init__(self, mode: str):
        self.mode = mode
        self.elapsed: float = 0.0
        self._t0 = None
        self._evt_start = None
        self._evt_end = None

    def __enter__(self) -> "BenchmarkTimer":
        if self.mode == "gpu" and _CUDA_AVAILABLE:
            if _GPU_TIMER_BACKEND == "cupy":
                self._evt_start = _cp.cuda.Event()
                self._evt_end   = _cp.cuda.Event()
                self._evt_start.record()
            else:
                self._evt_start = _pycuda.Event()
                self._evt_end   = _pycuda.Event()
                self._evt_start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.mode == "gpu" and _CUDA_AVAILABLE:
            self._evt_end.record()
            if _GPU_TIMER_BACKEND == "cupy":
                self._evt_end.synchronize()
                self.elapsed = _cp.cuda.get_elapsed_time(
                    self._evt_start, self._evt_end
                ) / 1000.0
            else:
                self._evt_end.synchronize()
                self.elapsed = self._evt_start.time_till(self._evt_end) / 1000.0
        else:
            self.elapsed = time.perf_counter() - self._t0
        return False


# ---------------------------------------------------------------------------
# Node-label attachment
# ---------------------------------------------------------------------------

# Result-dict fields that contain lists of integer node indices.  These
# are converted to {"index": int, "label": str} pairs so the frontend
# can display gene names without a second lookup.
_INDEXED_FIELDS = (
    "top_regulators",
    "top_targets",
    "top_nodes",
    "top_tfs",
    "top_hubs",
    "top_authorities",
    "hub_authority_overlap",
    "visited_order",
)


def _attach_labels(
    result: dict,
    node_index_map: dict,
) -> dict:
    """
    Walk the result dict and replace integer indices in known top-K fields
    with ``{"index": int, "label": str}`` pairs.

    Also attaches labels inside ``result['result']['top_communities'][*]['member_nodes']``
    when present (Louvain output).

    The original integer arrays are PRESERVED at sibling keys with a
    ``_indices`` suffix so downstream code that needs raw indices (e.g.
    graph-viz highlighting) still has them.
    """
    if not isinstance(result, dict) or "result" not in result:
        return result

    # Build reverse map: int → label
    index_to_label = {int(v): str(k) for k, v in node_index_map.items()}

    def _label(idx: int) -> str:
        return index_to_label.get(int(idx), f"node_{idx}")

    inner = result["result"]
    if not isinstance(inner, dict):
        return result

    for field in _INDEXED_FIELDS:
        if field in inner and isinstance(inner[field], list):
            indices = inner[field]
            inner[f"{field}_indices"] = list(indices)
            inner[field] = [
                {"index": int(i), "label": _label(i)} for i in indices
            ]

    # Louvain: top_communities[].member_nodes
    if "top_communities" in inner and isinstance(inner["top_communities"], list):
        for entry in inner["top_communities"]:
            if isinstance(entry, dict) and "member_nodes" in entry:
                indices = entry["member_nodes"]
                entry["member_nodes_indices"] = list(indices)
                entry["member_nodes"] = [
                    {"index": int(i), "label": _label(i)} for i in indices
                ]

    return result


# ---------------------------------------------------------------------------
# Public: run_algorithm
# ---------------------------------------------------------------------------

def run_algorithm(
    algorithm_name: str,
    graph_csr: sp.csr_matrix,
    node_index_map: dict,
    mode: str = "gpu",
    params: Optional[dict] = None,
    progress_reporter: Optional[ProgressReporter] = None,
) -> dict:
    """
    Run an algorithm end-to-end and return the standardised result dict.

    Parameters
    ----------
    algorithm_name : str
        One of ``pagerank``, ``louvain``, ``rwr``, ``hits``, ``bfs``, ``mcl``.
    graph_csr : scipy.sparse.csr_matrix
        The graph in CSR format (output of ``src/graph/converter.graphdata_to_csr``).
    node_index_map : dict
        ``{label: index}`` mapping used to attach human-readable node names
        to result fields like ``top_regulators``.
    mode : str
        ``"cpu_single"``, ``"cpu_multi"``, or ``"gpu"``.  GPU mode falls
        back to ``cpu_single`` (with a warning) if no CUDA device is present.
    params : dict, optional
        User-supplied algorithm parameters.  Merged on top of the GPU-tuned
        recommended defaults.
    progress_reporter : ProgressReporter, optional
        If provided, progress events are emitted at every stage.
    """
    reporter = progress_reporter or ProgressReporter(callback=None)
    reporter.update(ProgressReporter.STAGE_STARTING, 0,
                    f"Preparing to run {algorithm_name} ({mode})")

    # ---- 1. Resolve algorithm class ----
    algo_cls = ALGORITHM_REGISTRY.get(algorithm_name)
    if algo_cls is None:
        reporter.error(f"Unknown algorithm '{algorithm_name}'")
        raise ValueError(
            f"Unknown algorithm '{algorithm_name}'.  "
            f"Available: {sorted(ALGORITHM_REGISTRY.keys())}"
        )

    if mode not in ("cpu_single", "cpu_multi", "gpu"):
        reporter.error(f"Invalid mode '{mode}'")
        raise ValueError(
            f"Invalid mode '{mode}'.  "
            f"Must be one of: cpu_single, cpu_multi, gpu"
        )

    # ---- 2. GPU configuration (hardware-aware param tuning) ----
    reporter.update(ProgressReporter.STAGE_CONFIG, 10, "Tuning GPU parameters")
    user_params = dict(params or {})
    tuned_params = apply_config(algorithm_name, graph_csr, user_params)

    # ---- 3. Validate / merge against algorithm's PARAM_SCHEMA ----
    reporter.update(ProgressReporter.STAGE_VALIDATE, 20,
                    "Validating algorithm parameters")
    final_params = algo_cls.validate_params(tuned_params)

    # ---- 4. Resolve mode function & run timed ----
    mode_fn = getattr(algo_cls, mode, None)
    if mode_fn is None:
        reporter.error(f"{algorithm_name} has no {mode} method")
        raise AttributeError(f"{algorithm_name} has no {mode}() method")

    reporter.update(ProgressReporter.STAGE_RUNNING, 30,
                    f"Running {algorithm_name}.{mode}()")
    try:
        with BenchmarkTimer(mode) as timer:
            result = mode_fn(graph_csr, final_params)
    except Exception as exc:
        reporter.error(f"{algorithm_name}.{mode} raised "
                       f"{type(exc).__name__}: {exc}")
        raise

    # ---- 5. Override execution_time with the runner's timer ----
    reporter.update(ProgressReporter.STAGE_PACKING, 80, "Packing results")
    if isinstance(result, dict):
        result["execution_time"] = float(timer.elapsed)
        result["mode"] = mode   # GPU fallback to cpu_single overwrites this back

    # ---- 6. Attach node labels to known index-containing fields ----
    result = _attach_labels(result, node_index_map)

    # ---- 7. Validate result dict structure ----
    AlgorithmBase.validate_result(result)

    # ---- 8. Final stage ----
    reporter.update(ProgressReporter.STAGE_DONE, 100,
                    f"Completed in {result['execution_time']:.4f}s")
    reporter.done(result)
    return result


# ---------------------------------------------------------------------------
# Public: algorithm introspection
# ---------------------------------------------------------------------------

def get_algorithm_info(algorithm_name: str) -> dict:
    """
    Return ``{"name", "param_schema", "description"}`` for one algorithm.
    Used by the frontend to dynamically build parameter input forms.
    """
    algo_cls = ALGORITHM_REGISTRY.get(algorithm_name)
    if algo_cls is None:
        raise ValueError(
            f"Unknown algorithm '{algorithm_name}'.  "
            f"Available: {sorted(ALGORITHM_REGISTRY.keys())}"
        )
    return algo_cls.describe()


def list_algorithms() -> list[dict]:
    """Return :func:`get_algorithm_info` for every registered algorithm."""
    return [algo_cls.describe() for algo_cls in ALGORITHM_REGISTRY.values()]


# ---------------------------------------------------------------------------
# Convenience — expose GPU info for callers without importing the optimizer
# ---------------------------------------------------------------------------

def get_runtime_info() -> dict:
    """Return current GPU configuration — handy for /status endpoints."""
    return get_gpu_config()
