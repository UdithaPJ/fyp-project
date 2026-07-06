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
    run_algorithm(...)            -> standardised result dict
    get_algorithm_info(name)      -> schema for one algorithm
    list_algorithms()             -> schemas for all 6 algorithms

The runner intentionally does NOT do any file I/O — it returns the
result dict for the caller (web service, CLI, notebook) to persist.
"""

from __future__ import annotations

from contextlib import contextmanager
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
_PYCUDA_PRIMARY_CONTEXT = None

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
            _pycuda = _pc_mod
            _GPU_TIMER_BACKEND = "pycuda"
            _CUDA_AVAILABLE = True
    except Exception:
        pass


def _needs_pycuda_context(algorithm_name: str, mode: str) -> bool:
    """Return True when a PyCUDA context must be held for the whole call.

    All GPU modes now return True because every algorithm uses
    ``retain_primary_context().push()/pop()`` internally, and
    ``BenchmarkTimer`` creates CUDA events in ``__enter__`` that must
    remain valid through ``__exit__`` (after the algorithm's own pop).
    Holding a parent push here keeps the context alive through the
    BenchmarkTimer boundary.
    """
    if mode not in ("gpu", "gpu_baseline"):
        return False
    # All GPU modes need the context held for the BenchmarkTimer boundary.
    return True


@contextmanager
def _cuda_context_guard(algorithm_name: str, mode: str):
    """Ensure the executing thread has a current CUDA context when needed.

    For all GPU modes we push the PyCUDA primary context here and pop it
    AFTER ``BenchmarkTimer.__exit__`` has recorded its end event.  The
    algorithm's own ``retain_primary_context().push()/pop()`` calls are
    nested inside and are safe because ``retain_primary_context`` is
    ref-counted — the context only becomes non-current when ALL pushes
    have been matched by pops.  Without this outer push, the algorithm's
    pop would make the context non-current while ``BenchmarkTimer.__exit__``
    still needs to call ``cuEventRecord``.
    """
    pushed = False

    # Ensure a PyCUDA context is current for all GPU runs.
    if _needs_pycuda_context(algorithm_name, mode):
        global _PYCUDA_PRIMARY_CONTEXT
        try:
            import pycuda.driver as cuda

            cuda.init()
            if _PYCUDA_PRIMARY_CONTEXT is None:
                if cuda.Device.count() <= 0:
                    raise RuntimeError("no CUDA devices detected")
                # Use the PRIMARY context so it coexists with CuPy.
                _PYCUDA_PRIMARY_CONTEXT = cuda.Device(0).retain_primary_context()

            try:
                current = cuda.Context.get_current()
            except Exception:
                current = None

            if current is None:
                _PYCUDA_PRIMARY_CONTEXT.push()
                pushed = True
        except Exception:
            # No PyCUDA context available.  Let downstream code surface the
            # error or fall back.
            pushed = False

    # Ensure CuPy's primary context is current in this thread (no-op if
    # already set).  Done AFTER the PyCUDA push so both APIs bind to the
    # same underlying primary context.
    if mode in ("gpu", "gpu_baseline") and _cp is not None:
        try:
            _cp.cuda.Device(0).use()
        except Exception:
            pass

    try:
        yield
    finally:
        if pushed and _PYCUDA_PRIMARY_CONTEXT is not None:
            try:
                _PYCUDA_PRIMARY_CONTEXT.pop()
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
        if self.mode in ("gpu", "gpu_baseline") and _CUDA_AVAILABLE:
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
        if self.mode in ("gpu", "gpu_baseline") and _CUDA_AVAILABLE:
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

    # Build reverse map: int -> label
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
        ``"cpu_single"``, ``"cpu_multi"``, ``"gpu"``, or ``"gpu_baseline"``.
        ``gpu`` runs the heavily optimized cuda_optimized implementation;
        ``gpu_baseline`` runs the simple cuGraph / CuPy baseline used as a
        benchmark reference.  GPU mode falls back to ``cpu_single`` (with
        a warning) if no CUDA device is present.
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

    if mode not in ("cpu_single", "cpu_multi", "gpu", "gpu_baseline"):
        reporter.error(f"Invalid mode '{mode}'")
        raise ValueError(
            f"Invalid mode '{mode}'.  "
            f"Must be one of: cpu_single, cpu_multi, gpu, gpu_baseline"
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
        with _cuda_context_guard(algorithm_name, mode):
            with BenchmarkTimer(mode) as timer:
                result = mode_fn(graph_csr, final_params)
    except ImportError as exc:
        # gpu_baseline algorithms hard-fail with ImportError when their
        # required backend (cuGraph for 5 algorithms, CuPy for MCL) is
        # not installed.  Convert to RuntimeError with install instructions
        # so the caller always sees RuntimeError from the runner.
        if mode == "gpu_baseline":
            reporter.error(
                f"{algorithm_name}.{mode} backend not installed: {exc}"
            )
            raise RuntimeError(
                f"{algorithm_name}_gpu_baseline requires RAPIDS (cuGraph + "
                f"cuDF) for cuGraph algorithms, or CuPy for MCL.  "
                f"Install RAPIDS via:\n"
                f"  conda install -c rapidsai -c nvidia -c conda-forge "
                f"rapids=24.02 python=3.10 cudatoolkit=11.8\n"
                f"Original ImportError: {exc}"
            ) from exc
        reporter.error(f"{algorithm_name}.{mode} raised "
                       f"{type(exc).__name__}: {exc}")
        raise
    except Exception as exc:
        reporter.error(f"{algorithm_name}.{mode} raised "
                       f"{type(exc).__name__}: {exc}")
        raise

    # ---- 5. Override execution_time with the runner's timer ----
    reporter.update(ProgressReporter.STAGE_PACKING, 80, "Packing results")
    if isinstance(result, dict):
        result["execution_time"] = float(timer.elapsed)
        if mode == "gpu_baseline":
            # Preserve the algorithm's backend-specific mode string
            # ("gpu_baseline_cugraph" or "gpu_baseline_cupy") rather than
            # overwriting with the generic "gpu_baseline".
            pass
        else:
            result["mode"] = mode   # gpu fallback to cpu_single overwrites this back

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

# ---------------------------------------------------------------------------
# UI metadata schema for the algorithm parameter form
# ---------------------------------------------------------------------------
#
# UI_SCHEMA is *purely* metadata used by the frontend to build a typed
# parameter form (sliders, dropdowns, preset pills, node selectors).
# It is intentionally kept separate from the algorithm-side ``PARAM_SCHEMA``
# (which the runner uses for validation) so algorithm code never has to
# carry UI concerns.
#
# Each entry has:
#   display_name : str
#   description  : str            (one-paragraph algorithm summary)
#   category     : str            (used by the algorithm card chip)
#   params       : list[dict]     (rendered top-down in the form)
#
# Each param dict has at least:
#   key      : str   (matches the backend param name exactly)
#   label    : str   (human-readable form label)
#   type     : str   ("slider" | "number" | "select" | "preset_select"
#                     | "node_selector" | "multi_node_selector")
#   default  : Any   (initial form value; matches the backend default)
#   tooltip  : str   (shown on hover)
#   advanced : bool  (True -> behind the "Advanced Settings" toggle)
#
# Type-specific additional keys:
#   slider:              min, max, step, display_format ("scientific" optional)
#   number:              min, max, step (step may be None for free entry)
#   select:              options: [{value, label}, ...]
#   preset_select:       presets: [{label, value}, ...]
#                        (presets are display affordances; the raw
#                         float is always what gets submitted)
#   node_selector:       searchable: bool, placeholder: str
#   multi_node_selector: searchable: bool, placeholder: str
# ---------------------------------------------------------------------------

UI_SCHEMA: dict[str, dict] = {
    "pagerank": {
        "display_name": "PageRank",
        "description": (
            "Ranks nodes by influence propagation.  Higher scores "
            "= more regulatory influence."
        ),
        "category": "Ranking",
        "params": [
            {
                "key": "damping", "label": "Damping Factor",
                "type": "slider",
                "default": 0.85, "min": 0.1, "max": 0.99, "step": 0.01,
                "tooltip": (
                    "Probability of following a regulatory edge at each "
                    "step.  0.85 is the standard value for most "
                    "biological networks."
                ),
                "advanced": False,
            },
            {
                "key": "max_iter", "label": "Max Iterations",
                "type": "number",
                "default": 100, "min": 10, "max": 1000, "step": 10,
                "tooltip": (
                    "Maximum number of update steps before stopping.  "
                    "Increase if results seem unstable."
                ),
                "advanced": False,
            },
            {
                "key": "tolerance", "label": "Convergence Tolerance",
                "type": "number",
                "default": 1e-6, "min": 1e-10, "max": 1e-3, "step": None,
                "tooltip": (
                    "How precisely the algorithm must converge before "
                    "stopping.  Smaller = more precise but slower.  "
                    "Default 1e-6 suits most analyses."
                ),
                "advanced": True,
            },
        ],
    },
    "bfs": {
        "display_name": "Breadth-First Search",
        "description": (
            "Traces regulatory cascades from a starting gene, showing "
            "how influence spreads step by step."
        ),
        "category": "Traversal",
        "params": [
            {
                "key": "source", "label": "Source Node",
                "type": "node_selector",
                "default": 0,
                "searchable": True,
                "placeholder": "Search by gene name…",
                "tooltip": (
                    "The starting gene or transcription factor for "
                    "cascade tracing.  Try a known regulator like TP53."
                ),
                "advanced": False,
            },
            {
                "key": "max_depth", "label": "Maximum Cascade Depth",
                "type": "slider",
                "default": 5, "min": 1, "max": 10, "step": 1,
                "tooltip": (
                    "How many regulatory steps to trace outward from "
                    "the source.  Depth 1 = direct targets, depth 2 "
                    "= targets of targets, etc."
                ),
                "advanced": False,
            },
        ],
    },
    "louvain": {
        "display_name": "Louvain Community Detection",
        "description": (
            "Groups genes into functional modules based on how densely "
            "they are connected."
        ),
        "category": "Clustering",
        "params": [
            {
                "key": "resolution", "label": "Resolution",
                "type": "slider",
                "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1,
                "tooltip": (
                    "Higher values produce smaller, more detailed "
                    "communities.  Lower values produce larger "
                    "biological modules.  Start with 1.0."
                ),
                "advanced": False,
            },
            {
                "key": "max_levels", "label": "Maximum Hierarchy Levels",
                "type": "slider",
                "default": 10, "min": 1, "max": 20, "step": 1,
                "tooltip": (
                    "Number of coarsening rounds.  Higher values allow "
                    "deeper hierarchical structure but take longer.  "
                    "10 is sufficient for most networks."
                ),
                "advanced": False,
            },
            {
                "key": "min_delta_q", "label": "Minimum Modularity Gain",
                "type": "number",
                "default": 1e-4, "min": 1e-6, "max": 0.1, "step": None,
                "tooltip": (
                    "Minimum improvement required to move a node to a "
                    "new community.  Smaller values allow finer "
                    "optimisation."
                ),
                "advanced": True,
            },
        ],
    },
    "rwr": {
        "display_name": "Random Walk with Restart",
        "description": (
            "Spreads influence from seed genes through the network to "
            "find functionally related genes."
        ),
        "category": "Propagation",
        "params": [
            {
                "key": "restart_prob", "label": "Restart Probability",
                "type": "slider",
                "default": 0.3, "min": 0.1, "max": 0.9, "step": 0.05,
                "tooltip": (
                    "Controls how strongly the walk stays near the "
                    "seed nodes.  Higher values keep results closer to "
                    "seeds.  0.3 is a standard setting."
                ),
                "advanced": False,
            },
            {
                "key": "seed_nodes", "label": "Seed Nodes",
                "type": "multi_node_selector",
                "default": [],
                "searchable": True,
                "placeholder": "Search and add seed genes…",
                "tooltip": (
                    "Starting genes for the walk.  For GRNs, use known "
                    "transcription factors.  Leave empty to use all "
                    "nodes equally."
                ),
                "advanced": False,
            },
            {
                "key": "max_iter", "label": "Max Iterations",
                "type": "number",
                "default": 100, "min": 10, "max": 500, "step": 10,
                "tooltip": "Maximum walk iterations before stopping.",
                "advanced": False,
            },
            {
                "key": "tolerance", "label": "Convergence Tolerance",
                "type": "number",
                "default": 1e-6, "min": 1e-10, "max": 1e-3, "step": None,
                "tooltip": (
                    "How precisely the walk must stabilise before "
                    "stopping."
                ),
                "advanced": True,
            },
            {
                "key": "precision", "label": "Precision Mode",
                "type": "select",
                "default": "fp32",
                "options": [
                    {"value": "fp32",
                     "label": "FP32 — Accurate"},
                    {"value": "fp16_storage",
                     "label": "Mixed Precision — Faster"},
                ],
                "tooltip": (
                    "FP32 gives full precision.  Mixed Precision uses "
                    "less memory and runs faster but may have minor "
                    "numerical differences on large networks."
                ),
                "advanced": True,
            },
        ],
    },
    "hits": {
        "display_name": "HITS",
        "description": (
            "Scores each node as a hub (regulates others) and an "
            "authority (is regulated by others)."
        ),
        "category": "Ranking",
        "params": [
            {
                "key": "max_iter", "label": "Max Iterations",
                "type": "number",
                "default": 100, "min": 10, "max": 500, "step": 10,
                "tooltip": (
                    "Maximum number of HITS update iterations before "
                    "stopping.  100 is sufficient for most biological "
                    "networks."
                ),
                "advanced": False,
            },
            {
                "key": "tolerance", "label": "Precision",
                "type": "preset_select",
                "default": 1e-6,
                "presets": [
                    {"label": "Fast",           "value": 1e-4},
                    {"label": "Balanced",       "value": 1e-6},
                    {"label": "High Precision", "value": 1e-8},
                ],
                "tooltip": (
                    "How precisely hub and authority scores must "
                    "stabilise before stopping.  Balanced is "
                    "recommended for most analyses."
                ),
                "advanced": False,
            },
        ],
    },
    "mcl": {
        "display_name": "Markov Clustering (MCL)",
        "description": (
            "Finds protein complexes and gene modules by simulating "
            "random walks on the network."
        ),
        "category": "Clustering",
        "params": [
            {
                "key": "inflation", "label": "Inflation",
                "type": "slider",
                "default": 2.0, "min": 1.4, "max": 6.0, "step": 0.1,
                "tooltip": (
                    "Higher values produce smaller, tighter clusters.  "
                    "Lower values produce larger, more overlapping "
                    "communities.  Suggested range: 1.4 – 6.0.  Start "
                    "with 2.0 for most biological networks."
                ),
                "advanced": False,
            },
            {
                "key": "expansion", "label": "Expansion",
                "type": "select",
                "default": 2,
                "options": [
                    {"value": 2, "label": "2 — Standard (recommended)"},
                    {"value": 3, "label": "3 — Deeper diffusion"},
                    {"value": 4, "label": "4 — Maximum diffusion (slow)"},
                ],
                "tooltip": (
                    "Controls how far random walks spread at each "
                    "step.  Higher values allow deeper diffusion but "
                    "use significantly more memory.  Value of 2 is "
                    "standard for MCL."
                ),
                "advanced": False,
            },
            {
                "key": "prune_threshold", "label": "Prune Threshold",
                "type": "slider",
                "default": 0.001, "min": 0.0001, "max": 0.05, "step": 0.0001,
                "display_format": "scientific",
                "tooltip": (
                    "Removes weak connections below this value to save "
                    "memory.  Higher values improve speed and reduce "
                    "memory usage but may remove weak biological "
                    "relationships.  0.001 is the recommended starting "
                    "point."
                ),
                "advanced": False,
            },
            {
                "key": "max_iter", "label": "Max Iterations",
                "type": "number",
                "default": 100, "min": 10, "max": 500, "step": 10,
                "tooltip": "Maximum clustering iterations before stopping.",
                "advanced": True,
            },
            {
                "key": "convergence_tol", "label": "Convergence Tolerance",
                "type": "preset_select",
                "default": 1e-4,
                "presets": [
                    {"label": "Fast",     "value": 1e-3},
                    {"label": "Standard", "value": 1e-4},
                    {"label": "Precise",  "value": 1e-5},
                ],
                "tooltip": (
                    "How stable the clustering must be before "
                    "stopping.  Standard suits most analyses."
                ),
                "advanced": True,
            },
        ],
    },
}


def get_algorithm_info(algorithm_name: str) -> dict:
    """
    Return ``{"name", "param_schema", "description",
    "display_name", "category", "ui_schema"}`` for one algorithm.

    ``param_schema`` is kept (and unchanged) for backend validators;
    ``ui_schema`` carries the new metadata the frontend uses to build
    the parameter form.  Algorithms with no UI_SCHEMA entry fall back
    to an empty ``ui_schema`` list — the legacy form-from-param_schema
    rendering path still works.
    """
    algo_cls = ALGORITHM_REGISTRY.get(algorithm_name)
    if algo_cls is None:
        raise ValueError(
            f"Unknown algorithm '{algorithm_name}'.  "
            f"Available: {sorted(ALGORITHM_REGISTRY.keys())}"
        )
    base = dict(algo_cls.describe())
    ui_entry = UI_SCHEMA.get(algorithm_name, {})
    base["display_name"] = ui_entry.get("display_name",
                                        base.get("name", algorithm_name))
    if "description" not in base or not base.get("description"):
        base["description"] = ui_entry.get("description", "")
    elif ui_entry.get("description"):
        # Prefer the richer UI description when both exist.
        base["description"] = ui_entry["description"]
    base["category"]  = ui_entry.get("category", "")
    base["ui_schema"] = list(ui_entry.get("params", []))
    return base


def list_algorithms() -> list[dict]:
    """Return :func:`get_algorithm_info` for every registered algorithm."""
    return [get_algorithm_info(name) for name in ALGORITHM_REGISTRY.keys()]


# ---------------------------------------------------------------------------
# Convenience — expose GPU info for callers without importing the optimizer
# ---------------------------------------------------------------------------

def get_runtime_info() -> dict:
    """Return current GPU configuration — handy for /status endpoints."""
    return get_gpu_config()
