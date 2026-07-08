"""
benchmark/runner.py — Central Benchmarking & Timing Framework
==============================================================
Runs an algorithm across one or more execution modes (cpu_single,
cpu_multi, gpu) on a preprocessed graph, measures *only* algorithm
execution time (no I/O), and writes structured results.

TIMING
------
  CPU modes : time.perf_counter() (highest-resolution monotonic clock)
  GPU mode  : CUDA events via cp.cuda.Event (CuPy) when available,
              falling back to pycuda.driver.Event.
              Both measure device-side execution only, not host overhead.
              CuPy is preferred because most algorithm modules use CuPy;
              mixing pycuda.autoinit with a CuPy context causes a
              "invalid resource handle" error on event record.

OUTPUTS
-------
  benchmark/results/<algorithm>_<dataset>_<timestamp>.csv
      columns: algorithm, dataset, mode, execution_time_seconds,
               num_nodes, num_edges, timestamp, extra_params

  benchmark/results/<algorithm>_<dataset>_<mode>_output.json
      biological result (cluster labels, ranked list, motif counts, ...)

ALGORITHM CONTRACT
------------------
Each module under algorithms/<name>.py must expose:
    _cpu_single(graph, **kwargs)
    _cpu_multi (graph, **kwargs)
    _gpu       (graph, **kwargs)

Each function receives a scipy.sparse.csr_matrix and returns either:
    * a JSON-serialisable biological result, OR
    * a dict {"output": <result>, "extra_params": {...}}

USAGE
-----
  python benchmark/runner.py --algorithm pagerank --dataset trrust
  python benchmark/runner.py --algorithm bfs      --dataset trrust --modes cpu_single gpu

  python benchmark/runner.py --algorithm mcl --dataset trrust --modes cpu_single
  python benchmark/runner.py --algorithm mcl --dataset trrust --modes gpu
  python benchmark/runner.py --algorithm mcl --dataset trrust --modes cpu_single gpu
  python benchmark/runner.py --algorithm mcl --dataset trrust --modes cpu_single cpu_multi
"""

import argparse
import csv
import importlib
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PRE_DIR = os.path.join(_ROOT, "data", "preprocessed")
_RES_DIR = os.path.join(_ROOT, "benchmark", "results")

# Make the project root importable so `algorithms.<name>` resolves when
# the runner is launched as a script.
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from src.optimization.gpu_config import apply_config, print_gpu_summary


def _force_utf8_streams() -> None:
    """Make stdout/stderr encode UTF-8 so benchmark prints never crash.

    Result notes and progress messages contain non-ASCII characters
    (em-dash, ``×``, ``…``, ``Aᵀ``).  When the interpreter's stdout uses a
    narrow codec — ``ascii`` under a C/POSIX locale, or when output is piped
    on some Windows setups — ``print()`` raises ``UnicodeEncodeError``.
    Reconfiguring to UTF-8 (with ``errors="replace"`` as a final safety net)
    fixes every such character at once.  No-op on interpreters without
    ``TextIOWrapper.reconfigure`` (pre-3.7) or already-UTF-8 streams.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        enc = (getattr(stream, "encoding", "") or "").lower()
        if enc in ("utf-8", "utf8"):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# Force UTF-8 output BEFORE the first print (print_gpu_summary below prints
# a banner at import time).
_force_utf8_streams()


# Print the detected GPU configuration once, at runner startup.
# Cached at module level so subsequent apply_config() calls are free.
print_gpu_summary()


# ---------------------------------------------------------------------------
# Optional GPU setup — CuPy preferred, PyCUDA as fallback
#
# Why CuPy first: algorithm modules (mcl, pagerank, …) use CuPy, which
# initialises its own CUDA context.  Creating a *second* context via
# pycuda.autoinit and then recording PyCUDA events while CuPy work is
# running on the CuPy context produces "cuEventRecord failed: invalid
# resource handle".  Using cp.cuda.Event keeps everything in one context.
# ---------------------------------------------------------------------------

cp = None                   # CuPy module reference (or None)
cuda = None                 # PyCUDA driver reference (or None)
_GPU_TIMER_BACKEND = None   # 'cupy' | 'pycuda' | None
_CUDA_AVAILABLE = False
_CUDA_ERROR = None

# --- Attempt 1: CuPy (preferred) ---
try:
    import cupy as _cp
    _ = _cp.zeros(1)        # force context initialisation; raises if no GPU
    cp = _cp
    _GPU_TIMER_BACKEND = "cupy"
    _CUDA_AVAILABLE = True
except Exception as _exc:
    _CUDA_ERROR = f"cupy unavailable: {_exc}"

# --- Attempt 2: PyCUDA fallback (used by algorithms that write raw kernels) ---
if not _CUDA_AVAILABLE:
    try:
        import pycuda.driver as _cuda_mod
        _cuda_mod.init()
        if _cuda_mod.Device.count() > 0:
            # Use the PRIMARY context (retain_primary_context) rather than
            # `pycuda.autoinit` which calls `make_context()` and creates a
            # SEPARATE non-primary context.  All six GPU algorithms also use
            # `retain_primary_context().push()/pop()`.  When both the
            # benchmarker and an algorithm use the same primary context,
            # nested push/pop pairs are reference-counted and the context
            # stays active for the lifetime of each BenchmarkTimer block.
            # With `autoinit`'s separate context, the algorithm's pop
            # switches to the non-primary context and `BenchmarkTimer`
            # events (created in the primary ctx) become invalid.
            _primary_ctx = _cuda_mod.Device(0).retain_primary_context()
            _primary_ctx.push()
            cuda = _cuda_mod
            _GPU_TIMER_BACKEND = "pycuda"
            _CUDA_AVAILABLE = True
            _CUDA_ERROR = None
        else:
            _CUDA_ERROR = "no CUDA devices detected"
    except Exception as _exc:
        _CUDA_ERROR = f"pycuda unavailable: {_exc}"


# ---------------------------------------------------------------------------
# CUDA context helper — imported by algorithm _gpu() functions
# ---------------------------------------------------------------------------

def _ensure_cuda_context() -> bool:
    """
    Guarantee that a CUDA context is current before any GPU operation.

    Algorithm modules call this at the top of every ``_gpu()`` function so
    that the CuPy context is active regardless of whether the function is
    invoked through the benchmark runner or directly in a test / script.

    When called through the runner the context is already live (runner.py
    forces it at import time via ``cp.zeros(1)``).  When called standalone
    this makes device 0 current so the first cupyx.scipy.sparse call does
    not raise ``CUDADriverError: CUDA_ERROR_INVALID_CONTEXT``.

    Returns
    -------
    bool
        True  — a CUDA device is available and its context is now current.
        False — no GPU / CuPy absent; caller should fall back to CPU.
    """
    if not _CUDA_AVAILABLE:
        return False
    if _GPU_TIMER_BACKEND == "cupy" and cp is not None:
        try:
            cp.cuda.Device(0).use()   # no-op if device 0 is already current
        except Exception:
            return False
    return True


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class BenchmarkTimer:
    """
    Context manager that measures algorithm execution time.

    GPU timing uses CUDA events in the *same* context as the algorithm:
            - CuPy backend  -> cp.cuda.Event  (preferred; matches CuPy algorithms)
            - PyCUDA backend -> cuda.Event    (fallback for raw-kernel algorithms)
    CPU timing uses time.perf_counter().

    Usage:
        with BenchmarkTimer("gpu") as t:
            run_kernel(...)
        print(t.elapsed)   # seconds (float)
    """

    def __init__(self, mode: str):
        self.mode = mode
        self.elapsed: float | None = None
        self._t0 = None
        self._evt_start = None
        self._evt_end = None

    def __enter__(self) -> "BenchmarkTimer":
        if self.mode == "gpu" and _CUDA_AVAILABLE:
            if _GPU_TIMER_BACKEND == "cupy":
                self._evt_start = cp.cuda.Event()
                self._evt_end = cp.cuda.Event()
                self._evt_start.record()
            else:  # pycuda
                self._evt_start = cuda.Event()
                self._evt_end = cuda.Event()
                self._evt_start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.mode == "gpu" and _CUDA_AVAILABLE:
            self._evt_end.record()
            if _GPU_TIMER_BACKEND == "cupy":
                self._evt_end.synchronize()
                # get_elapsed_time returns milliseconds -> convert to seconds
                self.elapsed = cp.cuda.get_elapsed_time(
                    self._evt_start, self._evt_end
                ) / 1000.0
            else:  # pycuda
                self._evt_end.synchronize()
                self.elapsed = self._evt_start.time_till(self._evt_end) / 1000.0
        else:
            self.elapsed = time.perf_counter() - self._t0
        return False  # never suppress exceptions


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def _to_jsonable(obj):
    """Recursively convert numpy / scipy objects into JSON-friendly types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, sp.spmatrix):
        return {"_sparse_shape": list(obj.shape), "_sparse_nnz": int(obj.nnz)}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]
    return obj


def _save_output_json(output, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(output), fh, indent=2)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(dataset_name: str) -> tuple[sp.csr_matrix, dict, dict]:
    """Load the preprocessed CSR matrix and node-index mapping."""
    npz_path = os.path.join(_PRE_DIR, f"{dataset_name}.npz")
    nodes_path = os.path.join(_PRE_DIR, f"{dataset_name}_nodes.json")

    if not os.path.isfile(npz_path):
        sys.exit(
            f"[ERROR] Preprocessed graph not found: {npz_path}\n"
            f"        Generate it via the preprocessing pipeline (src/preprocessing/) first."
        )

    graph = sp.load_npz(npz_path).tocsr()

    node2idx = {}
    if os.path.isfile(nodes_path):
        with open(nodes_path, "r", encoding="utf-8") as fh:
            node2idx = json.load(fh)

    meta = {
        "num_nodes": int(graph.shape[0]),
        "num_edges": int(graph.nnz),  # for symmetric matrices counts each edge twice
    }
    return graph, node2idx, meta


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "algorithm",
    "dataset",
    "mode",
    "execution_time_seconds",
    "num_nodes",
    "num_edges",
    "timestamp",
    "extra_params",
]


def _write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def run_benchmark(
    algorithm_name: str,
    dataset_name: str,
    modes: list[str] | None = None,
    **algo_kwargs,
) -> list[dict]:
    """
    Benchmark `algorithm_name` against `dataset_name` across one or more modes.

    Parameters
    ----------
    algorithm_name : module name under algorithms/  (e.g. 'pagerank')
    dataset_name   : preprocessed dataset name      (e.g. 'trrust')
    modes          : subset of ['cpu_single', 'cpu_multi', 'gpu']
    **algo_kwargs  : forwarded verbatim to each algorithm function

    Returns
    -------
    list of result dicts (one per mode that executed successfully).
    """
    if modes is None:
        modes = ["cpu_single", "cpu_multi", "gpu"]

    os.makedirs(_RES_DIR, exist_ok=True)

    # ---- Load preprocessed graph ----
    graph, node2idx, meta = _load_dataset(dataset_name)
    print(
        f"[INFO]  Loaded '{dataset_name}': "
        f"{meta['num_nodes']:,} nodes, {meta['num_edges']:,} nnz"
    )

    # ---- Dynamically import the algorithm module ----
    try:
        module = importlib.import_module(f"src.algorithms.gpu.cuda_optimized.{algorithm_name}")
    except ImportError as exc:
        sys.exit(f"[ERROR] Cannot import src.algorithms.gpu.cuda_optimized.{algorithm_name}: {exc}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: list[dict] = []

    for mode in modes:
        func_name = f"_{mode}"
        fn = getattr(module, func_name, None)

        if fn is None:
            print(f"[WARN]  algorithms.{algorithm_name} has no {func_name}() — skipping.")
            continue

        if mode == "gpu" and not _CUDA_AVAILABLE:
            print(f"[WARN]  Skipping 'gpu' mode — {_CUDA_ERROR}")
            continue

        print(f"[RUN]   {algorithm_name}.{func_name}({dataset_name}) …")

        # Architecture-aware parameter injection.
        # Only the GPU mode genuinely needs hardware-tuned recommendations;
        # CPU modes still get the merged dict so any user-supplied params
        # carry through unchanged.
        call_kwargs = dict(algo_kwargs)
        if mode == "gpu":
            tuned_params = apply_config(
                algorithm_name,
                graph,
                call_kwargs.get("params", {}),
            )
            call_kwargs["params"] = tuned_params

        try:
            with BenchmarkTimer(mode) as timer:
                raw = fn(graph, **call_kwargs)
        except Exception as exc:  # noqa: BLE001 — surface algorithm errors
            print(f"[ERROR] '{mode}' mode raised {type(exc).__name__}: {exc}")
            continue

        # Unpack: support both `result` and `{"output": ..., "extra_params": ...}`
        if isinstance(raw, dict) and "output" in raw:
            bio_output = raw["output"]
            extra_params = raw.get("extra_params", {})
        else:
            bio_output = raw
            extra_params = {}

        # Persist biological output
        out_path = os.path.join(
            _RES_DIR, f"{algorithm_name}_{dataset_name}_{mode}_output.json"
        )
        _save_output_json(bio_output, out_path)

        results.append({
            "algorithm": algorithm_name,
            "dataset": dataset_name,
            "mode": mode,
            "execution_time_seconds": f"{timer.elapsed:.6f}",
            "num_nodes": meta["num_nodes"],
            "num_edges": meta["num_edges"],
            "timestamp": timestamp,
            "extra_params": json.dumps(_to_jsonable(extra_params)),
        })

        print(f"[DONE]  {mode}: {timer.elapsed:.4f}s - {out_path}")

    if not results:
        print("[WARN]  No modes executed successfully — nothing to write.")
        return results

    csv_path = os.path.join(
        _RES_DIR, f"{algorithm_name}_{dataset_name}_{timestamp}.csv"
    )
    _write_csv(results, csv_path)
    print(f"[INFO]  Timing results saved - {csv_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark a graph algorithm across CPU/GPU modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--algorithm", required=True,
        help="Algorithm module name under algorithms/ (e.g. 'pagerank').",
    )
    p.add_argument(
        "--dataset", required=True,
        help="Preprocessed dataset name (matches <name>.npz).",
    )
    p.add_argument(
        "--modes", nargs="+",
        default=["cpu_single", "cpu_multi", "gpu"],
        choices=["cpu_single", "cpu_multi", "gpu"],
        help="Execution modes to benchmark (default: all three).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        algorithm_name=args.algorithm,
        dataset_name=args.dataset,
        modes=args.modes,
    )
