"""
src/benchmarking/_scalability_run_worker.py
===========================================

Isolated single-unit worker for the ScalabilityBenchmarker.

Runs ONE ``(algorithm, mode)`` benchmark unit (warmup runs + timed runs) on a
single graph, in its own process, and writes the timing result to a JSON file.

Why a separate process
----------------------
A naive out-of-memory in a benchmark run (notably the CuPy ``gpu_baseline`` MCL
on dense-ing Erdős–Rényi graphs) can exhaust host RAM and be **SIGKILL-ed by the
OS / cgroup OOM killer** (exit code -9).  SIGKILL cannot be caught in Python, so
an in-process run takes the whole sweep down with it.  By running each unit in a
child process, an OOM-kill terminates only the child; the parent
(:class:`~src.benchmarking.scalability_benchmark.ScalabilityBenchmarker`) sees
the non-zero exit, records the unit as failed, and continues with the next one.

Unlike ``multi_threaded/_networkit_worker.py`` (which must NOT touch CUDA), this
worker DOES want a live CUDA context — it runs the GPU algorithms — so importing
from ``src.*`` is intentional here.

Invocation
----------
    python -m src.benchmarking._scalability_run_worker <config_json> <result_json>

``config_json`` (input) keys:
    graph_npz      : str   — path to a scipy .npz CSR of the graph
    algorithm      : str
    mode           : str   — "gpu", "gpu_baseline", "cpu_single", "cpu_multi"
    network_type   : str
    params         : dict
    n_warmup       : int
    n_runs         : int

``result_json`` (output) keys:
    ok     : bool
    times  : list[float]
    mems   : list[float]
    note   : str
    error  : str | None

The result file is written ONLY on a clean finish.  Its absence after the
process exits is how the parent detects an OOM-kill / hard crash.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``src`` importable regardless of the launcher's cwd.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: _scalability_run_worker <config_json> <result_json>",
              file=sys.stderr)
        return 2

    config_path = sys.argv[1]
    result_path = sys.argv[2]

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    import scipy.sparse as sp
    # Reuse the benchmarker's exact timing helper so isolated numbers are
    # identical to the in-process path (GPU modes → runner CUDA-event timing).
    from src.benchmarking.scalability_benchmark import _run_once_timed

    graph = sp.load_npz(cfg["graph_npz"]).tocsr()
    algorithm    = cfg["algorithm"]
    mode         = cfg["mode"]
    params       = dict(cfg["params"])
    params["network_type"] = cfg["network_type"]
    n_warmup     = int(cfg.get("n_warmup", 1))
    n_runs       = int(cfg.get("n_runs", 1))

    # Warmup — pays JIT/kernel-compile + cache population inside THIS process so
    # the timed runs below hit the warm kernel cache (matching the in-process
    # warmup semantics).
    for _ in range(n_warmup):
        try:
            _run_once_timed(algorithm, mode, graph, params)
        except Exception:                                   # noqa: BLE001
            pass

    times: list[float] = []
    mems:  list[float] = []
    note:  str = ""
    error: str | None = None
    for _ in range(n_runs):
        try:
            t, mb, nt = _run_once_timed(algorithm, mode, graph, params)
            times.append(float(t))
            mems.append(float(mb))
            if nt:
                note = nt
        except Exception as exc:                            # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            break

    out = {
        "ok":    bool(times),
        "times": times,
        "mems":  mems,
        "note":  note,
        "error": error,
    }
    # Written only on a clean finish; if the process is OOM-killed before this,
    # the file never appears and the parent treats the unit as failed.
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
