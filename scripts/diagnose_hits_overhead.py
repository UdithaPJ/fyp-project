"""
scripts/diagnose_hits_overhead.py
==================================

Diagnose why GPU-optimised HITS trails the cuGraph GPU baseline.

Run on the GPU box (pycuda + a working CUDA toolchain; RAPIDS/cuGraph for
the baseline comparison).

Two competing hypotheses, both testable here:

  H1 (iteration count / convergence criterion):
      cuGraph converges when  Σ|Δhub| < n_vertices * epsilon  — a threshold
      that SCALES with n.  The optimised path converges when
      sqrt(Σ(Δh² + Δa²)) < tolerance — a FIXED threshold.  For the same
      `tolerance` param the optimised path may run many more iterations,
      especially on barabasi_albert (large spectral gap → cuGraph stops
      almost immediately, we keep going).

  H2 (uncached per-call setup):
      Every timed call rebuilds A^T, symmetrizes (ppi), runs a CPU argsort
      for node reordering, and re-uploads H2D — none cached across the
      warmup→timed transition (unlike the BFS resident-graph cache).

What it measures, per graph family × size:

  * GPU optimised: wall time (what the benchmark records) split into
    setup (= wall − iteration-loop) vs iteration-loop (CUDA-event timed),
    plus iterations_run, converged, reorder_cost_ms.
  * A tolerance sweep {1e-4, 1e-3, 1e-2, 5e-2} on the SAME graph, showing
    iterations vs wall time — if loosening tolerance collapses both
    (especially on BA), H1 is confirmed.
  * cuGraph baseline wall time for reference.

Usage
-----
    python scripts/diagnose_hits_overhead.py
    python scripts/diagnose_hits_overhead.py --sizes 100000,1000000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.sparse as sp

DEFAULT_SIZES = [100_000, 1_000_000]
GRAPH_TYPES = ["barabasi_albert", "erdos_renyi", "watts_strogatz"]
TOL_SWEEP = [1e-4, 1e-3, 1e-2, 5e-2]

# Match the benchmark's HITS params (network_type + the new iter cap).
NETWORK_TYPE = "ppi"
MAX_ITER = 500


def _gen_graph(graph_type: str, n: int) -> sp.csr_matrix:
    import networkx as nx
    if graph_type == "barabasi_albert":
        G = nx.barabasi_albert_graph(n, 3, seed=42)
    elif graph_type == "erdos_renyi":
        p = min(6.0 / max(n - 1, 1), 1.0)
        G = nx.fast_gnp_random_graph(n, p, seed=42)
    elif graph_type == "watts_strogatz":
        k = min(6, max(2, n // 4))
        if k % 2 != 0:
            k += 1
        G = nx.watts_strogatz_graph(n, k, 0.1, seed=42)
    else:
        raise ValueError(graph_type)
    csr = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float32)
    csr.sum_duplicates()
    csr.eliminate_zeros()
    return csr


def _run_gpu_optimised(csr: sp.csr_matrix, label: str) -> None:
    from src.algorithms.gpu.cuda_optimized.hits import (
        hits_gpu, clear_hits_buffer_pool, clear_hits_graph_cache,
    )

    print(f"  [GPU optimised] {label}")

    # ---- Setup vs iteration-loop split, at the benchmark tolerance ------
    params = {
        "max_iter": MAX_ITER, "tolerance": 1e-4,
        "network_type": NETWORK_TYPE,
    }
    # Start cold so the warmup call is a genuine graph-cache MISS (pays full
    # setup) and the timed call is a HIT (setup should collapse to H2D only).
    clear_hits_buffer_pool()
    clear_hits_graph_cache()
    hits_gpu(csr, params)                       # warmup (compile, pool + cache fill)

    t0 = time.perf_counter()
    res = hits_gpu(csr, params)                 # timed
    wall = (time.perf_counter() - t0) * 1000.0

    inner = res["result"]
    loop_ms  = res["execution_time"] * 1000.0   # CUDA-event iteration loop
    setup_ms = max(wall - loop_ms, 0.0)         # symmetrize+transpose+reorder+H2D
    print(f"    tol=1e-4: wall={wall:8.2f} ms  "
          f"loop={loop_ms:8.2f} ms  setup={setup_ms:7.2f} ms  "
          f"iters={inner['iterations']:4d}  converged={inner['converged']}  "
          f"reorder={inner.get('reorder_cost_ms', 0.0):.1f} ms")

    # ---- Tolerance sweep: iterations & wall vs tolerance ----------------
    # If BA collapses (few iters, low wall) as tolerance loosens, the gap to
    # cuGraph is a convergence-criterion mismatch, not raw kernel speed.
    for tol in TOL_SWEEP:
        p = {"max_iter": MAX_ITER, "tolerance": tol, "network_type": NETWORK_TYPE}
        clear_hits_buffer_pool()
        hits_gpu(csr, p)                         # warmup for this tol
        t0 = time.perf_counter()
        r = hits_gpu(csr, p)
        w = (time.perf_counter() - t0) * 1000.0
        ii = r["result"]
        print(f"      sweep tol={tol:<6g}: wall={w:8.2f} ms  "
              f"iters={ii['iterations']:4d}  converged={ii['converged']}")

    clear_hits_buffer_pool()
    clear_hits_graph_cache()


def _run_gpu_baseline(csr: sp.csr_matrix, label: str) -> None:
    try:
        from src.algorithms.gpu.basic.hits import hits_gpu_baseline
    except Exception as exc:                            # noqa: BLE001
        print(f"  [GPU baseline] {label}: unavailable ({exc})")
        return

    params = {"max_iter": MAX_ITER, "tolerance": 1e-4, "network_type": NETWORK_TYPE}
    try:
        hits_gpu_baseline(csr, params)                  # warmup
        t0 = time.perf_counter()
        res = hits_gpu_baseline(csr, params)
        wall = (time.perf_counter() - t0) * 1000.0
        print(f"  [GPU baseline]  {label}: wall={wall:8.2f} ms  "
              f"backend={res['result'].get('backend')}  "
              f"iters={res['result'].get('iterations')}")
    except Exception as exc:                            # noqa: BLE001
        print(f"  [GPU baseline]  {label}: FAILED ({type(exc).__name__}: {exc})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default=",".join(map(str, DEFAULT_SIZES)))
    ap.add_argument("--graph-types", type=str, default=",".join(GRAPH_TYPES))
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    graph_types = [g.strip() for g in args.graph_types.split(",") if g.strip()]

    for gt in graph_types:
        for n in sizes:
            print(f"\n=== {gt}  n={n:,} ===")
            csr = _gen_graph(gt, n)
            label = f"{gt} n={csr.shape[0]:,} m={csr.nnz:,}"
            _run_gpu_optimised(csr, label)
            _run_gpu_baseline(csr, label)


if __name__ == "__main__":
    main()
