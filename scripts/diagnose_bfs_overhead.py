"""
scripts/diagnose_bfs_overhead.py
==================================

Run this on the GPU box (with pycuda + a working CUDA toolchain, and
ideally python-suitesparse-graphblas for the cpu_multi comparison) to
diagnose where GPU-optimised BFS time goes relative to CPU GraphBLAS,
after two fixes in ``src/algorithms/gpu/cuda_optimized/bfs.py``:

  * Opt-2 fast-path fix: removed a per-level synchronous ``memcpy_htod``
    + a wasted O(N) bitmap-zero that were undermining the "no per-level
    sync" design.
  * Profiler Event-pool fix: ``enable_profiling=True`` used to allocate
    two fresh ``cuda.Event()`` objects EVERY level, and Event construction
    is a real (non-trivial) driver call not captured by the events' own
    elapsed-time measurement — this could make profiled wall-clock time
    look 10-40x worse than the actual unprofiled algorithm.  Events are
    now pooled and reused.

What it does
------------
For each of the three benchmark graph families (barabasi_albert,
erdos_renyi, watts_strogatz) at each requested size:

  1. Generates the graph with the SAME generators the scalability
     benchmark uses (avg out-degree ~= 6).
  2. Runs GPU BFS through TWO phases:
       Phase A — enable_profiling=False, exactly matching the real
                 benchmark (src/benchmarking/scalability_benchmark.py).
                 This is the number to compare against CPU GraphBLAS.
       Phase B — enable_profiling=True, for the per-kernel breakdown.
                 Also prints levels_run so we can confirm the no_sync_
                 levels early-termination logic is firing as designed.
  3. Runs cpu_multi (GraphBLAS) with the matching collect_cascade=False
     for an apples-to-apples wall-clock comparison.

Usage
-----
    python scripts/diagnose_bfs_overhead.py
    python scripts/diagnose_bfs_overhead.py --sizes 100000,2000000,8000000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.sparse as sp

DEFAULT_SIZES = [100_000, 2_000_000, 8_000_000]
GRAPH_TYPES = ["barabasi_albert", "erdos_renyi", "watts_strogatz"]


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


def _run_gpu(csr: sp.csr_matrix, label: str) -> None:
    from src.algorithms.gpu.cuda_optimized.bfs import (
        _bfs_gpu_optimized, clear_bfs_caches,
    )

    base_params = {
        "source": 0,
        "max_depth": 999,
        "network_type": "ppi",
        "collect_cascade": False,
        "direction_mode": "push_only",
        "cache_graph": True,
    }

    print(f"  [GPU optimised] {label}")

    # ---- Phase A: PRODUCTION path (enable_profiling=False) --------------
    # This matches src/benchmarking/scalability_benchmark.py exactly — no
    # profiling instrumentation, so no Event()-construction overhead.  This
    # is the number that should be compared against CPU GraphBLAS.
    clear_bfs_caches()
    params_prod = {**base_params, "enable_profiling": False}

    t0 = time.perf_counter()
    warm = _bfs_gpu_optimized(csr, params_prod)
    t_warm = time.perf_counter() - t0

    t0 = time.perf_counter()
    timed = _bfs_gpu_optimized(csr, params_prod)
    t_timed = time.perf_counter() - t0

    n_levels = len(timed.get("traversal_modes", []))
    print(f"    [production, no profiling]")
    print(f"      warmup : {t_warm*1000:8.2f} ms  "
          f"(num_reachable={warm['num_reachable']})")
    print(f"      timed  : {t_timed*1000:8.2f} ms  "
          f"(num_reachable={timed['num_reachable']}, "
          f"levels_run={n_levels})")

    # ---- Phase B: DIAGNOSTIC path (enable_profiling=True) ---------------
    # Same graph, resident-graph cache still warm from Phase A — but now
    # with per-phase CUDA-event timing.  Phase A never touched the event
    # pool (profiling was off), so this phase's warmup call pays Event
    # construction once; the timed call reuses the pool and should show
    # wall time much closer to sum(profile) than before the pooling fix.
    params_diag = {**base_params, "enable_profiling": True}

    t0 = time.perf_counter()
    warm2 = _bfs_gpu_optimized(csr, params_diag)
    t_warm2 = time.perf_counter() - t0

    t0 = time.perf_counter()
    timed2 = _bfs_gpu_optimized(csr, params_diag)
    t_timed2 = time.perf_counter() - t0

    print(f"    [diagnostic, profiling on]")
    print(f"      warmup : {t_warm2*1000:8.2f} ms  "
          f"(levels_run={len(warm2.get('traversal_modes', []))})")
    print(f"      timed  : {t_timed2*1000:8.2f} ms  "
          f"(levels_run={len(timed2.get('traversal_modes', []))})")
    prof = timed2.get("profiling")
    if prof:
        parts = ", ".join(f"{k}={v:.3f}" for k, v in prof.items() if v)
        print(f"      profile: {parts}")
        accounted = sum(v for v in prof.values())
        print(f"      sum(profile)={accounted:.3f} ms vs wall={t_timed2*1000:.3f} ms "
              f"(gap should now be small — was profiler Event-churn before the fix)")

    clear_bfs_caches()


def _run_cpu_graphblas(csr: sp.csr_matrix, label: str) -> None:
    try:
        from src.algorithms.cpu.multi_threaded.bfs import bfs_cpu_multi
    except Exception as exc:                            # noqa: BLE001
        print(f"  [CPU GraphBLAS] {label}: unavailable ({exc})")
        return

    params = {"source": 0, "max_depth": 999, "collect_cascade": False}

    t0 = time.perf_counter()
    bfs_cpu_multi(csr, params)  # warmup (thread pool / JIT-ish costs)
    t_warm = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = bfs_cpu_multi(csr, params)
    t_timed = time.perf_counter() - t0

    print(f"  [CPU GraphBLAS] {label}")
    print(f"    warmup : {t_warm*1000:8.2f} ms")
    print(f"    timed  : {t_timed*1000:8.2f} ms  "
          f"(num_reachable={res['num_reachable']})")


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
            _run_gpu(csr, label)
            _run_cpu_graphblas(csr, label)


if __name__ == "__main__":
    main()
