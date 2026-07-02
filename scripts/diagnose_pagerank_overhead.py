"""
scripts/diagnose_pagerank_overhead.py
======================================

Diagnose why GPU-optimised PageRank trails the other modes, after the
chunk-trigger fix in src/algorithms/gpu/cuda_optimized/pagerank.py
(the chunked path re-streamed the entire CSR host->device every
iteration; it is now only used when the graph genuinely does not fit).

Run on the GPU box (pycuda + CUDA toolchain; RAPIDS/cuGraph for the
baseline comparison).

What it measures per graph family x size, for the GPU-optimised path:
  * wall time (what the benchmark records)
  * loop time (CUDA-event timed iteration loop, from result execution_time)
  * setup   = wall - loop  (degrees + ELLPACK build + transpose + H2D,
              all currently inside the timed region and uncached)
  * iterations, converged
  * chunked? (inferred from result note) — should now be False for graphs
    that fit in VRAM

and the cuGraph baseline wall time for reference.

If `setup` is a large fraction of `wall` on the small sizes (where
chunking was never active), a resident-graph cache (as added for BFS and
HITS) is the remaining lever.  If `loop` dominates and iterations is high,
convergence / kernel efficiency is the lever instead.

Usage
-----
    python scripts/diagnose_pagerank_overhead.py
    python scripts/diagnose_pagerank_overhead.py --sizes 100000,1000000
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
NETWORK_TYPE = "ppi"
# NB: include use_chunking=True to REPLICATE what the runner's apply_config
# injects into params before pagerank_gpu sees them.  Calling pagerank_gpu
# directly with clean params previously hid a bug where this injected flag
# forced the (catastrophic) re-streaming chunked path on graphs that fit.
# With use_chunking=True here, chunked MUST stay False for fitting graphs.
PARAMS = {"damping": 0.85, "max_iter": 100, "tolerance": 1e-6,
          "network_type": NETWORK_TYPE, "use_chunking": True}


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
    from src.algorithms.gpu.cuda_optimized.pagerank import (
        pagerank_gpu, clear_pagerank_graph_cache,
    )

    print(f"  [GPU optimised] {label}")
    # Start cold so warmup is a genuine graph-cache MISS (pays full setup)
    # and the timed call is a HIT (setup should collapse toward H2D only).
    clear_pagerank_graph_cache()
    pagerank_gpu(csr, PARAMS)                    # warmup (compile + cache fill)

    t0 = time.perf_counter()
    res = pagerank_gpu(csr, PARAMS)              # timed
    wall = (time.perf_counter() - t0) * 1000.0

    inner   = res["result"]
    loop_ms = res["execution_time"] * 1000.0     # CUDA-event iteration loop
    setup_ms = max(wall - loop_ms, 0.0)          # degrees+ELLPACK+transpose+H2D
    note    = str(inner.get("note", ""))
    chunked = "chunk" in note.lower()
    print(f"    wall={wall:8.2f} ms  loop={loop_ms:8.2f} ms  "
          f"setup={setup_ms:7.2f} ms  iters={inner.get('iterations')}  "
          f"converged={inner.get('converged')}  chunked={chunked}")
    if note:
        print(f"      note: {note[:110]}")

    clear_pagerank_graph_cache()


def _run_gpu_baseline(csr: sp.csr_matrix, label: str) -> None:
    try:
        from src.algorithms.gpu.basic.pagerank import pagerank_gpu_baseline
    except Exception as exc:                            # noqa: BLE001
        print(f"  [GPU baseline] {label}: unavailable ({exc})")
        return
    try:
        pagerank_gpu_baseline(csr, PARAMS)              # warmup
        t0 = time.perf_counter()
        res = pagerank_gpu_baseline(csr, PARAMS)
        wall = (time.perf_counter() - t0) * 1000.0
        print(f"  [GPU baseline]  {label}: wall={wall:8.2f} ms  "
              f"backend={res['result'].get('backend')}")
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
