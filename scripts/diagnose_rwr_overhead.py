"""
scripts/diagnose_rwr_overhead.py
=================================

Diagnose why GPU-optimised RWR trails the other modes at large n, after the
chunk-trigger fix in src/algorithms/gpu/cuda_optimized/rwr.py (the chunked
path re-streamed the entire W CSR host->device every iteration; it is now
only used when the graph genuinely does not fit).

Run on the GPU box (pycuda + CUDA toolchain; RAPIDS/cuGraph for the
baseline, which uses cugraph.personalized_pagerank).

Per graph family x size, for the GPU-optimised path:
  * wall time (what the benchmark records)
  * loop time (CUDA-event iteration loop, from result execution_time)
  * setup   = wall - loop  (W build + transpose + H2D — mostly CACHED across
              calls for RWR, so this should already be small on the timed run)
  * iterations, converged
  * chunked? (parsed from result note) — MUST be False for graphs that fit

and the cuGraph baseline wall time for reference.

Params include use_chunking=True to REPLICATE what the runner's apply_config
injects; with the fix, chunked must stay False for fitting graphs.

Usage
-----
    python scripts/diagnose_rwr_overhead.py
    python scripts/diagnose_rwr_overhead.py --sizes 100000,1000000
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.sparse as sp

DEFAULT_SIZES = [100_000, 1_000_000]
GRAPH_TYPES = ["barabasi_albert", "erdos_renyi", "watts_strogatz"]
NETWORK_TYPE = "ppi"
# use_chunking=True replicates the runner's apply_config injection (see above).
PARAMS = {"restart_prob": 0.3, "max_iter": 100, "tolerance": 1e-6,
          "seed_nodes": [0], "network_type": NETWORK_TYPE, "use_chunking": True}


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
    from src.algorithms.gpu.cuda_optimized.rwr import rwr_gpu

    print(f"  [GPU optimised] {label}")
    rwr_gpu(csr, PARAMS)                         # warmup (compile + cache fill)

    t0 = time.perf_counter()
    res = rwr_gpu(csr, PARAMS)                   # timed
    wall = (time.perf_counter() - t0) * 1000.0

    inner   = res["result"]
    loop_ms = res["execution_time"] * 1000.0     # CUDA-event iteration loop
    setup_ms = max(wall - loop_ms, 0.0)
    note    = str(inner.get("note", ""))
    m = re.search(r"chunked=(\w+)", note)
    chunked = (m.group(1) if m else "?")
    print(f"    wall={wall:8.2f} ms  loop={loop_ms:8.2f} ms  "
          f"setup={setup_ms:7.2f} ms  iters={inner.get('iterations')}  "
          f"converged={inner.get('converged')}  chunked={chunked}")


def _run_gpu_baseline(csr: sp.csr_matrix, label: str) -> None:
    try:
        from src.algorithms.gpu.basic.rwr import rwr_gpu_baseline
    except Exception as exc:                            # noqa: BLE001
        print(f"  [GPU baseline] {label}: unavailable ({exc})")
        return
    try:
        rwr_gpu_baseline(csr, PARAMS)                   # warmup
        t0 = time.perf_counter()
        res = rwr_gpu_baseline(csr, PARAMS)
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
