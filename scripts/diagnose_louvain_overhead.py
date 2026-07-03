"""
scripts/diagnose_louvain_overhead.py
=====================================

Diagnose Louvain across modes on the GPU box, after the chunk-trigger fix in
src/algorithms/gpu/cuda_optimized/louvain.py (the chunked Phase-1 path streams
CSR rows and drops the freeze optimisation; it is now used only when a level's
working set genuinely does not fit in VRAM).

Three things this surfaces:

  1. GPU-optimised: wall + loop (CUDA-event) + setup, num_communities,
     modularity, levels.  Params include use_chunking=True to REPLICATE the
     runner's apply_config injection — the fix must keep it on the normal path.

  2. cpu_single: wall time.  (cpu_multi delegates to cpu_single because
     SuiteSparse:GraphBLAS has no Louvain primitive, so its line ~= cpu_single.)

  3. gpu_baseline (cuGraph louvain): run with the error CAUGHT and PRINTED, so
     we can see WHY it is missing from the scalability plot (a failed run is
     recorded as NaN and simply doesn't draw).

Usage
-----
    python scripts/diagnose_louvain_overhead.py
    python scripts/diagnose_louvain_overhead.py --sizes 100000,1000000
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.sparse as sp

DEFAULT_SIZES = [100_000, 1_000_000]
GRAPH_TYPES = ["barabasi_albert", "erdos_renyi", "watts_strogatz"]
NETWORK_TYPE = "ppi"
PARAMS = {"min_delta_q": 1e-4, "max_levels": 10, "resolution": 1.0,
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
    from src.algorithms.gpu.cuda_optimized.louvain import louvain_gpu

    print(f"  [GPU optimised] {label}")
    louvain_gpu(csr, PARAMS)                     # warmup
    t0 = time.perf_counter()
    res = louvain_gpu(csr, PARAMS)               # timed
    wall = (time.perf_counter() - t0) * 1000.0

    inner   = res["result"]
    loop_ms = res["execution_time"] * 1000.0
    setup_ms = max(wall - loop_ms, 0.0)
    note    = str(inner.get("note", ""))
    chunked = "chunk" in note.lower()
    print(f"    wall={wall:8.2f} ms  loop={loop_ms:8.2f} ms  setup={setup_ms:7.2f} ms  "
          f"num_comm={inner.get('num_communities')}  "
          f"modularity={inner.get('modularity'):.4f}  chunked~={chunked}")


def _run_cpu_single(csr: sp.csr_matrix, label: str) -> None:
    from src.algorithms.cpu.single_threaded.louvain import louvain_cpu_single
    t0 = time.perf_counter()
    res = louvain_cpu_single(csr, PARAMS)
    wall = (time.perf_counter() - t0) * 1000.0
    nc = res.get("num_communities", res.get("result", {}).get("num_communities"))
    print(f"  [CPU single]    {label}: wall={wall:8.2f} ms  num_comm={nc}")


def _run_gpu_baseline(csr: sp.csr_matrix, label: str) -> None:
    try:
        from src.algorithms.gpu.basic.louvain import louvain_gpu_baseline
    except Exception as exc:                            # noqa: BLE001
        print(f"  [GPU baseline]  {label}: import unavailable ({exc})")
        return
    try:
        t0 = time.perf_counter()
        res = louvain_gpu_baseline(csr, PARAMS)
        wall = (time.perf_counter() - t0) * 1000.0
        print(f"  [GPU baseline]  {label}: wall={wall:8.2f} ms  "
              f"backend={res['result'].get('backend')}")
    except Exception as exc:                            # noqa: BLE001
        # This is the key output — WHY the baseline is missing from the plot.
        print(f"  [GPU baseline]  {label}: FAILED — {type(exc).__name__}: {exc}")
        tb = traceback.format_exc().strip().splitlines()
        for line in tb[-4:]:
            print(f"      {line}")


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
            _run_cpu_single(csr, label)
            _run_gpu_baseline(csr, label)


if __name__ == "__main__":
    main()
