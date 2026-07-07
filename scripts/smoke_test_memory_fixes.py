"""Smoke test for the post-audit memory fixes.

Builds a synthetic 50K-node BA graph and runs the four core ranking
algorithms in both cpu_single and gpu modes (gpu is skipped if PyCUDA
is absent or the GPU is busy).  Prints memory snapshots and timings.
"""
from __future__ import annotations

import gc
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import scipy.sparse  # noqa: F401  (ensures CSR codec is registered)

from src.utils.memory_logger import log_memory


def main() -> int:
    n = 50_000
    import networkx as nx
    G = nx.barabasi_albert_graph(n, 20, seed=42)
    csr = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float32)
    print(f"Test graph: {csr.shape[0]} nodes, {csr.nnz} edges")
    log_memory("After graph creation")

    from src.runner.algorithm_runner import run_algorithm

    node_map = {str(i): i for i in range(n)}

    overall_failures = 0
    for algo in ["pagerank", "bfs", "hits", "rwr"]:
        for mode in ["cpu_single", "gpu"]:
            try:
                log_memory(f"Before {algo} {mode}")
                t0 = time.perf_counter()
                result = run_algorithm(
                    algo, csr, node_map, mode=mode,
                    params={
                        "network_type": "ppi",
                        "source": 0,
                        "seed_nodes": [0, 1, 2],
                    },
                )
                wall = time.perf_counter() - t0
                log_memory(f"After  {algo} {mode}")
                exec_t = float(result.get("execution_time", 0.0))
                print(f"{algo}/{mode}: exec={exec_t:.4f}s "
                      f"wall={wall:.4f}s OK")
                del result
                gc.collect()
            except Exception as exc:                          # noqa: BLE001
                print(f"{algo}/{mode}: FAILED — {type(exc).__name__}: {exc}")
                overall_failures += 1
    return overall_failures


if __name__ == "__main__":
    sys.exit(main())
