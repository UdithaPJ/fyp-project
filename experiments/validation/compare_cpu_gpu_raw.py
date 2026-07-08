"""Compare CPU single, CPU multi, and GPU algorithm results on a raw dataset.

Example:
    python experiments/validation/compare_cpu_gpu_raw.py --sample-rows 50000
    python experiments/validation/compare_cpu_gpu_raw.py --algorithm pagerank --sample-rows 100000

The default raw file is the STRING PPI edge list in data/raw.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.cpu import (  # noqa: E402
    bfs_cpu_multi,
    bfs_cpu_single,
    hits_cpu_multi,
    hits_cpu_single,
    louvain_cpu_multi,
    louvain_cpu_single,
    mcl_cpu_multi,
    mcl_cpu_single,
    pagerank_cpu_multi,
    pagerank_cpu_single,
    rwr_cpu_multi,
    rwr_cpu_single,
)
from src.graph.converter import graphdata_to_csr  # noqa: E402
from src.preprocessing.pipeline import PreprocessingPipeline  # noqa: E402


AlgorithmFn = Callable[[Any, dict], dict]


CPU_ALGORITHMS: dict[str, tuple[AlgorithmFn, AlgorithmFn]] = {
    "pagerank": (pagerank_cpu_single, pagerank_cpu_multi),
    "bfs": (bfs_cpu_single, bfs_cpu_multi),
    "rwr": (rwr_cpu_single, rwr_cpu_multi),
    "hits": (hits_cpu_single, hits_cpu_multi),
    "louvain": (louvain_cpu_single, louvain_cpu_multi),
    "mcl": (mcl_cpu_single, mcl_cpu_multi),
}


DEFAULT_PARAMS: dict[str, dict] = {
    "pagerank": {"damping": 0.85, "max_iter": 100, "tolerance": 1e-6},
    "bfs": {"source": 0, "max_depth": 5},
    "rwr": {"restart_prob": 0.3, "max_iter": 100, "tolerance": 1e-6, "seed_nodes": [0]},
    "hits": {"max_iter": 100, "tolerance": 1e-6},
    "louvain": {"min_delta_q": 1e-4, "max_levels": 10, "resolution": 1.0},
    "mcl": {
        "expansion": 2,
        "inflation": 2.0,
        "prune_threshold": 0.001,
        "max_iter": 100,
        "convergence_tol": 1e-4,
    },
}


def load_graph(raw_path: Path, sample_rows: int | None):
    print(f"Loading raw data: {raw_path}")
    dataframe = pd.read_csv(
        raw_path,
        sep=None,
        engine="python",
        nrows=sample_rows,
        low_memory=True,
    )
    print(f"Rows loaded: {len(dataframe):,}")

    pipeline = PreprocessingPipeline()
    graph_data, report = pipeline.run_dataframe(
        dataframe,
        user_override={
            "source": "protein1",
            "target": "protein2",
            "weight": "combined_score",
        },
        duplicate_strategy="mean",
    )
    graph_csr, node_index_map = graphdata_to_csr(graph_data)
    print(
        "Graph built: "
        f"{graph_csr.shape[0]:,} nodes, {graph_csr.nnz:,} edges, "
        f"mapping={report['applied_mapping']}"
    )
    return graph_csr, node_index_map


def get_gpu_fn(name: str) -> AlgorithmFn:
    """Import GPU modules lazily so Windows CPU worker processes stay quiet."""

    if name == "pagerank":
        from src.algorithms.gpu.cuda_optimized.pagerank import pagerank_gpu

        return pagerank_gpu
    if name == "bfs":
        from src.algorithms.gpu.cuda_optimized.bfs import bfs_gpu

        return bfs_gpu
    if name == "rwr":
        from src.algorithms.gpu.cuda_optimized.rwr import rwr_gpu

        return rwr_gpu
    if name == "hits":
        from src.algorithms.gpu.cuda_optimized.hits import hits_gpu

        return hits_gpu
    if name == "louvain":
        from src.algorithms.gpu.cuda_optimized.louvain import louvain_gpu

        return louvain_gpu
    if name == "mcl":
        from src.algorithms.gpu.cuda_optimized.mcl import mcl_gpu

        return mcl_gpu
    raise ValueError(f"Unknown algorithm: {name}")


def top_overlap(a: list[int], b: list[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(set(a) & set(b)) / max(len(set(a)), len(set(b)))


def compare_vector(label: str, a: list[float], b: list[float], atol: float, rtol: float) -> bool:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    ok = np.allclose(arr_a, arr_b, atol=atol, rtol=rtol)
    max_abs = float(np.max(np.abs(arr_a - arr_b))) if arr_a.size else 0.0
    print(f"  {label:<38} {'PASS' if ok else 'FAIL'} max_abs={max_abs:.3e}")
    return bool(ok)


def compare_exact(label: str, a: Any, b: Any) -> bool:
    ok = a == b
    print(f"  {label:<38} {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def unwrap_result(result: dict) -> dict:
    """Accept either direct algorithm output or the standard wrapped result."""

    if isinstance(result, dict) and isinstance(result.get("result"), dict):
        return result["result"]
    if isinstance(result, dict) and isinstance(result.get("output"), dict):
        return unwrap_result(result["output"])
    return result


def compare_algorithm(name: str, graph_csr, params: dict, atol: float, rtol: float) -> None:
    cpu_single_fn, cpu_multi_fn = CPU_ALGORITHMS[name]
    gpu_fn = get_gpu_fn(name)
    print(f"\n=== {name} ===")

    results = {}
    for mode, fn in (
        ("cpu_single", cpu_single_fn),
        ("cpu_multi", cpu_multi_fn),
        ("gpu", gpu_fn),
    ):
        print(f"Running {mode}...")
        results[mode] = unwrap_result(fn(graph_csr, params))

    single = results["cpu_single"]
    multi = results["cpu_multi"]
    gpu = results["gpu"]

    if name in {"pagerank", "rwr"}:
        compare_vector("cpu_single vs cpu_multi scores", single["scores"], multi["scores"], atol, rtol)
        compare_vector("cpu_single vs gpu scores", single["scores"], gpu["scores"], atol, rtol)
        if "top_nodes" in single and "top_nodes" in gpu:
            print(f"  top_nodes overlap cpu/gpu              {top_overlap(single['top_nodes'], gpu['top_nodes']):.2%}")

    elif name == "hits":
        compare_vector("cpu_single vs cpu_multi hubs", single["hub_scores"], multi["hub_scores"], atol, rtol)
        compare_vector("cpu_single vs gpu hubs", single["hub_scores"], gpu["hub_scores"], atol, rtol)
        compare_vector("cpu_single vs cpu_multi auth", single["authority_scores"], multi["authority_scores"], atol, rtol)
        compare_vector("cpu_single vs gpu auth", single["authority_scores"], gpu["authority_scores"], atol, rtol)

    elif name == "bfs":
        compare_exact("cpu_single vs cpu_multi distances", single["distances"], multi["distances"])
        compare_exact("cpu_single vs gpu distances", single["distances"], gpu["distances"])
        compare_exact("cpu_single vs cpu_multi reachable", single["num_reachable"], multi["num_reachable"])
        compare_exact("cpu_single vs gpu reachable", single["num_reachable"], gpu["num_reachable"])

    elif name == "louvain":
        print("  Raw community IDs can differ even when partitions are similar.")
        print(f"  cpu_single communities/modularity      {single['num_communities']} / {single['modularity']:.6f}")
        print(f"  cpu_multi  communities/modularity      {multi['num_communities']} / {multi['modularity']:.6f}")
        print(f"  gpu        communities/modularity      {gpu['num_communities']} / {gpu['modularity']:.6f}")

    elif name == "mcl":
        print("  Raw cluster IDs can differ even when clusters are similar.")
        print(f"  cpu_single clusters/iterations         {single['num_clusters']} / {single['iterations']}")
        print(f"  cpu_multi  clusters/iterations         {multi['num_clusters']} / {multi['iterations']}")
        print(f"  gpu        clusters/iterations         {gpu['num_clusters']} / {gpu['iterations']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "9606.protein.links.v12.0.txt",
    )
    parser.add_argument(
        "--algorithm",
        choices=[*CPU_ALGORITHMS.keys(), "all"],
        default="all",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=50_000,
        help="Use a subset first. Pass 0 to load the whole raw file.",
    )
    parser.add_argument("--network-type", choices=["grn", "ppi", "mirna"], default="ppi")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Override max_iter for quick smoke runs of iterative algorithms.",
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_rows = None if args.sample_rows == 0 else args.sample_rows
    graph_csr, _node_index_map = load_graph(args.raw_path, sample_rows)

    algorithms = list(CPU_ALGORITHMS) if args.algorithm == "all" else [args.algorithm]
    for name in algorithms:
        params = {**DEFAULT_PARAMS[name], "network_type": args.network_type}
        if args.max_iter is not None and "max_iter" in params:
            params["max_iter"] = args.max_iter
        if name in {"bfs", "rwr"} and graph_csr.shape[0] == 0:
            print(f"Skipping {name}: graph has no nodes")
            continue
        if name == "rwr":
            params["seed_nodes"] = [0]
        if name == "bfs":
            params["source"] = 0
        try:
            compare_algorithm(name, graph_csr, params, args.atol, args.rtol)
        except Exception as exc:
            print(f"  ERROR running {name}: {type(exc).__name__}: {exc}")
            if math.isclose(args.sample_rows or 0, 0):
                print("  Try a sampled run first, for example --sample-rows 50000.")


if __name__ == "__main__":
    main()
