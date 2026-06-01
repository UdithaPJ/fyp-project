"""
experiments/benchmark/run_benchmark.py
=======================================

CLI driver that wires the preprocessing pipeline into the two benchmarkers:

    RuntimeBenchmarker     — repeated-run timing across all four modes
    ScalabilityBenchmarker — scalability across synthetic graph sizes

Examples
--------
    # Runtime benchmark on a real dataset (3 timed runs per mode)
    python experiments/benchmark/run_benchmark.py runtime \\
        --raw-path data/raw/9606.protein.links.v12.0.txt \\
        --network-type ppi --n-runs 3

    # Scalability on synthetic graphs (10K and 50K only, pagerank + bfs)
    python experiments/benchmark/run_benchmark.py scalability \\
        --sizes 10000,50000 --algorithms pagerank,bfs --modes cpu_single,gpu

    # Both benchmarks in one call
    python experiments/benchmark/run_benchmark.py all \\
        --raw-path data/raw/9606.protein.links.v12.0.txt \\
        --sizes 10000,50000 --algorithms pagerank,bfs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.runtime_benchmark    import RuntimeBenchmarker      # noqa: E402
from src.benchmarking.scalability_benchmark import ScalabilityBenchmarker  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_graph(raw_path: Path, sample_rows: int | None,
                source_col: str, target_col: str, weight_col: str):
    from src.graph.converter import graphdata_to_csr
    from src.preprocessing.pipeline import PreprocessingPipeline

    print(f"[benchmark] loading {raw_path} (sample_rows={sample_rows})")
    df = pd.read_csv(raw_path, sep=None, engine="python",
                     nrows=sample_rows, low_memory=True)
    print(f"[benchmark] rows: {len(df):,}")
    pipeline = PreprocessingPipeline()
    graph_data, report = pipeline.run_dataframe(
        df,
        user_override={"source": source_col, "target": target_col,
                       "weight": weight_col},
        duplicate_strategy="mean",
    )
    graph_csr, node_index_map = graphdata_to_csr(graph_data)
    print(f"[benchmark] nodes={graph_csr.shape[0]:,} edges={graph_csr.nnz:,}")
    return graph_csr, node_index_map


def _parse_mode_list(s: str, default_tuple: tuple) -> tuple:
    if s.lower() == "all":
        return default_tuple
    return tuple(m.strip() for m in s.split(",") if m.strip())


def _parse_algo_list(s: str, default_tuple: tuple) -> tuple:
    if s.lower() == "all":
        return default_tuple
    return tuple(a.strip() for a in s.split(",") if a.strip())


# ---------------------------------------------------------------------------
# Sub-command: runtime
# ---------------------------------------------------------------------------

def _run_runtime(args: argparse.Namespace) -> None:
    from src.benchmarking.runtime_benchmark import ALGORITHMS as ALL_ALGOS, MODES as ALL_MODES
    graph_csr, node_index_map = _load_graph(
        args.raw_path,
        None if args.sample_rows == 0 else args.sample_rows,
        args.source_col, args.target_col, args.weight_col,
    )
    ds_name = args.dataset_name or args.raw_path.stem
    algos   = _parse_algo_list(args.algorithms, ALL_ALGOS)
    modes   = _parse_mode_list(args.modes,      ALL_MODES)

    rwr_seeds = [int(s) for s in args.rwr_seeds.split(",") if s.strip()]
    params_override = {
        "bfs": {"source": int(args.source_node)},
        "rwr": {"seed_nodes": rwr_seeds or [0]},
    }

    rb = RuntimeBenchmarker(
        n_runs=args.n_runs,
        warmup_runs=args.warmup_runs,
        output_dir=args.output_dir,
        algorithms=algos,
        modes=modes,
    )
    rb.add_dataset(
        name=ds_name,
        graph_csr=graph_csr,
        network_type=args.network_type,
        params_override=params_override,
        node_index_map=node_index_map,
    )
    print(f"[benchmark] RuntimeBenchmarker: "
          f"{rb.n_runs} runs × {len(rb.algorithms)} algos × {len(rb.modes)} modes")
    rb.run()

    csv_path   = rb.write_csv()
    plot_paths = rb.write_plots()
    _print_summary("runtime", csv_path, plot_paths, rb.records)


# ---------------------------------------------------------------------------
# Sub-command: scalability
# ---------------------------------------------------------------------------

def _run_scalability(args: argparse.Namespace) -> None:
    from src.benchmarking.scalability_benchmark import (
        ALGORITHMS as ALL_ALGOS, MODES as ALL_MODES,
    )
    algos = _parse_algo_list(args.algorithms, ALL_ALGOS)
    modes = _parse_mode_list(args.modes,      ALL_MODES)
    sizes = [int(s) for s in str(args.sizes).split(",") if s.strip()]
    types = [t.strip() for t in args.graph_types.split(",") if t.strip()]

    sb = ScalabilityBenchmarker(
        graph_sizes=sizes,
        graph_types=types,
        algorithms=algos,
        modes=modes,
        output_dir=args.output_dir,
        network_type=args.network_type,
        n_runs=args.n_runs,
    )
    print(f"[benchmark] ScalabilityBenchmarker: "
          f"{len(sb.graph_types)} types × {len(sb.graph_sizes)} sizes × "
          f"{len(sb.algorithms)} algos × {len(sb.modes)} modes")
    sb.run()
    csv_path   = sb.write_csv()
    plot_paths = sb.write_plots()
    _print_summary("scalability", csv_path, plot_paths, sb.records)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(kind: str, csv_path: Path,
                   plot_paths: dict, records: list) -> None:
    print()
    print("=" * 70)
    print(f"{kind.upper()} BENCHMARK COMPLETE")
    print(f"CSV  : {csv_path}")
    for k, v in plot_paths.items():
        print(f"PLOT : {k:<32} {v}")
    succeeded = [r for r in records if r.success]
    failed    = [r for r in records if not r.success]
    print(f"Runs : {len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        print("Failed (first 5):")
        for r in failed[:5]:
            print(f"  {r.algorithm}/{r.mode}: {(r.error or '')[:80]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "experiments" / "outputs")
    p.add_argument("--algorithms", default="all",
                   help="comma-separated list or 'all'")
    p.add_argument("--modes", default="all",
                   help="comma-separated list of modes or 'all'")
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--network-type", choices=["grn", "ppi", "mirna"],
                   default="ppi")
    p.add_argument("--verbose", action="store_true")


def _add_real_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--raw-path", type=Path,
                   default=PROJECT_ROOT / "data" / "raw" /
                            "9606.protein.links.v12.0.txt")
    p.add_argument("--sample-rows", type=int, default=50_000)
    p.add_argument("--source-col", default="protein1")
    p.add_argument("--target-col", default="protein2")
    p.add_argument("--weight-col", default="combined_score")
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--source-node", type=int, default=0)
    p.add_argument("--rwr-seeds", default="0")


def _add_scalability_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sizes", default="10000,50000,100000,500000",
                   help="comma-separated target node counts")
    p.add_argument("--graph-types",
                   default="barabasi_albert,erdos_renyi,watts_strogatz")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FYP benchmarking driver — runtime and scalability.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_rt = sub.add_parser("runtime", help="Repeated-run timing benchmark")
    _add_shared_args(p_rt)
    _add_real_dataset_args(p_rt)

    p_sc = sub.add_parser("scalability", help="Scalability across graph sizes")
    _add_shared_args(p_sc)
    _add_scalability_args(p_sc)

    p_all = sub.add_parser("all", help="Run both runtime and scalability")
    _add_shared_args(p_all)
    _add_real_dataset_args(p_all)
    _add_scalability_args(p_all)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    if args.command == "runtime":
        _run_runtime(args)
    elif args.command == "scalability":
        _run_scalability(args)
    elif args.command == "all":
        _run_runtime(args)
        _run_scalability(args)


if __name__ == "__main__":
    main()
