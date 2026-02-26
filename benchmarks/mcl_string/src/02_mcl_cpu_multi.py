"""
MCL Algorithm - Multi-threaded CPU Benchmark & Analysis
========================================================

This script runs the MCL (Markov Clustering) algorithm on preprocessed STRING PPI data
across different thread counts to benchmark performance and analyze clustering results.

Features:
  - Benchmark on 1, 2, 4, 8, 12 threads
  - Track time, memory, and clustering metrics for each run
  - Compare speedup across thread configurations
  - Generate detailed performance comparison report
  - Analyze and rank top proteins by cluster influence
  - Complete logging and result persistence

Input: Preprocessed CSR matrix (ppi_csr.npz from preprocessing step)
Output:
  - Combined comparison report (results/mcl_cpu_comparison.txt)
  - Per-thread execution logs (results/mcl_cpu_threads_<T>.log)
  - Per-thread summary statistics (results/summary_threads_<T>.txt)
  - Top proteins analysis (results/top_proteins_analysis.txt)
  - Detailed metrics (results/mcl_metrics.json)
  - Raw stdout/stderr for debugging (results/stdout_threads_<T>.txt, etc.)

Author: FYP Project
Date: 2026
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List


# ==============================
# CONFIGURATION
# ==============================

# Thread counts to benchmark (no timestamp concept)
THREAD_LIST = [1, 2, 4, 8, 12]

# MCL algorithm parameters
MCL_INFLATION = 3
MCL_TAU = 1e-13
MCL_MAX_ITERS = 5000
MCL_TOL = 0.0

# Output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path(f"results_cpu_{timestamp}")
# RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Logging verbosity
VERBOSE = True


# ==============================
# UTILITY FUNCTIONS
# ==============================

def log_message(msg: str):
    """Print message with timestamp."""
    if VERBOSE:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")


def run_mcl_worker(results_dir: Path, threads: int) -> Tuple[int, float, Dict]:
    """
    Run MCL algorithm with specified thread count.

    Args:
        results_dir: Directory for output files
        threads: Number of threads to use

    Returns:
        Tuple of (return_code, wall_clock_time, parsed_summary)
    """
    env = os.environ.copy()

    # Set ALL common thread environment variables
    thread_str = str(threads)
    env["OMP_NUM_THREADS"] = thread_str
    env["MKL_NUM_THREADS"] = thread_str
    env["OPENBLAS_NUM_THREADS"] = thread_str
    env["BLIS_NUM_THREADS"] = thread_str
    env["NUMEXPR_NUM_THREADS"] = thread_str

    # Tell worker where to write output
    env["RESULTS_DIR"] = str(results_dir)

    # Run worker in separate process so thread env vars actually apply
    cmd = [sys.executable, "src/03_mcl_cpu_worker.py"]

    log_message(f"  Running with {threads} thread(s)...")
    t0 = time.perf_counter()
    process = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    wall_time = time.perf_counter() - t0

    # Save stdout/stderr for debugging
    stdout_file = results_dir / f"stdout_threads_{threads}.txt"
    stderr_file = results_dir / f"stderr_threads_{threads}.txt"
    stdout_file.write_text(process.stdout)
    stderr_file.write_text(process.stderr)

    # Parse summary from worker
    summary = {}
    summary_file = results_dir / f"summary_threads_{threads}.txt"
    if summary_file.exists():
        try:
            summary_line = summary_file.read_text().strip()
            for item in summary_line.split():
                if "=" in item:
                    k, v = item.split("=", 1)
                    summary[k] = v
        except Exception as e:
            log_message(f"  Warning: Could not parse summary: {e}")

    return process.returncode, wall_time, summary


def parse_summary_dict(summary: Dict) -> Dict:
    """Parse and convert summary values to appropriate types."""
    parsed = {}
    for k, v in summary.items():
        try:
            if k in ["threads", "n", "nnz_in", "nnz_out"]:
                parsed[k] = int(v)
            elif k in ["load_time_s", "total_time_s", "analysis_time_s"]:
                parsed[k] = float(v)
            else:
                parsed[k] = v
        except (ValueError, TypeError):
            parsed[k] = v
    return parsed


def calculate_statistics(results_list: List[Dict]) -> Dict:
    """Calculate performance statistics across runs."""
    if not results_list:
        return {}

    times = [r["total_time"] for r in results_list if "total_time" in r]
    if not times:
        return {}

    base_time = times[0] if times else 0
    speedups = [base_time / t if t > 0 else 0 for t in times]
    efficiencies = [s / t for s, t in zip(speedups, [r["threads"] for r in results_list if "threads" in r])]

    return {
        "min_time": min(times),
        "max_time": max(times),
        "avg_time": sum(times) / len(times),
        "min_speedup": min(speedups) if speedups else 0,
        "max_speedup": max(speedups) if speedups else 0,
        "avg_efficiency": sum(efficiencies) / len(efficiencies) if efficiencies else 0,
    }


# ==============================
# MAIN EXECUTION
# ==============================

def main():
    """Main benchmark orchestrator."""

    log_message("\n" + "=" * 80)
    log_message("MCL ALGORITHM - MULTI-THREADED CPU BENCHMARK & ANALYSIS")
    log_message("=" * 80 + "\n")

    log_message("Configuration:")
    log_message(f"  Thread counts: {THREAD_LIST}")
    log_message(f"  MCL inflation: {MCL_INFLATION}")
    log_message(f"  MCL tau: {MCL_TAU}")
    log_message(f"  MCL max iterations: {MCL_MAX_ITERS}")
    log_message(f"  MCL tolerance: {MCL_TOL}")
    log_message(f"  Output directory: {RESULTS_DIR}\n")

    # Header for result table
    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append("MCL MULTI-THREADED BENCHMARK REPORT")
    report_lines.append("=" * 90)
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append(f"Thread counts tested: {THREAD_LIST}")
    report_lines.append(f"MCL Parameters: inflation={MCL_INFLATION}, tau={MCL_TAU}, "
                       f"max_iters={MCL_MAX_ITERS}, tol={MCL_TOL}")
    report_lines.append("")
    report_lines.append("PERFORMANCE RESULTS")
    report_lines.append("-" * 90)
    report_lines.append("Threads | Load(s) | Total(s) | Speedup | Efficiency | Nnz_in    | Nnz_out")
    report_lines.append("-" * 90)

    all_results = []
    base_time = None
    metrics_dict = {
        "metadata": {
            "algorithm": "MCL (Markov Clustering)",
            "database": "STRING v12.0 Human PPI",
            "timestamp": datetime.now().isoformat(),
            "thread_counts": THREAD_LIST,
            "mcl_parameters": {
                "inflation": MCL_INFLATION,
                "tau": MCL_TAU,
                "max_iterations": MCL_MAX_ITERS,
                "tolerance": MCL_TOL,
            }
        },
        "results": []
    }

    # Run benchmark for each thread count
    log_message("Starting benchmark runs...\n")

    for thread_count in THREAD_LIST:
        log_message(f"Running MCL with {thread_count} thread(s)...")

        rc, wall_time, summary = run_mcl_worker(RESULTS_DIR, thread_count)

        if rc != 0:
            log_message(f"  ✗ ERROR: Worker returned code {rc}")
            report_lines.append(f"{thread_count:>7} | ERROR (returncode={rc}) - see stderr_threads_{thread_count}.txt")
            continue

        # Parse summary values
        summary = parse_summary_dict(summary)

        if "total_time_s" not in summary:
            log_message(f"  ✗ ERROR: Missing total_time_s in summary")
            report_lines.append(f"{thread_count:>7} | ERROR (missing total_time_s)")
            continue

        load_time = float(summary.get("load_time_s", 0))
        total_time = float(summary.get("total_time_s", 0))
        nnz_in = int(summary.get("nnz_in", 0))
        nnz_out = int(summary.get("nnz_out", 0))
        n_nodes = int(summary.get("n", 0))

        if base_time is None:
            base_time = total_time

        speedup = (base_time / total_time) if total_time > 0 else 0.0
        efficiency = (speedup / thread_count) if thread_count > 0 else 0.0

        log_message(f"  ✓ Completed: {total_time:.2f}s (speedup: {speedup:.2f}x)")

        # Store for analysis
        result_row = {
            "threads": thread_count,
            "load_time": load_time,
            "total_time": total_time,
            "wall_time": wall_time,
            "speedup": speedup,
            "efficiency": efficiency,
            "nnz_in": nnz_in,
            "nnz_out": nnz_out,
            "n_nodes": n_nodes,
            "summary": summary
        }
        all_results.append(result_row)
        metrics_dict["results"].append(result_row)

        # Add to report
        report_lines.append(
            f"{thread_count:>7} | {load_time:>7.3f} | {total_time:>8.3f} | "
            f"{speedup:>7.2f} | {efficiency:>10.2f} | {nnz_in:>9,} | {nnz_out:>9,}"
        )

    # Add statistics section
    report_lines.append("-" * 90)
    stats = calculate_statistics(all_results)
    if stats:
        report_lines.append("\nSTATISTICS:")
        report_lines.append(f"  Min execution time:    {stats.get('min_time', 0):.3f}s")
        report_lines.append(f"  Max execution time:    {stats.get('max_time', 0):.3f}s")
        report_lines.append(f"  Avg execution time:    {stats.get('avg_time', 0):.3f}s")
        report_lines.append(f"  Max speedup:           {stats.get('max_speedup', 0):.2f}x")
        report_lines.append(f"  Avg efficiency:        {stats.get('avg_efficiency', 0):.2%}")

    # Add analysis results from worker
    report_lines.append("\n" + "=" * 90)
    report_lines.append("TOP 20 PROTEINS BY CLUSTER INFLUENCE (From 1-thread run)")
    report_lines.append("=" * 90)

    # Try to load top proteins analysis from worker output
    top_proteins_file = RESULTS_DIR / "top_proteins_analysis.txt"
    if top_proteins_file.exists():
        try:
            top_proteins_content = top_proteins_file.read_text()
            report_lines.append(top_proteins_content)
        except Exception as e:
            report_lines.append(f"Note: Could not load top proteins analysis ({e})")
    else:
        report_lines.append("Note: Top proteins analysis not yet generated. Run with single thread first.")

    # Add notes
    report_lines.append("\n" + "=" * 90)
    report_lines.append("NOTES")
    report_lines.append("=" * 90)
    report_lines.append("- Each run executes in a fresh Python process so thread env vars apply correctly")
    report_lines.append("- Speedup = base_time / current_time (base_time is 1-thread run)")
    report_lines.append("- Efficiency = speedup / threads (ideally close to 1.0)")
    report_lines.append("- Full per-run logs available in mcl_cpu_threads_<T>.log files")
    report_lines.append("- Raw stdout/stderr captured for reproducibility and debugging")
    report_lines.append("- Top proteins ranked by their influence in MCL clustering")
    report_lines.append("- Cluster statistics and sizes included in detailed output")
    report_lines.append("")

    # Write combined report
    report_file = RESULTS_DIR / "mcl_cpu_comparison.txt"
    report_file.write_text("\n".join(report_lines))

    # Write metrics in JSON format
    metrics_file = RESULTS_DIR / "mcl_metrics.json"
    metrics_file.write_text(json.dumps(metrics_dict, indent=2))

    log_message(f"\n✓ BENCHMARK COMPLETE")
    log_message(f"  Report file: {report_file}")
    log_message(f"  Metrics file: {metrics_file}")
    log_message(f"  Results directory: {RESULTS_DIR}")
    log_message("")

    # Print summary to console
    print("\n" + "\n".join(report_lines))

    return 0


if __name__ == "__main__":
    exit(main())