# src/03_mcl_cpu_multithread_compare.py
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path


THREAD_LIST = [1, 2, 4, 8, 12]


def run_one(results_dir: Path, threads: int):
    env = os.environ.copy()

    # Set ALL common thread env vars (covers MKL/OpenBLAS/BLIS/NumExpr)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["BLIS_NUM_THREADS"] = str(threads)
    env["NUMEXPR_NUM_THREADS"] = str(threads)

    # Tell worker where to write output
    env["RESULTS_DIR"] = str(results_dir)

    cmd = [sys.executable, "src/03_mcl_cpu_worker.py"]

    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0

    # Save stdout/stderr for debugging
    (results_dir / f"stdout_threads_{threads}.txt").write_text(p.stdout)
    (results_dir / f"stderr_threads_{threads}.txt").write_text(p.stderr)

    return p.returncode, wall


def parse_summary_line(line: str) -> dict:
    # threads=2 n=... nnz_in=... nnz_out=... load_time_s=... total_time_s=...
    out = {}
    for kv in line.strip().split():
        k, v = kv.split("=", 1)
        out[k] = v
    return out


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results_mcl_cpu_threads_{timestamp}")
    results_dir.mkdir(exist_ok=True)

    combined = []
    combined.append(f"STRING MCL CPU MULTI-THREAD COMPARISON")
    combined.append(f"timestamp={timestamp}")
    combined.append(f"threads_tested={THREAD_LIST}")
    combined.append("")

    table = []
    combined.append("SUMMARY:")
    combined.append("threads | load_time_s | total_time_s | speedup_vs_2t | nnz_out")
    combined.append("-" * 72)

    base_time = None

    for t in THREAD_LIST:
        rc, wall = run_one(results_dir, t)

        if rc != 0:
            combined.append(f"{t:>7} | ERROR (returncode={rc}) | see stderr_threads_{t}.txt")
            continue

        summary_file = results_dir / f"summary_threads_{t}.txt"
        if not summary_file.exists():
            combined.append(f"{t:>7} | ERROR (missing summary file) | check stdout_threads_{t}.txt")
            continue

        d = parse_summary_line(summary_file.read_text().strip())
        load_time = float(d["load_time_s"])
        total_time = float(d["total_time_s"])
        nnz_out = int(d["nnz_out"])

        if base_time is None:
            base_time = total_time

        speedup = (base_time / total_time) if total_time > 0 else 0.0

        combined.append(f"{t:>7} | {load_time:>11.3f} | {total_time:>12.3f} | {speedup:>12.3f} | {nnz_out:,}")

        table.append((t, load_time, total_time, speedup, nnz_out, wall))

    combined.append("")
    combined.append("NOTES:")
    combined.append("- Each run is executed in a fresh Python process so thread env vars actually apply.")
    combined.append("- Full per-run logs: mcl_cpu_threads_<t>.log")
    combined.append("- Raw stdout/stderr captured for each run for reproducibility/debugging.")
    combined.append("")

    out_file = results_dir / f"mcl_cpu_multithread_comparison_{timestamp}.txt"
    out_file.write_text("\n".join(combined))

    print(f"\nDONE. Results folder: {results_dir}")
    print(f"Combined report: {out_file}\n")


if __name__ == "__main__":
    main()