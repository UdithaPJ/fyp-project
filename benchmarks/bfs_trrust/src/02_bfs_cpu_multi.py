import time
import json
import numpy as np
from pathlib import Path
from scipy import sparse
import multiprocessing as mp
from datetime import datetime

# ==================== Configuration ====================
CSR_NPZ = "data_processed/grn_bfs/grn_csr.npz"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
OUT_FILE_TXT = RESULTS_DIR / "bfs_cpu_multi.txt"
OUT_FILE_JSON = RESULTS_DIR / "bfs_cpu_multi.json"
OUT_FILE_CSV = RESULTS_DIR / "bfs_cpu_multi.csv"

# Workload configuration
NUM_SOURCES = 4096   # increase for more time
SOURCE_MODE = "top_degree"  # or "random"
SEED = 42
N_PROCS_LIST = [1, 2, 4, 8, 12]  # Thread counts to benchmark
REPEAT_BFS = 6
EXTRA_COMPUTE = 12
# =========================================================

log_lines = []

def log(msg):
    """Log message to stdout and buffer for file output."""
    print(msg)
    log_lines.append(msg)

# Globals for workers
A_global = None

def init_worker(csr_path):
    """Initialize worker process with CSR matrix."""
    global A_global
    A_global = sparse.load_npz(csr_path).tocsr()

def pick_sources(A: sparse.csr_matrix, num_sources: int, mode: str, seed: int):
    """Select source nodes for BFS runs."""
    n = A.shape[0]
    out_deg = np.diff(A.indptr)
    if mode == "top_degree":
        return np.argsort(-out_deg)[:min(num_sources, n)].astype(np.int32)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(n, size=min(num_sources, n), replace=False).astype(np.int32)
    else:
        raise ValueError(f"Unknown SOURCE_MODE: {mode}")

def bfs_single(A: sparse.csr_matrix, src: int):
    """Run BFS from a single source node.
    
    Returns:
        tuple: (visited_count, max_depth)
    """
    n = A.shape[0]
    dist = np.full(n, -1, dtype=np.int32)
    q = np.empty(n, dtype=np.int32)
    head = 0
    tail = 0
    dist[src] = 0
    q[tail] = src
    tail += 1

    max_depth = 0
    visited = 1
    indptr = A.indptr
    indices = A.indices

    while head < tail:
        u = q[head]
        head += 1
        du = dist[u]
        start, end = indptr[u], indptr[u+1]
        for v in indices[start:end]:
            for _ in range(EXTRA_COMPUTE):
                _dummy = (v * 31) % 7

            if dist[v] == -1:
                dist[v] = du + 1
                q[tail] = v
                tail += 1
                visited += 1
                if dist[v] > max_depth:
                    max_depth = dist[v]
    return visited, max_depth

def worker_run(src_list):
    A = A_global
    total_visited = 0
    total_depth = 0
    max_depth_seen = 0
    per_source = []

    for s in src_list:
        for _ in range(REPEAT_BFS):   # 🔥 repeat BFS
            v, d = bfs_single(A, int(s))
            total_visited += v
            total_depth += d
            if d > max_depth_seen:
                max_depth_seen = d

        per_source.append((int(s), v, d))

    return (
        total_visited,
        total_depth,
        max_depth_seen,
        len(src_list) * REPEAT_BFS,  # IMPORTANT
        per_source,
    )

def chunk_list(arr, k):
    """Split array into k chunks."""
    return [arr[i::k] for i in range(k)]

def run_for_nprocs(n_procs, csr_path, sources):
    """Run BFS benchmark with specified number of processes.
    
    Returns both aggregate results and per-source details.
    """
    chunks = chunk_list(sources, n_procs)
    t1 = time.perf_counter()
    with mp.Pool(processes=n_procs, initializer=init_worker, initargs=(csr_path,)) as pool:
        results = pool.map(worker_run, chunks)
    total_t = time.perf_counter() - t1

    total_visited = sum(r[0] for r in results)
    total_depth = sum(r[1] for r in results)
    max_depth_seen = max((r[2] for r in results), default=0)
    total_runs = sum(r[3] for r in results)
    avg_depth = total_depth / total_runs if total_runs else 0
    
    # Aggregate per-source results from all workers
    all_per_source = []
    for r in results:
        if len(r) > 4:
            all_per_source.extend(r[4])

    return {
        "n_procs": int(n_procs),
        "total_runs": int(total_runs),
        "total_visited": int(total_visited),
        "avg_depth": float(avg_depth),
        "max_depth": int(max_depth_seen),
        "time_s": float(total_t),
        "throughput": float(total_runs / total_t) if total_t > 0 else 0,
        "per_source_results": all_per_source,  # New: detailed per-source results
    }


def compute_analysis(results_list):
    """Compute speedup and efficiency metrics."""
    if not results_list:
        return None
    baseline_time = results_list[0]["time_s"]
    analysis = []
    for res in results_list:
        speedup = baseline_time / res["time_s"] if res["time_s"] > 0 else 1.0
        efficiency = speedup / res["n_procs"] if res["n_procs"] > 0 else 0.0
        analysis.append({
            "n_procs": res["n_procs"],
            "speedup": float(speedup),
            "efficiency": float(efficiency),
        })
    return analysis

def rank_sources(per_source_results, top_n=20):
    """Rank source nodes by visited count and depth.
    
    Args:
        per_source_results: List of (source_id, visited, depth) tuples
        top_n: Number of top sources to return
        
    Returns:
        Tuple of (by_visited, by_depth)
    """
    if not per_source_results:
        return [], []
    
    # Rank by visited count (descending)
    by_visited = sorted(per_source_results, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Rank by max depth (descending)
    by_depth = sorted(per_source_results, key=lambda x: x[2], reverse=True)[:top_n]
    
    return by_visited, by_depth

def print_results_table(results_list, analysis_list):
    """Print formatted results table."""
    log("\n" + "="*100)
    log("BFS MULTI-THREADED BENCHMARK RESULTS")
    log("="*100)
    log(f"{'Procs':<8} {'Runs':<8} {'Visited':<15} {'Avg Depth':<12} {'Max Depth':<12} {'Time (s)':<12} {'Throughput':<15} {'Speedup':<10} {'Efficiency':<12}")
    log("-"*100)
    for res, ana in zip(results_list, analysis_list):
        log(f"{res['n_procs']:<8} {res['total_runs']:<8} {res['total_visited']:<15,} {res['avg_depth']:<12.2f} {res['max_depth']:<12} {res['time_s']:<12.4f} {res['throughput']:<15.2f} {ana['speedup']:<10.2f}x {ana['efficiency']:<12.2%}")
    log("="*100)

def save_results_json(results_list, analysis_list, graph_info, config):
    """Save results to JSON format."""
    # Convert per_source_results to standard Python types for JSON serialization
    results_json = []
    for res in results_list:
        res_copy = res.copy()
        if "per_source_results" in res_copy:
            # Convert numpy types to Python native types
            per_src = [(int(s), int(v), int(d)) for s, v, d in res_copy["per_source_results"]]
            res_copy["per_source_results"] = per_src
        results_json.append(res_copy)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "configuration": config,
        "graph_info": graph_info,
        "results": results_json,
        "analysis": analysis_list,
    }
    with open(OUT_FILE_JSON, "w") as f:
        json.dump(output, f, indent=2)
    log(f"✓ JSON results saved to {OUT_FILE_JSON}")

def save_results_csv(results_list, analysis_list):
    """Save results to CSV format."""
    csv_lines = []
    csv_lines.append("n_procs,total_runs,total_visited,avg_depth,max_depth,time_s,throughput,speedup,efficiency")
    for res, ana in zip(results_list, analysis_list):
        csv_lines.append(f"{res['n_procs']},{res['total_runs']},{res['total_visited']},{res['avg_depth']:.4f},{res['max_depth']},{res['time_s']:.6f},{res['throughput']:.4f},{ana['speedup']:.4f},{ana['efficiency']:.4f}")
    with open(OUT_FILE_CSV, "w") as f:
        f.write("\n".join(csv_lines))
    log(f"CSV results saved to {OUT_FILE_CSV}")
def save_algorithm_outputs(results_list):
    """Save algorithm outputs and rankings for single-thread run."""
    if not results_list or len(results_list) < 1:
        return
    
    # Use single-thread results (most reliable)
    single_thread_res = results_list[0]
    per_source = single_thread_res.get("per_source_results", [])
    
    if not per_source:
        log("⚠ No per-source results available")
        return
    
    by_visited, by_depth = rank_sources(per_source, top_n=20)
    
    # Save to text file
    algo_output_file = RESULTS_DIR / "bfs_algorithm_outputs.txt"
    with open(algo_output_file, "w") as f:
        f.write("BFS ALGORITHM OUTPUTS AND RANKINGS\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total sources analyzed: {len(per_source)}\n\n")
        
        f.write("TOP 20 SOURCES BY NODES VISITED\n")
        f.write("=" * 100 + "\n")
        f.write("Rank | Source ID |   Visited   | Max Depth | % Coverage\n")
        f.write("-" * 100 + "\n")
        total_nodes = 2861  # From our dataset
        for rank, (src_id, visited, depth) in enumerate(by_visited, 1):
            coverage = (visited / total_nodes) * 100
            f.write(f"{rank:4d} | {src_id:9d} | {visited:11,d} | {depth:9d} | {coverage:9.2f}%\n")
        
        f.write("\n")
        f.write("TOP 20 SOURCES BY MAX DEPTH REACHED\n")
        f.write("=" * 100 + "\n")
        f.write("Rank | Source ID | Max Depth |   Visited   | Nodes/Depth Ratio\n")
        f.write("-" * 100 + "\n")
        for rank, (src_id, visited, depth) in enumerate(by_depth, 1):
            ratio = visited / max(depth, 1)
            f.write(f"{rank:4d} | {src_id:9d} | {depth:9d} | {visited:11,d} | {ratio:17.2f}\n")
        
        f.write("\n")
        f.write("ALGORITHM STATISTICS\n")
        f.write("=" * 100 + "\n")
        visited_counts = [r[1] for r in per_source]
        depth_counts = [r[2] for r in per_source]
        f.write(f"  Total sources:        {len(per_source)}\n")
        f.write(f"  Avg nodes visited:    {np.mean(visited_counts):.2f}\n")
        f.write(f"  Min nodes visited:    {np.min(visited_counts):,}\n")
        f.write(f"  Max nodes visited:    {np.max(visited_counts):,}\n")
        f.write(f"  Avg max depth:        {np.mean(depth_counts):.2f}\n")
        f.write(f"  Min depth reached:    {np.min(depth_counts)}\n")
        f.write(f"  Max depth reached:    {np.max(depth_counts)}\n")
    
    log(f"✓ Algorithm outputs saved to {algo_output_file}")
def main():
    """Main benchmark execution."""
    log(f"\n{'='*80}")
    log("BFS MULTI-THREADED CPU BENCHMARK")
    log(f"Started: {datetime.now().isoformat()}")
    log(f"{'='*80}\n")
    
    # Load graph
    log("[1/3] Loading graph...")
    t0 = time.perf_counter()
    try:
        A = sparse.load_npz(CSR_NPZ).tocsr()
    except FileNotFoundError:
        log(f"ERROR: Graph file not found at {CSR_NPZ}")
        return
    
    load_time = time.perf_counter() - t0
    graph_info = {
        "nodes": int(A.shape[0]),
        "edges": int(A.nnz),
        "load_time_s": float(load_time),
    }
    log(f"✓ Graph loaded: {graph_info['nodes']:,} nodes, {graph_info['edges']:,} edges (load_time={load_time:.3f}s)")

    # Select sources
    log("\n[2/3] Selecting source nodes...")
    sources = pick_sources(A, NUM_SOURCES, SOURCE_MODE, SEED)
    log(f"✓ Selected {len(sources)} source nodes ({SOURCE_MODE} mode)")

    # Run benchmarks
    log("\n[3/3] Running benchmarks...")
    all_results = []
    for n in N_PROCS_LIST:
        procs = max(1, min(n, len(sources)))
        log(f"\n  Running with {procs} process(es)...")
        res = run_for_nprocs(procs, CSR_NPZ, sources)
        all_results.append(res)
        log(f"    ✓ Time: {res['time_s']:.4f}s | Throughput: {res['throughput']:.2f} runs/sec")

    # Compute analysis
    analysis = compute_analysis(all_results)

    # Display results
    print_results_table(all_results, analysis)

    # Store configuration
    config = {
        "num_sources": NUM_SOURCES,
        "source_mode": SOURCE_MODE,
        "seed": SEED,
        "thread_counts": N_PROCS_LIST,
    }

    # Save all output formats
    log(f"\n[4/4] Saving results...")
    save_results_json(all_results, analysis, graph_info, config)
    save_results_csv(all_results, analysis)
    save_algorithm_outputs(all_results)
    
    # Save text summary
    with open(OUT_FILE_TXT, "w") as f:
        f.write("\n".join(log_lines))
    log(f"✓ Text summary saved to {OUT_FILE_TXT}")

    log(f"\n{'='*80}")
    log(f"Benchmark completed at {datetime.now().isoformat()}")
    log(f"{'='*80}\n")


if __name__ == "__main__":
    main()
