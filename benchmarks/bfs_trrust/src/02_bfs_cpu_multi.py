import time
import numpy as np
from pathlib import Path
from scipy import sparse
import multiprocessing as mp

CSR_NPZ = "data_processed/grn_bfs/grn_csr.npz"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
OUT_FILE = RESULTS_DIR / "bfs_cpu_multi.txt"

# ---- Workload knobs ----
NUM_SOURCES = 2048   # increase for more time
SOURCE_MODE = "top_degree"  # or "random"
SEED = 42
N_PROCS = 4          # set to your CPU cores (e.g., 8)
# ------------------------

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

# Globals for workers
A_global = None

def init_worker(csr_path):
    global A_global
    A_global = sparse.load_npz(csr_path).tocsr()

def pick_sources(A: sparse.csr_matrix, num_sources: int, mode: str, seed: int):
    n = A.shape[0]
    out_deg = np.diff(A.indptr)
    if mode == "top_degree":
        return np.argsort(-out_deg)[:min(num_sources, n)].astype(np.int32)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(n, size=min(num_sources, n), replace=False).astype(np.int32)
    else:
        raise ValueError("Unknown SOURCE_MODE")

def bfs_single(A: sparse.csr_matrix, src: int):
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
        u = q[head]; head += 1
        du = dist[u]
        start, end = indptr[u], indptr[u+1]
        for v in indices[start:end]:
            if dist[v] == -1:
                dist[v] = du + 1
                q[tail] = v; tail += 1
                visited += 1
                if dist[v] > max_depth:
                    max_depth = dist[v]
    return visited, max_depth

def worker_run(src_list):
    A = A_global
    total_visited = 0
    total_depth = 0
    max_depth_seen = 0
    for s in src_list:
        v, d = bfs_single(A, int(s))
        total_visited += v
        total_depth += d
        if d > max_depth_seen:
            max_depth_seen = d
    return total_visited, total_depth, max_depth_seen, len(src_list)

def chunk_list(arr, k):
    return [arr[i::k] for i in range(k)]

def main():
    t0 = time.perf_counter()
    A = sparse.load_npz(CSR_NPZ).tocsr()
    log(f"Loaded CSR: n={A.shape[0]:,} nnz={A.nnz:,} load_time={time.perf_counter()-t0:.2f}s")
    log(f"Workload: NUM_SOURCES={NUM_SOURCES} SOURCE_MODE={SOURCE_MODE} N_PROCS={N_PROCS}")

    sources = pick_sources(A, NUM_SOURCES, SOURCE_MODE, SEED)
    log(f"Selected sources: {len(sources)}")

    chunks = chunk_list(sources, N_PROCS)

    t1 = time.perf_counter()
    with mp.Pool(processes=N_PROCS, initializer=init_worker, initargs=(CSR_NPZ,)) as pool:
        results = pool.map(worker_run, chunks)

    total_t = time.perf_counter() - t1

    total_visited = sum(r[0] for r in results)
    total_depth = sum(r[1] for r in results)
    max_depth_seen = max(r[2] for r in results) if results else 0
    total_runs = sum(r[3] for r in results)

    avg_depth = total_depth / total_runs if total_runs else 0

    log("\nCPU MULTI BFS SUMMARY")
    log(f"Total BFS runs: {total_runs}")
    log(f"Total visited (sum over runs): {total_visited:,}")
    log(f"Avg depth (per run): {avg_depth:.2f}")
    log(f"Max depth seen: {max_depth_seen}")
    log(f"Total BFS time: {total_t:.2f}s")

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(log_lines))

if __name__ == "__main__":
    main()
