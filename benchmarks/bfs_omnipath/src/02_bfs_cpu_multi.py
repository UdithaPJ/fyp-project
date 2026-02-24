import time
import numpy as np
from pathlib import Path
from scipy import sparse
import multiprocessing as mp

VAR_DIR = Path("data_processed/grn_bfs/variants")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
OUT_FILE = RESULTS_DIR / "bfs_cpu_multi.txt"

# ---- Workload knobs ----
NUM_SOURCES = 4096
SOURCE_MODE = "top_degree"     # "top_degree" or "random"
SEED = 42
N_PROCS = 4

VARIANT_FILES = [
    "grn_all.npz",
    "grn_directed.npz",
    "grn_stimulation.npz",
    "grn_inhibition.npz",
    "grn_consensus_direction.npz",
    "grn_consensus_stimulation.npz",
    "grn_consensus_inhibition.npz",
]
# ------------------------

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

A_global = None

def init_worker(csr_path):
    global A_global
    A_global = sparse.load_npz(csr_path).tocsr()

def pick_sources(A: sparse.csr_matrix, num_sources: int, mode: str, seed: int):
    n = A.shape[0]
    out_deg = np.diff(A.indptr)
    if n == 0:
        return np.array([], dtype=np.int32)
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

    visited = 1
    max_depth = 0
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

def run_variant_multi(path: Path):
    A = sparse.load_npz(path).tocsr()
    sources = pick_sources(A, NUM_SOURCES, SOURCE_MODE, SEED)
    chunks = chunk_list(sources, N_PROCS)

    t = time.perf_counter()
    with mp.Pool(processes=N_PROCS, initializer=init_worker, initargs=(str(path),)) as pool:
        res = pool.map(worker_run, chunks)
    elapsed = time.perf_counter() - t

    total_visited = sum(x[0] for x in res)
    total_depth = sum(x[1] for x in res)
    max_depth_seen = max(x[2] for x in res) if res else 0
    total_runs = sum(x[3] for x in res)
    avg_depth = (total_depth / total_runs) if total_runs else 0.0

    return A.shape[0], A.nnz, total_runs, total_visited, avg_depth, max_depth_seen, elapsed

def main():
    log("CPU MULTI MBFS (multiprocessing)")
    log(f"NUM_SOURCES={NUM_SOURCES} SOURCE_MODE={SOURCE_MODE} SEED={SEED} N_PROCS={N_PROCS}")
    log(f"VARIANTS={VARIANT_FILES}\n")

    grand = 0.0
    for vf in VARIANT_FILES:
        p = VAR_DIR / vf
        n, m, runs, tot_vis, avg_d, max_d, elapsed = run_variant_multi(p)
        grand += elapsed
        log(f"\n[{vf}] n={n:,} edges={m:,} runs={runs}")
        log(f"  total_visited_sum={tot_vis:,} avg_depth={avg_d:.2f} max_depth={max_d}")
        log(f"  variant_time={elapsed:.2f}s")

    log(f"\nTOTAL CPU MULTI TIME (sum variants): {grand:.2f}s")

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(log_lines))

if __name__ == "__main__":
    main()
