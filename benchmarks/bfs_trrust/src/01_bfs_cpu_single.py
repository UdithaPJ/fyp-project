import time
import json
import numpy as np
from pathlib import Path
from scipy import sparse

CSR_NPZ  = "data_processed/grn_bfs/grn_csr.npz"
MAP_JSON = "data_processed/grn_bfs/node_map.json"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
OUT_FILE = RESULTS_DIR / "bfs_cpu_single.txt"

# ---- Workload knobs (increase runtime) ----
NUM_SOURCES = 1024
SOURCE_MODE = "top_degree"   # "top_degree" | "random"
SEED = 42
# -----------------------------------------

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

def pick_sources(A: sparse.csr_matrix, num_sources: int, mode: str, seed: int):
    n = A.shape[0]
    out_deg = np.diff(A.indptr)

    if mode == "top_degree":
        idx = np.argsort(-out_deg)[:min(num_sources, n)]
        return idx.astype(np.int32)
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
        u = q[head]
        head += 1
        du = dist[u]
        start, end = indptr[u], indptr[u+1]
        for v in indices[start:end]:
            if dist[v] == -1:
                dist[v] = du + 1
                q[tail] = v
                tail += 1
                visited += 1
                if dist[v] > max_depth:
                    max_depth = dist[v]

    return visited, max_depth

def main():
    t0 = time.perf_counter()
    A = sparse.load_npz(CSR_NPZ).tocsr()
    load_t = time.perf_counter() - t0
    log(f"Loaded CSR: n={A.shape[0]:,} nnz={A.nnz:,} load_time={load_t:.2f}s")
    log(f"Workload: NUM_SOURCES={NUM_SOURCES} SOURCE_MODE={SOURCE_MODE} SEED={SEED}")

    sources = pick_sources(A, NUM_SOURCES, SOURCE_MODE, SEED)
    log(f"Selected sources: {len(sources)}")

    t1 = time.perf_counter()
    total_visited = 0
    total_depth = 0
    max_depth_seen = 0

    for k, src in enumerate(sources, 1):
        visited, depth = bfs_single(A, int(src))
        total_visited += visited
        total_depth += depth
        max_depth_seen = max(max_depth_seen, depth)

        if k % 100 == 0:
            log(f"progress {k}/{len(sources)} | last_visited={visited} last_depth={depth}")

    total_t = time.perf_counter() - t1
    avg_depth = total_depth / len(sources) if len(sources) else 0

    log("\nCPU SINGLE BFS SUMMARY")
    log(f"Total BFS runs: {len(sources)}")
    log(f"Total visited (sum over runs): {total_visited:,}")
    log(f"Avg depth (per run): {avg_depth:.2f}")
    log(f"Max depth seen: {max_depth_seen}")
    log(f"Total BFS time: {total_t:.2f}s")

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(log_lines))

if __name__ == "__main__":
    main()
