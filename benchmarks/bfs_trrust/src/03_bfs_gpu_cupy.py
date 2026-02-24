import time
import numpy as np
from pathlib import Path
from scipy import sparse

try:
    import cupy as cp
except Exception as e:
    raise SystemExit("CuPy not installed. Install cupy-cuda11x/cupy-cuda12x.") from e

CSR_NPZ = "data_processed/grn_bfs/grn_csr.npz"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
OUT_FILE = RESULTS_DIR / "bfs_gpu.txt"

# ---- Workload knobs ----
NUM_SOURCES = 1024
SOURCE_MODE = "top_degree"  # or "random"
SEED = 42
# ------------------------

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

def pick_sources_cpu(A: sparse.csr_matrix, num_sources: int, mode: str, seed: int):
    n = A.shape[0]
    out_deg = np.diff(A.indptr)
    if mode == "top_degree":
        return np.argsort(-out_deg)[:min(num_sources, n)].astype(np.int32)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(n, size=min(num_sources, n), replace=False).astype(np.int32)
    else:
        raise ValueError("Unknown SOURCE_MODE")

def bfs_gpu_csr(indptr, indices, src, n):
    # frontier/visited as boolean arrays
    visited = cp.zeros(n, dtype=cp.bool_)
    frontier = cp.zeros(n, dtype=cp.bool_)
    visited[src] = True
    frontier[src] = True

    depth = 0
    visited_count = 1

    while True:
        frontier_nodes = cp.where(frontier)[0]
        if frontier_nodes.size == 0:
            break

        # Build next frontier
        next_frontier = cp.zeros(n, dtype=cp.bool_)

        # For each node in frontier, visit neighbors
        # NOTE: This is not fully optimized; good as prototype baseline.
        for u in frontier_nodes.tolist():
            start = indptr[u]
            end = indptr[u + 1]
            nbrs = indices[start:end]
            new = nbrs[~visited[nbrs]]
            if new.size > 0:
                visited[new] = True
                next_frontier[new] = True

        visited_count = int(cp.count_nonzero(visited).get())
        frontier = next_frontier
        depth += 1

        # Safety stop if something weird happens
        if depth > n:
            break

    return visited_count, depth

def main():
    # Load CSR on CPU
    t0 = time.perf_counter()
    A = sparse.load_npz(CSR_NPZ).tocsr()
    log(f"Loaded CSR: n={A.shape[0]:,} nnz={A.nnz:,} load_time={time.perf_counter()-t0:.2f}s")

    sources = pick_sources_cpu(A, NUM_SOURCES, SOURCE_MODE, SEED)
    log(f"Workload: NUM_SOURCES={len(sources)} SOURCE_MODE={SOURCE_MODE}")

    # Transfer CSR to GPU
    t1 = time.perf_counter()
    indptr = cp.asarray(A.indptr, dtype=cp.int32)
    indices = cp.asarray(A.indices, dtype=cp.int32)
    cp.cuda.Stream.null.synchronize()
    log(f"Transferred CSR to GPU | time={time.perf_counter()-t1:.2f}s")

    t2 = time.perf_counter()
    total_visited = 0
    total_depth = 0
    max_depth_seen = 0

    n = A.shape[0]

    for k, s in enumerate(sources, 1):
        v, d = bfs_gpu_csr(indptr, indices, int(s), n)
        total_visited += v
        total_depth += d
        max_depth_seen = max(max_depth_seen, d)

        if k % 50 == 0:
            log(f"progress {k}/{len(sources)} | last_visited={v} last_depth={d}")

    cp.cuda.Stream.null.synchronize()
    total_t = time.perf_counter() - t2
    avg_depth = total_depth / len(sources) if len(sources) else 0

    log("\nGPU BFS SUMMARY")
    log(f"Total BFS runs: {len(sources)}")
    log(f"Total visited (sum over runs): {total_visited:,}")
    log(f"Avg depth (per run): {avg_depth:.2f}")
    log(f"Max depth seen: {max_depth_seen}")
    log(f"Total BFS time: {total_t:.2f}s")

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(log_lines))

if __name__ == "__main__":
    main()
