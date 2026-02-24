import time
import numpy as np
from pathlib import Path
from scipy import sparse

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
except Exception as e:
    raise SystemExit("CuPy not installed. Install cupy-cuda11x/cupy-cuda12x.") from e

VAR_DIR = Path("data_processed/grn_bfs/variants")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
OUT_FILE = RESULTS_DIR / "bfs_gpu.txt"

# ---- Workload knobs (increase runtime) ----
NUM_SOURCES = 2048
SOURCE_MODE = "top_degree"   # "top_degree" or "random"
SEED = 42

VARIANT_FILES = [
    "grn_all.npz",
    "grn_directed.npz",
    "grn_stimulation.npz",
    "grn_inhibition.npz",
    "grn_consensus_direction.npz",
    "grn_consensus_stimulation.npz",
    "grn_consensus_inhibition.npz",
]
# -----------------------------------------

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

def pick_sources_cpu(A: sparse.csr_matrix, num_sources: int, mode: str, seed: int):
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

def bfs_gpu_levelsync(A_gpu: cpsp.csr_matrix, src: int, max_depth_cap: int = 1_000_000):
    n = A_gpu.shape[0]

    visited = cp.zeros(n, dtype=cp.bool_)
    frontier = cp.zeros(n, dtype=cp.float32)

    visited[src] = True
    frontier[src] = 1.0

    depth = 0
    while True:
        # next = A * frontier
        nxt = A_gpu @ frontier  # float32 vector
        next_frontier = (nxt != 0) & (~visited)

        cnt = int(cp.count_nonzero(next_frontier).get())
        if cnt == 0:
            break

        visited[next_frontier] = True

        # update frontier
        frontier = cp.zeros(n, dtype=cp.float32)
        frontier[next_frontier] = 1.0

        depth += 1
        if depth >= max_depth_cap:
            break

    visited_count = int(cp.count_nonzero(visited).get())
    return visited_count, depth

def run_variant(path: Path):
    # CPU load for source picking (cheap)
    A_cpu = sparse.load_npz(path).tocsr()
    sources = pick_sources_cpu(A_cpu, NUM_SOURCES, SOURCE_MODE, SEED)

    # Transfer CSR to GPU
    t0 = time.perf_counter()
    A_gpu = cpsp.csr_matrix(A_cpu.astype(np.float32))  # store as float for matvec
    cp.cuda.Stream.null.synchronize()
    transfer_t = time.perf_counter() - t0

    # BFS many times
    t1 = time.perf_counter()
    total_visited = 0
    total_depth = 0
    max_depth_seen = 0

    for k, s in enumerate(sources, 1):
        v, d = bfs_gpu_levelsync(A_gpu, int(s))
        total_visited += v
        total_depth += d
        max_depth_seen = max(max_depth_seen, d)
        if k % 200 == 0:
            log(f"  {path.name}: progress {k}/{len(sources)} last_visited={v} last_depth={d}")

    cp.cuda.Stream.null.synchronize()
    bfs_t = time.perf_counter() - t1
    avg_depth = (total_depth / len(sources)) if len(sources) else 0.0

    return A_cpu.shape[0], A_cpu.nnz, len(sources), total_visited, avg_depth, max_depth_seen, transfer_t, bfs_t

def main():
    log("GPU MBFS (CuPy sparse matvec BFS)")
    log(f"NUM_SOURCES={NUM_SOURCES} SOURCE_MODE={SOURCE_MODE} SEED={SEED}")
    log(f"VARIANTS={VARIANT_FILES}\n")

    grand = 0.0
    for vf in VARIANT_FILES:
        p = VAR_DIR / vf
        n, m, runs, tot_vis, avg_d, max_d, transfer_t, bfs_t = run_variant(p)
        grand += (transfer_t + bfs_t)
        log(f"\n[{vf}] n={n:,} edges={m:,} runs={runs}")
        log(f"  total_visited_sum={tot_vis:,} avg_depth={avg_d:.2f} max_depth={max_d}")
        log(f"  gpu_transfer_time={transfer_t:.2f}s gpu_bfs_time={bfs_t:.2f}s")

    log(f"\nTOTAL GPU TIME (sum variants): {grand:.2f}s")

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(log_lines))

if __name__ == "__main__":
    main()
