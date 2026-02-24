import time
import numpy as np
from pathlib import Path
from scipy import sparse

VAR_DIR = Path("data_processed/grn_bfs/variants")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
OUT_FILE = RESULTS_DIR / "bfs_cpu_single.txt"

# ---- Workload knobs to run longer ----
NUM_SOURCES = 2048
SOURCE_MODE = "top_degree"   # "top_degree" or "random"
SEED = 42

# Which variants to include (more variants => longer)
VARIANT_FILES = [
    "grn_all.npz",
    "grn_directed.npz",
    "grn_stimulation.npz",
    "grn_inhibition.npz",
    "grn_consensus_direction.npz",
    "grn_consensus_stimulation.npz",
    "grn_consensus_inhibition.npz",
]
# -------------------------------------

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

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

def run_variant(path: Path):
    A = sparse.load_npz(path).tocsr()
    sources = pick_sources(A, NUM_SOURCES, SOURCE_MODE, SEED)

    t = time.perf_counter()
    total_visited = 0
    total_depth = 0
    max_depth_seen = 0

    for k, s in enumerate(sources, 1):
        v, d = bfs_single(A, int(s))
        total_visited += v
        total_depth += d
        max_depth_seen = max(max_depth_seen, d)
        if k % 200 == 0:
            log(f"  {path.name}: progress {k}/{len(sources)} last_visited={v} last_depth={d}")

    elapsed = time.perf_counter() - t
    avg_depth = (total_depth / len(sources)) if len(sources) else 0.0
    return A.shape[0], A.nnz, len(sources), total_visited, avg_depth, max_depth_seen, elapsed

def main():
    log("CPU SINGLE MBFS (many BFS runs)")
    log(f"NUM_SOURCES={NUM_SOURCES} SOURCE_MODE={SOURCE_MODE} SEED={SEED}")
    log(f"VARIANTS={VARIANT_FILES}\n")

    grand_time = 0.0
    for vf in VARIANT_FILES:
        p = VAR_DIR / vf
        t0 = time.perf_counter()
        n, m, runs, tot_vis, avg_d, max_d, elapsed = run_variant(p)
        grand_time += elapsed
        log(f"\n[{vf}] n={n:,} edges={m:,} runs={runs}")
        log(f"  total_visited_sum={tot_vis:,} avg_depth={avg_d:.2f} max_depth={max_d}")
        log(f"  variant_time={elapsed:.2f}s (load+setup={time.perf_counter()-t0-elapsed:.2f}s)")

    log(f"\nTOTAL CPU SINGLE TIME (sum variants): {grand_time:.2f}s")

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(log_lines))

if __name__ == "__main__":
    main()
