import time, json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import sparse

# ==============================
# CONFIG
# ==============================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RAW_FILE = "data_raw/string/9606.protein.links.full.v12.0.txt"  # can also be .gz
OUT_DIR = Path(f"data_processed/ppi_mcl_{timestamp}")
RESULTS_DIR = Path(f"results_{timestamp}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Increase runtime: keep MORE edges by lowering this.
# combined_score is usually 0..1000
SCORE_MIN = 0          # set 0 for max edges (slowest)
MAX_EDGES = None       # e.g., 5_000_000 if you need a cap
SAVE_EDGES_CSV = True  # set False if file becomes too large

EDGES_CSV = OUT_DIR / f"ppi_edges_{timestamp}.csv"
MAP_JSON  = OUT_DIR / f"node_map_{timestamp}.json"
CSR_NPZ   = OUT_DIR / f"ppi_csr_{timestamp}.npz"
STATS_TXT = RESULTS_DIR / f"stats_{timestamp}.txt"


def detect_columns(sample_path: str) -> list[str]:
    # Read first non-comment line to determine header
    with open(sample_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    return header.split()


def main():
    t_total = time.perf_counter()

    print("Detecting STRING columns...")
    cols = detect_columns(RAW_FILE)
    print("Columns:", cols)

    # Determine which column is combined_score
    if "combined_score" not in cols:
        raise ValueError("combined_score column not found. Please check file header.")
    score_col = "combined_score"

    # Only keep needed columns
    usecols = ["protein1", "protein2", score_col]

    print("\nLoading STRING (only protein1, protein2, combined_score)...")
    t0 = time.perf_counter()
    df = pd.read_csv(
        RAW_FILE,
        sep=r"\s+",
        usecols=usecols,
        low_memory=False
    )
    print(f"Loaded rows: {len(df):,} | time={time.perf_counter()-t0:.2f}s")

    print("\nFiltering human-only (prefix 9606.) and SCORE_MIN...")
    t1 = time.perf_counter()

    # Enforce human prefix (file is human but safe)
    df = df[df["protein1"].str.startswith("9606.") & df["protein2"].str.startswith("9606.")]

    # Score filter
    df = df[df[score_col] >= SCORE_MIN]

    # Optional cap
    if MAX_EDGES is not None and len(df) > MAX_EDGES:
        df = df.iloc[:MAX_EDGES].copy()
        print(f"Applied MAX_EDGES={MAX_EDGES:,}")

    print(f"Rows after filters: {len(df):,} | time={time.perf_counter()-t1:.2f}s")

    print("\nBuilding undirected edge list...")
    t2 = time.perf_counter()

    # Make undirected by sorting endpoints
    s = df[["protein1", "protein2"]].min(axis=1)
    t = df[["protein1", "protein2"]].max(axis=1)
    w = df[score_col].to_numpy(dtype=np.float32)

    edges = pd.DataFrame({"source": s, "target": t, "combined_score": w})

    # Remove self loops
    edges = edges[edges["source"] != edges["target"]]

    # If duplicates exist, keep the max score
    edges = edges.groupby(["source", "target"], as_index=False)["combined_score"].max()

    print(f"Edges after undirected+dedup: {len(edges):,} | time={time.perf_counter()-t2:.2f}s")

    print("\nCreating node map...")
    t3 = time.perf_counter()
    nodes = pd.unique(edges[["source", "target"]].values.ravel())
    node_map = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    print(f"Unique nodes: {n:,} | time={time.perf_counter()-t3:.2f}s")

    if SAVE_EDGES_CSV:
        print("\nSaving edges.csv (can be large)...")
        t4 = time.perf_counter()
        edges.to_csv(EDGES_CSV, index=False)
        print(f"Saved {EDGES_CSV} | time={time.perf_counter()-t4:.2f}s")

    print("\nSaving node_map.json...")
    t5 = time.perf_counter()
    with open(MAP_JSON, "w") as f:
        json.dump(node_map, f)
    print(f"Saved {MAP_JSON} | time={time.perf_counter()-t5:.2f}s")

    print("\nBuilding weighted CSR adjacency...")
    t6 = time.perf_counter()

    # Map to integer ids
    i = edges["source"].map(node_map).to_numpy(dtype=np.int32)
    j = edges["target"].map(node_map).to_numpy(dtype=np.int32)

    # Normalize combined_score to [0,1]
    weight = (edges["combined_score"].to_numpy(dtype=np.float32)) / 1000.0

    # Undirected adjacency: add both directions
    row = np.concatenate([i, j])
    col = np.concatenate([j, i])
    data = np.concatenate([weight, weight]).astype(np.float32)

    A = sparse.csr_matrix((data, (row, col)), shape=(n, n))
    sparse.save_npz(CSR_NPZ, A)

    print(f"CSR saved: {CSR_NPZ} | nnz={A.nnz:,} | time={time.perf_counter()-t6:.2f}s")

    # --------------------------
    # ANALYSIS (save stats.txt)
    # --------------------------
    t7 = time.perf_counter()
    undirected_edges = A.nnz // 2
    avg_deg = float(A.nnz) / n
    density = (2.0 * undirected_edges) / (n * (n - 1)) if n > 1 else 0.0

    # Degree stats (approx from CSR indptr)
    degrees = np.diff(A.indptr)
    deg_max = int(degrees.max()) if degrees.size else 0
    deg_mean = float(degrees.mean()) if degrees.size else 0.0

    # Score stats
    score_vals = edges["combined_score"].to_numpy(dtype=np.float32)
    q = np.quantile(score_vals, [0.0, 0.25, 0.5, 0.75, 1.0])

    stats = (
        "STRING HUMAN PPI STATS (for MCL)\n"
        "-------------------------------\n"
        f"RAW_FILE: {RAW_FILE}\n"
        f"SCORE_MIN: {SCORE_MIN}\n"
        f"Nodes: {n:,}\n"
        f"Undirected edges: {undirected_edges:,}\n"
        f"CSR nnz (directed): {A.nnz:,}\n"
        f"Avg degree (CSR directed): {avg_deg:.2f}\n"
        f"Degree mean (row): {deg_mean:.2f}\n"
        f"Degree max (row): {deg_max}\n"
        f"Density: {density:.8f}\n"
        f"combined_score quantiles [min,25%,50%,75%,max]: {q}\n"
        f"Analysis time: {time.perf_counter()-t7:.2f}s\n"
    )

    with open(STATS_TXT, "w") as f:
        f.write(stats)

    print("\n" + stats)
    print(f"TOTAL preprocess time: {time.perf_counter()-t_total:.2f}s")


if __name__ == "__main__":
    main()
