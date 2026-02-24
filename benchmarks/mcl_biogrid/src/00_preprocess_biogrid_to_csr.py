import time, json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import sparse

# ==============================
# CONFIG (edit these)
# ==============================
RAW_FILE = "data_raw/biogrid/BIOGRID-ALL.tab3.txt"

# Set to None to include ALL organisms (max nodes/edges)
# Set to 9606 for human-only
TAXID_FILTER = 9606   # Human only

# Optional: cap edges to avoid blowing RAM (set None for full)
MAX_EDGES = None  # e.g., 5_000_000

OUT_DIR = Path("data_processed/ppi_mcl")
RESULTS_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

EDGES_CSV = OUT_DIR / "ppi_edges.csv"
MAP_JSON  = OUT_DIR / "node_map.json"
CSR_NPZ   = OUT_DIR / "ppi_csr.npz"
STATS_TXT = RESULTS_DIR / "stats.txt"


def main():
    t_total = time.perf_counter()

    # Read only required columns to reduce RAM
    usecols = [
        "Organism ID Interactor A",
        "Organism ID Interactor B",
        "Official Symbol Interactor A",
        "Official Symbol Interactor B",
    ]

    print("Loading BioGRID (only needed columns)...")
    t0 = time.perf_counter()
    df = pd.read_csv(RAW_FILE, sep="\t", usecols=usecols, low_memory=False)
    print(f"Loaded rows: {len(df):,} | time={time.perf_counter()-t0:.2f}s")

    # Filter taxid if needed
    if TAXID_FILTER is not None:
        print(f"\nFiltering organism taxid={TAXID_FILTER}...")
        t1 = time.perf_counter()
        df = df[
            (df["Organism ID Interactor A"] == TAXID_FILTER) &
            (df["Organism ID Interactor B"] == TAXID_FILTER)
        ]
        print(f"Rows after filter: {len(df):,} | time={time.perf_counter()-t1:.2f}s")
    else:
        print("\nNo taxid filter: using ALL organisms (max nodes/edges).")

    print("\nExtracting undirected edges...")
    t2 = time.perf_counter()

    edges = df[["Official Symbol Interactor A", "Official Symbol Interactor B"]].dropna()
    edges.columns = ["A", "B"]

    # Remove self-loops
    edges = edges[edges["A"] != edges["B"]]

    # Undirected: sort endpoints
    src = edges[["A", "B"]].min(axis=1)
    dst = edges[["A", "B"]].max(axis=1)
    edges_u = pd.DataFrame({"source": src, "target": dst})

    # Optional edge cap for huge ALL-organism runs
    if MAX_EDGES is not None and len(edges_u) > MAX_EDGES:
        edges_u = edges_u.iloc[:MAX_EDGES].copy()
        print(f"Edge cap applied: MAX_EDGES={MAX_EDGES:,}")

    # Drop duplicates
    edges_u = edges_u.drop_duplicates()

    print(f"Edges after cleaning: {len(edges_u):,} | time={time.perf_counter()-t2:.2f}s")

    print("\nBuilding node map...")
    t3 = time.perf_counter()
    nodes = pd.unique(edges_u[["source", "target"]].values.ravel())
    node_map = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    print(f"Unique nodes: {n:,} | time={time.perf_counter()-t3:.2f}s")

    print("\nSaving edges + node_map...")
    t4 = time.perf_counter()
    edges_u.to_csv(EDGES_CSV, index=False)
    with open(MAP_JSON, "w") as f:
        json.dump(node_map, f)
    print(f"Saved: {EDGES_CSV} and {MAP_JSON} | time={time.perf_counter()-t4:.2f}s")

    print("\nBuilding CSR adjacency (unweighted, undirected)...")
    t5 = time.perf_counter()

    i = edges_u["source"].map(node_map).to_numpy(dtype=np.int32)
    j = edges_u["target"].map(node_map).to_numpy(dtype=np.int32)

    # undirected adjacency => add both directions
    row = np.concatenate([i, j])
    col = np.concatenate([j, i])
    data = np.ones_like(row, dtype=np.float32)

    A = sparse.csr_matrix((data, (row, col)), shape=(n, n))
    sparse.save_npz(CSR_NPZ, A)

    print(f"CSR saved: {CSR_NPZ} | nnz={A.nnz:,} | time={time.perf_counter()-t5:.2f}s")

    # Basic stats (for results/stats.txt)
    num_edges_undirected = A.nnz // 2
    avg_deg = float(A.nnz) / n
    density = (2.0 * num_edges_undirected) / (n * (n - 1)) if n > 1 else 0.0

    stats = (
        "BIOGRID PPI STATS (for MCL)\n"
        "--------------------------\n"
        f"TAXID_FILTER: {TAXID_FILTER}\n"
        f"Nodes: {n:,}\n"
        f"Undirected edges: {num_edges_undirected:,}\n"
        f"CSR nnz (directed): {A.nnz:,}\n"
        f"Average degree (directed): {avg_deg:.2f}\n"
        f"Density: {density:.6f}\n"
    )

    with open(STATS_TXT, "w") as f:
        f.write(stats)

    print("\n" + stats)
    print(f"TOTAL preprocess time: {time.perf_counter()-t_total:.2f}s")


if __name__ == "__main__":
    main()
