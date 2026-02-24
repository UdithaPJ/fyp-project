import time, json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import sparse

RAW_FILE = "data_raw/trrust/trrust_rawdata.human.tsv"

OUT_DIR = Path("data_processed/grn_bfs")
RESULTS_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

EDGES_CSV = OUT_DIR / "grn_edges.csv"
MAP_JSON  = OUT_DIR / "node_map.json"
CSR_NPZ   = OUT_DIR / "grn_csr.npz"
STATS_TXT = RESULTS_DIR / "stats.txt"

def main():
    t_total = time.perf_counter()

    print("Loading TRRUST...")
    t0 = time.perf_counter()

    # TRRUST is tab-separated, usually no header
    # Columns: TF, Target, Mode, PMID
    df = pd.read_csv(RAW_FILE, sep="\t", header=None, names=["TF","Target","Mode","PMID"], low_memory=False)
    print(f"Loaded rows: {len(df):,} | time={time.perf_counter()-t0:.2f}s")

    print("\nCleaning + extracting directed edges (TF -> Target)...")
    t1 = time.perf_counter()

    edges = df[["TF","Target"]].dropna()
    edges["TF"] = edges["TF"].astype(str).str.strip()
    edges["Target"] = edges["Target"].astype(str).str.strip()

    # Remove empty strings
    edges = edges[(edges["TF"] != "") & (edges["Target"] != "")]
    # Remove self loops
    edges = edges[edges["TF"] != edges["Target"]]
    # Remove duplicates
    edges = edges.drop_duplicates()

    edges.columns = ["source","target"]

    print(f"Edges after clean: {len(edges):,} | time={time.perf_counter()-t1:.2f}s")

    print("\nCreating node map...")
    t2 = time.perf_counter()
    nodes = pd.unique(edges[["source","target"]].values.ravel())
    node_map = {n:i for i,n in enumerate(nodes)}
    n = len(nodes)
    print(f"Unique nodes: {n:,} | time={time.perf_counter()-t2:.2f}s")

    print("\nSaving edges + node_map...")
    t3 = time.perf_counter()
    edges.to_csv(EDGES_CSV, index=False)
    with open(MAP_JSON, "w") as f:
        json.dump(node_map, f)
    print(f"Saved: {EDGES_CSV}, {MAP_JSON} | time={time.perf_counter()-t3:.2f}s")

    print("\nBuilding directed CSR adjacency...")
    t4 = time.perf_counter()
    i = edges["source"].map(node_map).to_numpy(dtype=np.int32)
    j = edges["target"].map(node_map).to_numpy(dtype=np.int32)
    data = np.ones_like(i, dtype=np.int8)

    A = sparse.csr_matrix((data, (i, j)), shape=(n, n))
    sparse.save_npz(CSR_NPZ, A)
    print(f"CSR saved: {CSR_NPZ} | nnz={A.nnz:,} | time={time.perf_counter()-t4:.2f}s")

    # Graph stats
    out_deg = np.diff(A.indptr)
    out_deg_mean = float(out_deg.mean()) if out_deg.size else 0.0
    out_deg_max = int(out_deg.max()) if out_deg.size else 0

    stats = (
        "TRRUST HUMAN GRN STATS (for BFS)\n"
        "-------------------------------\n"
        f"Nodes: {n:,}\n"
        f"Edges (directed): {A.nnz:,}\n"
        f"Avg out-degree: {out_deg_mean:.2f}\n"
        f"Max out-degree: {out_deg_max}\n"
    )

    with open(STATS_TXT, "w") as f:
        f.write(stats)

    print("\n" + stats)
    print(f"TOTAL preprocess time: {time.perf_counter()-t_total:.2f}s")

if __name__ == "__main__":
    main()
