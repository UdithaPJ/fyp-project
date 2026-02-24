import time, json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import sparse

RAW_FILE = "data_raw/omnipath/omnipathdb.org.tsv"

OUT_DIR = Path("data_processed/grn_bfs")
VAR_DIR = OUT_DIR / "variants"
RESULTS_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VAR_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

NODE_MAP_JSON = OUT_DIR / "node_map.json"
EDGES_CSV = OUT_DIR / "edges_clean.csv"
STATS_TXT = RESULTS_DIR / "stats.txt"

# ---- Controls ----
# If your file contains multiple species, you need a tax_id column to filter human.
# Your example doesn't show tax_id; so by default we assume the file is already human.
HUMAN_TAXID = 9606
TAXID_COLUMN = None   # set to "tax_id" if your TSV has it

MAX_EDGES = None      # set to an int if you want to cap extremely large datasets
# ------------------

EXPECTED_COLS = [
    "source", "target",
    "is_directed", "is_stimulation", "is_inhibition",
    "consensus_direction", "consensus_stimulation", "consensus_inhibition"
]

VARIANTS = {
    "grn_all": lambda df: df,  # everything
    "grn_directed": lambda df: df[df["is_directed"] == 1],
    "grn_stimulation": lambda df: df[df["is_stimulation"] == 1],
    "grn_inhibition": lambda df: df[df["is_inhibition"] == 1],
    "grn_consensus_direction": lambda df: df[df["consensus_direction"] == 1],
    "grn_consensus_stimulation": lambda df: df[df["consensus_stimulation"] == 1],
    "grn_consensus_inhibition": lambda df: df[df["consensus_inhibition"] == 1],
}

def safe_int_col(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int8)
    else:
        df[col] = np.int8(0)
    return df

def build_csr(edges_df, node_map):
    # Directed adjacency: source -> target
    i = edges_df["source"].map(node_map).to_numpy(dtype=np.int32)
    j = edges_df["target"].map(node_map).to_numpy(dtype=np.int32)

    data = np.ones_like(i, dtype=np.int8)
    n = len(node_map)
    A = sparse.csr_matrix((data, (i, j)), shape=(n, n))
    return A

def graph_stats(A: sparse.csr_matrix, name: str):
    n = A.shape[0]
    m = A.nnz
    out_deg = np.diff(A.indptr)
    avg_out = float(out_deg.mean()) if out_deg.size else 0.0
    max_out = int(out_deg.max()) if out_deg.size else 0
    density = (m / (n * (n - 1))) if n > 1 else 0.0

    # top hubs (row out-degree)
    top_idx = np.argsort(-out_deg)[:5] if n > 0 else []
    return {
        "name": name,
        "nodes": n,
        "edges": m,
        "avg_out": avg_out,
        "max_out": max_out,
        "density": density,
        "top_hubs": [(int(i), int(out_deg[i])) for i in top_idx]
    }

def main():
    t_total = time.perf_counter()

    print("Loading OmniPath TSV...")
    t0 = time.perf_counter()
    df = pd.read_csv(RAW_FILE, sep="\t", low_memory=False)
    print(f"Loaded rows: {len(df):,} | time={time.perf_counter()-t0:.2f}s")

    # Ensure required columns exist (create missing flags as 0)
    for c in EXPECTED_COLS:
        if c not in df.columns:
            if c in ("source", "target"):
                raise ValueError(f"Missing required column: {c}")
            df[c] = 0

    # Coerce flags to int8
    for col in ["is_directed","is_stimulation","is_inhibition",
                "consensus_direction","consensus_stimulation","consensus_inhibition"]:
        df = safe_int_col(df, col)

    # Optional human filter if tax id exists
    if TAXID_COLUMN and TAXID_COLUMN in df.columns:
        print(f"\nFiltering human only where {TAXID_COLUMN} == {HUMAN_TAXID}...")
        t1 = time.perf_counter()
        df = df[df[TAXID_COLUMN] == HUMAN_TAXID]
        print(f"After human filter: {len(df):,} | time={time.perf_counter()-t1:.2f}s")
    else:
        print("\nNo tax_id column provided/found; assuming dataset is already human-only.")

    # Clean source/target strings
    print("\nCleaning source/target, dropping invalid rows, removing self-loops, deduplicating...")
    t2 = time.perf_counter()
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df = df[(df["source"] != "") & (df["target"] != "")]
    df = df[df["source"] != df["target"]]

    if MAX_EDGES is not None and len(df) > MAX_EDGES:
        df = df.iloc[:MAX_EDGES].copy()
        print(f"Applied MAX_EDGES={MAX_EDGES:,}")

    # Keep one row per (source,target) by OR-ing flags (max)
    agg_cols = ["is_directed","is_stimulation","is_inhibition",
                "consensus_direction","consensus_stimulation","consensus_inhibition"]
    df = df.groupby(["source","target"], as_index=False)[agg_cols].max()

    print(f"Edges after clean: {len(df):,} | time={time.perf_counter()-t2:.2f}s")

    # Node map (all possible nodes)
    print("\nBuilding node map (all unique nodes)...")
    t3 = time.perf_counter()
    nodes = pd.unique(df[["source","target"]].values.ravel())
    node_map = {n:i for i,n in enumerate(nodes)}
    print(f"Unique nodes: {len(node_map):,} | time={time.perf_counter()-t3:.2f}s")

    print("\nSaving node_map.json + edges_clean.csv ...")
    t4 = time.perf_counter()
    with open(NODE_MAP_JSON, "w") as f:
        json.dump(node_map, f)
    df.to_csv(EDGES_CSV, index=False)
    print(f"Saved: {NODE_MAP_JSON} and {EDGES_CSV} | time={time.perf_counter()-t4:.2f}s")

    # Build & save CSR variants + stats
    stats_lines = []
    stats_lines.append("OMNIPATH HUMAN GRN STATS (for MBFS)\n----------------------------------\n")
    stats_lines.append(f"RAW_FILE: {RAW_FILE}\n")
    stats_lines.append(f"TOTAL CLEAN EDGES: {len(df):,}\n")
    stats_lines.append(f"TOTAL NODES: {len(node_map):,}\n\n")

    print("\nBuilding CSR variants...")
    for name, fn in VARIANTS.items():
        tv = time.perf_counter()
        sub = fn(df)

        A = build_csr(sub, node_map)
        out_path = VAR_DIR / f"{name}.npz"
        sparse.save_npz(out_path, A)

        st = graph_stats(A, name)
        stats_lines.append(f"[{name}]\n")
        stats_lines.append(f"  nodes={st['nodes']:,} edges={st['edges']:,}\n")
        stats_lines.append(f"  avg_out={st['avg_out']:.2f} max_out={st['max_out']} density={st['density']:.8f}\n")
        stats_lines.append(f"  top_hubs(node_id,out_deg)={st['top_hubs']}\n")
        stats_lines.append(f"  build_time={time.perf_counter()-tv:.2f}s\n\n")

        print(f"  {name}: nnz={A.nnz:,} saved={out_path.name} time={time.perf_counter()-tv:.2f}s")

    with open(STATS_TXT, "w") as f:
        f.write("".join(stats_lines))

    print(f"\nSaved stats: {STATS_TXT}")
    print(f"TOTAL preprocess time: {time.perf_counter()-t_total:.2f}s")

if __name__ == "__main__":
    main()
