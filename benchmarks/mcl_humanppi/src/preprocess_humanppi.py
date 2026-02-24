import argparse
import os
import pandas as pd
import numpy as np
from scipy import sparse

def preprocess(input_tsv: str, out_csr: str, out_nodes: str,
               weight_col: str = "RFprob", min_weight: float = 0.0, max_edges: int = 0):
    # IMPORTANT FIX:
    # comment="#" skips the metadata lines starting with "# ..."
    # engine="python" is more tolerant for weird TSV formatting on Windows
    df = pd.read_csv(
        input_tsv,
        sep="\t",
        dtype=str,
        comment="#",
        engine="python"
    )

    # Validate columns
    required = {"Protein1", "Protein2", weight_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")

    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)

    df = df[["Protein1", "Protein2", weight_col]].rename(columns={weight_col: "w"})
    df = df[(df["Protein1"].notna()) & (df["Protein2"].notna())]
    df = df[(df["Protein1"] != "none") & (df["Protein2"] != "none")]
    df = df[df["Protein1"].str.len() > 0]
    df = df[df["Protein2"].str.len() > 0]

    df = df[df["w"] >= min_weight]

    if max_edges and max_edges > 0 and len(df) > max_edges:
        df = df.sort_values("w", ascending=False).head(max_edges)

    # Map proteins to IDs
    nodes = pd.Index(pd.unique(df[["Protein1", "Protein2"]].values.ravel("K")))
    node_to_id = {p: i for i, p in enumerate(nodes)}
    n = len(nodes)

    src = df["Protein1"].map(node_to_id).to_numpy()
    dst = df["Protein2"].map(node_to_id).to_numpy()
    w = df["w"].to_numpy(dtype=np.float32)

    # Symmetric (undirected stored as both directions)
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    data = np.concatenate([w, w])

    A = sparse.coo_matrix((data, (row, col)), shape=(n, n), dtype=np.float32).tocsr()
    A.sum_duplicates()

    os.makedirs(os.path.dirname(out_csr), exist_ok=True)
    os.makedirs(os.path.dirname(out_nodes), exist_ok=True)

    sparse.save_npz(out_csr, A)
    pd.DataFrame({"id": np.arange(n, dtype=np.int32), "protein": nodes.values}).to_csv(out_nodes, sep="\t", index=False)

    degrees = np.diff(A.indptr)
    print("=== Preprocess Done ===")
    print(f"Input: {input_tsv}")
    print(f"Nodes: {n:,}")
    print(f"NNZ (directed entries): {A.nnz:,}")
    print(f"Estimated undirected edges: ~{A.nnz // 2:,}")
    print(f"Degree: min={degrees.min()}, mean={degrees.mean():.2f}, max={degrees.max()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_csr", required=True)
    ap.add_argument("--out_nodes", required=True)
    ap.add_argument("--weight", default="RFprob", choices=["RFprob", "AFprob", "AFprob5", "AFMprob"])
    ap.add_argument("--min_weight", type=float, default=0.0)
    ap.add_argument("--max_edges", type=int, default=0)
    args = ap.parse_args()

    preprocess(args.input, args.out_csr, args.out_nodes, args.weight, args.min_weight, args.max_edges)

if __name__ == "__main__":
    main()