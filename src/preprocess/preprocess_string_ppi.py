#!/usr/bin/env python3
import argparse
import os
import json
import time
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import sparse


def read_string_links(path: str) -> pd.DataFrame:
    """
    Reads STRING protein.links-like file:
    columns: protein1 protein2 combined_score
    whitespace-separated, header present.
    """
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    expected = {"protein1", "protein2", "combined_score"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Expected columns {expected}, got {set(df.columns)}")

    # Clean
    df["protein1"] = df["protein1"].astype(str).str.strip()
    df["protein2"] = df["protein2"].astype(str).str.strip()
    df["combined_score"] = pd.to_numeric(df["combined_score"], errors="coerce")

    df = df.dropna(subset=["protein1", "protein2", "combined_score"])
    df = df[df["protein1"] != ""]
    df = df[df["protein2"] != ""]
    df = df[df["protein1"] != df["protein2"]]  # remove self-loops

    # Ensure score is within expected range
    df = df[(df["combined_score"] >= 0) & (df["combined_score"] <= 1000)]
    df["combined_score"] = df["combined_score"].astype(int)

    return df


def make_undirected_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    For PPI, treat as undirected. Deduplicate edges by sorting endpoints.
    Keep maximum score if duplicates exist.
    """
    a = df["protein1"].to_numpy()
    b = df["protein2"].to_numpy()
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    df2 = pd.DataFrame(
        {
            "u": lo,
            "v": hi,
            "combined_score": df["combined_score"].to_numpy(),
        }
    )
    # If duplicates exist, keep max score
    df2 = df2.groupby(["u", "v"], as_index=False)["combined_score"].max()
    return df2


def build_node_mapping(edges_uv: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    nodes = pd.Index(pd.concat([edges_uv["u"], edges_uv["v"]], ignore_index=True).unique())
    nodes = nodes.sort_values()

    node2id = {node: i for i, node in enumerate(nodes.tolist())}
    nodes_df = pd.DataFrame({"node_str": nodes, "node_id": np.arange(len(nodes), dtype=np.int64)})
    return nodes_df, node2id


def build_sparse_matrices(edges_uv: pd.DataFrame, node2id: dict, make_symmetric: bool = True):
    """
    Build COO and CSR adjacency matrices.
    Weight = combined_score / 1000.0 (float32)
    """
    ui = edges_uv["u"].map(node2id).to_numpy(dtype=np.int64)
    vi = edges_uv["v"].map(node2id).to_numpy(dtype=np.int64)
    w = (edges_uv["combined_score"].to_numpy(dtype=np.float32) / 1000.0)

    n = len(node2id)

    if make_symmetric:
        # Add both directions
        rows = np.concatenate([ui, vi])
        cols = np.concatenate([vi, ui])
        data = np.concatenate([w, w]).astype(np.float32, copy=False)
    else:
        rows, cols, data = ui, vi, w.astype(np.float32, copy=False)

    coo = sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    csr = coo.tocsr()
    csr.sum_duplicates()
    return coo, csr


def main():
    ap = argparse.ArgumentParser(description="Preprocess STRING protein.links PPI into cleaned edges + CSR/COO.")
    ap.add_argument("--input", required=True, help="Path to protein.links file (e.g., protein.links.v12.0.txt)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--min-score", type=int, default=0, help="Filter edges with combined_score < min-score (0..1000)")
    ap.add_argument("--no-symmetric", action="store_true", help="Do NOT make adjacency symmetric")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.time()
    df = read_string_links(args.input)
    if args.min_score > 0:
        df = df[df["combined_score"] >= args.min_score]

    # Undirected dedup for PPI edge list (u, v)
    edges = make_undirected_dedup(df)

    # Node mapping
    nodes_df, node2id = build_node_mapping(edges)

    # Sparse matrices
    coo, csr = build_sparse_matrices(edges, node2id, make_symmetric=(not args.no_symmetric))

    # Save outputs
    edges_path = os.path.join(args.outdir, "edges_clean.csv")
    nodes_path = os.path.join(args.outdir, "nodes.csv")
    coo_path = os.path.join(args.outdir, "adj_coo.npz")
    csr_path = os.path.join(args.outdir, "adj_csr.npz")
    meta_path = os.path.join(args.outdir, "meta.json")

    # Add integer columns too (useful later)
    edges_int = edges.copy()
    edges_int["src"] = edges_int["u"].map(node2id).astype(np.int64)
    edges_int["dst"] = edges_int["v"].map(node2id).astype(np.int64)
    edges_int["weight"] = (edges_int["combined_score"].astype(np.float32) / 1000.0)

    edges_int.to_csv(edges_path, index=False)
    nodes_df.to_csv(nodes_path, index=False)
    sparse.save_npz(coo_path, coo)
    sparse.save_npz(csr_path, csr)

    meta = {
        "input": os.path.abspath(args.input),
        "num_nodes": int(csr.shape[0]),
        "num_edges_undirected_unique": int(len(edges)),
        "num_edges_in_adjacency": int(csr.nnz),
        "min_score": int(args.min_score),
        "symmetric": bool(not args.no_symmetric),
        "columns_raw": ["protein1", "protein2", "combined_score"],
        "columns_clean": ["u", "v", "combined_score", "src", "dst", "weight"],
        "weight_definition": "combined_score / 1000.0",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    t1 = time.time()
    print("[OK] Wrote:")
    print(" -", edges_path)
    print(" -", nodes_path)
    print(" -", coo_path)
    print(" -", csr_path)
    print(" -", meta_path)
    print(f"[DONE] Time: {t1 - t0:.2f}s | Nodes={meta['num_nodes']} | UndirectedEdges={meta['num_edges_undirected_unique']} | CSR.nnz={meta['num_edges_in_adjacency']}")


if __name__ == "__main__":
    main()
