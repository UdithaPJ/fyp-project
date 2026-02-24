import pandas as pd
import numpy as np
from scipy import sparse
import os
import json


def preprocess_string(input_path, output_dir, min_score=0):

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path, sep=" ")

    df = df[df["combined_score"] >= min_score]

    df = df[df["protein1"] != df["protein2"]]

    # Undirected dedup
    a = df["protein1"].values
    b = df["protein2"].values
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)

    df2 = pd.DataFrame({
        "u": lo,
        "v": hi,
        "weight": df["combined_score"] / 1000.0
    })

    df2 = df2.groupby(["u", "v"], as_index=False)["weight"].max()

    nodes = pd.Index(pd.concat([df2["u"], df2["v"]]).unique())
    node2id = {node: i for i, node in enumerate(nodes)}

    df2["src"] = df2["u"].map(node2id)
    df2["dst"] = df2["v"].map(node2id)

    edges = df2[["src", "dst"]].values
    reverse_edges = edges[:, ::-1]
    all_edges = np.vstack([edges, reverse_edges])

    rows = all_edges[:, 0]
    cols = all_edges[:, 1]
    weights = np.concatenate([df2["weight"], df2["weight"]])

    csr = sparse.csr_matrix(
        (weights, (rows, cols)),
        shape=(len(nodes), len(nodes))
    )

    sparse.save_npz(os.path.join(output_dir, "adj_csr.npz"), csr)
    df2.to_csv(os.path.join(output_dir, "edges.csv"), index=False)

    meta = {
        "nodes": len(nodes),
        "edges": len(df2),
        "min_score": min_score
    }

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Preprocessing complete.")
