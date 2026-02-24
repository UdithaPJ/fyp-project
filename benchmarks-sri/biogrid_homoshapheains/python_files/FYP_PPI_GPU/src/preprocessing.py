import pandas as pd
import numpy as np
from scipy import sparse
import json
import os


def preprocess_biogrid(input_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path, sep="\t", low_memory=False)

    # Filter physical interactions
    df = df[df["Experimental System Type"] == "physical"].copy()

    ppi = df[[
        "Official Symbol Interactor A",
        "Official Symbol Interactor B"
    ]].copy()

    # Remove self-loops
    ppi = ppi[
        ppi["Official Symbol Interactor A"] !=
        ppi["Official Symbol Interactor B"]
    ]

    # Remove duplicate undirected edges
    ppi["sorted_pair"] = ppi.apply(
        lambda row: tuple(sorted([
            row["Official Symbol Interactor A"],
            row["Official Symbol Interactor B"]
        ])),
        axis=1
    )

    ppi = ppi.drop_duplicates(subset="sorted_pair")
    ppi = ppi[[
        "Official Symbol Interactor A",
        "Official Symbol Interactor B"
    ]]

    # Map nodes to integers
    unique_nodes = pd.unique(
        ppi.values.ravel()
    )

    node2id = {node: i for i, node in enumerate(unique_nodes)}

    ppi["src"] = ppi.iloc[:, 0].map(node2id)
    ppi["dst"] = ppi.iloc[:, 1].map(node2id)

    # Save edge list
    ppi[["src", "dst"]].to_csv(
        os.path.join(output_dir, "edges.csv"),
        index=False
    )

    # Build symmetric CSR
    edges = ppi[["src", "dst"]].values
    reverse_edges = edges[:, ::-1]
    all_edges = np.vstack([edges, reverse_edges])

    rows = all_edges[:, 0]
    cols = all_edges[:, 1]
    data = np.ones(len(rows))

    csr = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(unique_nodes), len(unique_nodes))
    )

    sparse.save_npz(
        os.path.join(output_dir, "adj_csr.npz"),
        csr
    )

    meta = {
        "num_nodes": int(csr.shape[0]),
        "num_edges": int(len(ppi))
    }

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Preprocessing complete.")
