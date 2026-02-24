#!/usr/bin/env python3

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

import cudf
import cugraph


# ==============================
# CONFIG
# ==============================
CSR_PATH = "data/processed/adj_csr.npz"
EDGES_PATH = "data/processed/edges.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():

    print("\n==============================")
    print("LOADING GRAPH")
    print("==============================")

    csr = sparse.load_npz(CSR_PATH)
    edges_df = pd.read_csv(EDGES_PATH)

    num_nodes = csr.shape[0]
    num_edges = csr.nnz // 2

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of undirected edges: {num_edges}")

    # ==============================
    # CONVERT TO GPU GRAPH
    # ==============================
    print("\nMoving graph to GPU...")

    coo = csr.tocoo()

    gdf = cudf.DataFrame({
        "src": coo.row,
        "dst": coo.col
    })

    G_gpu = cugraph.Graph()
    G_gpu.from_cudf_edgelist(gdf, source="src", destination="dst")

    print("GPU graph constructed.\n")

    # ==========================================================
    # ================== GPU PAGERANK ==========================
    # ==========================================================

    print("===================================")
    print("RUNNING GPU PAGERANK")
    print("===================================")

    start = time.time()
    pr_gpu = cugraph.pagerank(G_gpu, alpha=0.85)
    end = time.time()

    pr_time = end - start
    print(f"GPU PageRank Execution Time: {pr_time:.4f} seconds")

    pr_pd = pr_gpu.to_pandas()

    # Save full results
    pr_pd.to_csv(os.path.join(RESULTS_DIR, "pagerank_results.csv"), index=False)

    # Top 10
    top_10 = pr_pd.sort_values("pagerank", ascending=False).head(10)

    print("\nTop 10 Proteins by PageRank (node IDs):")
    print(top_10)

    # PageRank distribution
    plt.figure(figsize=(8,5))
    plt.hist(pr_pd["pagerank"], bins=50)
    plt.title("Distribution of PageRank Scores")
    plt.xlabel("PageRank Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pagerank_distribution.png"))
    plt.close()

    print("\nInterpretation:")
    print("- High PageRank proteins represent globally influential hubs.")
    print("- Distribution should show few dominant nodes (scale-free behavior).")

    # ==========================================================
    # ================== GPU LOUVAIN ===========================
    # ==========================================================

    print("\n===================================")
    print("RUNNING GPU LOUVAIN")
    print("===================================")

    start = time.time()
    parts_gpu, modularity = cugraph.louvain(G_gpu)
    end = time.time()

    louvain_time = end - start

    print(f"GPU Louvain Execution Time: {louvain_time:.4f} seconds")
    print(f"Modularity Score: {modularity:.4f}")

    parts_pd = parts_gpu.to_pandas()

    # Save community assignments
    parts_pd.to_csv(os.path.join(RESULTS_DIR, "louvain_results.csv"), index=False)

    # Community statistics
    community_sizes = parts_pd["partition"].value_counts()

    print("\nNumber of communities detected:", len(community_sizes))
    print("\nTop 10 Largest Communities:")
    print(community_sizes.head(10))

    # Plot community size distribution
    plt.figure(figsize=(8,5))
    plt.hist(community_sizes.values, bins=50)
    plt.title("Community Size Distribution")
    plt.xlabel("Community Size")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "community_distribution.png"))
    plt.close()

    print("\nInterpretation:")
    print("- Louvain groups proteins into functional modules.")
    print("- High modularity (>0.3) indicates strong modular structure.")
    print("- Large communities may represent protein complexes.")

    # ==========================================================
    # ================== SUMMARY ===============================
    # ==========================================================

    print("\n===================================")
    print("SUMMARY")
    print("===================================")

    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    print(f"GPU PageRank Time: {pr_time:.4f}s")
    print(f"GPU Louvain Time: {louvain_time:.4f}s")
    print(f"Modularity: {modularity:.4f}")

    print("\nResults saved in 'results/' folder.")
    print("Analysis complete.\n")


if __name__ == "__main__":
    main()
