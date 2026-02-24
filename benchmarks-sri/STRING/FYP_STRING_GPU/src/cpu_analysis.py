import networkx as nx
import time
import pandas as pd
import os
import community as community_louvain
from scipy import sparse


def run_cpu_analysis(csr_path, results_dir):

    os.makedirs(results_dir, exist_ok=True)

    csr = sparse.load_npz(csr_path)
    G = nx.from_scipy_sparse_array(csr)

    # ------------------------------
    # PageRank Parameter Sweep
    # ------------------------------
    alphas = [0.70, 0.85, 0.95]
    tolerances = [1e-4, 1e-6]

    pr_results = []

    for alpha in alphas:
        for tol in tolerances:
            start = time.time()
            pr = nx.pagerank(G, alpha=alpha, tol=tol)
            end = time.time()

            pr_results.append({
                "alpha": alpha,
                "tol": tol,
                "runtime": end - start
            })

    pd.DataFrame(pr_results).to_csv(
        os.path.join(results_dir, "pagerank_cpu_benchmark.csv"),
        index=False
    )

    # ------------------------------
    # Louvain
    # ------------------------------
    start = time.time()
    partition = community_louvain.best_partition(G)
    modularity = community_louvain.modularity(partition, G)
    end = time.time()

    with open(os.path.join(results_dir, "louvain_cpu.txt"), "w") as f:
        f.write(f"Runtime: {end-start}\n")
        f.write(f"Modularity: {modularity}\n")

    print("CPU analysis complete.")
