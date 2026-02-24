import cudf
import cugraph
import time
import pandas as pd
import os
from scipy import sparse


def run_gpu_analysis(csr_path, results_dir):

    os.makedirs(results_dir, exist_ok=True)

    csr = sparse.load_npz(csr_path)
    coo = csr.tocoo()

    gdf = cudf.DataFrame({
        "src": coo.row,
        "dst": coo.col,
        "weight": coo.data
    })

    G = cugraph.Graph()
    G.from_cudf_edgelist(gdf, source="src", destination="dst", edge_attr="weight")

    # ------------------------------
    # PageRank Parameter Sweep
    # ------------------------------
    alphas = [0.70, 0.85, 0.95]
    tolerances = [1e-4, 1e-6]

    pr_results = []

    for alpha in alphas:
        for tol in tolerances:
            start = time.time()
            # pr = cugraph.pagerank(G, alpha=alpha, tol=tol)

            pr = cugraph.pagerank(
                    G,
                    alpha=alpha,
                    tol=tol,
                    max_iter=1000
                )

            end = time.time()

            pr_results.append({
                "alpha": alpha,
                "tol": tol,
                "runtime": end - start
            })

    pd.DataFrame(pr_results).to_csv(
        os.path.join(results_dir, "pagerank_gpu_benchmark.csv"),
        index=False
    )

    # ------------------------------
    # Louvain
    # ------------------------------
    start = time.time()
    parts, modularity = cugraph.louvain(G)
    end = time.time()

    with open(os.path.join(results_dir, "louvain_gpu.txt"), "w") as f:
        f.write(f"Runtime: {end-start}\n")
        f.write(f"Modularity: {modularity}\n")

    print("GPU analysis complete.")
