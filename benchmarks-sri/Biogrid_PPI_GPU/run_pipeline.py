import os
import pandas as pd
import matplotlib.pyplot as plt

from src.graph_loader import load_csr
from src.cpu_pagerank import run_cpu_pagerank
from src.cpu_louvain import run_cpu_louvain
from src.pagerank_gpu import run_gpu_pagerank
from src.louvain_gpu import run_gpu_louvain
from src.benchmark import benchmark_all


CSR_PATH = "data/processed/adj_csr.npz"
RESULTS_DIR = "results"


def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    csr = load_csr(CSR_PATH)

    mapping_df = pd.read_csv("data/processed/id_to_protein.csv")

    comparison_df, pr_gpu, part_gpu = benchmark_all(
        csr,
        run_cpu_pagerank,
        run_gpu_pagerank,
        run_cpu_louvain,
        run_gpu_louvain,
        thread_list=[1, 2, 4, 8]
    )

    print("\n=== Performance Comparison ===")
    print(comparison_df)
    print("\n=== CPU Thread Scaling Details ===")
    cpu_thread_df = pd.read_csv("results/cpu_thread_scaling_raw.csv")
    print(cpu_thread_df)

    comparison_df.to_csv(
        os.path.join(RESULTS_DIR, "performance_comparison.csv"),
        index=False
    )

    # ===========================
    # PAGE RANK SECTION
    # ===========================

    pr_gpu_sorted = pr_gpu.sort_values("pagerank", ascending=False)

    pr_gpu_sorted_named = pr_gpu_sorted.merge(
        mapping_df,
        left_on="vertex",
        right_on="Node_ID",
        how="left"
    )

    pr_gpu_sorted_named.head(20).to_csv(
        os.path.join(RESULTS_DIR, "gpu_pagerank_top20_named.csv"),
        index=False
    )

    print("\n=== Top 20 GPU PageRank Proteins ===")
    print(pr_gpu_sorted_named.head(20)[["Protein", "pagerank"]])

    # PageRank distribution plot
    plt.figure()
    plt.hist(pr_gpu_sorted["pagerank"], bins=50)
    plt.title("Distribution of PageRank Scores (GPU)")
    plt.xlabel("PageRank Score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(RESULTS_DIR, "pagerank_distribution.png"))
    plt.close()

    # ===========================
    # LOUVAIN SECTION
    # ===========================

    # Merge community assignments with protein names
    part_gpu_named = part_gpu.merge(
        mapping_df,
        left_on="vertex",
        right_on="Node_ID",
        how="left"
    )

    # Clean columns
    part_gpu_named = part_gpu_named[["Protein", "partition"]]
    part_gpu_named.columns = ["Protein", "Community"]

    # Save named communities
    part_gpu_named.to_csv(
        os.path.join(RESULTS_DIR, "gpu_louvain_communities_named.csv"),
        index=False
    )

    # Compute community sizes
    gpu_community_sizes = part_gpu_named["Community"].value_counts()
    gpu_community_sizes = gpu_community_sizes.reset_index()
    gpu_community_sizes.columns = ["Community", "Size"]

    gpu_community_sizes.to_csv(
        os.path.join(RESULTS_DIR, "gpu_louvain_community_sizes.csv"),
        index=False
    )

    num_communities = len(gpu_community_sizes)
    largest_community = gpu_community_sizes["Size"].max()
    avg_community_size = gpu_community_sizes["Size"].mean()

    print("\n=== Louvain Summary (GPU) ===")
    print(f"Number of Communities: {num_communities}")
    print(f"Largest Community Size: {largest_community}")
    print(f"Average Community Size: {avg_community_size:.2f}")

    gpu_community_sizes = part_gpu["partition"].value_counts()
    gpu_community_sizes = gpu_community_sizes.reset_index()
    gpu_community_sizes.columns = ["Community", "Size"]

    gpu_community_sizes.to_csv(
        os.path.join(RESULTS_DIR, "gpu_louvain_community_sizes.csv"),
        index=False
    )

    num_communities = len(gpu_community_sizes)
    largest_community = gpu_community_sizes["Size"].max()
    avg_community_size = gpu_community_sizes["Size"].mean()

    print("\n=== Louvain Summary (GPU) ===")
    print(f"Number of Communities: {num_communities}")
    print(f"Largest Community Size: {largest_community}")
    print(f"Average Community Size: {avg_community_size:.2f}")

    # Community size distribution plot
    plt.figure()
    plt.hist(gpu_community_sizes["Size"], bins=50)
    plt.title("Louvain Community Size Distribution")
    plt.xlabel("Community Size")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(RESULTS_DIR, "community_size_distribution.png"))
    plt.close()

    # ===========================
    # PERFORMANCE PLOTS
    # ===========================

    # PageRank
    plt.figure()
    plt.bar(["Best CPU", "GPU"],
            [comparison_df.loc[0, "CPU Time (s)"],
            comparison_df.loc[0, "GPU Time (s)"]])
    plt.title("PageRank: Best CPU vs GPU Execution Time")
    plt.ylabel("Time (seconds)")
    plt.savefig(os.path.join(RESULTS_DIR, "pagerank_performance.png"))
    plt.close()

    # Louvain
    plt.figure()
    plt.bar(["Best CPU", "GPU"],
            [comparison_df.loc[1, "CPU Time (s)"],
            comparison_df.loc[1, "GPU Time (s)"]])
    plt.title("Louvain: Best CPU vs GPU Execution Time")
    plt.ylabel("Time (seconds)")
    plt.savefig(os.path.join(RESULTS_DIR, "louvain_performance.png"))
    plt.close()

    # ===========================
    # FINAL SUMMARY
    # ===========================

    num_nodes = csr.shape[0]
    num_edges = csr.nnz // 2

    summary = f"""
    ===== NETWORK SUMMARY =====
    Total Proteins (Nodes): {num_nodes}
    Total Interactions (Edges): {num_edges}

    ===== LOUVAIN SUMMARY =====
    Number of Communities: {num_communities}
    Largest Community Size: {largest_community}
    Average Community Size: {avg_community_size:.2f}

    ===== PERFORMANCE =====
    PageRank Speedup (vs Best CPU): {comparison_df.loc[0, "GPU Speedup vs Best CPU"]}x
    Louvain Speedup (vs Best CPU): {comparison_df.loc[1, "GPU Speedup vs Best CPU"]}x
    """

    print(summary)

    with open(os.path.join(RESULTS_DIR, "analysis_summary.txt"), "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()