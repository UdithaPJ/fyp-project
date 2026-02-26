import pandas as pd
import os


def benchmark_all(csr,
                  cpu_pr_func,
                  gpu_pr_func,
                  cpu_louvain_func,
                  gpu_louvain_func,
                  thread_list=[1, 2, 4, 8]):

    cpu_results = []

    for threads in thread_list:
        os.environ["OMP_NUM_THREADS"] = str(threads)

        pr_cpu, pr_time = cpu_pr_func(csr)
        part_cpu, mod_cpu, louvain_time = cpu_louvain_func(csr)

        cpu_results.append({
            "Threads": threads,
            "PageRank_Time (s)": round(pr_time, 4),
            "Louvain_Time (s)": round(louvain_time, 4)
        })

    cpu_df = pd.DataFrame(cpu_results)

    # Save detailed thread results separately
    cpu_df.to_csv("results/cpu_thread_scaling_raw.csv", index=False)

    # Find best CPU configuration
    best_pr_row = cpu_df.loc[cpu_df["PageRank_Time (s)"].idxmin()]
    best_louvain_row = cpu_df.loc[cpu_df["Louvain_Time (s)"].idxmin()]

    best_pr_threads = int(best_pr_row["Threads"])
    best_pr_time = best_pr_row["PageRank_Time (s)"]

    best_louvain_threads = int(best_louvain_row["Threads"])
    best_louvain_time = best_louvain_row["Louvain_Time (s)"]

    # GPU
    pr_gpu, pr_gpu_time = gpu_pr_func(csr)
    part_gpu, mod_gpu, louvain_gpu_time = gpu_louvain_func(csr)

    # Clean summary table
    comparison_df = pd.DataFrame([
        {
            "Algorithm": "PageRank",
            "CPU Best Threads": best_pr_threads,
            "CPU Time (s)": round(best_pr_time, 4),
            "GPU Time (s)": round(pr_gpu_time, 4),
            "GPU Speedup vs Best CPU": round(best_pr_time / pr_gpu_time, 2)
        },
        {
            "Algorithm": "Louvain",
            "CPU Best Threads": best_louvain_threads,
            "CPU Time (s)": round(best_louvain_time, 4),
            "GPU Time (s)": round(louvain_gpu_time, 4),
            "GPU Speedup vs Best CPU": round(best_louvain_time / louvain_gpu_time, 2)
        }
    ])

    return comparison_df, pr_gpu, part_gpu