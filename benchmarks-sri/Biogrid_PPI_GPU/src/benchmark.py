import pandas as pd


def benchmark_all(csr,
                  cpu_pr_func,
                  gpu_pr_func,
                  cpu_louvain_func,
                  gpu_louvain_func):

    pr_cpu, pr_cpu_time = cpu_pr_func(csr)
    pr_gpu, pr_gpu_time = gpu_pr_func(csr)

    part_cpu, mod_cpu, louvain_cpu_time = cpu_louvain_func(csr)
    part_gpu, mod_gpu, louvain_gpu_time = gpu_louvain_func(csr)

    pr_speedup = pr_cpu_time / pr_gpu_time
    louvain_speedup = louvain_cpu_time / louvain_gpu_time

    pr_improvement = (1 - pr_gpu_time / pr_cpu_time) * 100
    louvain_improvement = (1 - louvain_gpu_time / louvain_cpu_time) * 100

    comparison_df = pd.DataFrame([
        {
            "Algorithm": "PageRank",
            "CPU Time (s)": round(pr_cpu_time, 4),
            "GPU Time (s)": round(pr_gpu_time, 4),
            "Speedup (x)": round(pr_speedup, 2),
            "Improvement (%)": round(pr_improvement, 2),
            "CPU Modularity": "-",
            "GPU Modularity": "-"
        },
        {
            "Algorithm": "Louvain",
            "CPU Time (s)": round(louvain_cpu_time, 4),
            "GPU Time (s)": round(louvain_gpu_time, 4),
            "Speedup (x)": round(louvain_speedup, 2),
            "Improvement (%)": round(louvain_improvement, 2),
            "CPU Modularity": round(mod_cpu, 4),
            "GPU Modularity": round(mod_gpu, 4)
        }
    ])

    return comparison_df, pr_cpu, pr_gpu, part_cpu, part_gpu