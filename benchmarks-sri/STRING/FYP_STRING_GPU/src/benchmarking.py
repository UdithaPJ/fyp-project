import pandas as pd
import os


def compare_benchmarks(cpu_file, gpu_file, output_path):

    # 🔹 Create parent directory if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cpu = pd.read_csv(cpu_file)
    gpu = pd.read_csv(gpu_file)

    merged = cpu.merge(gpu, on=["alpha", "tol"], suffixes=("_cpu", "_gpu"))
    merged["speedup"] = merged["runtime_cpu"] / merged["runtime_gpu"]

    merged.to_csv(output_path, index=False)

    print("Benchmark comparison saved.")
