from src.preprocessing_string import preprocess_string
from src.cpu_analysis import run_cpu_analysis
from src.gpu_analysis import run_gpu_analysis
from src.benchmarking import compare_benchmarks


RAW = "data/raw/9606.protein.links.v12.0.txt"
PROCESSED = "data/processed"
CPU_RESULTS = "results/cpu"
GPU_RESULTS = "results/gpu"
BENCH_RESULTS = "results/benchmarks/comparison.csv"


def main():

    preprocess_string(RAW, PROCESSED, min_score=0)

    csr_path = f"{PROCESSED}/adj_csr.npz"

    run_cpu_analysis(csr_path, CPU_RESULTS)
    run_gpu_analysis(csr_path, GPU_RESULTS)

    compare_benchmarks(
        f"{CPU_RESULTS}/pagerank_cpu_benchmark.csv",
        f"{GPU_RESULTS}/pagerank_gpu_benchmark.csv",
        BENCH_RESULTS
    )


if __name__ == "__main__":
    main()
