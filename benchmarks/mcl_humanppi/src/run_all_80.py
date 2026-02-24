import os
import subprocess
import sys

def main():

    inp = "data_raw/final_predictions_80.tsv"

    csr = "data_processed/humanppi_80_csr.npz"
    nodes = "data_processed/humanppi_80_nodes.tsv"

    # Output folders
    out_dir = "results/80"
    os.makedirs(out_dir, exist_ok=True)

    # =========================
    # 1️⃣ PREPROCESS
    # =========================
    print("\n=== STEP 1: PREPROCESS ===")
    subprocess.check_call([
        sys.executable, "src/preprocess_humanppi.py",
        "--input", inp,
        "--out_csr", csr,
        "--out_nodes", nodes,
        "--weight", "RFprob",
        "--min_weight", "0.0"
    ])

    # =========================
    # 2️⃣ SINGLE THREAD
    # =========================
    print("\n=== STEP 2: MCL SINGLE THREAD ===")
    subprocess.check_call([
        sys.executable, "src/run_mcl_cpu_single.py",
        "--csr", csr,
        "--nodes", nodes,
        "--out_clusters", f"{out_dir}/clusters_single.tsv",
        "--out_stats", f"{out_dir}/stats_single.txt",
        "--prune_k", "100",
        "--inflation", "2.0",
        "--max_iter", "40"
    ])

    # =========================
    # 3️⃣ MULTI THREAD (8)
    # =========================
    print("\n=== STEP 3: MCL MULTI THREAD (8) ===")
    subprocess.check_call([
        sys.executable, "src/run_mcl_cpu_multi.py",
        "--csr", csr,
        "--nodes", nodes,
        "--out_clusters", f"{out_dir}/clusters_multi.tsv",
        "--out_stats", f"{out_dir}/stats_multi.txt",
        "--prune_k", "100",
        "--inflation", "2.0",
        "--max_iter", "40"
    ])

    print("\n✅ ALL DONE FOR 80 DATASET")

if __name__ == "__main__":
    main()