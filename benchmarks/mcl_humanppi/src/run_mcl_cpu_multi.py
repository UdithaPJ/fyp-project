import os
# ---- FORCE MULTI THREAD (DEFAULT 8) ----
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import argparse
import time
import numpy as np
import pandas as pd
from scipy import sparse
from mcl_sparse import mcl, extract_clusters

def run(csr_path: str, nodes_path: str, out_clusters: str, out_stats: str,
        expansion=2, inflation=2.0, max_iter=40, tol=1e-3, prune_k=100, prune_eps=1e-6):

    A = sparse.load_npz(csr_path).tocsr()
    nodes_df = pd.read_csv(nodes_path, sep="\t")
    proteins = nodes_df["protein"].tolist()

    t0 = time.perf_counter()
    M = mcl(A, expansion=expansion, inflation_r=inflation, max_iter=max_iter,
            tol=tol, prune_k=prune_k, prune_eps=prune_eps, verbose=True)
    t1 = time.perf_counter()

    clusters = extract_clusters(M)
    clusters_sorted = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
    sizes = [len(v) for _, v in clusters_sorted]

    os.makedirs(os.path.dirname(out_clusters), exist_ok=True)
    os.makedirs(os.path.dirname(out_stats), exist_ok=True)

    # Save clusters
    rows = []
    for cid, members in clusters_sorted:
        for nid in members:
            rows.append((cid, nid, proteins[nid]))
    pd.DataFrame(rows, columns=["cluster_id", "node_id", "protein"]).to_csv(out_clusters, sep="\t", index=False)

    # Save stats
    with open(out_stats, "w", encoding="utf-8") as f:
        f.write("=== MCL CPU MULTI (8 threads) ===\n")
        f.write(f"CSR: {csr_path}\n")
        f.write(f"Nodes: {A.shape[0]}\n")
        f.write(f"NNZ: {A.nnz}\n")
        f.write(f"Clusters: {len(sizes)}\n")
        f.write(f"Largest cluster: {max(sizes) if sizes else 0}\n")
        f.write(f"Mean cluster: {float(np.mean(sizes)) if sizes else 0.0:.4f}\n")
        f.write(f"Runtime_sec: {(t1 - t0):.6f}\n")

    print(f"\nSaved clusters: {out_clusters}")
    print(f"Saved stats: {out_stats}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csr", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--out_clusters", required=True)
    ap.add_argument("--out_stats", required=True)
    ap.add_argument("--expansion", type=int, default=2)
    ap.add_argument("--inflation", type=float, default=2.0)
    ap.add_argument("--max_iter", type=int, default=40)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--prune_k", type=int, default=100)
    ap.add_argument("--prune_eps", type=float, default=1e-6)
    args = ap.parse_args()

    run(args.csr, args.nodes, args.out_clusters, args.out_stats,
        args.expansion, args.inflation, args.max_iter, args.tol, args.prune_k, args.prune_eps)

if __name__ == "__main__":
    main()