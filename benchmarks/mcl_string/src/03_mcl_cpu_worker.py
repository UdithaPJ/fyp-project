"""
MCL Algorithm - CPU Worker Process
===================================

This worker performs MCL (Markov Clustering) clustering on STRING PPI network data.
It runs with specified thread count and generates detailed analysis results.

Clustering Algorithm:
  1. Load CSR sparse matrix of protein interactions
  2. Initialize MCL with column normalization
  3. Iterate expansion (M @ M) and inflation (M^r) steps
  4. Apply pruning and normalization until convergence
  5. Extract clusters from final matrix

Output:
  - per-thread execution log (mcl_cpu_threads_<T>.log)
  - clustering summary (summary_threads_<T>.txt)
  - top proteins analysis (top_proteins_analysis.txt)
  - cluster statistics (cluster_stats.json)
  - node cluster mappings (node_cluster_mapping.json)

Author: FYP Project
Date: 2026
"""

import os
import sys
import time
import json
import tracemalloc
from pathlib import Path
from typing import Tuple, Dict, List
from datetime import datetime

import numpy as np
from scipy import sparse


# ==============================
# CONFIGURATION
# ==============================

# MCL Algorithm Parameters
R = 2.0                 # Inflation parameter (Markov parameter)
TAU = 1e-6              # Pruning threshold
MAX_ITERS = 60          # Maximum iterations
TOL = 1e-9              # Convergence tolerance

# Output Options
VERBOSE = True
SAVE_CLUSTER_DETAILS = True  # Save detailed cluster memberships


# ==============================
# UTILITY FUNCTIONS
# ==============================

class Logger:
    """Unified logging to console and file."""
    
    def __init__(self):
        self.lines = []
    
    def log(self, msg: str):
        """Log message to console and buffer."""
        if VERBOSE:
            print(msg, flush=True)
        self.lines.append(msg)
    
    def get_text(self) -> str:
        """Get all logged text."""
        return "\n".join(self.lines)


def col_normalize(M: sparse.csr_matrix) -> sparse.csr_matrix:
    """Normalize columns to sum to 1."""
    col_sum = np.asarray(M.sum(axis=0)).ravel()
    col_sum[col_sum == 0] = 1.0
    inv = 1.0 / col_sum
    return M @ sparse.diags(inv, format="csr")


def prune(M: sparse.csr_matrix, tau: float) -> sparse.csr_matrix:
    """Remove small values below threshold."""
    M = M.tocsr(copy=True)
    M.data[M.data < tau] = 0.0
    M.eliminate_zeros()
    return M


def extract_clusters(M: sparse.csr_matrix) -> Dict[int, List[int]]:
    """
    Extract clusters from MCL matrix.
    Each node belongs to cluster of node it has highest weight to.
    
    Args:
        M: Final MCL matrix (sparse)
        
    Returns:
        Dictionary mapping cluster_id -> list of node indices
    """
    clusters = {}
    M_coo = M.tocoo()
    
    for node_idx in range(M.shape[0]):
        # Get weights for this node
        row_data = M.getrow(node_idx)
        if row_data.nnz > 0:
            # Find column with max value
            col_indices = row_data.nonzero()[1]
            col_values = row_data.data
            max_idx = col_indices[np.argmax(col_values)]
            
            if max_idx not in clusters:
                clusters[max_idx] = []
            clusters[max_idx].append(node_idx)
    
    return clusters


def rank_top_proteins_by_influence(
    clusters: Dict[int, List[int]],
    node_to_protein: Dict[int, str],
    M_final: sparse.csr_matrix,
    top_n: int = 20
) -> List[Tuple[str, int, int, float]]:
    """
    Rank proteins by their influence in clustering.
    Influence = cluster_size + average_connectivity_within_cluster
    
    Args:
        clusters: Cluster assignments
        node_to_protein: Mapping node index to protein ID
        M_final: Final MCL matrix
        top_n: Number of top proteins to return
        
    Returns:
        List of (protein_id, cluster_id, cluster_size, influence_score)
    """
    protein_influence = []
    
    for cluster_id, nodes_in_cluster in clusters.items():
        cluster_size = len(nodes_in_cluster)
        
        # Calculate average internal connectivity
        internal_edges = 0
        for node_i in nodes_in_cluster:
            row = M_final.getrow(node_i)
            for j in row.nonzero()[1]:
                if j in nodes_in_cluster:
                    internal_edges += 1
        
        avg_connectivity = internal_edges / max(cluster_size, 1)
        
        # Calculate influence for each node in cluster
        for node_idx in nodes_in_cluster:
            row = M_final.getrow(node_idx)
            node_connectivity = row.nnz / max(row.sum(), 1e-10) if row.sum() > 0 else 0
            
            # Influence = (cluster_size + node internal connections) / node total connections
            influence = (cluster_size + node_connectivity) * cluster_size
            
            protein_id = node_to_protein.get(node_idx, f"node_{node_idx}")
            protein_influence.append((protein_id, cluster_id, cluster_size, influence))
    
    # Sort by influence (descending)
    protein_influence.sort(key=lambda x: x[3], reverse=True)
    
    return protein_influence[:top_n]


def mcl_cpu(
    A: sparse.csr_matrix,
    r: float = 2.0,
    tau: float = 1e-6,
    max_iters: int = 60,
    tol: float = 1e-9,
    logger: Logger = None
) -> Tuple[sparse.csr_matrix, float, List[str]]:
    """
    Run MCL algorithm on sparse matrix.
    
    Args:
        A: Input CSR sparse matrix
        r: Inflation parameter
        tau: Pruning threshold
        max_iters: Maximum iterations
        tol: Convergence tolerance
        logger: Logger instance
        
    Returns:
        Tuple of (result_matrix, execution_time, iteration_logs)
    """
    if logger is None:
        logger = Logger()
    
    # Initialize
    M = col_normalize(A)
    iter_logs = []
    
    t_start = time.perf_counter()
    
    for it in range(1, max_iters + 1):
        t_iter = time.perf_counter()
        M_prev = M.copy()
        
        # Expansion: M = M @ M
        M = M @ M
        
        # Inflation: Raise non-zero entries to power r
        M.data = np.power(M.data, r, dtype=np.float32)
        
        # Prune: Remove small values
        M = prune(M, tau)
        
        # Normalize: Normalize columns
        M = col_normalize(M)
        
        # Check convergence
        diff = (M - M_prev)
        if diff.nnz > 0:
            err = float(np.max(np.abs(diff.data)))
        else:
            err = 0.0
        
        iter_time = time.perf_counter() - t_iter
        msg = f"  iter={it:02d} | nnz={M.nnz:,} | err={err:.3e} | time={iter_time:.3f}s"
        logger.log(msg)
        iter_logs.append(msg)
        
        if err < tol:
            logger.log(f"  Converged at iteration {it}")
            break
    
    total_time = time.perf_counter() - t_start
    
    return M, total_time, iter_logs


def pick_latest_csr() -> str:
    """Find the latest preprocessed CSR matrix."""
    csr_path = Path("data_processed/ppi_mcl/ppi_csr.npz")
    if csr_path.exists():
        return str(csr_path)
    
    # Fallback: try to find any ppi_csr.npz
    data_dir = Path("data_processed")
    if data_dir.exists():
        csr_files = list(data_dir.rglob("ppi_csr.npz"))
        if csr_files:
            return str(csr_files[0])
    
    return "data_processed/ppi_mcl/ppi_csr.npz"


def load_node_mapping() -> Dict[int, str]:
    """Load node ID to protein ID mapping."""
    import json
    mapping_file = Path("data_processed/ppi_mcl/node_map.json")
    
    if not mapping_file.exists():
        return {}
    
    try:
        with open(mapping_file) as f:
            forward_map = json.load(f)
        # Invert: protein_id -> node_idx becomes node_idx -> protein_id
        return {idx: protein_id for protein_id, idx in forward_map.items()}
    except Exception:
        return {}


# ==============================
# MAIN EXECUTION
# ==============================

def main():
    """Main MCL worker process."""
    
    # Initialize memory tracking
    tracemalloc.start()
    
    # Setup logger
    logger = Logger()
    
    # Get configuration from environment
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    mkl_threads = os.environ.get("MKL_NUM_THREADS", str(threads))
    openblas_threads = os.environ.get("OPENBLAS_NUM_THREADS", str(threads))
    blis_threads = os.environ.get("BLIS_NUM_THREADS", str(threads))
    numexpr_threads = os.environ.get("NUMEXPR_NUM_THREADS", str(threads))
    
    # Get paths
    csr_path = pick_latest_csr()
    results_dir = Path(os.environ.get("RESULTS_DIR", "results"))
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Log file (no timestamp in name)
    log_file = results_dir / f"mcl_cpu_threads_{threads}.log"
    
    # Print header
    logger.log("=" * 80)
    logger.log("MCL ALGORITHM - CPU WORKER PROCESS")
    logger.log("=" * 80)
    logger.log("")
    logger.log("Configuration:")
    logger.log(f"  Threads: {threads}")
    logger.log(f"    OMP_NUM_THREADS: {threads}")
    logger.log(f"    MKL_NUM_THREADS: {mkl_threads}")
    logger.log(f"    OPENBLAS_NUM_THREADS: {openblas_threads}")
    logger.log(f"    BLIS_NUM_THREADS: {blis_threads}")
    logger.log(f"    NUMEXPR_NUM_THREADS: {numexpr_threads}")
    logger.log("")
    logger.log("MCL Parameters:")
    logger.log(f"  Inflation (R): {R}")
    logger.log(f"  Pruning threshold (TAU): {TAU}")
    logger.log(f"  Max iterations: {MAX_ITERS}")
    logger.log(f"  Convergence tolerance: {TOL}")
    logger.log("")
    logger.log("Input/Output:")
    logger.log(f"  CSR file: {csr_path}")
    logger.log(f"  Results dir: {results_dir}")
    logger.log("")
    
    # Load data
    logger.log("Loading data...")
    t_load_start = time.perf_counter()
    
    try:
        A = sparse.load_npz(csr_path)
        if A.dtype != np.float32:
            A = A.astype(np.float32)
        A = A.tocsr()
    except Exception as e:
        logger.log(f"ERROR: Failed to load CSR: {e}")
        return 1
    
    load_time = time.perf_counter() - t_load_start
    mem_after_load = tracemalloc.get_traced_memory()[0] / 1024 / 1024
    
    n_nodes = A.shape[0]
    nnz_in = A.nnz
    
    logger.log(f"✓ Loaded: {n_nodes:,} nodes, {nnz_in:,} edges")
    logger.log(f"  Load time: {load_time:.3f}s")
    logger.log(f"  Memory: {mem_after_load:.1f} MB")
    logger.log("")
    
    # Run MCL algorithm
    logger.log("Running MCL algorithm...")
    logger.log("")
    
    M_final, mcl_time, iter_logs = mcl_cpu(
        A, r=R, tau=TAU, max_iters=MAX_ITERS, tol=TOL, logger=logger
    )
    
    nnz_out = M_final.nnz
    mem_peak = tracemalloc.get_traced_memory()[0] / 1024 / 1024
    
    logger.log("")
    logger.log("=" * 80)
    logger.log("MCL CLUSTERING RESULTS")
    logger.log("=" * 80)
    logger.log(f"Final matrix NNZ: {nnz_out:,}")
    logger.log(f"MCL time: {mcl_time:.3f}s")
    logger.log(f"Total time: {load_time + mcl_time:.3f}s")
    logger.log(f"Peak memory: {mem_peak:.1f} MB")
    logger.log("")
    
    # Extract and analyze clusters
    logger.log("Extracting clusters...")
    t_cluster_start = time.perf_counter()
    
    clusters = extract_clusters(M_final)
    n_clusters = len(clusters)
    cluster_sizes = [len(nodes) for nodes in clusters.values()]
    cluster_sizes.sort(reverse=True)
    
    analysis_time = time.perf_counter() - t_cluster_start
    
    logger.log(f"✓ Found {n_clusters} clusters")
    logger.log(f"  Cluster sizes (top 10): {cluster_sizes[:10]}")
    logger.log(f"  Min cluster size: {min(cluster_sizes) if cluster_sizes else 0}")
    logger.log(f"  Max cluster size: {max(cluster_sizes) if cluster_sizes else 0}")
    logger.log(f"  Avg cluster size: {np.mean(cluster_sizes):.1f}" if cluster_sizes else "N/A")
    logger.log(f"  Analysis time: {analysis_time:.3f}s")
    logger.log("")
    
    # Rank top proteins
    logger.log("Ranking top proteins by cluster influence...")
    node_to_protein = load_node_mapping()
    
    top_proteins = rank_top_proteins_by_influence(
        clusters, node_to_protein, M_final, top_n=20
    )
    
    logger.log("Top 20 proteins by cluster influence:")
    logger.log("-" * 80)
    logger.log(f"{'Rank':<5} {'Protein ID':<30} {'Cluster':<8} {'Cluster Size':<14} {'Influence':<12}")
    logger.log("-" * 80)
    
    for rank, (protein_id, cluster_id, cluster_size, influence) in enumerate(top_proteins, 1):
        logger.log(
            f"{rank:<5} {protein_id:<30} {cluster_id:<8} {cluster_size:<14} {influence:.6f}"
        )
    
    logger.log("-" * 80)
    logger.log("")
    
    # Save results
    logger.log("Saving results...")
    
    # Save log file
    with open(log_file, "w") as f:
        f.write(logger.get_text())
    logger.log(f"✓ Saved log: {log_file}")
    
    # Save summary (machine-readable for coordinator)
    summary_file = results_dir / f"summary_threads_{threads}.txt"
    with open(summary_file, "w") as f:
        f.write(
            f"threads={threads} "
            f"n={n_nodes} "
            f"nnz_in={nnz_in} "
            f"nnz_out={nnz_out} "
            f"load_time_s={load_time:.6f} "
            f"total_time_s={load_time + mcl_time:.6f} "
            f"analysis_time_s={analysis_time:.6f}\n"
        )
    logger.log(f"✓ Saved summary: {summary_file}")
    
    # Save top proteins analysis (only for thread=1, shared across all runs)
    if threads == 1:
        top_proteins_file = results_dir / "top_proteins_analysis.txt"
        top_proteins_lines = []
        top_proteins_lines.append("")
        top_proteins_lines.append(f"Rank | Protein ID                     | Cluster | Cluster Size | Influence")
        top_proteins_lines.append("-" * 80)
        for rank, (protein_id, cluster_id, cluster_size, influence) in enumerate(top_proteins, 1):
            top_proteins_lines.append(
                f"{rank:<4} | {protein_id:<28} | {cluster_id:<7} | {cluster_size:<12} | {influence:.6f}"
            )
        top_proteins_lines.append("-" * 80)
        
        with open(top_proteins_file, "w") as f:
            f.write("\n".join(top_proteins_lines))
        logger.log(f"✓ Saved top proteins: {top_proteins_file}")
    
    # Save cluster statistics JSON
    if SAVE_CLUSTER_DETAILS:
        cluster_stats = {
            "algorithm": "MCL",
            "timestamp": datetime.now().isoformat(),
            "threads": threads,
            "mcl_parameters": {
                "inflation": R,
                "tau": TAU,
                "max_iterations": MAX_ITERS,
                "tolerance": TOL
            },
            "input": {
                "n_nodes": n_nodes,
                "nnz_in": nnz_in
            },
            "output": {
                "nnz_out": nnz_out,
                "n_clusters": n_clusters,
                "cluster_sizes": cluster_sizes,
                "cluster_sizes_mean": float(np.mean(cluster_sizes)) if cluster_sizes else 0,
                "cluster_sizes_std": float(np.std(cluster_sizes)) if cluster_sizes else 0
            },
            "timing": {
                "load_time_s": load_time,
                "mcl_time_s": mcl_time,
                "analysis_time_s": analysis_time,
                "total_time_s": load_time + mcl_time + analysis_time
            },
            "memory": {
                "after_load_mb": mem_after_load,
                "peak_mb": mem_peak
            },
            "top_20_proteins": [
                {
                    "rank": i,
                    "protein_id": protein_id,
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(cluster_size),
                    "influence": float(influence)
                }
                for i, (protein_id, cluster_id, cluster_size, influence) in enumerate(top_proteins, 1)
            ]
        }
        
        stats_file = results_dir / f"cluster_stats_threads_{threads}.json"
        with open(stats_file, "w") as f:
            json.dump(cluster_stats, f, indent=2)
        logger.log(f"✓ Saved cluster stats: {stats_file}")
    
    logger.log("")
    logger.log("=" * 80)
    logger.log("MCL WORKER COMPLETE")
    logger.log("=" * 80)
    
    tracemalloc.stop()
    
    return 0


if __name__ == "__main__":
    exit(main())