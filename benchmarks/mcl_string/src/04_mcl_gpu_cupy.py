"""
MCL Algorithm - GPU (CuPy) Implementation with Full Analysis
=============================================================

This script performs Markov Clustering (MCL) on STRING PPI network data using GPU acceleration.

Features:
  - GPU-accelerated MCL clustering using CuPy
  - Detailed timing analysis (load, transfer, execution, per-iteration)
  - Cluster extraction and protein ranking
  - Comprehensive result files with timing, clusters, and statistics
  - Top proteins analysis by influence and connectivity
  - Complete logging and profiling

Input: Preprocessed CSR matrix (data_processed/ppi_mcl/ppi_csr.npz)
       Node mapping (data_processed/ppi_mcl/node_map.json)

Output:
  - Timing results (results_*/mcl_gpu_timing.txt)
  - Cluster statistics (results_*/cluster_stats.json)
  - Top proteins analysis (results_*/top_proteins.txt)
  - Iteration logs (results_*/mcl_gpu_iterations.txt)
  - All proteins cluster assignments (results_*/protein_clusters.txt)
  - Summary statistics (results_*/summary.txt)

Author: FYP Project
Date: 2026
"""

import time
from datetime import datetime
import json
import numpy as np
from pathlib import Path
from scipy import sparse
from typing import Dict, List, Tuple, Optional

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
except Exception as e:
    raise SystemExit("CuPy not installed. Install cupy-cuda11x or cupy-cuda12x matching your CUDA.") from e

# Optional: find the latest CSR from data_processed/ppi_mcl_*
_data_dir = Path("data_processed")
_mcl_dirs = sorted([d for d in _data_dir.glob("ppi_mcl_*") if d.is_dir()])
if _mcl_dirs:
    _latest_dir = _mcl_dirs[-1]
    _csr_files = sorted(_latest_dir.glob("ppi_csr_*.npz"))
    CSR_NPZ = str(_csr_files[-1]) if _csr_files else "data_processed/ppi_mcl/ppi_csr.npz"
else:
    CSR_NPZ = "data_processed/ppi_mcl/ppi_csr.npz"

# Node mapping
NODE_MAP_FILE = Path("data_processed/ppi_mcl/node_map.json")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path(f"results_gpu_{timestamp}")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ---- MCL Algorithm Parameters ----
R = 1.5              # Inflation parameter (reduced to preserve connectivity)
TAU = 1e-6           # Pruning threshold (standard - applied after inflation)
MAX_ITERS = 60       # Maximum iterations
TOL = 1e-9           # Convergence tolerance
# ----------------------------------

log_lines = []
timing_data = {}

def preprune_network(A: sparse.csr_matrix, min_degree: int = 2) -> Tuple[sparse.csr_matrix, Dict[int, int]]:
    """Pre-prune network to remove isolated and low-degree nodes.
    
    Returns:
        - Pruned adjacency matrix
        - Mapping from original indices to new indices (for proteins not removed)
    """
    # Calculate node degrees (number of connections each node has)
    degrees = np.asarray(A.sum(axis=1)).ravel()  # sum across rows
    
    # Keep only nodes with degree >= min_degree
    keep_mask = degrees >= min_degree
    keep_indices = np.where(keep_mask)[0]
    
    n_removed = len(degrees) - np.sum(keep_mask)
    log(f"Pre-pruning: Removing {n_removed} nodes with degree < {min_degree}")
    log(f"Keeping {np.sum(keep_mask)} nodes with degree >= {min_degree}")
    
    if np.sum(keep_mask) == 0:
        return A, {i: i for i in range(len(degrees))}
    
    # Create mapping
    idx_map = {}
    new_idx = 0
    for orig_idx in range(len(degrees)):
        if keep_mask[orig_idx]:
            idx_map[orig_idx] = new_idx
            new_idx += 1
    
    # Subset matrix to keep only selected nodes, both rows and columns
    A_pruned = A[keep_indices, :][:, keep_indices].tocsr().astype(np.float64)
    log(f"New matrix size: n={A_pruned.shape[0]:,} nnz={A_pruned.nnz:,}\n")
    
    return A_pruned, idx_map

def log(msg: str):
    """Log message to console and buffer."""
    print(msg)
    log_lines.append(msg)

def col_normalize_gpu(M: cpsp.csr_matrix) -> cpsp.csr_matrix:
    """Normalize columns to sum to 1."""
    col_sum = cp.asarray(M.sum(axis=0)).ravel()
    col_sum = cp.where(col_sum == 0, 1.0, col_sum)
    inv = 1.0 / col_sum
    Dinv = cpsp.diags(inv, format="csr")
    return M @ Dinv

def prune_gpu(M: cpsp.csr_matrix, tau: float) -> cpsp.csr_matrix:
    """Remove small values below threshold."""
    M = M.tocsr(copy=True)
    M.data = cp.where(M.data < tau, 0.0, M.data)
    M.eliminate_zeros()
    return M

def load_node_mapping(node_map_file: Path) -> Dict[int, str]:
    """Load node index to protein ID mapping."""
    if not node_map_file.exists():
        log(f"Warning: Node map file not found at {node_map_file}")
        return {}
    
    try:
        with open(node_map_file, 'r') as f:
            protein_to_idx = json.load(f)
        # Reverse mapping: index -> protein
        idx_to_protein = {v: k for k, v in protein_to_idx.items()}
        return idx_to_protein
    except Exception as e:
        log(f"Error loading node map: {e}")
        return {}

def extract_clusters_gpu(M_gpu: cpsp.csr_matrix) -> Dict[int, List[int]]:
    """
    Extract clusters from MCL result matrix.
    Each node belongs to the cluster of the node it has the highest weight to.
    
    Args:
        M_gpu: Final MCL matrix in GPU format
        
    Returns:
        Dictionary mapping cluster_id -> list of node indices
    """
    # Transfer to CPU for cluster extraction
    M_cpu = M_gpu.get().tocsr()
    clusters = {}
    
    for node_idx in range(M_cpu.shape[0]):
        row_data = M_cpu.getrow(node_idx)
        if row_data.nnz > 0:
            col_indices = row_data.nonzero()[1]
            col_values = row_data.data
            max_idx = col_indices[np.argmax(col_values)]
            
            if max_idx not in clusters:
                clusters[max_idx] = []
            clusters[max_idx].append(node_idx)
    
    return clusters

def rank_proteins_by_influence(
    clusters: Dict[int, List[int]],
    idx_to_protein: Dict[int, str],
    M_cpu: sparse.csr_matrix,
    top_n: int = 50
) -> List[Tuple[str, int, int, float, float]]:
    """
    Rank proteins by their influence in clustering.
    Influence considers cluster size and internal connectivity.
    
    Args:
        clusters: Cluster assignments
        idx_to_protein: Index to protein ID mapping
        M_cpu: Final MCL matrix (CPU)
        top_n: Number of top proteins to return
        
    Returns:
        List of (protein_id, cluster_id, cluster_size, connection_strength, influence_score)
    """
    protein_influence = []
    
    for cluster_id, nodes_in_cluster in clusters.items():
        cluster_size = len(nodes_in_cluster)
        
        for node_idx in nodes_in_cluster:
            protein_id = idx_to_protein.get(node_idx, f"node_{node_idx}")
            
            # Get connection strength within cluster
            row = M_cpu.getrow(node_idx)
            if row.nnz > 0:
                # Connection strength: sum of weights to nodes in same cluster
                internal_weight = 0.0
                for j_idx in row.nonzero()[1]:
                    if j_idx in nodes_in_cluster:
                        # Get the value
                        val = row.data[row.nonzero()[1] == j_idx]
                        if len(val) > 0:
                            internal_weight += float(val[0])
                
                connection_strength = internal_weight / max(row.sum(), 1e-10)
            else:
                connection_strength = 0.0
            
            # Influence score = cluster size * connection strength
            influence_score = cluster_size * (1.0 + connection_strength)
            
            protein_influence.append((
                protein_id,
                cluster_id,
                cluster_size,
                connection_strength,
                influence_score
            ))
    
    # Sort by influence score (descending)
    protein_influence.sort(key=lambda x: x[4], reverse=True)
    
    return protein_influence[:top_n]

def calculate_cluster_statistics(
    clusters: Dict[int, List[int]],
    idx_to_protein: Dict[int, str],
    M_cpu: sparse.csr_matrix
) -> Dict:
    """Calculate statistics for each cluster."""
    stats = {
        "total_clusters": len(clusters),
        "cluster_details": []
    }
    
    for cluster_id, nodes_in_cluster in clusters.items():
        cluster_size = len(nodes_in_cluster)
        
        # Calculate internal density
        internal_edges = 0
        for node_i in nodes_in_cluster:
            row = M_cpu.getrow(node_i)
            for j in row.nonzero()[1]:
                if j in nodes_in_cluster:
                    internal_edges += 1
        
        max_edges = cluster_size * (cluster_size - 1)
        density = internal_edges / max(max_edges, 1)
        
        # Average weight
        avg_weight = 0.0
        if internal_edges > 0:
            for node_i in nodes_in_cluster:
                row = M_cpu.getrow(node_i)
                for j in row.nonzero()[1]:
                    if j in nodes_in_cluster:
                        val = row.data[row.nonzero()[1] == j]
                        if len(val) > 0:
                            avg_weight += float(val[0])
            avg_weight /= internal_edges
        
        cluster_info = {
            "cluster_id": int(cluster_id),
            "size": cluster_size,
            "internal_edges": internal_edges,
            "density": float(density),
            "avg_internal_weight": float(avg_weight),
            "proteins": [idx_to_protein.get(idx, f"node_{idx}") for idx in nodes_in_cluster[:20]]  # First 20
        }
        
        stats["cluster_details"].append(cluster_info)
    
    # Sort by cluster size
    stats["cluster_details"].sort(key=lambda x: x["size"], reverse=True)
    
    return stats

def mcl_gpu(A_gpu: cpsp.csr_matrix, r=2.0, tau=1e-6, max_iters=60, tol=1e-9):
    """
    Run MCL algorithm on GPU with detailed timing.
    
    Returns:
        M: Final MCL matrix
        timing: Dictionary with detailed timing information
        iteration_logs: List of iteration detail messages
    """
    timing = {
        "initialization": 0.0,
        "iterations": [],
        "total": 0.0
    }
    iteration_logs = []
    
    # Initialize
    t_init = time.perf_counter()
    M = col_normalize_gpu(A_gpu)
    cp.cuda.Stream.null.synchronize()
    timing["initialization"] = time.perf_counter() - t_init

    t_start = time.perf_counter()
    for it in range(1, max_iters + 1):
        t_it = time.perf_counter()
        M_prev = M.copy()

        # Expansion: M = M @ M  
        # NOTE: Using CPU sparse multiplication + pruning because:
        # 1. CuPy sparse mult produces empty result (GPU bug)
        # 2. Result is huge (365M nnz) and exceeds GPU memory limits anyway
        # 3. CPU sparse matrix mult is actually highly optimized
        M_cpu = M.get()
        M_cpu = M_cpu @ M_cpu
        
        # CRITICAL: Aggressively prune on CPU BEFORE transferring back to GPU
        # This is essential to avoid GPU memory overflow
        # Use a strong threshold to drastically reduce matrix size
        M_cpu.data[M_cpu.data < 1e-2] = 0.0  # Heavy early threshold (1e-2)
        M_cpu.eliminate_zeros()
        log(f"[DEBUG] iter {it}: After aggr prune on CPU: nnz={M_cpu.nnz:,}")
        
        # Transfer pruned result back to GPU
        M = cpsp.csr_matrix(M_cpu)
        
        # Check if matrix is empty
        if M.nnz == 0:
            log(f"WARNING: Matrix pruned to zero on CPU before GPU transfer at iteration {it}")
            log(f"Threshold 1e-4 was too aggressive. Values too small.")
            break

        # Prune with standard threshold
        M = prune_gpu(M, tau)
        
        # Check if matrix collapsed due to pruning
        if M.nnz == 0:
            log(f"WARNING: Matrix pruned to zero at iteration {it}")
            log(f"All M@M values are below TAU={tau}")
            break

        # Inflation: Raise non-zero entries to power r
        M.data = cp.power(M.data, r, dtype=cp.float64)

        # Prune again with standard threshold to clean up small values from inflation
        M = prune_gpu(M, tau)

        # Normalize: Columns sum to 1
        M = col_normalize_gpu(M)

        # Convergence check
        diff = (M - M_prev).data
        err = float(cp.max(cp.abs(diff)).get()) if diff.size else 0.0

        cp.cuda.Stream.null.synchronize()
        iter_time = time.perf_counter() - t_it
        timing["iterations"].append(iter_time)
        
        msg = f"iter={it:02d} nnz={M.nnz:,} err={err:.3e} iter_time={iter_time:.3f}s"
        log(msg)
        iteration_logs.append(msg)

        if err < tol:
            log(f"Converged at iteration {it}")
            break

    cp.cuda.Stream.null.synchronize()
    timing["total"] = time.perf_counter() - t_start
    
    return M, timing, iteration_logs

def main():
    """Main execution with comprehensive analysis."""
    
    log("\n" + "=" * 80)
    log("MCL ALGORITHM - GPU (CuPy) ANALYSIS")
    log("=" * 80 + "\n")
    
    # Timing: Load
    t0 = time.perf_counter()
    A = sparse.load_npz(CSR_NPZ).tocsr().astype(np.float64)
    load_time = time.perf_counter() - t0
    log(f"Loaded CPU CSR: n={A.shape[0]:,} nnz={A.nnz:,}")
    log(f"Load time: {load_time:.3f}s")
    
    # Load node mapping
    idx_to_protein = load_node_mapping(NODE_MAP_FILE)
    log(f"Loaded protein mapping: {len(idx_to_protein)} nodes")
    
    # Pre-prune: Remove low-degree nodes to reduce network size and memory pressure
    log("\nPre-processing network...")
    A, node_idx_map = preprune_network(A, min_degree=2)
    log(f"Sparsity after pre-pruning: {100.0 * (1.0 - A.nnz / (A.shape[0] * A.shape[1])):.2f}%")
    
    # Create new protein mapping for pruned network
    # Map: new_idx -> protein_id (skipping removed nodes)
    idx_to_protein_pruned = {}
    for orig_idx, protein_id in idx_to_protein.items():
        if orig_idx in node_idx_map:
            new_idx = node_idx_map[orig_idx]
            idx_to_protein_pruned[new_idx] = protein_id
    
    log(f"\nMCL Parameters: R={R} TAU={TAU} MAX_ITERS={MAX_ITERS} TOL={TOL}")
    log(f"Results directory: {RESULTS_DIR}\n")
    
    # Timing: Transfer to GPU
    t1 = time.perf_counter()
    A_gpu = cpsp.csr_matrix(A)
    cp.cuda.Stream.null.synchronize()
    transfer_time = time.perf_counter() - t1
    log(f"Transferred to GPU: {transfer_time:.3f}s\n")
    
    # Run MCL with detailed timing
    log("Running MCL Algorithm...")
    log("-" * 80)
    M_gpu, mcl_timing, iteration_logs = mcl_gpu(A_gpu, r=R, tau=TAU, max_iters=MAX_ITERS, tol=TOL)
    log("-" * 80)
    log(f"MCL initialization: {mcl_timing['initialization']:.3f}s")
    log(f"MCL iterations: {mcl_timing['total']:.3f}s")
    log(f"Final nnz: {M_gpu.nnz:,}\n")
    
    # Transfer result back to CPU
    t2 = time.perf_counter()
    M_cpu = M_gpu.get()
    transfer_back_time = time.perf_counter() - t2
    log(f"Transferred result back to CPU: {transfer_back_time:.3f}s\n")
    
    # Extract clusters
    log("Extracting clusters...")
    clusters = extract_clusters_gpu(M_gpu)
    log(f"Extracted {len(clusters)} clusters\n")
    
    # Rank proteins
    log("Ranking proteins by influence...")
    top_proteins = rank_proteins_by_influence(clusters, idx_to_protein, M_cpu, top_n=100)
    log(f"Generated influence rankings\n")
    
    # Calculate cluster statistics
    log("Calculating cluster statistics...")
    cluster_stats = calculate_cluster_statistics(clusters, idx_to_protein, M_cpu)
    log(f"Calculated statistics for {cluster_stats['total_clusters']} clusters\n")
    
    # =======================================
    # GENERATE RESULT FILES
    # =======================================
    
    log("Generating result files...")
    log("-" * 80)
    
    # 1. Timing Results with Cluster Rankings
    timing_file = RESULTS_DIR / "mcl_gpu_timing.txt"
    with open(timing_file, 'w', encoding='utf-8') as f:
        f.write("MCL GPU TIMING ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Results directory: {RESULTS_DIR}\n")
        f.write(f"Dataset: STRING PPI Network\n")
        f.write(f"Input size: {A.shape[0]:,} nodes, {A.nnz:,} edges\n\n")
        
        f.write("TIMING BREAKDOWN (seconds)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Data loading:              {load_time:10.4f}s\n")
        f.write(f"GPU transfer (upload):     {transfer_time:10.4f}s\n")
        f.write(f"MCL initialization:        {mcl_timing['initialization']:10.4f}s\n")
        f.write(f"MCL iterations:            {mcl_timing['total']:10.4f}s\n")
        f.write(f"GPU transfer (download):   {transfer_back_time:10.4f}s\n")
        f.write(f"{'TOTAL EXECUTION':30s} {load_time + transfer_time + mcl_timing['initialization'] + mcl_timing['total'] + transfer_back_time:10.4f}s\n\n")
        
        f.write("PER-ITERATION TIMING\n")
        f.write("=" * 80 + "\n")
        f.write("Iteration | Time (s)      | Cumulative (s)\n")
        f.write("-" * 80 + "\n")
        cumulative = 0.0
        for i, iter_time in enumerate(mcl_timing['iterations'], 1):
            cumulative += iter_time
            f.write(f"{i:9d} | {iter_time:13.4f} | {cumulative:13.4f}\n")
        
        f.write("\n")
        f.write("CLUSTERING RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total clusters found: {len(clusters)}\n")
        f.write(f"Total proteins assigned: {sum(len(c) for c in clusters.values())}\n\n")
        
        f.write("TOP 20 CLUSTERS BY SIZE AND CONNECTIVITY\n")
        f.write("-" * 80 + "\n")
        f.write("Rank | Cluster ID |   Size | Internal Edges |   Density | Avg Weight |  Connectivity\n")
        f.write("-" * 80 + "\n")
        
        # Sort clusters by metrics for ranking
        sorted_clusters = []
        for cluster_id, nodes_in_cluster in clusters.items():
            cluster_size = len(nodes_in_cluster)
            
            # Calculate internal edges and metrics
            internal_edges = 0
            total_weight = 0.0
            for node_i in nodes_in_cluster:
                row = M_cpu.getrow(node_i)
                for j in row.nonzero()[1]:
                    if j in nodes_in_cluster:
                        internal_edges += 1
                        val = row.data[row.nonzero()[1] == j]
                        if len(val) > 0:
                            total_weight += float(val[0])
            
            max_edges = cluster_size * (cluster_size - 1) if cluster_size > 1 else 1
            density = internal_edges / max_edges
            avg_weight = total_weight / max(internal_edges, 1)
            connectivity = cluster_size * density  # Combined metric
            
            sorted_clusters.append({
                'id': cluster_id,
                'size': cluster_size,
                'internal_edges': internal_edges,
                'density': density,
                'avg_weight': avg_weight,
                'connectivity': connectivity
            })
        
        # Sort by connectivity (size * density)
        sorted_clusters.sort(key=lambda x: x['connectivity'], reverse=True)
        
        for rank, cluster in enumerate(sorted_clusters[:20], 1):
            f.write(f"{rank:4d} | {cluster['id']:10d} | {cluster['size']:6d} | {cluster['internal_edges']:14d} | "
                   f"{cluster['density']:9.4f} | {cluster['avg_weight']:10.6f} | {cluster['connectivity']:13.4f}\n")
        
        f.write("\n")
        f.write("TOP 20 CLUSTERS BY SIZE ONLY\n")
        f.write("-" * 80 + "\n")
        f.write("Rank | Cluster ID |   Size | Metric: Size\n")
        f.write("-" * 80 + "\n")
        
        size_sorted = sorted(sorted_clusters, key=lambda x: x['size'], reverse=True)
        for rank, cluster in enumerate(size_sorted[:20], 1):
            f.write(f"{rank:4d} | {cluster['id']:10d} | {cluster['size']:6d} | {cluster['size']}\n")
        
        f.write("\n")
        f.write("ALGORITHM PARAMETERS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Inflation parameter (R):   {R}\n")
        f.write(f"Pruning threshold (TAU):   {TAU}\n")
        f.write(f"Max iterations:            {MAX_ITERS}\n")
        f.write(f"Convergence tolerance:     {TOL}\n")
        f.write(f"Actual iterations run:     {len(mcl_timing['iterations'])}\n")
        f.write(f"Final matrix nnz:          {M_gpu.nnz:,}\n")
    
    log(f"[✓] Timing results with cluster rankings: {timing_file}")
    
    # 2. Iteration Logs
    iter_log_file = RESULTS_DIR / "mcl_gpu_iterations.txt"
    with open(iter_log_file, 'w', encoding='utf-8') as f:
        f.write("MCL ALGORITHM ITERATION LOGS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Parameters: R={R}, TAU={TAU}, MAX_ITERS={MAX_ITERS}, TOL={TOL}\n")
        f.write(f"Input matrix: n={A.shape[0]:,}, nnz={A.nnz:,}\n\n")
        f.write("ITERATION DETAILS\n")
        f.write("=" * 80 + "\n")
        for log_line in iteration_logs:
            f.write(log_line + "\n")
    
    log(f"[✓] Iteration logs: {iter_log_file}")
    
    # 3. Cluster Statistics (JSON)
    stats_json_file = RESULTS_DIR / "cluster_stats.json"
    with open(stats_json_file, 'w') as f:
        json.dump(cluster_stats, f, indent=2)
    log(f"[✓] Cluster statistics (JSON): {stats_json_file}")
    
    # 4. Top Proteins Ranking (Top 100)
    top_proteins_file = RESULTS_DIR / "top_proteins.txt"
    with open(top_proteins_file, 'w', encoding='utf-8') as f:
        f.write("TOP PROTEINS RANKING BY INFLUENCE\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total clusters found: {len(clusters)}\n")
        f.write(f"Ranked by: Cluster size × Connection strength\n")
        f.write(f"Total proteins ranked: {len(top_proteins)}\n\n")
        f.write("TOP PROTEIN RANKINGS\n")
        f.write("=" * 100 + "\n")
        f.write("Rank | Protein ID                  | Cluster | Cluster Size | Connection | Influence Score\n")
        f.write("-" * 100 + "\n")
        for rank, (protein_id, cluster_id, cluster_size, conn_strength, influence) in enumerate(top_proteins, 1):
            f.write(f"{rank:4d} | {protein_id:27s} | {cluster_id:7d} | {cluster_size:12d} | "
                   f"{conn_strength:10.4f} | {influence:15.6f}\n")
    
    log(f"[✓] Top proteins ranking (top 100): {top_proteins_file}")
    
    # 5. All Protein Clusters Assignment
    protein_clusters_file = RESULTS_DIR / "protein_clusters.txt"
    with open(protein_clusters_file, 'w', encoding='utf-8') as f:
        f.write("PROTEIN CLUSTER ASSIGNMENTS\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total proteins: {A.shape[0]:,}\n")
        f.write(f"Total clusters: {len(clusters)}\n\n")
        
        for rank, cluster_id in enumerate(sorted(clusters.keys(), key=lambda c: len(clusters[c]), reverse=True), 1):
            nodes = sorted(clusters[cluster_id])
            proteins = [idx_to_protein.get(n, f"node_{n}") for n in nodes]
            f.write(f"[CLUSTER #{rank}] ID {cluster_id}: {len(proteins)} proteins\n")
            f.write("-" * 100 + "\n")
            for protein in proteins:
                f.write(f"  {protein}\n")
            f.write("\n")
    
    log(f"[✓] Protein cluster assignments: {protein_clusters_file}")
    
    # 6. Cluster Details by Size (Top 20 clusters)
    cluster_details_file = RESULTS_DIR / "cluster_details.txt"
    with open(cluster_details_file, 'w', encoding='utf-8') as f:
        f.write("DETAILED CLUSTER ANALYSIS (Top 20 Clusters)\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Sorted by: Cluster size (descending)\n")
        f.write(f"Total clusters in dataset: {len(cluster_stats['cluster_details'])}\n\n")
        
        for i, cluster_info in enumerate(cluster_stats["cluster_details"][:20], 1):
            f.write(f"[ CLUSTER #{i} ]\n")
            f.write("-" * 100 + "\n")
            f.write(f"  ID:                  {cluster_info['cluster_id']}\n")
            f.write(f"  Size:                {cluster_info['size']} proteins\n")
            f.write(f"  Internal edges:      {cluster_info['internal_edges']}\n")
            f.write(f"  Network density:     {cluster_info['density']:.6f}\n")
            f.write(f"  Avg internal weight: {cluster_info['avg_internal_weight']:.8f}\n")
            f.write(f"  Sample proteins (first 20):\n")
            for protein in cluster_info['proteins']:
                f.write(f"    • {protein}\n")
            f.write("\n")
    
    log(f"[✓] Cluster details (top 20): {cluster_details_file}")
    
    # 7. Summary Statistics
    summary_file = RESULTS_DIR / "summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        total_time = load_time + transfer_time + mcl_timing['initialization'] + mcl_timing['total'] + transfer_back_time
        total_iterations = len(mcl_timing['iterations'])
        avg_iter_time = mcl_timing['total'] / total_iterations if total_iterations > 0 else 0
        
        f.write("MCL GPU ANALYSIS SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Results directory: {RESULTS_DIR}\n\n")
        
        f.write("INPUT DATA\n")
        f.write("=" * 100 + "\n")
        f.write(f"  Nodes (proteins):     {A.shape[0]:,}\n")
        f.write(f"  Edges (interactions): {A.nnz:,}\n")
        f.write(f"  Sparsity:             {(1 - A.nnz / (A.shape[0] * A.shape[0])) * 100:.2f}%\n\n")
        
        f.write("MCL PARAMETERS\n")
        f.write("=" * 100 + "\n")
        f.write(f"  Inflation (R):        {R}\n")
        f.write(f"  Pruning threshold:    {TAU}\n")
        f.write(f"  Max iterations:       {MAX_ITERS}\n")
        f.write(f"  Convergence tol:      {TOL}\n\n")
        
        f.write("EXECUTION TIMING (seconds)\n")
        f.write("=" * 100 + "\n")
        f.write(f"  Data loading:                  {load_time:12.4f}s\n")
        f.write(f"  GPU transfer (upload):         {transfer_time:12.4f}s\n")
        f.write(f"  MCL initialization:            {mcl_timing['initialization']:12.4f}s\n")
        f.write(f"  MCL iterations ({total_iterations:2d}):                {mcl_timing['total']:12.4f}s\n")
        f.write(f"  GPU transfer (download):       {transfer_back_time:12.4f}s\n")
        f.write(f"  {'TOTAL EXECUTION TIME':40s} {total_time:12.4f}s\n")
        f.write(f"  Average time per iteration:    {avg_iter_time:12.4f}s\n\n")
        
        f.write("CLUSTERING RESULTS\n")
        f.write("=" * 100 + "\n")
        f.write(f"  Total clusters:       {len(clusters)}\n")
        if len(clusters) > 0:
            cluster_sizes = [len(c) for c in clusters.values()]
            f.write(f"  Largest cluster:      {max(cluster_sizes):,} proteins\n")
            f.write(f"  Smallest cluster:     {min(cluster_sizes):,} proteins\n")
            f.write(f"  Average cluster size: {np.mean(cluster_sizes):.1f}\n")
            f.write(f"  Median cluster size:  {np.median(cluster_sizes):.1f}\n")
            f.write(f"  Std. dev. cluster:    {np.std(cluster_sizes):.1f}\n")
        else:
            f.write(f"  ⚠ WARNING: No clusters found. Matrix may have converged to zero.\n")
            f.write(f"  Final matrix nnz: {M_gpu.nnz}\n")
        f.write("\n")
        
        f.write("OUTPUT FILES GENERATED\n")
        f.write("=" * 100 + "\n")
        f.write(f"  ✓ mcl_gpu_timing.txt        - Timing breakdown + top 20 cluster rankings\n")
        f.write(f"  ✓ mcl_gpu_iterations.txt    - Per-iteration algorithm logs\n")
        f.write(f"  ✓ cluster_stats.json        - Cluster statistics in JSON format\n")
        f.write(f"  ✓ top_proteins.txt          - Top 100 proteins ranked by influence\n")
        f.write(f"  ✓ protein_clusters.txt      - Complete protein cluster assignments\n")
        f.write(f"  ✓ cluster_details.txt       - Detailed analysis of top 20 clusters\n")
        f.write(f"  ✓ summary.txt               - This comprehensive summary\n")
    
    log(f"[✓] Summary: {summary_file}")
    
    log("-" * 80)
    log("\n[✓] All result files generated successfully!")
    log(f"Results location: {RESULTS_DIR}\n")
    
    # Also save the full log
    full_log_file = RESULTS_DIR / "full_log.txt"
    with open(full_log_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(log_lines))
    
    log(f"[✓] Full execution log: {full_log_file}")

if __name__ == "__main__":
    main()
