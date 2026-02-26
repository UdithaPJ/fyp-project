"""
STRING PPI Network Preprocessing for MCL Algorithm
===================================================

This script preprocesses the STRING protein-protein interaction database
and converts it to a CSR (Compressed Sparse Row) sparse matrix format
suitable for MCL clustering algorithm.

Input: STRING full links file (combined_score for all human interactions)
Output: 
  - CSR matrix (.npz): weighted undirected adjacency matrix
  - Node map (.json): protein ID to node index mapping
  - Edges CSV: edge list with weights
  - Statistics file: network metrics and metadata

Author: FYP Project
Date: 2026
"""

import time
import json
import sys
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from scipy import sparse

# ==============================
# CONFIGURATION
# ==============================

# Input file path (STRING database)
RAW_FILE = "data_raw/string/9606.protein.links.full.v12.0.txt"

# Output directories (no timestamp)
OUT_DIR = Path("data_processed/ppi_mcl")
RESULTS_DIR = Path("results")

# Create output directories
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Output file names (no timestamp)
EDGES_CSV = OUT_DIR / "ppi_edges.csv"
MAP_JSON = OUT_DIR / "node_map.json"
CSR_NPZ = OUT_DIR / "ppi_csr.npz"
STATS_TXT = RESULTS_DIR / "stats.txt"
LOGS_TXT = RESULTS_DIR / "preprocessing_log.txt"

# Filtering parameters
SCORE_MIN = 0               # Minimum combined score (0-1000). 0 = max edges (slowest)
MAX_EDGES = None            # Optional cap on edges (e.g., 5_000_000)
SAVE_EDGES_CSV = True       # Save edge list to CSV (can be large)

# Verbose output
VERBOSE = True


# ==============================
# UTILITY FUNCTIONS
# ==============================

def detect_columns(sample_path: str) -> list[str]:
    """
    Detect column names from STRING file header.
    
    Args:
        sample_path: Path to STRING data file
        
    Returns:
        List of column names
    """
    with open(sample_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    return header.split()


def format_number(n: float) -> str:
    """Format large numbers with thousands separators."""
    if isinstance(n, int):
        return f"{n:,}"
    return f"{n:,.2f}"


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        return tracemalloc.get_traced_memory()[0] / 1024 / 1024
    except:
        return 0.0


def log_message(msg: str, log_file=None):
    """Print and optionally log message."""
    if VERBOSE:
        print(msg)
    if log_file:
        with open(log_file, "a") as f:
            f.write(msg + "\n")


# ==============================
# MAIN PREPROCESSING
# ==============================

def main():
    """Main preprocessing pipeline."""
    
    # Start memory tracking
    tracemalloc.start()
    
    # Initialize log file
    with open(LOGS_TXT, "w") as f:
        f.write("STRING PPI PREPROCESSING LOG\n")
        f.write("=" * 60 + "\n\n")
    
    t_total = time.perf_counter()
    
    log_message(
        f"\n{'='*70}\n"
        f"STRING PROTEIN-PROTEIN INTERACTION PREPROCESSING\n"
        f"{'='*70}\n",
        LOGS_TXT
    )
    
    log_message(f"\nConfiguration:", LOGS_TXT)
    log_message(f"  Raw file: {RAW_FILE}", LOGS_TXT)
    log_message(f"  Output dir: {OUT_DIR}", LOGS_TXT)
    log_message(f"  Results dir: {RESULTS_DIR}", LOGS_TXT)
    log_message(f"  Score minimum: {SCORE_MIN}", LOGS_TXT)
    log_message(f"  Max edges: {MAX_EDGES if MAX_EDGES else 'unlimited'}", LOGS_TXT)
    log_message(f"  Save edges CSV: {SAVE_EDGES_CSV}\n", LOGS_TXT)
    
    # Step 1: Detect columns
    log_message("\n[1/7] Detecting STRING columns...", LOGS_TXT)
    try:
        cols = detect_columns(RAW_FILE)
        log_message(f"  Columns found: {', '.join(cols)}", LOGS_TXT)
    except Exception as e:
        log_message(f"ERROR: Failed to detect columns: {e}", LOGS_TXT)
        raise
    
    # Validate required columns
    if "combined_score" not in cols:
        raise ValueError("combined_score column not found. Please check file header.")
    
    score_col = "combined_score"
    
    # Step 2: Load data
    log_message("\n[2/7] Loading STRING data...", LOGS_TXT)
    t0 = time.perf_counter()
    usecols = ["protein1", "protein2", score_col]
    
    try:
        df = pd.read_csv(
            RAW_FILE,
            sep=r"\s+",
            usecols=usecols,
            low_memory=False,
            dtype={"protein1": str, "protein2": str, score_col: np.float32}
        )
        elapsed = time.perf_counter() - t0
        mem_mb = get_memory_usage()
        log_message(
            f"  Rows loaded: {format_number(len(df))}\n"
            f"  Time: {elapsed:.2f}s\n"
            f"  Memory: {mem_mb:.1f} MB",
            LOGS_TXT
        )
    except Exception as e:
        log_message(f"ERROR: Failed to load file: {e}", LOGS_TXT)
        raise
    
    # Step 3: Filter data
    log_message("\n[3/7] Filtering data...", LOGS_TXT)
    t1 = time.perf_counter()
    
    initial_rows = len(df)
    
    # Filter for human proteins (9606 = H. sapiens)
    df = df[
        df["protein1"].str.startswith("9606.") & 
        df["protein2"].str.startswith("9606.")
    ]
    after_species = len(df)
    
    # Filter by minimum score
    df = df[df[score_col] >= SCORE_MIN]
    after_score = len(df)
    
    # Optional edge cap
    if MAX_EDGES is not None and len(df) > MAX_EDGES:
        df = df.iloc[:MAX_EDGES].copy()
        log_message(f"  Applied MAX_EDGES cap: {MAX_EDGES:,}", LOGS_TXT)
    
    elapsed = time.perf_counter() - t1
    log_message(
        f"  Initial rows: {format_number(initial_rows)}\n"
        f"  After species filter: {format_number(after_species)} "
        f"(removed {format_number(initial_rows - after_species)})\n"
        f"  After score filter (>={SCORE_MIN}): {format_number(after_score)} "
        f"(removed {format_number(after_species - after_score)})\n"
        f"  Time: {elapsed:.2f}s",
        LOGS_TXT
    )
    
    # Step 4: Build undirected edges
    log_message("\n[4/7] Building undirected edge list...", LOGS_TXT)
    t2 = time.perf_counter()
    
    # Make undirected by sorting protein IDs
    s = df[["protein1", "protein2"]].min(axis=1)
    t = df[["protein1", "protein2"]].max(axis=1)
    w = df[score_col].to_numpy(dtype=np.float32)
    
    edges = pd.DataFrame({
        "source": s,
        "target": t,
        "combined_score": w
    })
    
    # Remove self-loops
    self_loops = len(edges[edges["source"] == edges["target"]])
    edges = edges[edges["source"] != edges["target"]]
    
    # Remove duplicates (keep max score)
    duplicates_before = len(edges)
    edges = edges.groupby(["source", "target"], as_index=False)[
        "combined_score"
    ].max()
    duplicates_removed = duplicates_before - len(edges)
    
    elapsed = time.perf_counter() - t2
    log_message(
        f"  Self-loops removed: {format_number(self_loops)}\n"
        f"  Duplicate edges removed: {format_number(duplicates_removed)}\n"
        f"  Final undirected edges: {format_number(len(edges))}\n"
        f"  Time: {elapsed:.2f}s",
        LOGS_TXT
    )
    
    # Step 5: Create node mapping
    log_message("\n[5/7] Creating node mapping...", LOGS_TXT)
    t3 = time.perf_counter()
    
    nodes = pd.unique(edges[["source", "target"]].values.ravel())
    node_map = {n: i for i, n in enumerate(sorted(nodes))}
    n_nodes = len(nodes)
    
    elapsed = time.perf_counter() - t3
    log_message(
        f"  Unique nodes: {format_number(n_nodes)}\n"
        f"  Time: {elapsed:.2f}s",
        LOGS_TXT
    )
    
    # Save edges CSV
    if SAVE_EDGES_CSV:
        log_message("\n[6/7] Saving edges CSV...", LOGS_TXT)
        t4 = time.perf_counter()
        try:
            edges.to_csv(EDGES_CSV, index=False)
            elapsed = time.perf_counter() - t4
            file_size_mb = EDGES_CSV.stat().st_size / 1024 / 1024
            log_message(
                f"  Saved: {EDGES_CSV}\n"
                f"  Size: {file_size_mb:.2f} MB\n"
                f"  Time: {elapsed:.2f}s",
                LOGS_TXT
            )
        except Exception as e:
            log_message(f"WARNING: Failed to save edges CSV: {e}", LOGS_TXT)
    else:
        log_message("\n[6/7] Skipping edges CSV (SAVE_EDGES_CSV=False)", LOGS_TXT)
    
    # Save node map
    log_message("\n[7/7] Building and saving CSR matrix...", LOGS_TXT)
    t5 = time.perf_counter()
    try:
        with open(MAP_JSON, "w") as f:
            json.dump(node_map, f, indent=2)
        elapsed = time.perf_counter() - t5
        file_size_kb = MAP_JSON.stat().st_size / 1024
        log_message(
            f"  Saved: {MAP_JSON}\n"
            f"  Size: {file_size_kb:.1f} KB\n"
            f"  Time: {elapsed:.2f}s",
            LOGS_TXT
        )
    except Exception as e:
        log_message(f"ERROR: Failed to save node map: {e}", LOGS_TXT)
        raise
    
    # Step 8: Build CSR matrix
    log_message("\n[8/8] Building weighted CSR adjacency matrix...", LOGS_TXT)
    t6 = time.perf_counter()
    
    # Map to integer indices
    i_idx = edges["source"].map(node_map).to_numpy(dtype=np.int32)
    j_idx = edges["target"].map(node_map).to_numpy(dtype=np.int32)
    
    # Normalize combined_score from [0,1000] to [0,1]
    weight = (edges["combined_score"].to_numpy(dtype=np.float32)) / 1000.0
    
    # Build undirected adjacency (add both directions)
    row = np.concatenate([i_idx, j_idx])
    col = np.concatenate([j_idx, i_idx])
    data = np.concatenate([weight, weight]).astype(np.float32)
    
    A = sparse.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    
    try:
        sparse.save_npz(str(CSR_NPZ), A)
        elapsed = time.perf_counter() - t6
        file_size_mb = CSR_NPZ.stat().st_size / 1024 / 1024
        log_message(
            f"  Saved: {CSR_NPZ}\n"
            f"  NNZ (non-zero): {format_number(A.nnz)}\n"
            f"  Size: {file_size_mb:.2f} MB\n"
            f"  Time: {elapsed:.2f}s",
            LOGS_TXT
        )
    except Exception as e:
        log_message(f"ERROR: Failed to save CSR matrix: {e}", LOGS_TXT)
        raise
    
    # ========================
    # COMPUTE STATISTICS
    # ========================
    log_message("\n[STATS] Computing network statistics...", LOGS_TXT)
    t7 = time.perf_counter()
    
    undirected_edges = A.nnz // 2
    avg_nnz = float(A.nnz) / n_nodes
    
    # Density calculation
    density = (2.0 * undirected_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    
    # Degree statistics from CSR format
    degrees = np.diff(A.indptr).astype(np.int32)
    deg_min = int(degrees.min()) if degrees.size else 0
    deg_max = int(degrees.max()) if degrees.size else 0
    deg_mean = float(degrees.mean()) if degrees.size else 0.0
    deg_median = float(np.median(degrees)) if degrees.size else 0.0
    deg_std = float(degrees.std()) if degrees.size else 0.0
    
    # Connected components (rough estimate using BFS from first node)
    try:
        from scipy.sparse.csgraph import connected_components as sp_components
        n_components, labels = sp_components(A, directed=False, return_labels=True)
    except:
        n_components = 0
    
    # Score statistics
    score_vals = edges["combined_score"].to_numpy(dtype=np.float32)
    score_min, score_q25, score_med, score_q75, score_max = np.quantile(
        score_vals, [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    score_mean = float(score_vals.mean())
    score_std = float(score_vals.std())
    
    elapsed = time.perf_counter() - t7
    
    # ========================
    # COMPILE STATISTICS
    # ========================
    
    stats_dict = {
        "metadata": {
            "algorithm": "MCL (Markov Clustering)",
            "database": "STRING v12.0 Human PPI",
            "timestamp": datetime.now().isoformat(),
            "organism": "Homo sapiens (NCBI ID: 9606)"
        },
        "input": {
            "raw_file": RAW_FILE,
            "file_size_mb": RAW_FILE if Path(RAW_FILE).exists() 
                else "N/A (file not found)"
        },
        "parameters": {
            "score_minimum": SCORE_MIN,
            "max_edges": MAX_EDGES if MAX_EDGES else "unlimited"
        },
        "output_files": {
            "csr_matrix": str(CSR_NPZ),
            "node_mapping": str(MAP_JSON),
            "edge_list": str(EDGES_CSV) if SAVE_EDGES_CSV else "not saved",
            "statistics": str(STATS_TXT),
            "logs": str(LOGS_TXT)
        },
        "network": {
            "nodes": int(n_nodes),
            "undirected_edges": int(undirected_edges),
            "csr_nnz_directed": int(A.nnz),
            "density": float(density),
            "connected_components": int(n_components) if n_components > 0 else "unknown"
        },
        "degrees": {
            "min": int(deg_min),
            "max": int(deg_max),
            "mean": float(deg_mean),
            "median": float(deg_median),
            "std_dev": float(deg_std)
        },
        "edge_weights": {
            "combined_score_min": float(score_min),
            "combined_score_quantile_25": float(score_q25),
            "combined_score_median": float(score_med),
            "combined_score_quantile_75": float(score_q75),
            "combined_score_max": float(score_max),
            "combined_score_mean": float(score_mean),
            "combined_score_std": float(score_std),
            "normalized_to_0_1": True
        },
        "performance": {
            "total_time_seconds": float(time.perf_counter() - t_total),
            "peak_memory_mb": float(get_memory_usage()),
            "statistics_time_seconds": float(elapsed)
        }
    }
    
    # ========================
    # SAVE STATISTICS
    # ========================
    
    stats_text = (
        f"{'='*70}\n"
        f"STRING HUMAN PPI NETWORK FOR MCL CLUSTERING\n"
        f"{'='*70}\n\n"
        
        f"METADATA\n"
        f"{'-'*70}\n"
        f"Algorithm:        MCL (Markov Clustering)\n"
        f"Database:         STRING v12.0 Human PPI\n"
        f"Organism:         Homo sapiens (NCBI ID: 9606)\n"
        f"Generated:        {stats_dict['metadata']['timestamp']}\n\n"
        
        f"INPUT\n"
        f"{'-'*70}\n"
        f"Raw file:         {RAW_FILE}\n"
        f"Score threshold:  {SCORE_MIN}\n"
        f"Max edges cap:    {MAX_EDGES if MAX_EDGES else 'None (unlimited)'}\n\n"
        
        f"OUTPUT FILES\n"
        f"{'-'*70}\n"
        f"CSR Matrix:       {CSR_NPZ}\n"
        f"Node Mapping:     {MAP_JSON}\n"
        f"Edge List CSV:    {EDGES_CSV if SAVE_EDGES_CSV else 'Not saved'}\n"
        f"Statistics:       {STATS_TXT}\n"
        f"Preprocessing Log: {LOGS_TXT}\n\n"
        
        f"NETWORK STATISTICS\n"
        f"{'-'*70}\n"
        f"Nodes:                        {format_number(n_nodes)}\n"
        f"Undirected edges:             {format_number(undirected_edges)}\n"
        f"CSR matrix NNZ (directed):    {format_number(A.nnz)}\n"
        f"Network density:              {density:.8f}\n"
        f"Connected components:         {n_components if n_components > 0 else 'unknown'}\n\n"
        
        f"DEGREE STATISTICS (CSR row perspective)\n"
        f"{'-'*70}\n"
        f"Minimum:                      {deg_min}\n"
        f"Maximum:                      {deg_max}\n"
        f"Mean:                         {deg_mean:.2f}\n"
        f"Median:                       {deg_median:.2f}\n"
        f"Std Dev:                      {deg_std:.2f}\n\n"
        
        f"EDGE WEIGHT STATISTICS (combined_score)\n"
        f"{'-'*70}\n"
        f"Minimum:                      {score_min:.2f}\n"
        f"25th percentile:              {score_q25:.2f}\n"
        f"Median (50th):                {score_med:.2f}\n"
        f"75th percentile:              {score_q75:.2f}\n"
        f"Maximum:                      {score_max:.2f}\n"
        f"Mean:                         {score_mean:.2f}\n"
        f"Std Dev:                      {score_std:.2f}\n"
        f"Normalization range:          [0, 1] (original [0, 1000])\n\n"
        
        f"PROCESSING METRICS\n"
        f"{'-'*70}\n"
        f"Total time:                   {time.perf_counter() - t_total:.2f} seconds\n"
        f"Peak memory usage:            {get_memory_usage():.1f} MB\n"
        f"Statistics computation time:  {elapsed:.2f} seconds\n\n"
        
        f"NOTES\n"
        f"{'-'*70}\n"
        f"- Network is represented as undirected (symmetric CSR matrix)\n"
        f"- Self-loops have been removed\n"
        f"- Duplicate edges kept with maximum combined_score\n"
        f"- Edge weights normalized to [0,1] range for MCL algorithm\n"
        f"- All proteins are homo sapiens with 9606. prefix\n"
        f"{'='*70}\n"
    )
    
    # Save stats file
    with open(STATS_TXT, "w") as f:
        f.write(stats_text)
    
    # Save JSON stats
    stats_json_file = RESULTS_DIR / "stats.json"
    with open(stats_json_file, "w") as f:
        json.dump(stats_dict, f, indent=2)
    
    # Print summary
    log_message(f"\n{stats_text}", LOGS_TXT)
    
    log_message(
        f"\n✓ PREPROCESSING COMPLETE\n"
        f"  - All output files saved to: {OUT_DIR} and {RESULTS_DIR}\n"
        f"  - JSON stats available: {stats_json_file}\n"
        f"  - Full logs available: {LOGS_TXT}\n",
        LOGS_TXT
    )
    
    # Print to console
    print(stats_text)
    print(f"\n✓ JSON stats: {stats_json_file}")
    print(f"✓ Full logs: {LOGS_TXT}")
    
    tracemalloc.stop()


if __name__ == "__main__":
    main()
