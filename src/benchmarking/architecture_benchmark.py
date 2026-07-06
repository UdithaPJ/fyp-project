"""
src/benchmarking/architecture_benchmark.py
===========================================

Architecture parameter sweep benchmark for the **gpu** mode ONLY.

The optimised CUDA implementations expose architecture-aware parameters
(block_size, chunk_size, use_shared_mem, traversal_mode, etc.) that are
set automatically by :func:`apply_config`.  This benchmarker sweeps
selected parameter values to characterise the effect of block-size and
chunking choices on:

    runtime_s          – wall-clock time in seconds (CUDA-event accurate)
    occupancy_proxy    – estimated occupancy  = block_size / max_threads_per_block
                         (0-1 float; proxy because true occupancy requires
                          a device-side query not available on all platforms)
    peak_vram_mb       – predicted peak VRAM from apply_config._memory_estimate

CSV output
----------
    experiments/outputs/reports/architecture_benchmark.csv
        columns: algorithm, dataset, block_size, chunk_size,
                 runtime_s, occupancy_proxy, peak_vram_mb, success, error

Summary keys (available on the ArchitectureBenchmarker instance)
    best_block_size     – block_size with lowest mean runtime (per algorithm)
    best_chunk_size     – chunk_size with lowest mean runtime (per algorithm)
    best_runtime        – the corresponding runtime_s value

Plots
-----
    experiments/outputs/plots/
        blocksize_vs_runtime.png   – runtime vs block_size line per algorithm
        optimization_heatmap.png   – 2-D heatmap block_size × chunk_size for
                                     runtime, one subplot per algorithm

Notes
-----
* Only the ``gpu`` mode is exercised — architecture-aware params exist only
  in the custom CUDA implementations.
* Block sizes that exceed ``max_threads_per_block`` (hardware limit, default
  1024) are silently skipped.
* Chunk sizes are expressed as a fraction of n (graph nodes): each value in
  ``CHUNK_SIZE_FRACTIONS`` is multiplied by graph n and rounded.
* If the algorithm fails (e.g. nvcc unavailable) the record is stored with
  success=False and NaN metrics.
"""

from __future__ import annotations

import csv
import gc
import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import scipy.sparse as sp

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALGORITHMS: tuple[str, ...] = (
    "pagerank", "bfs", "louvain", "rwr", "hits", "mcl",
)

BLOCK_SIZES: tuple[int, ...] = (32, 64, 128, 256)

# Fraction of graph nodes used as chunk_size for the sweep.
# 0 means "no chunking" (full graph in one pass).
CHUNK_SIZE_FRACTIONS: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)

MAX_THREADS_PER_BLOCK: int = 1024
WARP_SIZE: int = 32

_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "pagerank": {"damping": 0.85, "max_iter": 100, "tolerance": 1e-6},
    "bfs":      {"source": 0, "max_depth": 5},
    "rwr":      {"restart_prob": 0.3, "max_iter": 100,
                 "tolerance": 1e-6, "seed_nodes": [0]},
    "hits":     {"max_iter": 100, "tolerance": 1e-6},
    "louvain":  {"min_delta_q": 1e-4, "max_levels": 10, "resolution": 1.0},
    "mcl":      {"expansion": 2, "inflation": 2.0, "prune_threshold": 0.001,
                 "max_iter": 100, "convergence_tol": 1e-4},
}

# Recommended block sizes from CLAUDE.md for occupancy labelling
_RECOMMENDED_BLOCK_SIZE: dict[str, int] = {
    "pagerank": 128,
    "louvain":  64,
    "rwr":      64,
    "hits":     64,
    "bfs":      64,
    "mcl":      32,
}


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class ArchRecord:
    algorithm:       str
    dataset:         str
    block_size:      int
    chunk_size:      int          # 0 = full graph (no chunking)
    runtime_s:       float = float("nan")
    occupancy_proxy: float = float("nan")
    peak_vram_mb:    float = float("nan")
    success:         bool  = True
    error:           str   = ""


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class ArchDataset:
    name:           str
    graph_csr:      Any           # sp.csr_matrix
    network_type:   str = "ppi"
    params_override: dict = field(default_factory=dict)
    node_index_map:  dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_params(algorithm: str, network_type: str,
                  override: dict, block_size: int, chunk_size: int,
                  n_nodes: int) -> dict:
    """Build a params dict with block_size / chunk_size injected."""
    p: dict[str, Any] = dict(_DEFAULT_PARAMS.get(algorithm, {}))
    p["network_type"] = network_type
    p.update(override.get(algorithm, {}))
    # Inject architecture params (apply_config will merge these)
    p["block_size"] = block_size
    if chunk_size > 0:
        p["use_chunking"] = True
        p["chunk_size"]   = chunk_size
    else:
        p["use_chunking"] = False
        p["chunk_size"]   = n_nodes  # effectively "no chunking"
    return p


def _get_vram_estimate(algorithm: str, graph_csr: sp.csr_matrix,
                       params: dict) -> float:
    """Return predicted peak VRAM in MB from apply_config, or 0."""
    try:
        from src.optimization.gpu_config import apply_config
        merged = apply_config(algorithm, graph_csr, dict(params))
        est = merged.get("_memory_estimate", {})
        return float(est.get("total_mb", 0.0))
    except Exception:
        return 0.0


def _run_gpu(algorithm: str, graph_csr: sp.csr_matrix,
             params: dict,
             node_index_map: Optional[dict] = None) -> tuple[float, bool, str]:
    """
    Run ``algorithm`` in gpu mode.
    Returns ``(runtime_s, success, error_message)``.
    """
    try:
        from src.runner.algorithm_runner import run_algorithm
        result = run_algorithm(
            algorithm, graph_csr,
            node_index_map or {},
            mode="gpu",
            params=params,
        )
        rt = float(result.get("execution_time", float("nan")))
        return rt, True, ""
    except Exception as exc:
        return float("nan"), False, str(exc)[:200]


def _occupancy_proxy(block_size: int) -> float:
    return min(block_size / MAX_THREADS_PER_BLOCK, 1.0)


# ---------------------------------------------------------------------------
# ArchitectureBenchmarker
# ---------------------------------------------------------------------------

class ArchitectureBenchmarker:
    """
    Sweep block_size and chunk_size for each algorithm in ``gpu`` mode.

    Parameters
    ----------
    algorithms:
        Algorithms to benchmark.  Defaults to all six.
    block_sizes:
        Tuple of int block sizes to sweep.
    chunk_size_fractions:
        Fraction of graph n to use as chunk_size (0 = no chunking).
    output_dir:
        Root output directory.  CSV written to ``<output_dir>/reports/``,
        plots to ``<output_dir>/plots/``.
    """

    def __init__(
        self,
        algorithms: tuple[str, ...] = ALGORITHMS,
        block_sizes: tuple[int, ...] = BLOCK_SIZES,
        chunk_size_fractions: tuple[float, ...] = CHUNK_SIZE_FRACTIONS,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.algorithms           = tuple(algorithms)
        self.block_sizes          = tuple(block_sizes)
        self.chunk_size_fractions = tuple(chunk_size_fractions)

        _root = Path(__file__).resolve().parents[2]
        self.output_dir = Path(output_dir) if output_dir else (
            _root / "experiments" / "outputs"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

        self._datasets: list[ArchDataset] = []
        self.records:   list[ArchRecord]  = []

        # Best-param summaries, populated after run()
        self.best_block_size:  dict[str, int]   = {}
        self.best_chunk_size:  dict[str, int]   = {}
        self.best_runtime:     dict[str, float] = {}

    # ------------------------------------------------------------------
    # Dataset registration
    # ------------------------------------------------------------------

    def add_dataset(
        self,
        name: str,
        graph_csr: sp.csr_matrix,
        network_type: str = "ppi",
        params_override: Optional[dict] = None,
        node_index_map: Optional[dict] = None,
    ) -> None:
        self._datasets.append(ArchDataset(
            name            = name,
            graph_csr       = graph_csr,
            network_type    = network_type,
            params_override = params_override or {},
            node_index_map  = node_index_map or {},
        ))

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, algorithms: Optional[tuple] = None) -> None:
        algos = algorithms or self.algorithms
        if not self._datasets:
            _LOG.warning("No datasets registered; nothing to benchmark.")
            return

        for ds in self._datasets:
            n = ds.graph_csr.shape[0]
            _LOG.info("Architecture sweep | dataset=%s  n=%d", ds.name, n)

            for algo in algos:
                _LOG.info("  algorithm=%s", algo)
                valid_blocks = [b for b in self.block_sizes
                                if b <= MAX_THREADS_PER_BLOCK]
                for block_size in valid_blocks:
                    for frac in self.chunk_size_fractions:
                        chunk_size = max(0, int(round(frac * n)))
                        params = _build_params(
                            algo, ds.network_type,
                            ds.params_override, block_size, chunk_size, n,
                        )
                        vram_mb = _get_vram_estimate(algo, ds.graph_csr, params)
                        gc.collect()
                        rt, ok, err = _run_gpu(
                            algo, ds.graph_csr, params,
                            ds.node_index_map,
                        )
                        occ = _occupancy_proxy(block_size)
                        rec = ArchRecord(
                            algorithm       = algo,
                            dataset         = ds.name,
                            block_size      = block_size,
                            chunk_size      = chunk_size,
                            runtime_s       = rt,
                            occupancy_proxy = occ,
                            peak_vram_mb    = vram_mb,
                            success         = ok,
                            error           = err,
                        )
                        self.records.append(rec)
                        _LOG.info(
                            "    block=%d  chunk=%d  rt=%.4fs  ok=%s",
                            block_size, chunk_size, rt, ok,
                        )

        self._compute_best_params()

    # ------------------------------------------------------------------
    # Best-param extraction
    # ------------------------------------------------------------------

    def _compute_best_params(self) -> None:
        for algo in self.algorithms:
            good = [r for r in self.records
                    if r.algorithm == algo and r.success
                    and not math.isnan(r.runtime_s)]
            if not good:
                continue
            best = min(good, key=lambda r: r.runtime_s)
            self.best_block_size[algo] = best.block_size
            self.best_chunk_size[algo] = best.chunk_size
            self.best_runtime[algo]    = best.runtime_s

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------

    def write_csv(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.output_dir / "reports" / "architecture_benchmark.csv"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fields = [
            "algorithm", "dataset", "block_size", "chunk_size",
            "runtime_s", "occupancy_proxy", "peak_vram_mb",
            "success", "error",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.records:
                w.writerow({
                    "algorithm":       r.algorithm,
                    "dataset":         r.dataset,
                    "block_size":      r.block_size,
                    "chunk_size":      r.chunk_size,
                    "runtime_s":       "" if math.isnan(r.runtime_s) else f"{r.runtime_s:.6f}",
                    "occupancy_proxy": f"{r.occupancy_proxy:.4f}",
                    "peak_vram_mb":    "" if math.isnan(r.peak_vram_mb) else f"{r.peak_vram_mb:.2f}",
                    "success":         r.success,
                    "error":           r.error,
                })
        _LOG.info("Architecture CSV: %s", path)
        return path

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def write_plots(self) -> dict[str, Path]:
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        paths["blocksize_vs_runtime"]  = self._plot_blocksize_runtime(plot_dir)
        paths["optimization_heatmap"]  = self._plot_heatmap(plot_dir)
        return paths

    # --- block_size vs runtime (one line per algorithm) ----------------

    def _plot_blocksize_runtime(self, plot_dir: Path) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = plt.cm.tab10
        colors = [cmap(i / max(len(self.algorithms) - 1, 1))
                  for i in range(len(self.algorithms))]

        plotted_any = False
        for idx, algo in enumerate(self.algorithms):
            # Aggregate over all datasets and chunk_sizes: mean runtime per block_size
            bs_rt: dict[int, list[float]] = {}
            for r in self.records:
                if r.algorithm == algo and r.success and not math.isnan(r.runtime_s):
                    bs_rt.setdefault(r.block_size, []).append(r.runtime_s)
            if not bs_rt:
                continue
            xs = sorted(bs_rt)
            ys = [float(np.mean(bs_rt[x])) for x in xs]
            ax.plot(xs, ys, marker="o", label=algo, color=colors[idx])
            # Mark recommended block size
            rec_bs = _RECOMMENDED_BLOCK_SIZE.get(algo)
            if rec_bs and rec_bs in bs_rt:
                rec_y = float(np.mean(bs_rt[rec_bs]))
                ax.scatter([rec_bs], [rec_y], marker="*", s=120,
                           color=colors[idx], zorder=5)
            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "No successful runs", ha="center", va="center",
                    transform=ax.transAxes)

        ax.set_xlabel("Block Size (threads)")
        ax.set_ylabel("Mean Runtime (s)")
        ax.set_title("Block Size vs Runtime by Algorithm (GPU mode)")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xticks(sorted(set(r.block_size for r in self.records)) or BLOCK_SIZES)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out = plot_dir / "blocksize_vs_runtime.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _LOG.info("Plot: %s", out)
        return out

    # --- 2-D heatmap block_size × chunk_size --------------------------

    def _plot_heatmap(self, plot_dir: Path) -> Path:
        algos_with_data = [
            a for a in self.algorithms
            if any(r.algorithm == a and r.success for r in self.records)
        ]
        if not algos_with_data:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No successful runs", ha="center", va="center",
                    transform=ax.transAxes)
            fig.tight_layout()
            out = plot_dir / "optimization_heatmap.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return out

        n_algo = len(algos_with_data)
        ncols = min(3, n_algo)
        nrows = math.ceil(n_algo / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 4 * nrows), squeeze=False)

        for idx, algo in enumerate(algos_with_data):
            ax = axes[idx // ncols][idx % ncols]
            recs = [r for r in self.records
                    if r.algorithm == algo and r.success
                    and not math.isnan(r.runtime_s)]
            if not recs:
                ax.axis("off")
                continue

            # Build grid: rows = block_size, cols = chunk_size
            bsizes  = sorted({r.block_size  for r in recs})
            csizes  = sorted({r.chunk_size  for r in recs})
            grid = np.full((len(bsizes), len(csizes)), np.nan)
            for r in recs:
                bi = bsizes.index(r.block_size)
                ci = csizes.index(r.chunk_size)
                existing = grid[bi, ci]
                # aggregate: take mean over multiple datasets
                if np.isnan(existing):
                    grid[bi, ci] = r.runtime_s
                else:
                    grid[bi, ci] = (existing + r.runtime_s) / 2

            im = ax.imshow(grid, aspect="auto", cmap="viridis_r",
                           interpolation="nearest")
            ax.set_xticks(range(len(csizes)))
            ax.set_xticklabels([str(c) for c in csizes], rotation=45, fontsize=7)
            ax.set_yticks(range(len(bsizes)))
            ax.set_yticklabels([str(b) for b in bsizes], fontsize=8)
            ax.set_xlabel("Chunk Size (nodes)", fontsize=8)
            ax.set_ylabel("Block Size", fontsize=8)
            ax.set_title(algo, fontsize=9)
            plt.colorbar(im, ax=ax, label="Runtime (s)", fraction=0.046)

            # Mark best cell
            best_recs = min(recs, key=lambda r: r.runtime_s)
            bi = bsizes.index(best_recs.block_size)
            ci = csizes.index(best_recs.chunk_size)
            ax.add_patch(plt.Rectangle((ci - 0.5, bi - 0.5), 1, 1,
                                       fill=False, edgecolor="red", lw=2))

        # Hide unused axes
        for idx in range(len(algos_with_data), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        fig.suptitle("Runtime Heatmap: Block Size × Chunk Size (GPU mode)",
                     fontsize=11)
        fig.tight_layout()
        out = plot_dir / "optimization_heatmap.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _LOG.info("Plot: %s", out)
        return out
