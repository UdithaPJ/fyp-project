"""
src/benchmarking/memory_benchmark.py
======================================

Memory-efficiency benchmark comparing ``gpu_baseline`` against the
optimised ``gpu`` implementation for each (algorithm, dataset) pair.

What is measured
----------------
peak_vram_mb
    Estimated peak GPU VRAM consumption during the run.  Taken from
    ``_memory_estimate.total_mb`` injected by :func:`apply_config` —
    this is the predicted working-set size at the time the algorithm
    launches, *not* the post-run residual (which is 0 for both impls
    because both free their temporaries).

peak_ram_mb
    Peak host-RAM delta measured by sampling the process working-set
    (Windows) or ``/proc/self/status`` (Linux/macOS) before and after
    the algorithm call.

runtime_s
    Algorithm wall-clock time in seconds (CUDA-event accurate for GPU
    modes via the runner's :class:`BenchmarkTimer`).

max_graph_edges_supported
    Estimated maximum number of edges the implementation can handle
    within ``available_vram_mb``.  Derived analytically:

        bytes_per_edge_baseline  = CSR_BYTES + cuGraph_overhead_factor
        bytes_per_edge_optimized = CSR_BYTES (custom kernels, no cuGraph
                                   temporary frame, chunking enables
                                   graphs > VRAM)

    ``max_edges_baseline  = (available_vram_mb * 1e6) / bytes_per_edge_baseline``
    ``max_edges_optimized = (available_vram_mb * 1e6) / bytes_per_edge_optimized``

    For the optimised implementation, chunking removes the hard VRAM
    ceiling, so max_edges_optimized is multiplied by a ``chunking_factor``
    (ratio of available_vram to single-chunk budget, capped at
    ``MAX_CHUNKING_FACTOR``) when ``needs_chunking`` is True or when the
    algorithm's chunked path is known to be implemented.

Derived metrics
---------------
vram_reduction_pct
    ``(peak_vram_baseline − peak_vram_gpu) / peak_vram_baseline * 100``
    Positive means the optimised GPU uses less peak VRAM.

graph_capacity_gain
    ``max_edges_optimized / max_edges_baseline``
    > 1 means the optimised GPU can handle larger graphs.

runtime_overhead_pct
    ``(runtime_gpu − runtime_baseline) / runtime_baseline * 100``
    Negative means the optimised GPU is *faster* (common); positive means
    the custom kernels have overhead on this particular graph size.

Outputs
-------
    experiments/outputs/reports/memory_benchmark.csv
        columns: algorithm, dataset, mode, n_nodes, n_edges,
                 peak_vram_mb, peak_ram_mb, runtime_s,
                 max_graph_edges_supported,
                 vram_reduction_pct, graph_capacity_gain, runtime_overhead_pct

    experiments/outputs/plots/
        memory_usage.png          – peak_vram + peak_ram bars per algo
        max_graph_size.png        – max_edges_supported comparison
        vram_comparison.png       – baseline vs optimised VRAM side-by-side
        optimization_memory_gain.png – derived metrics: reduction & gain
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
from typing import Any, Iterable, Optional

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

MODES: tuple[str, ...] = ("gpu_baseline", "gpu")

_MODE_LABELS: dict[str, str] = {
    "gpu_baseline": "GPU Baseline",
    "gpu":          "GPU Optimised",
}

_MODE_COLOURS: dict[str, str] = {
    "gpu_baseline": "#bd10e0",
    "gpu":          "#7ed321",
}

# CSR edge storage cost in bytes: (int32 col_idx + float32 value) per nnz
#   + (int32 row_ptr) per node  ≈ 8 bytes/edge dominant for large graphs
_CSR_BYTES_PER_EDGE: float = 8.0

# cuGraph builds internal COO + vertex-renaming tables; empirically this
# is ~3-4× the raw CSR size for typical algorithms.
_CUGRAPH_OVERHEAD_FACTOR: float = 3.5

# Algorithms whose optimised implementation has a fully-implemented chunked
# execution path (see CLAUDE.md milestone 3.1).
_CHUNKED_ALGOS: frozenset[str] = frozenset({"pagerank", "rwr", "louvain"})

# Cap on how many chunk-widths the chunked path can extend max_edges.
MAX_CHUNKING_FACTOR: float = 8.0

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


# ---------------------------------------------------------------------------
# Host-RAM measurement (shared with scalability_benchmark)
# ---------------------------------------------------------------------------

import ctypes
import ctypes.wintypes as _W
import platform as _platform

_IS_WINDOWS = _platform.system() == "Windows"

if _IS_WINDOWS:
    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb",                      _W.DWORD),
            ("PageFaultCount",          _W.DWORD),
            ("PeakWorkingSetSize",      ctypes.c_size_t),
            ("WorkingSetSize",          ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage",     ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage",  ctypes.c_size_t),
            ("QuotaPagedPoolUsage2",    ctypes.c_size_t),
            ("PagefileUsage",           ctypes.c_size_t),
            ("PeakPagefileUsage",       ctypes.c_size_t),
        ]
    _K32   = ctypes.windll.kernel32
    _PSAPI = ctypes.windll.psapi

    def _rss_mb() -> float:
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        h = _K32.OpenProcess(0x0400 | 0x0010, False, _K32.GetCurrentProcessId())
        _PSAPI.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        _K32.CloseHandle(h)
        return pmc.WorkingSetSize / 1024 ** 2
else:
    def _rss_mb() -> float:  # type: ignore[misc]
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except Exception:
            pass
        return 0.0


# ---------------------------------------------------------------------------
# VRAM measurement helpers
# ---------------------------------------------------------------------------

def _free_vram_mb() -> float:
    """Return current free GPU VRAM in MB.  Returns 0 if no GPU available."""
    try:
        import cupy as cp
        free, _ = cp.cuda.runtime.memGetInfo()
        return free / 1024 ** 2
    except Exception:
        pass
    try:
        import pycuda.driver as cuda
        cuda.init()
        if cuda.Device.count() > 0:
            ctx = cuda.Device(0).retain_primary_context()
            ctx.push()
            free, _ = cuda.mem_get_info()
            ctx.pop()
            return free / 1024 ** 2
    except Exception:
        pass
    return 0.0


def _total_vram_mb() -> float:
    try:
        import cupy as cp
        _, total = cp.cuda.runtime.memGetInfo()
        return total / 1024 ** 2
    except Exception:
        pass
    try:
        import pycuda.driver as cuda
        cuda.init()
        if cuda.Device.count() > 0:
            ctx = cuda.Device(0).retain_primary_context()
            ctx.push()
            _, total = cuda.mem_get_info()
            ctx.pop()
            return total / 1024 ** 2
    except Exception:
        pass
    return 0.0


def _get_vram_estimate(algorithm: str, graph_csr: sp.csr_matrix,
                       params: dict) -> dict[str, float]:
    """Return the MemoryEstimator prediction for this algorithm + graph."""
    try:
        from src.optimization.gpu_config import apply_config
        p = apply_config(algorithm, graph_csr, dict(params))
        est = p.get("_memory_estimate", {})
        return {
            "total_mb":       float(est.get("total_mb", 0.0)),
            "base_mb":        float(est.get("base_mb", 0.0)),
            "algorithm_mb":   float(est.get("algorithm_mb", 0.0)),
            "available_mb":   float(est.get("available_mb", 0.0)),
            "needs_chunking": bool(est.get("needs_chunking", False)),
            "pressure":       str(est.get("pressure", "unknown")),
        }
    except Exception as exc:
        _LOG.debug("_get_vram_estimate failed: %s", exc)
        return {"total_mb": 0.0, "base_mb": 0.0, "algorithm_mb": 0.0,
                "available_mb": 0.0, "needs_chunking": False,
                "pressure": "unknown"}


def _compute_max_edges(
    algorithm: str,
    available_vram_mb: float,
    n_nodes: int,
    mode: str,
) -> int:
    """Analytically estimate the maximum number of edges that fit in VRAM.

    gpu_baseline
        Uses cuGraph which needs ~3.5× the raw CSR bytes for its internal
        tables.  No chunking.

    gpu (optimised)
        Uses custom kernels.  Raw bytes ≈ _CSR_BYTES_PER_EDGE per edge.
        Algorithms with a chunked path can exceed the single-pass limit by
        up to ``MAX_CHUNKING_FACTOR``; for others the limit equals the
        memory-pressure threshold (70 % of available VRAM).
    """
    available_bytes = available_vram_mb * 1024 ** 2

    if mode == "gpu_baseline":
        # node-side pointer array + cuGraph frame
        node_bytes = n_nodes * 4 * 2  # row_ptr + internal vertex array
        bytes_per_edge = _CSR_BYTES_PER_EDGE * _CUGRAPH_OVERHEAD_FACTOR
        usable = max(available_bytes - node_bytes, 0.0)
        return max(1, int(usable / bytes_per_edge))

    else:  # mode == "gpu" (optimised)
        # Custom kernels: raw CSR storage + algorithm working memory
        # Working memory multipliers from MemoryEstimator (CLAUDE.md):
        _ALGO_MULTIPLIERS = {
            "pagerank": 1.5, "bfs": 1.3, "rwr": 1.6,
            "louvain": 3.0, "hits": 2.5, "mcl": 4.0,
        }
        mult = _ALGO_MULTIPLIERS.get(algorithm, 2.0)
        node_bytes = n_nodes * 4 * 2
        bytes_per_edge = _CSR_BYTES_PER_EDGE * mult
        usable = max(available_bytes * 0.70 - node_bytes, 0.0)
        base_max = max(1, int(usable / bytes_per_edge))

        if algorithm in _CHUNKED_ALGOS:
            # Chunked path: full-size score arrays stay on GPU; only CSR
            # rows stream in chunks.  Max edges ≈ base_max × chunking_factor.
            node_only_bytes = n_nodes * 4 * 4   # score / new_score / p0 + mask
            bytes_per_edge_chunk = _CSR_BYTES_PER_EDGE  # raw CSR only
            chunk_budget = max(available_bytes * 0.70 - node_only_bytes, 0.0)
            chunk_size_edges = max(1, int(chunk_budget / bytes_per_edge_chunk))
            # Number of chunks that fit in one pass defines how many total
            # edges can be processed:
            chunking_factor = min(
                MAX_CHUNKING_FACTOR,
                max(1.0, chunk_budget / (bytes_per_edge_chunk * max(1, base_max // 4))),
            )
            return max(base_max, int(base_max * chunking_factor))

        return base_max


# ---------------------------------------------------------------------------
# Measurement record
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    algorithm:              str
    dataset:                str
    mode:                   str
    n_nodes:                int
    n_edges:                int
    peak_vram_mb:           float   # estimated from apply_config
    peak_ram_mb:            float   # measured host-RAM delta
    runtime_s:              float
    max_graph_edges_supported: int
    success:                bool
    error:                  Optional[str] = None


# ---------------------------------------------------------------------------
# MemoryBenchmarker
# ---------------------------------------------------------------------------

@dataclass
class MemoryDataset:
    name:            str
    graph_csr:       sp.csr_matrix
    network_type:    str = "grn"
    params_override: dict[str, dict[str, Any]] = field(default_factory=dict)
    node_index_map:  dict[str, int] = field(default_factory=dict)

    def params_for(self, algorithm: str) -> dict:
        base = dict(_DEFAULT_PARAMS.get(algorithm, {}))
        base["network_type"] = self.network_type
        base.update(self.params_override.get(algorithm, {}))
        return base


class MemoryBenchmarker:
    """Memory-efficiency benchmark: ``gpu_baseline`` vs ``gpu``.

    Usage
    -----
    .. code-block:: python

        mb = MemoryBenchmarker(output_dir="experiments/outputs")
        mb.add_dataset("my_graph", graph_csr, network_type="ppi")
        mb.run(algorithms=["pagerank", "bfs"])
        mb.write_csv()
        mb.write_plots()
    """

    def __init__(
        self,
        output_dir: str | Path = "experiments/outputs",
        algorithms: Iterable[str] = ALGORITHMS,
    ) -> None:
        self.output_dir   = Path(output_dir)
        self.reports_dir  = self.output_dir / "reports"
        self.plots_dir    = self.output_dir / "plots"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.algorithms   = tuple(algorithms)
        self.datasets:    list[MemoryDataset] = []
        self.records:     list[MemoryRecord]  = []

    def add_dataset(
        self,
        name: str,
        graph_csr: sp.csr_matrix,
        network_type: str = "grn",
        params_override: Optional[dict[str, dict[str, Any]]] = None,
        node_index_map: Optional[dict[str, int]] = None,
    ) -> None:
        if not sp.isspmatrix_csr(graph_csr):
            graph_csr = graph_csr.tocsr()
        self.datasets.append(MemoryDataset(
            name=str(name),
            graph_csr=graph_csr,
            network_type=str(network_type).lower(),
            params_override=params_override or {},
            node_index_map=node_index_map or {},
        ))

    # ---- core run ----

    def run(
        self,
        algorithms: Optional[Iterable[str]] = None,
        datasets:   Optional[Iterable[str]] = None,
    ) -> None:
        algos   = tuple(algorithms) if algorithms else self.algorithms
        ds_list = [d for d in self.datasets
                   if datasets is None or d.name in set(datasets)]
        if not ds_list:
            raise RuntimeError("MemoryBenchmarker.run: no datasets registered.")

        from src.runner.algorithm_runner import run_algorithm

        self.records = []
        available_vram = _free_vram_mb()
        _LOG.info("MemoryBenchmarker: available VRAM = %.1f MB", available_vram)

        for dataset in ds_list:
            n_nodes = int(dataset.graph_csr.shape[0])
            n_edges = int(dataset.graph_csr.nnz)
            for algorithm in algos:
                params = dataset.params_for(algorithm)
                for mode in MODES:
                    _LOG.info("memory_bench: %s / %s / %s",
                              algorithm, dataset.name, mode)
                    gc.collect()
                    # VRAM prediction
                    vram_est = _get_vram_estimate(
                        algorithm, dataset.graph_csr, params
                    )
                    peak_vram = vram_est["total_mb"]

                    # Host-RAM measurement
                    ram_before = _rss_mb()
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            envelope = run_algorithm(
                                algorithm_name=algorithm,
                                graph_csr=dataset.graph_csr,
                                node_index_map=dataset.node_index_map,
                                mode=mode,
                                params=params,
                            )
                        ram_after  = _rss_mb()
                        peak_ram   = max(ram_after - ram_before, 0.0)
                        runtime    = float(envelope.get("execution_time", 0.0))
                        max_edges  = _compute_max_edges(
                            algorithm, available_vram, n_nodes, mode
                        )
                        self.records.append(MemoryRecord(
                            algorithm=algorithm,
                            dataset=dataset.name,
                            mode=mode,
                            n_nodes=n_nodes,
                            n_edges=n_edges,
                            peak_vram_mb=peak_vram,
                            peak_ram_mb=peak_ram,
                            runtime_s=runtime,
                            max_graph_edges_supported=max_edges,
                            success=True,
                        ))
                    except Exception as exc:
                        err = f"{type(exc).__name__}: {exc}"
                        _LOG.warning("memory_bench %s/%s/%s: %s",
                                     algorithm, dataset.name, mode, err)
                        max_edges = _compute_max_edges(
                            algorithm, available_vram, n_nodes, mode
                        )
                        self.records.append(MemoryRecord(
                            algorithm=algorithm,
                            dataset=dataset.name,
                            mode=mode,
                            n_nodes=n_nodes,
                            n_edges=n_edges,
                            peak_vram_mb=peak_vram,
                            peak_ram_mb=0.0,
                            runtime_s=float("nan"),
                            max_graph_edges_supported=max_edges,
                            success=False,
                            error=err,
                        ))

    # ---- derived metrics ----

    def _derived(self) -> list[dict]:
        """Build rows with all raw + derived columns."""
        # key: (algorithm, dataset, mode) → record
        lookup: dict[tuple, MemoryRecord] = {
            (r.algorithm, r.dataset, r.mode): r for r in self.records
        }
        rows: list[dict] = []
        for r in self.records:
            base_rec = lookup.get((r.algorithm, r.dataset, "gpu_baseline"))
            gpu_rec  = lookup.get((r.algorithm, r.dataset, "gpu"))

            vram_reduction   = float("nan")
            cap_gain         = float("nan")
            runtime_overhead = float("nan")

            if base_rec and gpu_rec:
                if base_rec.peak_vram_mb > 0:
                    vram_reduction = (
                        (base_rec.peak_vram_mb - gpu_rec.peak_vram_mb)
                        / base_rec.peak_vram_mb * 100.0
                    )
                if base_rec.max_graph_edges_supported > 0:
                    cap_gain = (
                        gpu_rec.max_graph_edges_supported
                        / base_rec.max_graph_edges_supported
                    )
                if (base_rec.success and gpu_rec.success
                        and math.isfinite(base_rec.runtime_s)
                        and base_rec.runtime_s > 0
                        and math.isfinite(gpu_rec.runtime_s)):
                    runtime_overhead = (
                        (gpu_rec.runtime_s - base_rec.runtime_s)
                        / base_rec.runtime_s * 100.0
                    )

            rows.append({
                "algorithm":               r.algorithm,
                "dataset":                 r.dataset,
                "mode":                    r.mode,
                "n_nodes":                 r.n_nodes,
                "n_edges":                 r.n_edges,
                "peak_vram_mb":            r.peak_vram_mb,
                "peak_ram_mb":             r.peak_ram_mb,
                "runtime_s":               r.runtime_s,
                "max_graph_edges_supported": r.max_graph_edges_supported,
                "vram_reduction_pct":      vram_reduction,
                "graph_capacity_gain":     cap_gain,
                "runtime_overhead_pct":    runtime_overhead,
            })
        return rows

    # ---- CSV ----

    def write_csv(self, path: Optional[str | Path] = None) -> Path:
        out = Path(path) if path else (
            self.reports_dir / "memory_benchmark.csv"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = self._derived()
        fieldnames = [
            "algorithm", "dataset", "mode", "n_nodes", "n_edges",
            "peak_vram_mb", "peak_ram_mb", "runtime_s",
            "max_graph_edges_supported",
            "vram_reduction_pct", "graph_capacity_gain",
            "runtime_overhead_pct",
        ]
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow({
                    k: (f"{v:.4f}" if isinstance(v, float) and math.isfinite(v)
                        else ("" if isinstance(v, float) else v))
                    for k, v in row.items()
                })
        _LOG.info("MemoryBenchmarker: wrote %d rows to %s", len(rows), out)
        return out

    # ---- plots ----

    def write_plots(self) -> dict[str, Path]:
        return {
            "memory_usage":              self._plot_memory_usage(),
            "max_graph_size":            self._plot_max_graph_size(),
            "vram_comparison":           self._plot_vram_comparison(),
            "optimization_memory_gain":  self._plot_opt_gain(),
        }

    # ---- plot helpers ----

    def _agg_by_algo_mode(
        self, metric: str
    ) -> dict[tuple[str, str], float]:
        """Average ``metric`` over datasets, keyed by (algo, mode)."""
        buckets: dict[tuple[str, str], list[float]] = {}
        for r in self.records:
            v = getattr(r, metric, float("nan"))
            if math.isfinite(float(v)):
                buckets.setdefault((r.algorithm, r.mode), []).append(float(v))
        return {k: float(np.mean(vs)) for k, vs in buckets.items() if vs}

    def _plot_memory_usage(self) -> Path:
        """Grouped bar: peak_vram_mb + peak_ram_mb per algorithm × mode."""
        outpath = self.plots_dir / "memory_usage.png"
        algos   = sorted({r.algorithm for r in self.records})
        modes   = [m for m in MODES if m in {r.mode for r in self.records}]
        if not algos:
            return _blank_plot(outpath, "memory_usage: no data")

        vram = self._agg_by_algo_mode("peak_vram_mb")
        ram  = self._agg_by_algo_mode("peak_ram_mb")

        x      = np.arange(len(algos))
        width  = 0.35
        n_mode = len(modes)
        fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.8 * len(algos)), 5))

        for ax_idx, (title, data) in enumerate([
            ("Peak VRAM estimate (MB)", vram),
            ("Peak host-RAM delta (MB)", ram),
        ]):
            ax = axes[ax_idx]
            for mi, mode in enumerate(modes):
                vals = [data.get((a, mode), np.nan) for a in algos]
                offset = (mi - (n_mode - 1) / 2) * width
                bars = ax.bar(x + offset, vals, width,
                              label=_MODE_LABELS.get(mode, mode),
                              color=_MODE_COLOURS.get(mode, "#888"),
                              edgecolor="white", linewidth=0.4)
                for bar, v in zip(bars, vals):
                    if np.isfinite(v) and v > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() * 1.03,
                                f"{v:.1f}", ha="center",
                                va="bottom", fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(algos, rotation=25, ha="right")
            ax.set_title(title, fontsize=11)
            ax.set_ylabel("MB", fontsize=9)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            ax.legend(fontsize=8)

        fig.suptitle("Memory usage: GPU baseline vs GPU optimised", fontsize=13)
        fig.tight_layout()
        fig.savefig(outpath, dpi=120)
        plt.close(fig)
        return outpath

    def _plot_max_graph_size(self) -> Path:
        """Bar chart: max_graph_edges_supported per algorithm × mode."""
        outpath  = self.plots_dir / "max_graph_size.png"
        algos    = sorted({r.algorithm for r in self.records})
        modes    = [m for m in MODES if m in {r.mode for r in self.records}]
        if not algos:
            return _blank_plot(outpath, "max_graph_size: no data")

        data     = self._agg_by_algo_mode("max_graph_edges_supported")
        x        = np.arange(len(algos))
        width    = 0.35
        n_mode   = len(modes)
        fig, ax  = plt.subplots(figsize=(max(8, 1.6 * len(algos)), 5))

        for mi, mode in enumerate(modes):
            vals   = [data.get((a, mode), np.nan) for a in algos]
            offset = (mi - (n_mode - 1) / 2) * width
            bars   = ax.bar(x + offset, vals, width,
                            label=_MODE_LABELS.get(mode, mode),
                            color=_MODE_COLOURS.get(mode, "#888"),
                            edgecolor="white", linewidth=0.4)
            for bar, v in zip(bars, vals):
                if np.isfinite(v) and v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.02,
                            _fmt_edges(v), ha="center",
                            va="bottom", fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=25, ha="right")
        ax.set_title(
            "Max graph edges supported within available VRAM\n"
            "(optimised GPU benefits from chunking on pagerank/rwr/louvain)",
            fontsize=11,
        )
        ax.set_ylabel("max edges supported")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(outpath, dpi=120)
        plt.close(fig)
        return outpath

    def _plot_vram_comparison(self) -> Path:
        """Side-by-side VRAM bars: baseline vs optimised, annotated with
        the percentage reduction.
        """
        outpath = self.plots_dir / "vram_comparison.png"
        algos   = sorted({r.algorithm for r in self.records})
        modes   = [m for m in MODES if m in {r.mode for r in self.records}]
        if not algos:
            return _blank_plot(outpath, "vram_comparison: no data")

        vram     = self._agg_by_algo_mode("peak_vram_mb")
        x        = np.arange(len(algos))
        width    = 0.35
        n_mode   = len(modes)
        fig, ax  = plt.subplots(figsize=(max(8, 1.6 * len(algos)), 5))

        for mi, mode in enumerate(modes):
            vals   = [vram.get((a, mode), np.nan) for a in algos]
            offset = (mi - (n_mode - 1) / 2) * width
            bars   = ax.bar(x + offset, vals, width,
                            label=_MODE_LABELS.get(mode, mode),
                            color=_MODE_COLOURS.get(mode, "#888"),
                            edgecolor="white", linewidth=0.4)
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.03,
                            f"{v:.1f}MB", ha="center",
                            va="bottom", fontsize=7)

        # Annotate % reduction per algorithm
        for ai, algo in enumerate(algos):
            base_v = vram.get((algo, "gpu_baseline"), float("nan"))
            opt_v  = vram.get((algo, "gpu"),          float("nan"))
            if math.isfinite(base_v) and math.isfinite(opt_v) and base_v > 0:
                pct = (base_v - opt_v) / base_v * 100.0
                colour = "#7ed321" if pct >= 0 else "#e74c3c"
                ax.text(x[ai], max(base_v, opt_v) * 1.15,
                        f"{pct:+.0f}%", ha="center",
                        va="bottom", fontsize=9,
                        color=colour, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=25, ha="right")
        ax.set_title("Peak VRAM: GPU baseline vs GPU optimised\n"
                     "(annotation = % reduction by optimised)",
                     fontsize=11)
        ax.set_ylabel("estimated peak VRAM (MB)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(outpath, dpi=120)
        plt.close(fig)
        return outpath

    def _plot_opt_gain(self) -> Path:
        """Bar chart of the three derived metrics:
        vram_reduction_pct, graph_capacity_gain, runtime_overhead_pct.
        One subplot per metric; bars = algorithms.
        """
        outpath = self.plots_dir / "optimization_memory_gain.png"
        algos   = sorted({r.algorithm for r in self.records})
        if not algos:
            return _blank_plot(outpath, "optimization_memory_gain: no data")

        derived_lookup: dict[str, dict[str, float]] = {
            "vram_reduction_pct":   {},
            "graph_capacity_gain":  {},
            "runtime_overhead_pct": {},
        }
        # Use gpu rows for the derived metrics (they're symmetric)
        for row in self._derived():
            if row["mode"] != "gpu":
                continue
            algo = row["algorithm"]
            for metric in derived_lookup:
                v = row[metric]
                if math.isfinite(float(v)) if isinstance(v, float) else False:
                    derived_lookup[metric][algo] = float(v)

        metrics_cfg = [
            ("vram_reduction_pct",   "VRAM reduction (%)",
             "Positive = optimised uses less VRAM", "#7ed321"),
            ("graph_capacity_gain",  "Graph capacity gain (×)",
             "Ratio: max_edges_optimised / max_edges_baseline\n>1 = optimised handles larger graphs",
             "#5b8def"),
            ("runtime_overhead_pct", "Runtime overhead (%)",
             "Negative = optimised is faster\nPositive = optimised has overhead on this size",
             "#f5a623"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        x = np.arange(len(algos))

        for ax, (metric, title, subtitle, colour) in zip(axes, metrics_cfg):
            vals = [derived_lookup[metric].get(a, np.nan) for a in algos]
            colours = []
            for v in vals:
                if not np.isfinite(v):
                    colours.append("#cccccc")
                elif metric == "runtime_overhead_pct":
                    colours.append("#7ed321" if v <= 0 else "#e74c3c")
                elif metric == "vram_reduction_pct":
                    colours.append("#7ed321" if v >= 0 else "#e74c3c")
                else:
                    colours.append(colour)
            bars = ax.bar(x, vals, 0.6, color=colours,
                          edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (
                                max((abs(vv) for vv in vals if np.isfinite(vv)),
                                    default=0) * 0.02),
                            f"{v:.2f}", ha="center",
                            va="bottom", fontsize=8)
            if metric != "graph_capacity_gain":
                ax.axhline(0, color="black", linewidth=0.7,
                           linestyle="--", alpha=0.4)
            else:
                ax.axhline(1.0, color="black", linewidth=0.7,
                           linestyle="--", alpha=0.4, label="1× (no gain)")
            ax.set_xticks(x)
            ax.set_xticklabels(algos, rotation=30, ha="right")
            ax.set_title(f"{title}\n{subtitle}", fontsize=9)
            ax.set_ylabel(title, fontsize=8)
            ax.grid(axis="y", linestyle=":", alpha=0.4)

        fig.suptitle(
            "GPU optimisation memory metrics\n"
            "(all derived by comparing gpu vs gpu_baseline on the same graph)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(outpath, dpi=120)
        plt.close(fig)
        return outpath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_edges(n: float) -> str:
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(int(n))


def _blank_plot(path: Path, msg: str = "no data") -> Path:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


__all__ = [
    "MemoryBenchmarker",
    "MemoryDataset",
    "MemoryRecord",
    "ALGORITHMS",
    "MODES",
]
