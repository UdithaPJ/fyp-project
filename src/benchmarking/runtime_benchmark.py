"""
src/benchmarking/runtime_benchmark.py
======================================

Repeated-run timing benchmark across all four execution modes:

    cpu_single  cpu_multi  gpu_baseline  gpu

For every (algorithm, dataset) combination the benchmarker runs each
mode ``n_runs`` times and records:

    mean_runtime  std_runtime  min_runtime  max_runtime  (all in seconds)

Then computes pairwise speedup ratios:

    speedup_vs_cpu_single   = cpu_single_mean  / mode_mean
    speedup_vs_cpu_multi    = cpu_multi_mean   / mode_mean
    speedup_vs_gpu_baseline = gpu_baseline_mean / gpu_mean   (optimisation gain)

The three speedup columns quantify:

    * CPU parallelisation benefit   (speedup_vs_cpu_single for cpu_multi)
    * GPU acceleration benefit      (speedup_vs_cpu_single for gpu / gpu_baseline)
    * GPU optimisation benefit      (speedup_vs_gpu_baseline for gpu)

Outputs
-------
    experiments/outputs/reports/runtime_benchmark.csv
        columns: algorithm, dataset, mode, n_nodes, n_edges,
                 mean_s, std_s, min_s, max_s,
                 speedup_vs_cpu_single, speedup_vs_cpu_multi,
                 speedup_vs_gpu_baseline

    experiments/outputs/plots/
        runtime_comparison.png       – mean ± std bars per algorithm × mode
        speedup_comparison.png       – speedup bars vs cpu_single
        algorithm_runtime_breakdown.png – stacked/grouped breakdown per algo
        optimization_gain.png        – gpu_baseline vs gpu head-to-head
"""

from __future__ import annotations

import csv
import gc
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np
import scipy.sparse as sp

# MEMORY_FIX (Fix Cat. 6): diagnostic memory logging around each run.
try:
    from src.utils.memory_logger import log_memory, force_gc
except Exception:  # pragma: no cover
    def log_memory(label: str, logger=None) -> None: return None
    def force_gc(label: str = "", logger=None) -> float: return 0.0

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALGORITHMS: tuple[str, ...] = (
    "pagerank", "bfs", "louvain", "rwr", "hits", "mcl",
)

MODES: tuple[str, ...] = ("cpu_single", "cpu_multi", "gpu_baseline", "gpu")

_MODE_LABELS: dict[str, str] = {
    "cpu_single":   "CPU Single",
    "cpu_multi":    "CPU Multi",
    "gpu_baseline": "GPU Baseline",
    "gpu":          "GPU Optimised",
}

_MODE_COLOURS: dict[str, str] = {
    "cpu_single":   "#5b8def",
    "cpu_multi":    "#f5a623",
    "gpu_baseline": "#bd10e0",
    "gpu":          "#7ed321",
}

_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "pagerank": {"damping": 0.85, "max_iter": 100, "tolerance": 1e-6},
    "bfs":      {"source": 0, "max_depth": 999},  # full reachable component
    "rwr":      {"restart_prob": 0.3, "max_iter": 100,
                 "tolerance": 1e-6, "seed_nodes": [0]},
    "hits":     {"max_iter": 100, "tolerance": 1e-6},
    "louvain":  {"min_delta_q": 1e-4, "max_levels": 10, "resolution": 1.0},
    "mcl":      {"expansion": 2, "inflation": 2.0, "prune_threshold": 0.001,
                 "max_iter": 100, "convergence_tol": 1e-4},
}


# ---------------------------------------------------------------------------
# Benchmark dataset container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkDataset:
    name: str
    graph_csr: sp.csr_matrix
    network_type: str = "grn"
    params_override: dict[str, dict[str, Any]] = field(default_factory=dict)
    node_index_map: dict[str, int] = field(default_factory=dict)

    def params_for(self, algorithm: str) -> dict:
        base = dict(_DEFAULT_PARAMS.get(algorithm, {}))
        base["network_type"] = self.network_type
        base.update(self.params_override.get(algorithm, {}))
        return base


# ---------------------------------------------------------------------------
# Timing record
# ---------------------------------------------------------------------------

@dataclass
class TimingRecord:
    algorithm:  str
    dataset:    str
    mode:       str
    n_nodes:    int
    n_edges:    int
    times_s:    list[float]
    success:    bool
    error:      Optional[str] = None

    @property
    def mean_s(self) -> float:
        return float(np.mean(self.times_s)) if self.times_s else float("nan")

    @property
    def std_s(self) -> float:
        return float(np.std(self.times_s, ddof=0)) if len(self.times_s) > 1 else 0.0

    @property
    def min_s(self) -> float:
        return float(np.min(self.times_s)) if self.times_s else float("nan")

    @property
    def max_s(self) -> float:
        return float(np.max(self.times_s)) if self.times_s else float("nan")


# ---------------------------------------------------------------------------
# Mode runner (reuses same dispatch pattern as CrossImplementationValidator)
# ---------------------------------------------------------------------------

def _cpu_fn(algorithm: str, mode: str) -> Callable[[sp.csr_matrix, dict], dict]:
    from src.algorithms import cpu as _pkg
    fn = getattr(_pkg, f"{algorithm}_{mode}", None)
    if fn is None:
        raise AttributeError(f"src.algorithms.cpu has no {algorithm}_{mode}")
    return fn


def _run_once(algorithm: str, mode: str, graph_csr: sp.csr_matrix,
              params: dict, node_index_map: dict) -> float:
    """Execute one run and return elapsed seconds.  Raises on error."""
    if mode in ("cpu_single", "cpu_multi"):
        fn = _cpu_fn(algorithm, mode)
        t0 = time.perf_counter()
        fn(graph_csr, params)
        return time.perf_counter() - t0

    # GPU modes via the runner so we inherit CUDA-event timing + context guard
    from src.runner.algorithm_runner import run_algorithm
    envelope = run_algorithm(
        algorithm_name=algorithm,
        graph_csr=graph_csr,
        node_index_map=node_index_map,
        mode=mode,
        params=params,
    )
    return float(envelope.get("execution_time", 0.0))


# ---------------------------------------------------------------------------
# RuntimeBenchmarker
# ---------------------------------------------------------------------------

class RuntimeBenchmarker:
    """Repeated-run timing benchmark across four execution modes.

    Usage
    -----
    .. code-block:: python

        rb = RuntimeBenchmarker(n_runs=3, output_dir="experiments/outputs")
        rb.add_dataset("my_graph", graph_csr, network_type="ppi")
        rb.run(algorithms=["pagerank", "bfs"])
        rb.write_csv()
        rb.write_plots()
    """

    def __init__(
        self,
        n_runs: int = 3,
        warmup_runs: int = 1,
        output_dir: str | Path = "experiments/outputs",
        algorithms: Iterable[str] = ALGORITHMS,
        modes: Iterable[str] = MODES,
    ) -> None:
        self.n_runs = max(1, int(n_runs))
        self.warmup_runs = max(0, int(warmup_runs))
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "reports"
        self.plots_dir   = self.output_dir / "plots"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.algorithms = tuple(algorithms)
        self.modes      = tuple(modes)

        self.datasets: list[BenchmarkDataset] = []
        self.records:  list[TimingRecord] = []

    # ---- dataset registration ----

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
        self.datasets.append(BenchmarkDataset(
            name=str(name),
            graph_csr=graph_csr,
            network_type=str(network_type).lower(),
            params_override=params_override or {},
            node_index_map=node_index_map or {},
        ))

    # ---- run ----

    def run(
        self,
        algorithms: Optional[Iterable[str]] = None,
        datasets:   Optional[Iterable[str]] = None,
    ) -> None:
        """Execute every (algorithm × dataset × mode) for ``n_runs`` repetitions."""
        algos = tuple(algorithms) if algorithms else self.algorithms
        ds_list = [d for d in self.datasets
                   if datasets is None or d.name in set(datasets)]
        if not ds_list:
            raise RuntimeError("RuntimeBenchmarker.run: no datasets registered.")

        self.records = []
        total = len(algos) * len(ds_list) * len(self.modes)
        done  = 0

        for dataset in ds_list:
            for algorithm in algos:
                params = dataset.params_for(algorithm)
                for mode in self.modes:
                    done += 1
                    _LOG.info("[%d/%d] %s / %s / %s",
                              done, total, algorithm, dataset.name, mode)

                    times: list[float] = []
                    err: Optional[str] = None

                    # warmup
                    for _ in range(self.warmup_runs):
                        try:
                            _run_once(algorithm, mode, dataset.graph_csr,
                                      params, dataset.node_index_map)
                        except Exception:
                            break

                    # timed runs
                    for run_idx in range(self.n_runs):
                        gc.collect()
                        log_memory(
                            f"Before {algorithm}/{dataset.name}/{mode}"
                            f" run {run_idx + 1}/{self.n_runs}", _LOG,
                        )
                        try:
                            t = _run_once(algorithm, mode, dataset.graph_csr,
                                          params, dataset.node_index_map)
                            times.append(t)
                        except Exception as exc:
                            err = f"{type(exc).__name__}: {exc}"
                            _LOG.warning("benchmark: %s/%s/%s failed: %s",
                                         algorithm, dataset.name, mode, err)
                            break
                        log_memory(
                            f"After  {algorithm}/{dataset.name}/{mode}"
                            f" run {run_idx + 1}/{self.n_runs}", _LOG,
                        )
                        force_gc(
                            f"Post {algorithm}/{dataset.name}/{mode}", _LOG,
                        )

                    self.records.append(TimingRecord(
                        algorithm=algorithm,
                        dataset=dataset.name,
                        mode=mode,
                        n_nodes=int(dataset.graph_csr.shape[0]),
                        n_edges=int(dataset.graph_csr.nnz),
                        times_s=times,
                        success=bool(times),
                        error=err,
                    ))

    # ---- speedup computation ----

    def _mean_lookup(self) -> dict[tuple[str, str, str], float]:
        """Return {(algo, dataset, mode): mean_s}."""
        return {(r.algorithm, r.dataset, r.mode): r.mean_s
                for r in self.records if r.success and r.times_s}

    def _speedup(self, mean_ref: float, mean_cand: float) -> float:
        if mean_cand <= 0 or np.isnan(mean_cand):
            return float("nan")
        if np.isnan(mean_ref):
            return float("nan")
        return mean_ref / mean_cand

    # ---- CSV output ----

    def write_csv(self, path: Optional[str | Path] = None) -> Path:
        out = Path(path) if path else (
            self.reports_dir / "runtime_benchmark.csv"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        means = self._mean_lookup()
        fieldnames = [
            "algorithm", "dataset", "mode", "n_nodes", "n_edges",
            "mean_s", "std_s", "min_s", "max_s",
            "speedup_vs_cpu_single", "speedup_vs_cpu_multi",
            "speedup_vs_gpu_baseline",
        ]
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for rec in self.records:
                ref_single   = means.get((rec.algorithm, rec.dataset, "cpu_single"),   float("nan"))
                ref_multi    = means.get((rec.algorithm, rec.dataset, "cpu_multi"),    float("nan"))
                ref_baseline = means.get((rec.algorithm, rec.dataset, "gpu_baseline"), float("nan"))
                w.writerow({
                    "algorithm":  rec.algorithm,
                    "dataset":    rec.dataset,
                    "mode":       rec.mode,
                    "n_nodes":    rec.n_nodes,
                    "n_edges":    rec.n_edges,
                    "mean_s":     f"{rec.mean_s:.6f}",
                    "std_s":      f"{rec.std_s:.6f}",
                    "min_s":      f"{rec.min_s:.6f}",
                    "max_s":      f"{rec.max_s:.6f}",
                    "speedup_vs_cpu_single":   f"{self._speedup(ref_single, rec.mean_s):.4f}",
                    "speedup_vs_cpu_multi":    f"{self._speedup(ref_multi,  rec.mean_s):.4f}",
                    "speedup_vs_gpu_baseline": f"{self._speedup(ref_baseline, rec.mean_s):.4f}",
                })
        _LOG.info("RuntimeBenchmarker: wrote %d rows to %s", len(self.records), out)
        return out

    # ---- plot helpers ----

    def write_plots(self) -> dict[str, Path]:
        out = {}
        out["runtime_comparison"]          = self._plot_runtime_comparison()
        out["speedup_comparison"]          = self._plot_speedup_comparison()
        out["algorithm_runtime_breakdown"] = self._plot_breakdown()
        out["optimization_gain"]           = self._plot_optimization_gain()
        return out

    # ---- plot: mean ± std grouped bar per algorithm ----

    def _plot_runtime_comparison(self) -> Path:
        outpath = self.plots_dir / "runtime_comparison.png"
        algos   = sorted({r.algorithm for r in self.records})
        modes   = [m for m in MODES if m in {r.mode for r in self.records}]
        if not algos:
            return _blank_plot(outpath, "runtime_comparison: no data")

        n_algo  = len(algos)
        n_mode  = len(modes)
        fig, axes = plt.subplots(
            1, n_algo,
            figsize=(max(7, 2.4 * n_algo), 5),
            squeeze=False,
        )

        for col, algo in enumerate(algos):
            ax = axes[0][col]
            x  = np.arange(n_mode)
            algo_recs = {r.mode: r for r in self.records
                         if r.algorithm == algo and r.success}
            means = [algo_recs[m].mean_s if m in algo_recs else np.nan
                     for m in modes]
            stds  = [algo_recs[m].std_s  if m in algo_recs else 0.0
                     for m in modes]
            colours = [_MODE_COLOURS.get(m, "#888") for m in modes]
            bars = ax.bar(x, means, 0.6,
                          yerr=stds, capsize=4,
                          color=colours, edgecolor="white", linewidth=0.5)
            # value labels
            for bar, v in zip(bars, means):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(stds) * 0.1,
                            _fmt_time(v),
                            ha="center", va="bottom", fontsize=7)
            ax.set_title(algo, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([_MODE_LABELS.get(m, m) for m in modes],
                               rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("mean runtime (s)", fontsize=8)
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            ax.set_ylim(bottom=0)

        # shared legend
        patches = [plt.Rectangle((0, 0), 1, 1,
                                  color=_MODE_COLOURS.get(m, "#888"),
                                  label=_MODE_LABELS.get(m, m))
                   for m in modes]
        fig.legend(handles=patches, loc="upper center", ncol=min(4, n_mode),
                   fontsize=9, bbox_to_anchor=(0.5, 1.03))
        fig.suptitle("Runtime comparison: mean ± std across modes", fontsize=13)
        fig.tight_layout()
        fig.savefig(outpath, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return outpath

    # ---- plot: speedup vs cpu_single ----

    def _plot_speedup_comparison(self) -> Path:
        outpath = self.plots_dir / "speedup_comparison.png"
        means   = self._mean_lookup()
        algos   = sorted({r.algorithm for r in self.records})
        modes   = [m for m in MODES if m != "cpu_single"
                   and m in {r.mode for r in self.records}]
        datasets = sorted({r.dataset for r in self.records})
        if not algos or not modes:
            return _blank_plot(outpath, "speedup_comparison: no data")

        fig, axes = plt.subplots(
            1, len(algos),
            figsize=(max(7, 2.4 * len(algos)), 5),
            squeeze=False,
        )

        for col, algo in enumerate(algos):
            ax = axes[0][col]
            x = np.arange(len(modes))
            # average speedup over datasets
            spd_by_mode: dict[str, list[float]] = {m: [] for m in modes}
            for ds in datasets:
                ref = means.get((algo, ds, "cpu_single"), float("nan"))
                if np.isnan(ref):
                    continue
                for m in modes:
                    cand = means.get((algo, ds, m), float("nan"))
                    if not np.isnan(cand) and cand > 0:
                        spd_by_mode[m].append(ref / cand)

            vals    = [float(np.mean(spd_by_mode[m])) if spd_by_mode[m]
                       else np.nan for m in modes]
            colours = [_MODE_COLOURS.get(m, "#888") for m in modes]
            bars = ax.bar(x, vals, 0.6, color=colours,
                          edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.02,
                            f"{v:.2f}×", ha="center", va="bottom", fontsize=8)
            ax.axhline(1.0, color="black", linestyle="--",
                       linewidth=0.8, alpha=0.5, label="baseline (1×)")
            ax.set_title(algo, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([_MODE_LABELS.get(m, m) for m in modes],
                               rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("speedup vs cpu_single (×)", fontsize=8)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", linestyle=":", alpha=0.4)

        patches = [plt.Rectangle((0, 0), 1, 1,
                                  color=_MODE_COLOURS.get(m, "#888"),
                                  label=_MODE_LABELS.get(m, m))
                   for m in modes]
        fig.legend(handles=patches, loc="upper center", ncol=min(4, len(modes)),
                   fontsize=9, bbox_to_anchor=(0.5, 1.03))
        fig.suptitle("Speedup vs CPU single-threaded baseline", fontsize=13)
        fig.tight_layout()
        fig.savefig(outpath, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return outpath

    # ---- plot: grouped bar breakdown (all algos on one chart) ----

    def _plot_breakdown(self) -> Path:
        outpath  = self.plots_dir / "algorithm_runtime_breakdown.png"
        algos    = sorted({r.algorithm for r in self.records})
        modes    = [m for m in MODES if m in {r.mode for r in self.records}]
        datasets = sorted({r.dataset for r in self.records})
        if not algos:
            return _blank_plot(outpath, "algorithm_runtime_breakdown: no data")

        # One bar group per algorithm, bars per mode
        x      = np.arange(len(algos))
        width  = 0.8 / max(1, len(modes))
        fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(algos)), 5))

        for mi, mode in enumerate(modes):
            vals: list[float] = []
            for algo in algos:
                times_across_ds: list[float] = []
                for ds in datasets:
                    m_dict = {r.mode: r for r in self.records
                              if r.algorithm == algo and r.dataset == ds}
                    if mode in m_dict and m_dict[mode].success:
                        times_across_ds.append(m_dict[mode].mean_s)
                vals.append(float(np.mean(times_across_ds))
                            if times_across_ds else np.nan)

            offset = mi * width - 0.4 + width / 2
            bars   = ax.bar(x + offset, vals, width,
                            label=_MODE_LABELS.get(mode, mode),
                            color=_MODE_COLOURS.get(mode, "#888"),
                            edgecolor="white", linewidth=0.4)
            for bar, v in zip(bars, vals):
                if np.isfinite(v) and v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.02,
                            _fmt_time(v), ha="center",
                            va="bottom", fontsize=6, rotation=60)

        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=20, ha="right")
        ax.set_ylabel("mean runtime (s)")
        ax.set_title("Algorithm runtime breakdown — all modes", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(outpath, dpi=120)
        plt.close(fig)
        return outpath

    # ---- plot: gpu_baseline vs gpu optimisation gain ----

    def _plot_optimization_gain(self) -> Path:
        """Head-to-head chart: gpu_baseline vs gpu, showing the ratio
        (optimization gain) as an annotation.  A ratio > 1 means the
        optimised GPU is faster than the baseline.
        """
        outpath = self.plots_dir / "optimization_gain.png"
        algos   = sorted({r.algorithm for r in self.records})
        datasets = sorted({r.dataset for r in self.records})
        if not algos:
            return _blank_plot(outpath, "optimization_gain: no data")

        fig, axes = plt.subplots(
            1, len(algos),
            figsize=(max(7, 2.4 * len(algos)), 5),
            squeeze=False,
        )
        COMPARE_MODES = ("gpu_baseline", "gpu")

        for col, algo in enumerate(algos):
            ax = axes[0][col]
            x = np.arange(2)
            vals: list[float] = []
            for mode in COMPARE_MODES:
                times: list[float] = []
                for ds in datasets:
                    rec = next((r for r in self.records
                                if r.algorithm == algo
                                and r.dataset == ds
                                and r.mode == mode
                                and r.success), None)
                    if rec:
                        times.append(rec.mean_s)
                vals.append(float(np.mean(times)) if times else np.nan)

            colours = [_MODE_COLOURS.get(m, "#888") for m in COMPARE_MODES]
            bars = ax.bar(x, vals, 0.5, color=colours,
                          edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.04,
                            _fmt_time(v), ha="center",
                            va="bottom", fontsize=8)

            # annotate gain ratio
            base, opt = vals
            if np.isfinite(base) and np.isfinite(opt) and opt > 0:
                gain = base / opt
                colour = "#7ed321" if gain >= 1.0 else "#e74c3c"
                ax.text(0.5, max(v for v in vals if np.isfinite(v)) * 0.6,
                        f"gain = {gain:.2f}×",
                        ha="center", va="center", fontsize=10,
                        color=colour, fontweight="bold",
                        transform=ax.transData)
                ax.annotate(
                    "", xy=(1, opt), xytext=(0, base),
                    arrowprops=dict(arrowstyle="->",
                                   color=colour, lw=1.4),
                )

            ax.set_title(algo, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [_MODE_LABELS.get(m, m) for m in COMPARE_MODES],
                rotation=20, ha="right", fontsize=9,
            )
            ax.set_ylabel("mean runtime (s)", fontsize=8)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", linestyle=":", alpha=0.4)

        patches = [plt.Rectangle((0, 0), 1, 1,
                                  color=_MODE_COLOURS.get(m, "#888"),
                                  label=_MODE_LABELS.get(m, m))
                   for m in COMPARE_MODES]
        fig.legend(handles=patches, loc="upper center", ncol=2,
                   fontsize=9, bbox_to_anchor=(0.5, 1.03))
        fig.suptitle(
            "GPU optimisation gain\n"
            "(gpu_baseline vs gpu; gain > 1× = optimised is faster)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(outpath, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return outpath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(s: float) -> str:
    if s >= 1.0:
        return f"{s:.2f}s"
    if s >= 1e-3:
        return f"{s*1e3:.1f}ms"
    return f"{s*1e6:.0f}µs"


def _blank_plot(path: Path, msg: str = "no data") -> Path:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


__all__ = [
    "RuntimeBenchmarker",
    "BenchmarkDataset",
    "TimingRecord",
    "ALGORITHMS",
    "MODES",
]
