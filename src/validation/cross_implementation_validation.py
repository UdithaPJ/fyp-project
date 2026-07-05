"""
src/validation/cross_implementation_validation.py
==================================================

Cross-implementation validation across the four execution modes:

    cpu_single  cpu_multi  gpu_baseline  gpu

For every (algorithm, dataset) pair the validator runs all four modes,
then forms pair-wise comparisons:

    cpu_single  vs  cpu_multi      → multi-threaded CPU correctness
    cpu_single  vs  gpu_baseline   → GPU baseline correctness
    cpu_single  vs  gpu            → optimised GPU correctness
    gpu_baseline vs  gpu           → optimisation preserves correctness

Metrics are computed by :mod:`src.validation.metrics`.  Results are
flattened into a long-format CSV at

    experiments/outputs/reports/computational_validation.csv

and three plots are written under ``experiments/outputs/plots/``:

    validation_correlation.png
    validation_error.png
    validation_summary.png

The summary plot shows the agreement score of each implementation
against ``cpu_single`` (the reference) per algorithm, with all four
modes side-by-side so the user can see at a glance that the
multi-threaded CPU, GPU baseline, and optimised GPU all produce
matching results.

This module orchestrates algorithm runs by calling each implementation
directly — CPU functions from :mod:`src.algorithms.cpu` and GPU runs
through :func:`src.runner.algorithm_runner.run_algorithm` so the
existing CUDA context guard and CUDA-event timer are reused.  No
algorithm code is modified.
"""

from __future__ import annotations

import csv
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import scipy.sparse as sp

# Backend selection for matplotlib must happen before pyplot import so the
# validator can run on headless GPU boxes.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from src.validation.metrics import compute_metrics


_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALGORITHMS: tuple[str, ...] = (
    "pagerank", "bfs", "louvain", "rwr", "hits", "mcl",
)

MODES: tuple[str, ...] = ("cpu_single", "cpu_multi", "gpu_baseline", "gpu")

# Pair-wise comparisons.  Order is (reference, candidate).
COMPARISON_PAIRS: tuple[tuple[str, str], ...] = (
    ("cpu_single", "cpu_multi"),
    ("cpu_single", "gpu_baseline"),
    ("cpu_single", "gpu"),
    ("gpu_baseline", "gpu"),
)

# Per-algorithm "agreement" metric used on the summary plot.  Picked so
# that 1.0 == perfect agreement and 0.0 == no agreement.
_PRIMARY_AGREEMENT_METRIC: dict[str, tuple[str, ...]] = {
    "pagerank": ("spearman",),
    "rwr":      ("spearman",),
    "hits":     ("hub_spearman", "auth_spearman"),  # averaged
    "bfs":      ("exact_match_pct",),
    "louvain":  ("nmi",),
    "mcl":      ("nmi",),
}

# Metric names split for the correlation- vs error-style plots
_CORRELATION_METRICS = {
    "spearman", "pearson", "hub_spearman", "hub_pearson",
    "auth_spearman", "auth_pearson", "top10_overlap",
    "exact_match_pct", "reachable_agreement", "nmi", "ari",
}
_ERROR_METRICS = {
    "mae", "rmse", "hub_mae", "auth_mae",
    "modularity_diff", "cluster_count_diff", "reachable_diff",
}

# ---------------------------------------------------------------------------
# Interpretation notes
# ---------------------------------------------------------------------------
# PageRank / BFS / HITS / RWR are deterministic: two correct implementations
# must agree to numerical precision, so a metric < 1.0 there is a real bug.
# Louvain and MCL are NOT deterministic — a sub-1.0 partition-similarity
# (NMI / ARI) between two independent implementations is EXPECTED, not a
# failure.  These notes are written into the CSV ``note`` column so the low
# NMI/ARI rows are read correctly.
#
#   louvain — stochastic: node visit order + parallel tie-breaking select
#             different (equally valid) partitions at different resolutions.
#   mcl     — chaotic: repeated squaring + inflation; FP32 (GPU) vs FP64
#             (CPU) precision sends the iteration to different fixed points.
#
# For these two, judge correctness by the QUALITY metric (louvain:
# modularity_diff — lower is better; mcl: cluster_count_diff / internal
# consistency) rather than by partition identity.
_STOCHASTIC_NOTE: dict[str, dict[str, str]] = {
    "louvain": {
        "nmi": "louvain is stochastic; NMI<1 across independent "
               "implementations is expected. Judge by modularity_diff.",
        "ari": "louvain is stochastic; ARI<1 across independent "
               "implementations is expected. Judge by modularity_diff.",
        "modularity_diff": "REAL correctness signal for louvain "
                           "(lower = same partition quality).",
    },
    "mcl": {
        "nmi": "mcl is a chaotic iteration; FP32-GPU vs FP64-CPU converge to "
               "different fixed points, so NMI<1 is expected. The two CPU "
               "backends agree (see cpu_single vs cpu_multi).",
        "ari": "mcl is a chaotic iteration; ARI<1 across CPU/GPU is expected. "
               "The two CPU backends agree (see cpu_single vs cpu_multi).",
        "cluster_count_diff": "context for mcl divergence "
                              "(CPU/GPU pruning pipelines differ).",
    },
}


def _interpretation_note(algorithm: str, metric_name: str) -> str:
    """Return an explanatory note for a metric row, or '' when none applies."""
    return _STOCHASTIC_NOTE.get(algorithm, {}).get(metric_name, "")


# ---------------------------------------------------------------------------
# Default parameter dicts per algorithm (mirror compare_cpu_gpu_raw.py)
# ---------------------------------------------------------------------------

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
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class ValidationDataset:
    """A single graph used for cross-implementation validation."""
    name: str
    graph_csr: sp.csr_matrix
    network_type: str = "grn"
    params_override: dict[str, dict[str, Any]] = field(default_factory=dict)
    node_index_map: dict[str, int] = field(default_factory=dict)

    def params_for(self, algorithm: str) -> dict:
        """Return the merged params for an algorithm on this dataset."""
        base = dict(_DEFAULT_PARAMS.get(algorithm, {}))
        base["network_type"] = self.network_type
        base.update(self.params_override.get(algorithm, {}))
        return base


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """Captured output of running a single (algorithm, dataset, mode)."""
    algorithm:    str
    dataset:      str
    mode:         str
    inner_result: Optional[dict]
    elapsed_s:    float
    success:      bool
    error:        Optional[str] = None


# ---------------------------------------------------------------------------
# Lazy CPU function lookup (avoids importing GPU symbols in CPU workers)
# ---------------------------------------------------------------------------

def _cpu_fn(algorithm: str, mode: str) -> Callable[[sp.csr_matrix, dict], dict]:
    """Resolve the CPU function for ``(algorithm, mode)`` from
    :mod:`src.algorithms.cpu`.

    The cpu_single / cpu_multi functions return the *inner* result dict
    (no envelope) — exactly the shape the metric functions expect.
    """
    from src.algorithms import cpu as _cpu_pkg
    fn_name = f"{algorithm}_{mode}"
    fn = getattr(_cpu_pkg, fn_name, None)
    if fn is None:
        raise AttributeError(
            f"src.algorithms.cpu has no function {fn_name!r}"
        )
    return fn


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class CrossImplementationValidator:
    """Run all four implementations of every selected algorithm on every
    selected dataset and quantify their agreement.

    Usage
    -----
    .. code-block:: python

        v = CrossImplementationValidator(output_dir="experiments/outputs")
        v.add_dataset("trrust_ppi", graph_csr, network_type="ppi")
        v.run(algorithms=["pagerank", "bfs", "louvain"])
        v.write_csv()           # → reports/computational_validation.csv
        v.write_plots()         # → plots/validation_*.png
    """

    def __init__(
        self,
        output_dir: str | Path = "experiments/outputs",
        algorithms: Iterable[str] = ALGORITHMS,
        modes:      Iterable[str] = MODES,
        comparison_pairs: Iterable[tuple[str, str]] = COMPARISON_PAIRS,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "reports"
        self.plots_dir   = self.output_dir / "plots"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.algorithms = tuple(algorithms)
        self.modes      = tuple(modes)
        self.comparison_pairs = tuple(comparison_pairs)

        self.datasets: list[ValidationDataset] = []

        # Filled by ``run``
        self.runs:   list[RunRecord] = []
        # rows in long format for the CSV
        # (algorithm, dataset, comparison_pair, metric_name, metric_value)
        self.metric_rows: list[dict[str, Any]] = []

    # ---- dataset registration ---------------------------------------------

    def add_dataset(
        self,
        name: str,
        graph_csr: sp.csr_matrix,
        network_type: str = "grn",
        params_override: Optional[dict[str, dict[str, Any]]] = None,
        node_index_map: Optional[dict[str, int]] = None,
    ) -> None:
        """Register a graph for validation under ``name``."""
        if not sp.issparse(graph_csr):
            raise TypeError(
                "graph_csr must be a scipy sparse matrix "
                f"(got {type(graph_csr).__name__})"
            )
        if not sp.isspmatrix_csr(graph_csr):
            graph_csr = graph_csr.tocsr()
        self.datasets.append(
            ValidationDataset(
                name=str(name),
                graph_csr=graph_csr,
                network_type=str(network_type).lower(),
                params_override=params_override or {},
                node_index_map=node_index_map or {},
            )
        )

    # ---- core execution ---------------------------------------------------

    def _run_one(
        self,
        algorithm: str,
        dataset:   ValidationDataset,
        mode:      str,
    ) -> RunRecord:
        params = dataset.params_for(algorithm)

        try:
            if mode in ("cpu_single", "cpu_multi"):
                fn = _cpu_fn(algorithm, mode)
                t0 = time.perf_counter()
                inner = fn(dataset.graph_csr, params)
                elapsed = time.perf_counter() - t0
                # CPU functions return the inner-result dict directly
                return RunRecord(
                    algorithm=algorithm,
                    dataset=dataset.name,
                    mode=mode,
                    inner_result=inner,
                    elapsed_s=float(elapsed),
                    success=True,
                )

            # gpu / gpu_baseline — route through the runner so we inherit
            # the existing CUDA context guard and CUDA-event timer.
            from src.runner.algorithm_runner import run_algorithm
            envelope = run_algorithm(
                algorithm_name=algorithm,
                graph_csr=dataset.graph_csr,
                node_index_map=dataset.node_index_map,
                mode=mode,
                params=params,
            )
            return RunRecord(
                algorithm=algorithm,
                dataset=dataset.name,
                mode=mode,
                inner_result=envelope.get("result"),
                elapsed_s=float(envelope.get("execution_time", 0.0)),
                success=True,
            )

        except Exception as exc:                            # noqa: BLE001
            tb = traceback.format_exc(limit=2)
            _LOG.warning(
                "validator: %s/%s/%s failed: %s\n%s",
                algorithm, dataset.name, mode, exc, tb,
            )
            return RunRecord(
                algorithm=algorithm,
                dataset=dataset.name,
                mode=mode,
                inner_result=None,
                elapsed_s=0.0,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def run(
        self,
        algorithms: Optional[Iterable[str]] = None,
        datasets:   Optional[Iterable[str]] = None,
    ) -> None:
        """Execute every (algorithm × dataset × mode) and compute metrics
        for every (algorithm × dataset × comparison_pair).
        """
        algos_to_run = tuple(algorithms) if algorithms else self.algorithms
        if datasets is not None:
            wanted = set(datasets)
            ds_list = [d for d in self.datasets if d.name in wanted]
        else:
            ds_list = list(self.datasets)
        if not ds_list:
            raise RuntimeError(
                "CrossImplementationValidator.run: no datasets registered."
            )

        self.runs = []
        self.metric_rows = []

        for dataset in ds_list:
            for algorithm in algos_to_run:
                # ---- run all four modes ----
                results_by_mode: dict[str, Optional[dict]] = {}
                for mode in self.modes:
                    rec = self._run_one(algorithm, dataset, mode)
                    self.runs.append(rec)
                    results_by_mode[mode] = rec.inner_result if rec.success else None

                # ---- compute pair-wise metrics ----
                for ref_mode, cand_mode in self.comparison_pairs:
                    ref = results_by_mode.get(ref_mode)
                    cand = results_by_mode.get(cand_mode)
                    pair_label = f"{ref_mode}__vs__{cand_mode}"
                    if ref is None or cand is None:
                        self.metric_rows.append({
                            "algorithm":     algorithm,
                            "dataset":       dataset.name,
                            "comparison_pair": pair_label,
                            "metric_name":   "status",
                            "metric_value":  float("nan"),
                            "note":          (
                                "skipped: missing "
                                + ("ref " if ref is None else "")
                                + ("cand" if cand is None else "")
                            ).strip(),
                        })
                        continue
                    try:
                        metrics = compute_metrics(algorithm, ref, cand)
                    except Exception as exc:                # noqa: BLE001
                        self.metric_rows.append({
                            "algorithm":     algorithm,
                            "dataset":       dataset.name,
                            "comparison_pair": pair_label,
                            "metric_name":   "error",
                            "metric_value":  float("nan"),
                            "note":          f"{type(exc).__name__}: {exc}",
                        })
                        continue

                    for mname, mval in metrics.items():
                        self.metric_rows.append({
                            "algorithm":     algorithm,
                            "dataset":       dataset.name,
                            "comparison_pair": pair_label,
                            "metric_name":   mname,
                            "metric_value":  float(mval)
                                              if isinstance(mval, (int, float, np.floating))
                                              else float("nan"),
                            "note":          _interpretation_note(algorithm, mname),
                        })

    # ---- outputs ----------------------------------------------------------

    def write_csv(self, path: Optional[str | Path] = None) -> Path:
        """Write the long-format validation CSV.

        Columns
        -------
        ``algorithm, dataset, comparison_pair, metric_name, metric_value``
        (plus an additional ``note`` column for diagnostics).
        """
        out = Path(path) if path else (
            self.reports_dir / "computational_validation.csv"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["algorithm", "dataset", "comparison_pair",
                      "metric_name", "metric_value", "note"]
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in self.metric_rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
        _LOG.info("validator: wrote %d metric rows to %s",
                  len(self.metric_rows), out)
        return out

    def write_plots(self) -> dict[str, Path]:
        """Generate the three validation plots.

        Returns a dict mapping plot name → path on disk.
        """
        out = {}
        out["correlation"] = self._plot_correlation()
        out["error"]       = self._plot_error()
        out["summary"]     = self._plot_summary()
        return out

    # ---- internal: plotting -----------------------------------------------

    def _metrics_dataframe(self) -> dict[tuple[str, str, str, str], float]:
        """Return ``{(algo, dataset, pair, metric): value}`` indexed lookup."""
        out: dict[tuple[str, str, str, str], float] = {}
        for row in self.metric_rows:
            key = (
                row["algorithm"], row["dataset"],
                row["comparison_pair"], row["metric_name"],
            )
            v = row["metric_value"]
            if isinstance(v, (int, float, np.floating)) and not np.isnan(v):
                out[key] = float(v)
        return out

    def _aggregate_over_datasets(
        self,
        keep_metrics: set[str],
    ) -> dict[tuple[str, str, str], float]:
        """Average ``metric_value`` over all datasets, keyed by
        ``(algo, pair, metric)``.
        """
        buckets: dict[tuple[str, str, str], list[float]] = {}
        for (algo, _ds, pair, metric), val in self._metrics_dataframe().items():
            if metric not in keep_metrics:
                continue
            buckets.setdefault((algo, pair, metric), []).append(val)
        return {k: float(np.mean(v)) for k, v in buckets.items() if v}

    def _plot_correlation(self) -> Path:
        """Grouped bar plot — correlation-style metrics per algorithm,
        all comparison pairs side-by-side.
        """
        agg = self._aggregate_over_datasets(_CORRELATION_METRICS)
        return self._plot_grouped(
            agg=agg,
            title="Cross-implementation correlation / agreement",
            ylabel="metric value (1.0 = perfect agreement)",
            ylim=(0.0, 1.05),
            outpath=self.plots_dir / "validation_correlation.png",
        )

    def _plot_error(self) -> Path:
        """Grouped bar plot — error-style metrics per algorithm, all
        comparison pairs side-by-side.  Y-axis is log-scaled because
        per-algorithm scales differ by orders of magnitude.
        """
        agg = self._aggregate_over_datasets(_ERROR_METRICS)
        return self._plot_grouped(
            agg=agg,
            title="Cross-implementation numerical error",
            ylabel="error metric (lower = better)",
            ylim=None,
            outpath=self.plots_dir / "validation_error.png",
            log_y=True,
        )

    def _plot_grouped(
        self,
        agg: dict[tuple[str, str, str], float],
        title: str,
        ylabel: str,
        ylim: Optional[tuple[float, float]],
        outpath: Path,
        log_y: bool = False,
    ) -> Path:
        # Build one subplot per algorithm so x-axis labels (metric names)
        # don't collide across algorithms with different metric sets.
        algos_present = sorted({a for (a, _p, _m) in agg.keys()})
        if not algos_present:
            # Nothing to plot — write a blank stub so callers still get a file
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "no metrics to plot",
                    ha="center", va="center")
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(outpath, dpi=120)
            plt.close(fig)
            return outpath

        n = len(algos_present)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(5.2 * cols, 3.6 * rows),
            squeeze=False,
        )

        pair_order = [f"{a}__vs__{b}" for a, b in self.comparison_pairs]
        pair_colours = {
            "cpu_single__vs__cpu_multi":     "#5b8def",
            "cpu_single__vs__gpu_baseline":  "#f5a623",
            "cpu_single__vs__gpu":           "#7ed321",
            "gpu_baseline__vs__gpu":         "#bd10e0",
        }

        for ax_idx, algo in enumerate(algos_present):
            ax = axes[ax_idx // cols][ax_idx % cols]
            # Metric names this algorithm has
            metrics_present = sorted(
                {m for (a, _p, m) in agg.keys() if a == algo}
            )
            x = np.arange(len(metrics_present))
            width = 0.8 / max(1, len(pair_order))

            for pi, pair in enumerate(pair_order):
                ys = [agg.get((algo, pair, m), np.nan)
                      for m in metrics_present]
                ax.bar(
                    x + pi * width - 0.4 + width / 2,
                    ys, width,
                    label=pair.replace("__vs__", " vs "),
                    color=pair_colours.get(pair),
                    edgecolor="white", linewidth=0.4,
                )
            ax.set_title(algo, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_present, rotation=30, ha="right",
                               fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if log_y:
                ax.set_yscale("symlog", linthresh=1e-6)
            ax.grid(axis="y", linestyle=":", alpha=0.4)

        # Hide unused axes
        for j in range(len(algos_present), rows * cols):
            axes[j // cols][j % cols].set_axis_off()

        # Single shared legend at the top
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels,
                loc="upper center",
                ncol=min(4, len(labels)),
                fontsize=9,
                bbox_to_anchor=(0.5, 1.02),
            )
        fig.suptitle(title, fontsize=13, y=1.07 if rows == 1 else 1.03)
        fig.tight_layout()
        fig.savefig(outpath, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return outpath

    def _plot_summary(self) -> Path:
        """One-shot summary plot comparing cpu_multi, gpu_baseline, and
        gpu against cpu_single (the reference) per algorithm.

        Each algorithm gets three bars showing the *primary agreement
        metric* (1.0 = perfect).  This is the headline plot the user
        should look at to confirm that:

            * multi-threaded CPU produces valid results
            * GPU baseline produces valid results
            * optimised GPU produces valid results
            * optimisations preserve correctness (also visible as the
              cpu_multi / gpu_baseline / gpu bars all clustering near 1)
        """
        df = self._metrics_dataframe()

        # algo -> {candidate_mode -> [scores across datasets]}
        candidate_modes = ("cpu_multi", "gpu_baseline", "gpu")
        scores: dict[str, dict[str, list[float]]] = {
            a: {m: [] for m in candidate_modes} for a in self.algorithms
        }
        for algo in self.algorithms:
            primaries = _PRIMARY_AGREEMENT_METRIC[algo]
            for cand in candidate_modes:
                pair = f"cpu_single__vs__{cand}"
                # average across the listed primary metrics, per dataset
                ds_values: dict[str, list[float]] = {}
                for (a, ds, p, m), v in df.items():
                    if a == algo and p == pair and m in primaries:
                        ds_values.setdefault(ds, []).append(v)
                for ds, vals in ds_values.items():
                    if vals:
                        scores[algo][cand].append(float(np.mean(vals)))

        algos_present = [a for a in self.algorithms
                         if any(scores[a][m] for m in candidate_modes)]
        if not algos_present:
            outpath = self.plots_dir / "validation_summary.png"
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "no successful runs to summarise",
                    ha="center", va="center")
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(outpath, dpi=120)
            plt.close(fig)
            return outpath

        fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(algos_present) + 4),
                                        4.5))
        x = np.arange(len(algos_present))
        width = 0.25
        colours = {
            "cpu_multi":    "#5b8def",
            "gpu_baseline": "#f5a623",
            "gpu":          "#7ed321",
        }
        labels = {
            "cpu_multi":    "cpu_multi vs cpu_single",
            "gpu_baseline": "gpu_baseline vs cpu_single",
            "gpu":          "gpu (optimised) vs cpu_single",
        }
        for i, cand in enumerate(candidate_modes):
            ys = [float(np.mean(scores[a][cand])) if scores[a][cand]
                  else np.nan
                  for a in algos_present]
            ax.bar(x + (i - 1) * width, ys, width,
                   label=labels[cand], color=colours[cand],
                   edgecolor="white", linewidth=0.5)

        # Highlight the 1.0 = perfect-agreement line
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
                   alpha=0.5)
        ax.set_ylim(0.0, 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels(algos_present, rotation=20, ha="right")
        ax.set_ylabel("primary agreement metric "
                      "(spearman / nmi / exact-match; 1.0 = perfect)")
        ax.set_title(
            "Cross-implementation correctness summary\n"
            "(cpu_single is the reference; all bars should be near 1.0)",
            fontsize=12,
        )
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        # Per-bar annotations so the user can read exact values
        for i, cand in enumerate(candidate_modes):
            for j, a in enumerate(algos_present):
                vals = scores[a][cand]
                if not vals:
                    continue
                y = float(np.mean(vals))
                if np.isnan(y):
                    continue
                ax.text(x[j] + (i - 1) * width, y + 0.015,
                        f"{y:.3f}",
                        ha="center", va="bottom", fontsize=7)

        outpath = self.plots_dir / "validation_summary.png"
        fig.tight_layout()
        fig.savefig(outpath, dpi=120)
        plt.close(fig)
        return outpath


__all__ = [
    "CrossImplementationValidator",
    "ValidationDataset",
    "RunRecord",
    "ALGORITHMS",
    "MODES",
    "COMPARISON_PAIRS",
]
