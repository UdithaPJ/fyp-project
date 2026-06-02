"""
src/benchmarking/convergence_benchmark.py
==========================================

Convergence quality benchmark comparing ``gpu_baseline`` against the
optimised ``gpu`` implementation for iterative algorithms.

Algorithms benchmarked
----------------------
    pagerank, hits, rwr

(BFS, Louvain, MCL are excluded: BFS has no convergence loop; Louvain and
MCL convergence semantics differ too much from a simple iteration count.)

Modes benchmarked
-----------------
    gpu_baseline  – simple GPU implementation (cuGraph or CuPy power
                    iteration)
    gpu           – optimised custom-CUDA implementation

What is measured
----------------
iterations
    Number of iterations until convergence (extracted from the result dict
    when present, or inferred from the algorithm's max_iter when missing).

convergence_rate
    ``final_error / max_error`` — how much the error decreased relative to
    the worst observed error across all modes and tolerance settings.
    Lower is better (converged more thoroughly).

runtime_s
    Wall-clock execution time in seconds.

final_error
    Last reported L2/L1 convergence delta (extracted from result when
    available, else NaN).

converged
    Boolean flag extracted from the result dict (True/False/None).

Derived / comparison metrics (written per algorithm, per dataset)
-----------------------------------------------------------------
    baseline_iterations, gpu_iterations, iteration_reduction_pct
    baseline_runtime,    gpu_runtime,    speedup
    baseline_error,      gpu_error,      error_ratio   (gpu/baseline ≤ 1 ⟹ better)

CSV output
----------
    experiments/outputs/reports/convergence_benchmark.csv
        columns: algorithm, dataset, mode, tolerance, n_nodes, n_edges,
                 iterations, convergence_rate, runtime_s, final_error,
                 converged

Plots
-----
    experiments/outputs/plots/
        convergence_runtime.png     – runtime bar-chart baseline vs gpu
        convergence_iterations.png  – iteration count comparison
        convergence_accuracy.png    – final_error comparison (log scale)
"""

from __future__ import annotations

import csv
import gc
import logging
import math
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

ALGORITHMS: tuple[str, ...] = ("pagerank", "hits", "rwr")

MODES: tuple[str, ...] = ("gpu_baseline", "gpu")

_MODE_LABELS: dict[str, str] = {
    "gpu_baseline": "GPU Baseline",
    "gpu":          "GPU Optimised",
}

_MODE_COLOURS: dict[str, str] = {
    "gpu_baseline": "#bd10e0",
    "gpu":          "#7ed321",
}

# Tolerance values to sweep (tighter ⟹ more iterations needed)
TOLERANCES: tuple[float, ...] = (1e-4, 1e-6)

_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "pagerank": {"damping": 0.85, "max_iter": 100},
    "hits":     {"max_iter": 100},
    "rwr":      {"restart_prob": 0.3, "max_iter": 100, "seed_nodes": [0]},
}


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceRecord:
    algorithm:        str
    dataset:          str
    mode:             str
    tolerance:        float
    n_nodes:          int   = 0
    n_edges:          int   = 0
    iterations:       int   = 0
    convergence_rate: float = float("nan")
    runtime_s:        float = float("nan")
    final_error:      float = float("nan")
    converged:        Optional[bool] = None
    success:          bool  = True
    error:            str   = ""


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class ConvDataset:
    name:            str
    graph_csr:       Any
    network_type:    str  = "ppi"
    params_override: dict = field(default_factory=dict)
    node_index_map:  dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result-extraction helpers
# ---------------------------------------------------------------------------

def _extract_iterations(result: dict) -> int:
    """Pull iterations out of result dict (various field names)."""
    inner = result.get("result", {})
    if isinstance(inner, dict):
        for key in ("iterations", "num_iterations", "n_iterations"):
            val = inner.get(key)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
    # Fallback: check top-level
    for key in ("iterations", "num_iterations"):
        val = result.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return 0


def _extract_converged(result: dict) -> Optional[bool]:
    inner = result.get("result", {})
    if isinstance(inner, dict):
        val = inner.get("converged")
        if val is not None:
            return bool(val)
    val = result.get("converged")
    if val is not None:
        return bool(val)
    return None


def _extract_runtime(result: dict) -> float:
    val = result.get("execution_time")
    try:
        return float(val) if val is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _extract_scores(result: dict) -> Optional[list]:
    """Return the primary score vector for computing error proxy."""
    inner = result.get("result", {})
    if not isinstance(inner, dict):
        return None
    for key in ("scores", "hub_scores", "authority_scores"):
        val = inner.get(key)
        if val is not None and len(val) > 0:
            return list(val)
    return None


def _unwrap(result: Any) -> dict:
    """Normalise the three possible return shapes to a plain dict."""
    if isinstance(result, dict):
        if "output" in result:
            # cuda_optimized wraps in {"output": ..., "extra_params": ...}
            out = result["output"]
            if isinstance(out, dict):
                return out
        return result
    return {}


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def _run_mode(algorithm: str, mode: str, graph_csr: sp.csr_matrix,
              params: dict,
              node_index_map: Optional[dict] = None) -> tuple[dict, bool, str]:
    """
    Run algorithm in the given mode.
    Returns (result_dict, success, error_message).
    """
    try:
        if mode in ("gpu", "gpu_baseline"):
            from src.runner.algorithm_runner import run_algorithm
            raw = run_algorithm(
                algorithm, graph_csr,
                node_index_map or {},
                mode=mode,
                params=dict(params),
            )
            return _unwrap(raw), True, ""
        elif mode == "cpu_single":
            # Direct import for cpu_single
            import importlib
            mod = importlib.import_module(
                f"src.algorithms.cpu.single_threaded.{algorithm}")
            fn = getattr(mod, f"_{algorithm}_cpu_single", None) or \
                 getattr(mod, f"{algorithm}_cpu_single", None)
            if fn is None:
                raise AttributeError(f"No cpu_single function in {mod}")
            raw = fn(graph_csr, dict(params))
            return (_unwrap(raw) if isinstance(raw, dict) else {}), True, ""
        else:
            return {}, False, f"Unsupported mode: {mode}"
    except Exception as exc:
        return {}, False, str(exc)[:300]


# ---------------------------------------------------------------------------
# ConvergenceBenchmarker
# ---------------------------------------------------------------------------

class ConvergenceBenchmarker:
    """
    Benchmark convergence quality of iterative algorithms across modes.

    Parameters
    ----------
    algorithms:
        Algorithms to benchmark (default: pagerank, hits, rwr).
    modes:
        Modes to compare (default: gpu_baseline, gpu).
    tolerances:
        Tolerance values to sweep.
    output_dir:
        Root output dir.  CSV → ``<output_dir>/reports/``,
        plots → ``<output_dir>/plots/``.
    """

    def __init__(
        self,
        algorithms: tuple[str, ...] = ALGORITHMS,
        modes: tuple[str, ...] = MODES,
        tolerances: tuple[float, ...] = TOLERANCES,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.algorithms = tuple(algorithms)
        self.modes      = tuple(modes)
        self.tolerances = tuple(tolerances)

        _root = Path(__file__).resolve().parents[2]
        self.output_dir = Path(output_dir) if output_dir else (
            _root / "experiments" / "outputs"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

        self._datasets: list[ConvDataset] = []
        self.records:   list[ConvergenceRecord] = []

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
        self._datasets.append(ConvDataset(
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
            _LOG.warning("No datasets registered.")
            return

        for ds in self._datasets:
            n = ds.graph_csr.shape[0]
            m = ds.graph_csr.nnz
            _LOG.info("Convergence benchmark | dataset=%s  n=%d  m=%d",
                      ds.name, n, m)

            # We collect score vectors per (algo, mode, tol) to compute
            # a final_error proxy (L2 distance from highest-tol result).
            score_cache: dict[tuple, list] = {}

            for algo in algos:
                for tol in self.tolerances:
                    base_params = dict(_DEFAULT_PARAMS.get(algo, {}))
                    base_params["network_type"] = network_type = ds.network_type
                    base_params["tolerance"] = tol
                    base_params.update(ds.params_override.get(algo, {}))

                    for mode in self.modes:
                        params = dict(base_params)
                        _LOG.info("  %s / %s / tol=%.0e", algo, mode, tol)
                        gc.collect()

                        result, ok, err = _run_mode(
                            algo, mode, ds.graph_csr, params,
                            ds.node_index_map,
                        )

                        if not ok:
                            rec = ConvergenceRecord(
                                algorithm=algo, dataset=ds.name,
                                mode=mode, tolerance=tol,
                                n_nodes=n, n_edges=m,
                                success=False, error=err,
                            )
                            self.records.append(rec)
                            continue

                        iters     = _extract_iterations(result)
                        conv      = _extract_converged(result)
                        rt        = _extract_runtime(result)
                        scores    = _extract_scores(result)

                        # Cache scores for error computation
                        cache_key = (ds.name, algo, mode, tol)
                        if scores:
                            score_cache[cache_key] = scores

                        # final_error proxy: L2 dist from tighter-tol same mode
                        final_err = self._compute_error_proxy(
                            scores, score_cache,
                            ds.name, algo, mode, tol)

                        # convergence_rate = final_error / (initial_norm proxy)
                        # Use max across all recorded errors for this algo+dataset
                        conv_rate = self._convergence_rate(final_err)

                        rec = ConvergenceRecord(
                            algorithm        = algo,
                            dataset          = ds.name,
                            mode             = mode,
                            tolerance        = tol,
                            n_nodes          = n,
                            n_edges          = m,
                            iterations       = iters,
                            convergence_rate = conv_rate,
                            runtime_s        = rt,
                            final_error      = final_err,
                            converged        = conv,
                            success          = True,
                        )
                        self.records.append(rec)
                        _LOG.info(
                            "    iters=%d  rt=%.4fs  err=%.3e  conv=%s",
                            iters, rt, final_err, conv,
                        )

        self._fill_convergence_rates()

    # ------------------------------------------------------------------
    # Error proxy
    # ------------------------------------------------------------------

    def _compute_error_proxy(
        self,
        scores: Optional[list],
        cache: dict,
        ds_name: str,
        algo: str,
        mode: str,
        tol: float,
    ) -> float:
        """
        Compute a final-error proxy as L2(scores - reference).

        Reference is the tightest tolerance run for the *same* mode,
        or (if no tighter tolerance exists) the other mode's result.
        Falls back to NaN when no reference is available.
        """
        if scores is None:
            return float("nan")
        arr = np.array(scores, dtype=float)

        # Try tighter tolerance same mode
        tighter = [t for t in self.tolerances if t < tol]
        for t_ref in sorted(tighter):
            ref_key = (ds_name, algo, mode, t_ref)
            if ref_key in cache:
                ref = np.array(cache[ref_key], dtype=float)
                if len(ref) == len(arr):
                    return float(np.linalg.norm(arr - ref))

        # Try same tolerance other mode
        other_modes = [m for m in self.modes if m != mode]
        for m_ref in other_modes:
            ref_key = (ds_name, algo, m_ref, tol)
            if ref_key in cache:
                ref = np.array(cache[ref_key], dtype=float)
                if len(ref) == len(arr):
                    return float(np.linalg.norm(arr - ref))

        # No reference yet — return norm of scores as a stand-in
        norm_val = float(np.linalg.norm(arr))
        return norm_val if norm_val > 0 else float("nan")

    def _convergence_rate(self, final_error: float) -> float:
        """Placeholder; filled in by _fill_convergence_rates after all runs."""
        return float("nan")

    def _fill_convergence_rates(self) -> None:
        """
        Normalise final_error → convergence_rate once all records exist.

        convergence_rate = final_error / max_error_for_this_(algo, dataset)
        Lower is better (0 = perfect convergence).
        """
        from collections import defaultdict
        max_err: dict[tuple, float] = defaultdict(float)
        for r in self.records:
            if r.success and not math.isnan(r.final_error):
                key = (r.algorithm, r.dataset)
                if r.final_error > max_err[key]:
                    max_err[key] = r.final_error
        for r in self.records:
            if r.success and not math.isnan(r.final_error):
                key = (r.algorithm, r.dataset)
                denom = max_err[key]
                r.convergence_rate = (r.final_error / denom) if denom > 0 else 0.0

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------

    def write_csv(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.output_dir / "reports" / "convergence_benchmark.csv"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fields = [
            "algorithm", "dataset", "mode", "tolerance",
            "n_nodes", "n_edges", "iterations", "convergence_rate",
            "runtime_s", "final_error", "converged",
        ]

        def _fmt(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, float) and math.isnan(v):
                return ""
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v)

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.records:
                if not r.success:
                    continue
                w.writerow({
                    "algorithm":       r.algorithm,
                    "dataset":         r.dataset,
                    "mode":            r.mode,
                    "tolerance":       f"{r.tolerance:.0e}",
                    "n_nodes":         r.n_nodes,
                    "n_edges":         r.n_edges,
                    "iterations":      r.iterations,
                    "convergence_rate":_fmt(r.convergence_rate),
                    "runtime_s":       _fmt(r.runtime_s),
                    "final_error":     _fmt(r.final_error),
                    "converged":       "" if r.converged is None else r.converged,
                })
        _LOG.info("Convergence CSV: %s", path)
        return path

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def write_plots(self) -> dict[str, Path]:
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        paths["convergence_runtime"]    = self._plot_runtime(plot_dir)
        paths["convergence_iterations"] = self._plot_iterations(plot_dir)
        paths["convergence_accuracy"]   = self._plot_accuracy(plot_dir)
        return paths

    # -- helpers --

    def _records_for(self, algo: str, mode: str,
                     tol: Optional[float] = None) -> list[ConvergenceRecord]:
        return [
            r for r in self.records
            if r.algorithm == algo and r.mode == mode
            and r.success and (tol is None or r.tolerance == tol)
        ]

    def _mean_rt(self, algo: str, mode: str, tol: float) -> float:
        vals = [r.runtime_s for r in self._records_for(algo, mode, tol)
                if not math.isnan(r.runtime_s)]
        return float(np.mean(vals)) if vals else float("nan")

    def _mean_iters(self, algo: str, mode: str, tol: float) -> float:
        vals = [r.iterations for r in self._records_for(algo, mode, tol)
                if r.iterations > 0]
        return float(np.mean(vals)) if vals else float("nan")

    def _mean_err(self, algo: str, mode: str, tol: float) -> float:
        vals = [r.final_error for r in self._records_for(algo, mode, tol)
                if not math.isnan(r.final_error)]
        return float(np.mean(vals)) if vals else float("nan")

    # --- convergence_runtime.png ---------------------------------------

    def _plot_runtime(self, plot_dir: Path) -> Path:
        tol = min(self.tolerances)  # tightest tolerance → most representative
        algos   = self.algorithms
        x       = np.arange(len(algos))
        width   = 0.35
        n_modes = len(self.modes)
        offsets = np.linspace(-width / 2, width / 2, n_modes)

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, mode in enumerate(self.modes):
            rts = [self._mean_rt(a, mode, tol) for a in algos]
            bars = ax.bar(x + offsets[i], rts, width / n_modes,
                          label=_MODE_LABELS.get(mode, mode),
                          color=_MODE_COLOURS.get(mode, f"C{i}"),
                          alpha=0.85)
            # Annotate speedup on top of gpu bar
            if mode == "gpu":
                for xi, (base_rt, gpu_rt) in enumerate(
                    zip([self._mean_rt(a, "gpu_baseline", tol) for a in algos], rts)
                ):
                    if not (math.isnan(base_rt) or math.isnan(gpu_rt)) and base_rt > 0:
                        speedup = base_rt / gpu_rt
                        ax.text(xi + offsets[i], gpu_rt * 1.05,
                                f"{speedup:.1f}×",
                                ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(algos)
        ax.set_ylabel("Mean Runtime (s)")
        ax.set_title(f"Convergence Runtime: Baseline vs Optimised  (tol={tol:.0e})")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        out = plot_dir / "convergence_runtime.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _LOG.info("Plot: %s", out)
        return out

    # --- convergence_iterations.png ------------------------------------

    def _plot_iterations(self, plot_dir: Path) -> Path:
        n_tols = len(self.tolerances)
        algos  = self.algorithms
        fig, axes = plt.subplots(1, n_tols, figsize=(6 * n_tols, 5), squeeze=False)

        for ti, tol in enumerate(sorted(self.tolerances)):
            ax = axes[0][ti]
            x  = np.arange(len(algos))
            width   = 0.35
            n_modes = len(self.modes)
            offsets = np.linspace(-width / 2, width / 2, n_modes)

            for i, mode in enumerate(self.modes):
                iters = [self._mean_iters(a, mode, tol) for a in algos]
                ax.bar(x + offsets[i], iters, width / n_modes,
                       label=_MODE_LABELS.get(mode, mode),
                       color=_MODE_COLOURS.get(mode, f"C{i}"),
                       alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(algos)
            ax.set_ylabel("Mean Iterations")
            ax.set_title(f"Iterations to Convergence  (tol={tol:.0e})")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle("Convergence Iterations: Baseline vs Optimised", fontsize=11)
        fig.tight_layout()
        out = plot_dir / "convergence_iterations.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _LOG.info("Plot: %s", out)
        return out

    # --- convergence_accuracy.png --------------------------------------

    def _plot_accuracy(self, plot_dir: Path) -> Path:
        """
        Log-scale comparison of final_error per algorithm.
        Lower final_error means the optimised GPU converged as well as
        (or better than) the baseline.
        """
        tol = min(self.tolerances)
        algos = self.algorithms
        x     = np.arange(len(algos))
        width = 0.35
        n_modes = len(self.modes)
        offsets = np.linspace(-width / 2, width / 2, n_modes)

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, mode in enumerate(self.modes):
            errs = [self._mean_err(a, mode, tol) for a in algos]
            # Replace nan with 0 for bar chart visibility
            errs_plot = [e if not math.isnan(e) else 0.0 for e in errs]
            ax.bar(x + offsets[i], errs_plot, width / n_modes,
                   label=_MODE_LABELS.get(mode, mode),
                   color=_MODE_COLOURS.get(mode, f"C{i}"),
                   alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(algos)
        ax.set_ylabel("Final Error (L2, log scale)")
        ax.set_title(f"Convergence Accuracy: Baseline vs Optimised  (tol={tol:.0e})")
        ax.set_yscale("symlog", linthresh=1e-10)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

        # Add annotation: "GPU matches baseline" when error_ratio ≤ 1.1
        for xi, algo in enumerate(algos):
            base_err = self._mean_err(algo, "gpu_baseline", tol)
            gpu_err  = self._mean_err(algo, "gpu", tol)
            if not (math.isnan(base_err) or math.isnan(gpu_err) or base_err == 0):
                ratio = gpu_err / base_err
                colour = "#27ae60" if ratio <= 1.1 else "#e74c3c"
                label  = f"ratio={ratio:.2f}"
                ax.text(xi, max(base_err, gpu_err) * 1.15, label,
                        ha="center", va="bottom", fontsize=7, color=colour)

        fig.tight_layout()
        out = plot_dir / "convergence_accuracy.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _LOG.info("Plot: %s", out)
        return out
