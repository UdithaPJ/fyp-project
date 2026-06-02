"""
src/benchmarking/scalability_benchmark.py
==========================================

Scalability benchmark across three synthetic graph types and four graph
sizes.  For every (algorithm, graph_type, size, mode) combination the
benchmarker records:

    runtime_s   — algorithm wall-clock time in seconds
    peak_mb     — peak process working-set size in MB (measured around
                  the algorithm call via Windows ctypes / tracemalloc)

Speedup at each size is computed relative to ``cpu_single`` at that
size.

Graph types
-----------
    barabasi_albert   — power-law degree distribution (biological nets)
    erdos_renyi       — random sparse graph
    watts_strogatz    — small-world network

Graph sizes (approximate node counts)
--------------------------------------
    10K   50K   100K   500K

All graphs are converted to the standard CSR format (``graphdata_to_csr``
is NOT used here since we have no biological metadata — we build the CSR
directly from scipy/networkx).

Outputs
-------
    experiments/outputs/reports/scalability_benchmark.csv
        columns: algorithm, graph_type, n_nodes, n_edges, mode,
                 runtime_s, peak_mb, speedup_vs_cpu_single

    experiments/outputs/plots/
        scalability_runtime.png  — runtime vs graph size per mode
        scalability_memory.png   — peak RSS vs graph size per mode
        scalability_speedup.png  — speedup vs cpu_single vs graph size
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as _W
import csv
import gc
import logging
import platform
import sys
import time
import traceback
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np
import scipy.sparse as sp

# MEMORY_FIX (Fix Cat. 6): emit RAM/VRAM samples around hot operations.
try:
    from src.utils.memory_logger import log_memory, force_gc
except Exception:  # pragma: no cover
    def log_memory(label: str, logger=None) -> None: return None
    def force_gc(label: str = "", logger=None) -> float: return 0.0

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAPH_TYPES: tuple[str, ...] = (
    "barabasi_albert",
    "erdos_renyi",
    "watts_strogatz",
)

# Target node counts — actual size may differ slightly due to generator
# rounding, but labels use these exact values.
GRAPH_SIZES: tuple[int, ...] = (10_000, 50_000, 100_000, 500_000)

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

_TYPE_MARKERS: dict[str, str] = {
    "barabasi_albert": "o",
    "erdos_renyi":     "s",
    "watts_strogatz":  "^",
}

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
# Memory measurement — Windows preferred, tracemalloc fallback
# ---------------------------------------------------------------------------

_IS_WINDOWS = platform.system() == "Windows"

if _IS_WINDOWS:
    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb",                         _W.DWORD),
            ("PageFaultCount",             _W.DWORD),
            ("PeakWorkingSetSize",         ctypes.c_size_t),
            ("WorkingSetSize",             ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage",    ctypes.c_size_t),
            ("QuotaPagedPoolUsage",        ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage",     ctypes.c_size_t),
            ("QuotaPagedPoolUsage2",       ctypes.c_size_t),
            ("PagefileUsage",              ctypes.c_size_t),
            ("PeakPagefileUsage",          ctypes.c_size_t),
        ]
    _K32   = ctypes.windll.kernel32
    _PSAPI = ctypes.windll.psapi
    _PROCESS_QUERY_INFORMATION = 0x0400
    _PROCESS_VM_READ           = 0x0010

    def _rss_mb() -> float:
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        h = _K32.OpenProcess(
            _PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ,
            False,
            _K32.GetCurrentProcessId(),
        )
        _PSAPI.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        _K32.CloseHandle(h)
        return pmc.WorkingSetSize / 1024 ** 2
else:
    def _rss_mb() -> float:  # type: ignore[misc]
        # Linux / macOS fallback via /proc/self/status
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except Exception:
            pass
        return 0.0


class _MemTracker:
    """Context manager that records peak resident memory MB during its body.

    Uses Windows working-set sampling on Windows, tracemalloc delta elsewhere.
    On Windows we poll the working set at entry and exit and report the max.
    """

    def __init__(self) -> None:
        self.peak_mb: float = 0.0
        self._before: float = 0.0

    def __enter__(self) -> "_MemTracker":
        gc.collect()
        self._before = _rss_mb()
        return self

    def __exit__(self, *_) -> None:
        after = _rss_mb()
        self.peak_mb = max(after - self._before, 0.0)


# ---------------------------------------------------------------------------
# Synthetic graph generators
# ---------------------------------------------------------------------------

def _ba_graph(n: int, m: int = 3, seed: int = 42) -> sp.csr_matrix:
    """Barabási–Albert preferential attachment graph.  Undirected, no self-loops."""
    try:
        import networkx as nx
        G = nx.barabasi_albert_graph(n, m, seed=seed)
        return nx.to_scipy_sparse_array(G, format="csr", dtype=np.float32)
    except Exception as exc:
        _LOG.warning("ba_graph fallback to random: %s", exc)
        return _random_sparse(n, m)


def _er_graph(n: int, seed: int = 42) -> sp.csr_matrix:
    """Erdős–Rényi random sparse graph with expected degree ~6."""
    p = min(6.0 / max(n - 1, 1), 1.0)
    try:
        import networkx as nx
        G = nx.fast_gnp_random_graph(n, p, seed=seed)
        return nx.to_scipy_sparse_array(G, format="csr", dtype=np.float32)
    except Exception as exc:
        _LOG.warning("er_graph fallback to random: %s", exc)
        return _random_sparse(n, 3)


def _ws_graph(n: int, k: int = 6, p: float = 0.1, seed: int = 42) -> sp.csr_matrix:
    """Watts–Strogatz small-world graph.  k nearest neighbours, p rewire prob."""
    # WS requires k < n; clamp k for tiny graphs
    k_use = min(k, max(2, n // 4))
    if k_use % 2 != 0:
        k_use += 1  # WS requires even k
    try:
        import networkx as nx
        G = nx.watts_strogatz_graph(n, k_use, p, seed=seed)
        return nx.to_scipy_sparse_array(G, format="csr", dtype=np.float32)
    except Exception as exc:
        _LOG.warning("ws_graph fallback to random: %s", exc)
        return _random_sparse(n, 3)


def _random_sparse(n: int, m: int) -> sp.csr_matrix:
    """Pure-scipy random sparse fallback when networkx is absent."""
    rng  = np.random.default_rng(42)
    nnz  = n * m
    rows = rng.integers(0, n, nnz)
    cols = rng.integers(0, n, nnz)
    mask = rows != cols
    rows, cols = rows[mask], cols[mask]
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


_GRAPH_GENERATORS: dict[str, Callable[[int], sp.csr_matrix]] = {
    "barabasi_albert": lambda n: _ba_graph(n),
    "erdos_renyi":     lambda n: _er_graph(n),
    "watts_strogatz":  lambda n: _ws_graph(n),
}


# ---------------------------------------------------------------------------
# Pre-generated graph loader
# ---------------------------------------------------------------------------

def _find_pregenerated(
    graphs_dir: Path,
    graph_type: str,
    target_n: int | None = None,
    target_m: int | None = None,
    tolerance: float = 0.20,
) -> tuple[sp.csr_matrix, int, int] | None:
    """Load the best-matching pre-generated graph from *graphs_dir*.

    Lookup priority
    ---------------
    1. Exact match on ``target_m`` (edge count) within ``tolerance``.
    2. Exact match on ``target_n`` (node count) within ``tolerance``.
    3. Closest-by-edge-count among all files of the requested type.

    Returns ``(csr, actual_n, actual_m)`` or ``None`` if no file found.

    File naming convention (produced by ``scripts/generate_benchmark_graphs.py``):
        ``{graph_type}_n{N}_m{M}.npz``
    """
    import json as _json

    pattern = f"{graph_type}_n*_m*.npz"
    candidates: list[Path] = sorted(graphs_dir.glob(pattern))
    if not candidates:
        return None

    def _parse(p: Path) -> tuple[int, int]:
        """Extract (n, m) from filename.  Returns (0, 0) on failure."""
        try:
            stem = p.stem           # e.g. "barabasi_albert_n500000_m2997000"
            n_part = stem.split("_n")[1].split("_m")[0]
            m_part = stem.split("_m")[1]
            return int(n_part), int(m_part)
        except Exception:
            return 0, 0

    scored: list[tuple[float, Path, int, int]] = []
    for cand in candidates:
        cn, cm = _parse(cand)
        if cn == 0:
            continue
        if target_m is not None:
            diff = abs(cm - target_m) / max(target_m, 1)
        elif target_n is not None:
            diff = abs(cn - target_n) / max(target_n, 1)
        else:
            diff = 0.0
        scored.append((diff, cand, cn, cm))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0])
    best_diff, best_path, best_n, best_m = scored[0]

    if best_diff > tolerance:
        _LOG.warning(
            "Best match for %s target_m=%s target_n=%s is %s "
            "(diff=%.1f%% > tolerance=%.0f%%) — loading anyway.",
            graph_type, target_m, target_n, best_path.name,
            best_diff * 100, tolerance * 100,
        )

    _LOG.info("Loading pre-generated graph: %s", best_path.name)
    csr = sp.load_npz(str(best_path)).tocsr()
    csr.sum_duplicates()
    csr.eliminate_zeros()
    return csr, int(csr.shape[0]), int(csr.nnz)


# ---------------------------------------------------------------------------
# CPU / GPU runner (same pattern as runtime_benchmark.py)
# ---------------------------------------------------------------------------

def _cpu_fn(algorithm: str, mode: str) -> Callable[[sp.csr_matrix, dict], Any]:
    from src.algorithms import cpu as _pkg
    fn = getattr(_pkg, f"{algorithm}_{mode}", None)
    if fn is None:
        raise AttributeError(f"src.algorithms.cpu has no {algorithm}_{mode}")
    return fn


def _run_once_timed(algorithm: str, mode: str, graph_csr: sp.csr_matrix,
                    params: dict) -> tuple[float, float]:
    """Return (elapsed_s, peak_delta_mb)."""
    with _MemTracker() as mem:
        if mode in ("cpu_single", "cpu_multi"):
            fn = _cpu_fn(algorithm, mode)
            t0 = time.perf_counter()
            fn(graph_csr, params)
            elapsed = time.perf_counter() - t0
        else:
            from src.runner.algorithm_runner import run_algorithm
            envelope = run_algorithm(
                algorithm_name=algorithm,
                graph_csr=graph_csr,
                node_index_map={},
                mode=mode,
                params=params,
            )
            elapsed = float(envelope.get("execution_time", 0.0))
    return elapsed, mem.peak_mb


# ---------------------------------------------------------------------------
# Scalability record
# ---------------------------------------------------------------------------

@dataclass
class ScalabilityRecord:
    algorithm:  str
    graph_type: str
    n_nodes:    int
    n_edges:    int
    mode:       str
    runtime_s:  float
    peak_mb:    float
    success:    bool
    error:      Optional[str] = None


# ---------------------------------------------------------------------------
# ScalabilityBenchmarker
# ---------------------------------------------------------------------------

class ScalabilityBenchmarker:
    """Benchmark scalability across synthetic graphs of increasing size.

    Usage
    -----
    .. code-block:: python

        sb = ScalabilityBenchmarker(
            graph_sizes=(10_000, 50_000),
            output_dir="experiments/outputs",
        )
        sb.run(algorithms=["pagerank", "bfs"])
        sb.write_csv()
        sb.write_plots()
    """

    def __init__(
        self,
        graph_sizes:     Iterable[int]  = GRAPH_SIZES,
        graph_types:     Iterable[str]  = GRAPH_TYPES,
        algorithms:      Iterable[str]  = ALGORITHMS,
        modes:           Iterable[str]  = MODES,
        network_type:    str = "ppi",
        output_dir:      str | Path = "experiments/outputs",
        n_runs:          int = 1,
        # Pre-generated graph support -----------------------------------
        pregenerated_dir: str | Path | None = None,
        edge_targets:    Iterable[int] | None = None,
    ) -> None:
        self.graph_sizes     = tuple(sorted(set(graph_sizes)))
        self.graph_types     = tuple(graph_types)
        self.algorithms      = tuple(algorithms)
        self.modes           = tuple(modes)
        self.network_type    = str(network_type).lower()
        self.output_dir      = Path(output_dir)
        self.reports_dir     = self.output_dir / "reports"
        self.plots_dir       = self.output_dir / "plots"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.n_runs          = max(1, int(n_runs))
        # Pre-generated graph directory; when set, graphs are loaded from
        # disk instead of being generated on the fly.
        self.pregenerated_dir: Path | None = (
            Path(pregenerated_dir) if pregenerated_dir else None
        )
        # Optional edge-count targets used when pregenerated_dir is set.
        # Each target maps to the closest pre-generated graph for each type.
        self.edge_targets: tuple[int, ...] = (
            tuple(sorted(set(edge_targets))) if edge_targets else ()
        )

        self.records: list[ScalabilityRecord] = []

    # ---- run ----

    def run(
        self,
        algorithms: Optional[Iterable[str]] = None,
    ) -> None:
        algos = tuple(algorithms) if algorithms else self.algorithms

        # Build the list of (label, loader) pairs to iterate.
        # When pregenerated_dir is set:
        #   • if edge_targets was given → one entry per (type, edge_target)
        #   • otherwise                 → one entry per (type, node_size),
        #     loading the closest matching file for each size
        # When pregenerated_dir is not set → on-the-fly generation as before.
        graph_jobs: list[tuple[str, int | None, int | None]] = []
        if self.pregenerated_dir and self.edge_targets:
            for gt in self.graph_types:
                for et in self.edge_targets:
                    graph_jobs.append((gt, None, et))
        else:
            for gt in self.graph_types:
                for sz in self.graph_sizes:
                    graph_jobs.append((gt, sz, None))

        total = len(algos) * len(graph_jobs) * len(self.modes)
        done  = 0

        for graph_type, target_n, target_m in graph_jobs:
            # ---- Graph acquisition ------------------------------------
            if self.pregenerated_dir is not None:
                label = (
                    f"{graph_type} m≈{target_m:,}"
                    if target_m else f"{graph_type} n≈{target_n:,}"
                )
                _LOG.info("Loading %s from %s …",
                          label, self.pregenerated_dir)
                log_memory(f"Before loading {label}", _LOG)
                loaded = _find_pregenerated(
                    self.pregenerated_dir,
                    graph_type,
                    target_n=target_n,
                    target_m=target_m,
                )
                if loaded is None:
                    _LOG.error(
                        "No pre-generated file found for %s "
                        "(target_n=%s, target_m=%s) in %s — skipping.",
                        graph_type, target_n, target_m, self.pregenerated_dir,
                    )
                    continue
                g, actual_n, actual_m = loaded
                log_memory(f"After loading {label}", _LOG)
            else:
                target_n_use = target_n or 10_000
                gen = _GRAPH_GENERATORS.get(graph_type)
                if gen is None:
                    _LOG.warning("unknown graph_type %r — skipping", graph_type)
                    continue
                _LOG.info("generating %s n=%d…", graph_type, target_n_use)
                log_memory(f"Before generating {graph_type} n={target_n_use}", _LOG)
                try:
                    g = gen(target_n_use).tocsr()
                    g.sum_duplicates()
                    g.eliminate_zeros()
                except Exception as exc:
                    _LOG.error("graph generation failed: %s", exc)
                    continue
                log_memory(f"After generating {graph_type} n={target_n_use}", _LOG)
                actual_n = int(g.shape[0])
                actual_m = int(g.nnz)

            # ---- Algorithm × mode loop (shared by both paths) --------
            for algorithm in algos:
                params = dict(_DEFAULT_PARAMS.get(algorithm, {}))
                params["network_type"] = self.network_type
                # Ensure BFS source / RWR seeds are valid for this graph
                if "source" in params:
                    params["source"] = 0
                if "seed_nodes" in params:
                    params["seed_nodes"] = [0]

                for mode in self.modes:
                    done += 1
                    _LOG.info("[%d/%d] %s/%s/%s n=%d m=%d",
                              done, total, algorithm, graph_type, mode,
                              actual_n, actual_m)

                    # Average over n_runs
                    times: list[float] = []
                    mems:  list[float] = []
                    err:   Optional[str] = None

                    for _ in range(self.n_runs):
                        gc.collect()
                        try:
                            t, mb = _run_once_timed(algorithm, mode, g, params)
                            times.append(t)
                            mems.append(mb)
                        except Exception as exc:
                            err = f"{type(exc).__name__}: {exc}"
                            _LOG.warning("%s/%s/%s n=%d: %s",
                                         algorithm, graph_type, mode,
                                         actual_n, err)
                            break

                    if times:
                        self.records.append(ScalabilityRecord(
                            algorithm=algorithm,
                            graph_type=graph_type,
                            n_nodes=actual_n,
                            n_edges=actual_m,
                            mode=mode,
                            runtime_s=float(np.mean(times)),
                            peak_mb=float(np.mean(mems)),
                            success=True,
                        ))
                    else:
                        self.records.append(ScalabilityRecord(
                            algorithm=algorithm,
                            graph_type=graph_type,
                            n_nodes=actual_n,
                            n_edges=actual_m,
                            mode=mode,
                            runtime_s=float("nan"),
                            peak_mb=float("nan"),
                            success=False,
                            error=err,
                        ))

    # ---- CSV output ----

    def write_csv(self, path: Optional[str | Path] = None) -> Path:
        out = Path(path) if path else (
            self.reports_dir / "scalability_benchmark.csv"
        )
        out.parent.mkdir(parents=True, exist_ok=True)

        # Pre-compute speedup vs cpu_single at each (algo, graph_type, n_nodes)
        ref_lookup: dict[tuple[str, str, int], float] = {}
        for r in self.records:
            if r.mode == "cpu_single" and r.success:
                ref_lookup[(r.algorithm, r.graph_type, r.n_nodes)] = r.runtime_s

        fieldnames = [
            "algorithm", "graph_type", "n_nodes", "n_edges", "mode",
            "runtime_s", "peak_mb", "speedup_vs_cpu_single",
        ]
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in self.records:
                ref = ref_lookup.get((r.algorithm, r.graph_type, r.n_nodes),
                                     float("nan"))
                spd = (ref / r.runtime_s
                       if np.isfinite(ref) and np.isfinite(r.runtime_s)
                          and r.runtime_s > 0
                       else float("nan"))
                w.writerow({
                    "algorithm":  r.algorithm,
                    "graph_type": r.graph_type,
                    "n_nodes":    r.n_nodes,
                    "n_edges":    r.n_edges,
                    "mode":       r.mode,
                    "runtime_s":  f"{r.runtime_s:.6f}" if np.isfinite(r.runtime_s) else "",
                    "peak_mb":    f"{r.peak_mb:.2f}"   if np.isfinite(r.peak_mb)   else "",
                    "speedup_vs_cpu_single": f"{spd:.4f}" if np.isfinite(spd) else "",
                })
        _LOG.info("ScalabilityBenchmarker: wrote %d rows to %s",
                  len(self.records), out)
        return out

    # ---- plots ----

    def write_plots(self) -> dict[str, Path]:
        return {
            "scalability_runtime": self._plot_scalability("runtime_s",
                                                          "Runtime (s)",
                                                          "scalability_runtime.png"),
            "scalability_memory":  self._plot_scalability("peak_mb",
                                                          "Peak memory delta (MB)",
                                                          "scalability_memory.png"),
            "scalability_speedup": self._plot_speedup_curves(),
        }

    # ---- plot: runtime or memory vs graph size ----

    def _plot_scalability(
        self,
        metric: str,
        ylabel: str,
        filename: str,
    ) -> Path:
        outpath = self.plots_dir / filename
        algos   = sorted({r.algorithm for r in self.records})
        modes   = [m for m in MODES if m in {r.mode for r in self.records}]
        types   = [t for t in GRAPH_TYPES if t in {r.graph_type for r in self.records}]

        if not algos:
            return _blank_plot(outpath, f"{filename}: no data")

        # rows = graph types, cols = algorithms
        n_rows = len(types)
        n_cols = len(algos)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(max(5, 2.5 * n_cols), max(4, 2.8 * n_rows)),
            squeeze=False,
            sharey=False,
        )

        for row, gt in enumerate(types):
            for col, algo in enumerate(algos):
                ax = axes[row][col]
                for mode in modes:
                    pts = sorted(
                        [(r.n_nodes, getattr(r, metric))
                         for r in self.records
                         if r.algorithm == algo
                         and r.graph_type == gt
                         and r.mode == mode
                         and r.success
                         and np.isfinite(getattr(r, metric))],
                        key=lambda x: x[0],
                    )
                    if pts:
                        xs, ys = zip(*pts)
                        ax.plot(xs, ys,
                                marker=_TYPE_MARKERS.get(gt, "o"),
                                color=_MODE_COLOURS.get(mode, "#888"),
                                label=_MODE_LABELS.get(mode, mode),
                                linewidth=1.5, markersize=5)
                if row == 0:
                    ax.set_title(algo, fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"{gt}\n{ylabel}", fontsize=8)
                else:
                    ax.set_ylabel("")
                ax.set_xlabel("n nodes", fontsize=7)
                ax.set_xscale("log")
                ax.set_yscale("symlog", linthresh=1e-3)
                ax.grid(linestyle=":", alpha=0.4)
                ax.tick_params(labelsize=7)

        # Collect legend handles from the first populated axis
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels,
                           loc="upper center", ncol=min(4, len(modes)),
                           fontsize=9, bbox_to_anchor=(0.5, 1.02))
                break

        fig.suptitle(f"Scalability: {ylabel}", fontsize=12, y=1.05)
        fig.tight_layout()
        fig.savefig(outpath, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return outpath

    # ---- plot: speedup vs cpu_single at each size ----

    def _plot_speedup_curves(self) -> Path:
        outpath = self.plots_dir / "scalability_speedup.png"
        algos   = sorted({r.algorithm for r in self.records})
        modes   = [m for m in MODES if m != "cpu_single"
                   and m in {r.mode for r in self.records}]
        types   = [t for t in GRAPH_TYPES if t in {r.graph_type for r in self.records}]

        if not algos or not modes:
            return _blank_plot(outpath, "scalability_speedup: no data")

        ref_lookup: dict[tuple[str, str, int], float] = {}
        for r in self.records:
            if r.mode == "cpu_single" and r.success and np.isfinite(r.runtime_s):
                ref_lookup[(r.algorithm, r.graph_type, r.n_nodes)] = r.runtime_s

        n_rows = len(types)
        n_cols = len(algos)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(max(5, 2.5 * n_cols), max(4, 2.8 * n_rows)),
            squeeze=False,
            sharey=False,
        )

        for row, gt in enumerate(types):
            for col, algo in enumerate(algos):
                ax = axes[row][col]
                for mode in modes:
                    pts_raw = [
                        (r.n_nodes, r.runtime_s)
                        for r in self.records
                        if r.algorithm == algo
                        and r.graph_type == gt
                        and r.mode == mode
                        and r.success
                        and np.isfinite(r.runtime_s)
                    ]
                    pts = sorted(
                        [(n, ref_lookup[(algo, gt, n)] / rt)
                         for n, rt in pts_raw
                         if (algo, gt, n) in ref_lookup and rt > 0],
                        key=lambda x: x[0],
                    )
                    if pts:
                        xs, ys = zip(*pts)
                        ax.plot(xs, ys,
                                marker=_TYPE_MARKERS.get(gt, "o"),
                                color=_MODE_COLOURS.get(mode, "#888"),
                                label=_MODE_LABELS.get(mode, mode),
                                linewidth=1.5, markersize=5)
                ax.axhline(1.0, color="black", linestyle="--",
                           linewidth=0.7, alpha=0.4)
                if row == 0:
                    ax.set_title(algo, fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"{gt}\nspeedup vs cpu_single (×)",
                                  fontsize=8)
                else:
                    ax.set_ylabel("")
                ax.set_xlabel("n nodes", fontsize=7)
                ax.set_xscale("log")
                ax.set_yscale("symlog", linthresh=0.5)
                ax.grid(linestyle=":", alpha=0.4)
                ax.tick_params(labelsize=7)

        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels,
                           loc="upper center", ncol=min(4, len(modes)),
                           fontsize=9, bbox_to_anchor=(0.5, 1.02))
                break

        fig.suptitle("Scalability speedup vs cpu_single\n"
                     "CPU scaling · GPU acceleration · Optimisation gain",
                     fontsize=12, y=1.06)
        fig.tight_layout()
        fig.savefig(outpath, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return outpath


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _blank_plot(path: Path, msg: str = "no data") -> Path:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


__all__ = [
    "ScalabilityBenchmarker",
    "ScalabilityRecord",
    "GRAPH_TYPES",
    "GRAPH_SIZES",
    "ALGORITHMS",
    "MODES",
]
