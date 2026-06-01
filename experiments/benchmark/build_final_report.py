"""
experiments/benchmark/build_final_report.py
============================================

Master-report builder.  Reads every CSV produced by the validation +
benchmarking pipeline under ``experiments/outputs/reports/`` and merges
them into a single consolidated view that directly supports the
dissertation's evaluation chapter:

    final_benchmark_summary.csv  – one row per (algorithm, dataset)
    final_benchmark_summary.md   – human-readable report grouped by
                                    the three project pillars

Columns
-------
    Algorithm
    Dataset

    cpu_single_runtime
    cpu_multi_runtime
    gpu_baseline_runtime
    gpu_runtime

    speedup_cpu_multi              (= cpu_single / cpu_multi)
    speedup_gpu_baseline           (= cpu_single / gpu_baseline)
    speedup_gpu                    (= cpu_single / gpu)

    optimization_gain              (= gpu_baseline_runtime / gpu_runtime)

    peak_ram_mb
    peak_vram_mb

    max_graph_size                 (max nodes the implementation handled
                                    successfully across all benchmark runs)

    computational_validation_metric (representative numeric — Spearman
                                     for ranking algorithms, NMI for
                                     clustering, exact-match for BFS;
                                     taken from cpu_single__vs__gpu pair)

    biological_validation_metric    (best representative — precision /
                                     jaccard / fisher p-value depending
                                     on availability)

The script is defensive: every source CSV is optional, missing columns
become blank cells, and the merged report is still produced if some
benchmarks were not run.

Usage
-----
    python experiments/benchmark/build_final_report.py
    python experiments/benchmark/build_final_report.py \
        --reports-dir experiments/outputs/reports
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV readers (each returns DataFrame or empty DataFrame)
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        _LOG.warning("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Builders for the per-(algorithm, dataset) key
# ---------------------------------------------------------------------------

def _runtime_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the runtime_benchmark CSV into one row per (algorithm, dataset)
    with one column per mode's mean runtime.
    """
    if df.empty or "mode" not in df.columns or "mean_s" not in df.columns:
        return pd.DataFrame(columns=["algorithm", "dataset"])

    keep = df[["algorithm", "dataset", "mode", "mean_s", "n_nodes", "n_edges"]]
    pivot = keep.pivot_table(
        index=["algorithm", "dataset"],
        columns="mode",
        values="mean_s",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None

    # Rename mode columns to <mode>_runtime
    rename = {m: f"{m}_runtime" for m in df["mode"].unique()}
    pivot = pivot.rename(columns=rename)

    # Attach max n_nodes / n_edges across modes (representative size)
    sizes = keep.groupby(["algorithm", "dataset"]).agg(
        n_nodes=("n_nodes", "max"),
        n_edges=("n_edges", "max"),
    ).reset_index()
    pivot = pivot.merge(sizes, on=["algorithm", "dataset"], how="left")
    return pivot


def _add_speedups(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    def _safe_div(num, den):
        try:
            n = float(num); d = float(den)
            if not (math.isfinite(n) and math.isfinite(d)) or d == 0:
                return float("nan")
            return n / d
        except (TypeError, ValueError):
            return float("nan")

    def _row(row):
        cs = row.get("cpu_single_runtime", float("nan"))
        cm = row.get("cpu_multi_runtime",  float("nan"))
        gb = row.get("gpu_baseline_runtime", float("nan"))
        gp = row.get("gpu_runtime",         float("nan"))
        return pd.Series({
            "speedup_cpu_multi":     _safe_div(cs, cm),
            "speedup_gpu_baseline":  _safe_div(cs, gb),
            "speedup_gpu":           _safe_div(cs, gp),
            "optimization_gain":     _safe_div(gb, gp),
        })

    speedups = df.apply(_row, axis=1)
    return pd.concat([df, speedups], axis=1)


def _memory_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate memory_benchmark to (algorithm, dataset) → peak_vram, peak_ram, max_edges."""
    if df.empty:
        return pd.DataFrame(columns=["algorithm", "dataset"])
    # Prefer the gpu (optimised) row for the canonical peak numbers;
    # fall back to gpu_baseline when only the baseline ran.
    gpu = df[df["mode"] == "gpu"] if "mode" in df.columns else pd.DataFrame()
    base = df[df["mode"] == "gpu_baseline"] if "mode" in df.columns else pd.DataFrame()

    def _pick_col(d: pd.DataFrame, col: str) -> pd.DataFrame:
        if d.empty or col not in d.columns:
            return pd.DataFrame(columns=["algorithm", "dataset", col])
        return d[["algorithm", "dataset", col]]

    pieces = []
    for col in ("peak_vram_mb", "peak_ram_mb", "max_graph_edges_supported"):
        a = _pick_col(gpu, col).rename(columns={col: col})
        if a.empty:
            a = _pick_col(base, col)
        pieces.append(a)

    if not pieces:
        return pd.DataFrame(columns=["algorithm", "dataset"])
    out = pieces[0]
    for p in pieces[1:]:
        if p.empty:
            continue
        out = out.merge(p, on=["algorithm", "dataset"], how="outer")
    return out


def _scalability_max(df: pd.DataFrame) -> pd.DataFrame:
    """Compute max_graph_size from scalability_benchmark (across graph_types)."""
    if df.empty or "n_nodes" not in df.columns:
        return pd.DataFrame(columns=["algorithm", "dataset"])
    # In scalability the 'dataset' isn't an explicit column — fold it as the
    # graph_type, since that's how the rows are produced.
    # Use the max successful n_nodes per (algorithm, mode).
    if "mode" in df.columns:
        gpu = df[df["mode"] == "gpu"]
    else:
        gpu = df
    if gpu.empty:
        gpu = df
    grouped = gpu.groupby("algorithm").agg(
        max_graph_size_scalability=("n_nodes", "max"),
    ).reset_index()
    return grouped


def _computational_metric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pick a representative computational validation metric per
    (algorithm, dataset) from the cpu_single__vs__gpu comparison.
    """
    if df.empty:
        return pd.DataFrame(columns=["algorithm", "dataset",
                                     "computational_validation_metric"])
    # Prefer the optimised-GPU comparison; fall back to gpu_baseline.
    pairs_pref = ["cpu_single__vs__gpu", "cpu_single__vs__gpu_baseline"]

    # Per-algorithm metric of interest
    metric_for_algo = {
        "pagerank": "spearman",
        "rwr":      "spearman",
        "hits":     "hub_spearman",
        "bfs":      "exact_match_pct",
        "louvain":  "nmi",
        "mcl":      "nmi",
    }

    rows = []
    for (algo, dataset), grp in df.groupby(["algorithm", "dataset"]):
        metric_name = metric_for_algo.get(algo, "spearman")
        val = float("nan")
        for pair in pairs_pref:
            cand = grp[(grp["comparison_pair"] == pair)
                       & (grp["metric_name"] == metric_name)]
            if not cand.empty:
                try:
                    val = float(cand.iloc[0]["metric_value"])
                except (TypeError, ValueError):
                    val = float("nan")
                if math.isfinite(val):
                    break
        rows.append({
            "algorithm": algo,
            "dataset":   dataset,
            "computational_validation_metric": val,
            "computational_metric_name":       metric_name,
        })
    return pd.DataFrame(rows)


def _biological_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Pick a representative biological metric per (algorithm, dataset)."""
    if df.empty:
        return pd.DataFrame(columns=["algorithm", "dataset",
                                     "biological_validation_metric"])
    ok = df[df.get("status", "ok") == "ok"] if "status" in df.columns else df
    if ok.empty:
        return pd.DataFrame(columns=["algorithm", "dataset",
                                     "biological_validation_metric"])

    rows = []
    for (algo, ds), grp in ok.groupby(["algorithm", "dataset"]):
        first = grp.iloc[0]
        # Choose the best-defined metric (jaccard > precision > p-value)
        jacc = float(first.get("jaccard", float("nan")))
        prec = float(first.get("precision", float("nan")))
        pval = float(first.get("p_value", float("nan")))
        nmi  = float(first.get("nmi",   float("nan")))

        if algo in ("louvain", "mcl") and math.isfinite(nmi):
            chosen = nmi
            name   = "nmi"
        elif math.isfinite(jacc):
            chosen = jacc
            name   = "jaccard"
        elif math.isfinite(prec):
            chosen = prec
            name   = "precision"
        elif math.isfinite(pval):
            chosen = pval
            name   = "p_value"
        else:
            chosen = float("nan")
            name   = ""

        rows.append({
            "algorithm": algo,
            "dataset":   ds,
            "biological_validation_metric":      chosen,
            "biological_metric_name":            name,
            "biological_reference":              first.get("reference", ""),
            "biological_status":                 first.get("status", ""),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Merge orchestrator
# ---------------------------------------------------------------------------

def build_summary(reports_dir: Path) -> pd.DataFrame:
    runtime  = _read_csv(reports_dir / "runtime_benchmark.csv")
    memory   = _read_csv(reports_dir / "memory_benchmark.csv")
    scalab   = _read_csv(reports_dir / "scalability_benchmark.csv")
    compv    = _read_csv(reports_dir / "computational_validation.csv")
    biov     = _read_csv(reports_dir / "biological_validation.csv")

    base = _runtime_pivot(runtime)
    base = _add_speedups(base)
    mem  = _memory_pivot(memory)
    sca  = _scalability_max(scalab)
    comp = _computational_metric(compv)
    bio  = _biological_metric(biov)

    merged = base
    for piece in (mem, comp, bio):
        if not piece.empty:
            merged = merged.merge(piece, on=["algorithm", "dataset"], how="left")

    # Attach scalability max-size per algorithm (no dataset key)
    if not sca.empty:
        merged = merged.merge(sca, on="algorithm", how="left")

    # Final max_graph_size: prefer the per-run nodes from runtime_benchmark,
    # else fall back to the scalability max.
    def _max_size(row):
        a = row.get("n_nodes", float("nan"))
        b = row.get("max_graph_size_scalability", float("nan"))
        vals = [v for v in (a, b)
                if isinstance(v, (int, float)) and math.isfinite(v)]
        return max(vals) if vals else float("nan")
    if not merged.empty:
        merged["max_graph_size"] = merged.apply(_max_size, axis=1)

    # ── Canonicalise column names and order ──
    desired = [
        "algorithm", "dataset",
        "cpu_single_runtime", "cpu_multi_runtime",
        "gpu_baseline_runtime", "gpu_runtime",
        "speedup_cpu_multi", "speedup_gpu_baseline", "speedup_gpu",
        "optimization_gain",
        "peak_ram_mb", "peak_vram_mb",
        "max_graph_size",
        "computational_validation_metric", "computational_metric_name",
        "biological_validation_metric", "biological_metric_name",
        "biological_reference", "biological_status",
    ]
    for col in desired:
        if col not in merged.columns:
            merged[col] = ""
    merged = merged[desired + [c for c in merged.columns if c not in desired]]
    # Drop helper columns we don't want in the final report
    drop_cols = [c for c in ("n_nodes", "n_edges",
                             "max_graph_size_scalability")
                 if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    # Capitalise the column headers for the report (CSV stays snake_case)
    return merged


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

_PILLAR_TEXT = """\
# FYP Final Benchmark Summary

This report consolidates every validation and benchmark output into a
single dissertation-evaluation view across the three project pillars:

1. **Computational Correctness** — verified by
   `CrossImplementationValidator` (cpu_single → cpu_multi /
   gpu_baseline / gpu pair-wise metrics).
2. **Performance & Scalability** — measured by `RuntimeBenchmarker`,
   `ScalabilityBenchmarker`, `MemoryBenchmarker`, and
   `ArchitectureBenchmarker`.
3. **Biological Relevance** — measured by `BiologicalValidator`
   against TRRUST (GRN), BioGRID (PPI), and miRTarBase (miRNA).

Empty cells indicate the corresponding benchmark / validation step
was not run, or its reference data was unavailable in this
environment.

"""


def _fmt_cell(v) -> str:
    if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float):
        if abs(v) >= 1000 or (0 < abs(v) < 1e-3):
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def _md_table(df: pd.DataFrame, cols: list[str], headers: list[str]) -> str:
    if df.empty:
        return "_(no data)_\n\n"
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt_cell(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n\n"


def write_markdown(summary: pd.DataFrame, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sections = [_PILLAR_TEXT]

    # ── Pillar 1: Computational Correctness ──
    sections.append("## 1. Computational Correctness\n\n")
    sections.append(
        "Representative metric per algorithm — Spearman correlation for "
        "ranking algorithms (PageRank / RWR / HITS), exact-match % for BFS, "
        "NMI for clustering (Louvain / MCL).  Drawn from the "
        "`cpu_single__vs__gpu` comparison pair where available.\n\n"
    )
    sections.append(_md_table(
        summary,
        cols=["algorithm", "dataset",
              "computational_metric_name", "computational_validation_metric"],
        headers=["Algorithm", "Dataset",
                 "Metric", "Value"],
    ))

    # ── Pillar 2: Performance & Scalability ──
    sections.append("## 2. Performance & Scalability\n\n")
    sections.append(
        "Mean runtime (s) per implementation mode, plus speedups vs. "
        "`cpu_single`.  `optimization_gain = gpu_baseline_runtime / "
        "gpu_runtime` — the practical lift from the custom CUDA kernels "
        "over the cuGraph/CuPy baseline.\n\n"
    )
    sections.append(_md_table(
        summary,
        cols=["algorithm", "dataset",
              "cpu_single_runtime", "cpu_multi_runtime",
              "gpu_baseline_runtime", "gpu_runtime",
              "speedup_cpu_multi", "speedup_gpu_baseline",
              "speedup_gpu", "optimization_gain"],
        headers=["Algorithm", "Dataset",
                 "cpu_single (s)", "cpu_multi (s)",
                 "gpu_baseline (s)", "gpu (s)",
                 "× vs cpu_multi", "× vs gpu_baseline",
                 "× vs cpu_single", "opt gain"],
    ))
    sections.append("### Memory & graph capacity\n\n")
    sections.append(_md_table(
        summary,
        cols=["algorithm", "dataset",
              "peak_ram_mb", "peak_vram_mb", "max_graph_size"],
        headers=["Algorithm", "Dataset",
                 "peak RAM (MB)", "peak VRAM (MB)",
                 "max graph size (nodes)"],
    ))

    # ── Pillar 3: Biological Relevance ──
    sections.append("## 3. Biological Relevance\n\n")
    sections.append(
        "Validation against TRRUST (GRN), BioGRID (PPI), and miRTarBase "
        "(miRNA).  Representative metric: NMI for clustering algorithms; "
        "Jaccard / precision / Fisher exact p-value for ranking algorithms.\n\n"
    )
    sections.append(_md_table(
        summary,
        cols=["algorithm", "dataset", "biological_reference",
              "biological_metric_name", "biological_validation_metric",
              "biological_status"],
        headers=["Algorithm", "Dataset", "Reference",
                 "Metric", "Value", "Status"],
    ))

    sections.append("\n---\n\n_Generated by `experiments/benchmark/build_final_report.py`_\n")
    out_path.write_text("".join(sections), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Build final benchmark summary report.")
    p.add_argument(
        "--reports-dir", type=Path,
        default=PROJECT_ROOT / "experiments" / "outputs" / "reports",
    )
    p.add_argument(
        "--csv-out", type=Path,
        default=None,
    )
    p.add_argument(
        "--md-out", type=Path,
        default=None,
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    reports_dir = args.reports_dir
    csv_out = args.csv_out or (reports_dir / "final_benchmark_summary.csv")
    md_out  = args.md_out  or (reports_dir / "final_benchmark_summary.md")

    summary = build_summary(reports_dir)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_out, index=False)
    write_markdown(summary, md_out)

    print()
    print("=" * 70)
    print("FINAL BENCHMARK SUMMARY")
    print(f"CSV : {csv_out}")
    print(f"MD  : {md_out}")
    print(f"Rows: {len(summary)}")
    if not summary.empty:
        print("Preview (first 5 rows):")
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(summary.head().to_string(index=False))


if __name__ == "__main__":
    main()
