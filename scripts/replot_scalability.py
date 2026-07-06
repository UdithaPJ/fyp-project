"""
scripts/replot_scalability.py
=============================

Regenerate the three scalability plots (runtime / memory / speedup) from an
existing ``scalability_benchmark.csv`` using the CURRENT plotting code — with
no benchmark re-run.  Use this to refresh archived plot folders after a change
to the plotting style (e.g. the symlog→log y-axis fix).

Usage
-----
    # Regenerate in place (overwrites the PNGs next to the CSV):
    python scripts/replot_scalability.py D:/FYP/plots/grn-louvain

    # Or point at a CSV explicitly, optionally writing PNGs elsewhere:
    python scripts/replot_scalability.py path/to/scalability_benchmark.csv [out_dir]

Only folders that still contain their ``scalability_benchmark.csv`` can be
regenerated this way; folders with PNGs only must be re-run through the
benchmark (which now uses the fixed plotting code).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")   # headless; must precede pyplot import

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.benchmarking.scalability_benchmark import (   # noqa: E402
    ScalabilityBenchmarker,
    ScalabilityRecord,
)


def _load_records(csv_path: Path) -> list[ScalabilityRecord]:
    """Reconstruct ScalabilityRecords from a scalability_benchmark.csv."""
    records: list[ScalabilityRecord] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            def _num(key: str) -> float:
                try:
                    return float(row.get(key, "") or "nan")
                except ValueError:
                    return float("nan")

            runtime = _num("runtime_s")
            records.append(ScalabilityRecord(
                algorithm=row["algorithm"],
                graph_type=row["graph_type"],
                n_nodes=int(row["n_nodes"]),
                n_edges=int(row["n_edges"]),
                mode=row["mode"],
                runtime_s=runtime,
                peak_mb=_num("peak_mb"),
                # No success column in the CSV: a finite runtime == it ran.
                success=bool(np.isfinite(runtime)),
                strategy_note=(row.get("strategy_note") or ""),
            ))
    return records


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    target = Path(sys.argv[1])
    csv_path = target / "scalability_benchmark.csv" if target.is_dir() else target
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1

    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(csv_path)
    if not records:
        print(f"ERROR: no rows parsed from {csv_path}")
        return 1

    # Point the benchmarker's own output at a throwaway temp dir (its __init__
    # creates output_dir/{reports,plots}); write the actual PNGs into out_dir.
    with tempfile.TemporaryDirectory(prefix="replot_") as tmp:
        sb = ScalabilityBenchmarker(output_dir=tmp)
        sb.records = records
        sb.plots_dir = out_dir
        paths = sb.write_plots()

    print(f"Loaded {len(records)} rows from {csv_path.name}; wrote:")
    for name, p in paths.items():
        print(f"  {name:22s} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
