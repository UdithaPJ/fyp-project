"""
experiments/validation/run_validation.py
========================================

Driver script for :class:`src.validation.CrossImplementationValidator`
and (optionally) :class:`src.validation.BiologicalValidator`.

Loads one or more datasets via the existing preprocessing pipeline,
registers them with the validator, runs every (algorithm × mode), and
writes ``computational_validation.csv`` + the three validation plots
under ``experiments/outputs/``.

Adding ``--biological`` also runs the biological validation step
against the project's reference databases (TRRUST / BioGRID /
miRTarBase) and writes ``biological_validation.csv`` plus the
three enrichment plots.  When ``--biological`` is set, the script
also emits a merged summary CSV
(``computational_and_biological_validation.csv``) that joins both
result tables for downstream reporting.

Examples
--------
    python experiments/validation/run_validation.py --sample-rows 50000
    python experiments/validation/run_validation.py \
        --raw-path data/raw/9606.protein.links.v12.0.txt \
        --network-type ppi --algorithms pagerank,bfs,louvain
    python experiments/validation/run_validation.py --biological \
        --network-type ppi --sample-rows 50000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.converter import graphdata_to_csr               # noqa: E402
from src.preprocessing.pipeline import PreprocessingPipeline   # noqa: E402
from src.validation import (                                    # noqa: E402
    ALGORITHMS,
    BiologicalValidator,
    CrossImplementationValidator,
    OrthogonalValidator,
    HoldoutValidator,
    GOEnrichmentValidator,
)


def _load_graph(raw_path: Path, sample_rows: int | None,
                mapping: dict[str, str] | None):
    print(f"[validate] loading {raw_path} (sample_rows={sample_rows})")
    df = pd.read_csv(raw_path, sep=None, engine="python",
                     nrows=sample_rows, low_memory=True)
    print(f"[validate] rows loaded: {len(df):,}")

    pipeline = PreprocessingPipeline()
    graph_data, report = pipeline.run_dataframe(
        df,
        user_override=mapping,
        duplicate_strategy="mean",
    )
    graph_csr, node_index_map = graphdata_to_csr(graph_data)
    print(
        f"[validate] graph built: nodes={graph_csr.shape[0]:,} "
        f"edges={graph_csr.nnz:,} mapping={report.get('applied_mapping')}"
    )
    return graph_csr, node_index_map


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "9606.protein.links.v12.0.txt",
    )
    p.add_argument("--sample-rows", type=int, default=50_000,
                   help="0 = load the whole file")
    p.add_argument("--network-type",
                   choices=["grn", "ppi", "mirna"], default="ppi")
    p.add_argument(
        "--algorithms",
        type=str, default="all",
        help=f"comma-separated subset of {sorted(ALGORITHMS)} or 'all'",
    )
    p.add_argument("--source-col", default="protein1")
    p.add_argument("--target-col", default="protein2")
    p.add_argument("--weight-col", default="combined_score")
    p.add_argument("--dataset-name", default=None,
                   help="label used in the CSV (default: file stem)")
    p.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "experiments" / "outputs",
    )
    p.add_argument(
        "--source-node", type=int, default=0,
        help="BFS source node index (default 0)",
    )
    p.add_argument(
        "--rwr-seeds", type=str, default="0",
        help="comma-separated RWR seed node indices",
    )
    p.add_argument(
        "--biological", action="store_true",
        help="also run BiologicalValidator against TRRUST/BioGRID/miRTarBase",
    )
    p.add_argument(
        "--bio-reference-mode", default="cpu_single",
        choices=["cpu_single", "cpu_multi", "gpu", "gpu_baseline"],
        help="Which mode's algorithm results to feed into BiologicalValidator "
             "(default: cpu_single, the reference implementation).",
    )
    p.add_argument(
        "--trrust-path",     type=Path, default=None,
        help="Override path to TRRUST reference file.",
    )
    p.add_argument(
        "--biogrid-path",    type=Path, default=None,
        help="Override path to BioGRID reference file.",
    )
    p.add_argument(
        "--mirtarbase-path", type=Path, default=None,
        help="Override path to miRTarBase reference file.",
    )
    # ── Independent (circularity-immune) validation methods ──
    p.add_argument(
        "--orthogonal", action="store_true",
        help="Run OrthogonalValidator against disease/essential/drug gene "
             "sets (DisGeNET / DEG-OGEE / DrugBank).",
    )
    p.add_argument(
        "--holdout", action="store_true",
        help="Run HoldoutValidator (edge hold-out link prediction / "
             "community recovery).  Needs no external files.",
    )
    p.add_argument(
        "--go", action="store_true",
        help="Run GOEnrichmentValidator against a local GO annotation "
             "(GAF) file.",
    )
    p.add_argument("--disgenet-path", type=Path, default=None)
    p.add_argument("--deg-path",      type=Path, default=None)
    p.add_argument("--drugbank-path", type=Path, default=None)
    p.add_argument("--go-path",       type=Path, default=None)
    p.add_argument(
        "--go-aspects", type=str, default=None,
        help="Comma-separated GO aspects to keep: P,F,C (default: all).",
    )
    p.add_argument(
        "--holdout-fraction", type=float, default=0.2,
        help="Fraction of edges held out for the hold-out validator.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _select_results_for_bio(
    validator: CrossImplementationValidator,
    dataset_name: str,
    mode: str,
) -> dict[str, dict]:
    """
    Extract algorithm result dicts captured by the cross-implementation
    validator for use as input to BiologicalValidator.

    Returns ``{algorithm: result_envelope_or_inner_dict}`` for every
    successful run on the given dataset and mode.
    """
    out: dict[str, dict] = {}
    for rec in validator.runs:
        if rec.dataset != dataset_name:
            continue
        if rec.mode != mode:
            continue
        if not rec.success or rec.inner_result is None:
            continue
        # Wrap in a standard envelope so BiologicalValidator finds the
        # 'result' key.
        out[rec.algorithm] = {"result": rec.inner_result}
    return out


def _merge_bio_into_comp(
    comp_csv: Path,
    bio_csv: Path,
    out_csv: Path,
) -> Path:
    """
    Produce a merged CSV that left-joins biological-validation rows
    onto the computational-validation table by (algorithm, dataset).
    """
    if not comp_csv.exists():
        return out_csv
    comp = pd.read_csv(comp_csv)
    if bio_csv.exists():
        bio = pd.read_csv(bio_csv)
        bio_subset = bio[["algorithm", "dataset", "reference",
                          "overlap_count", "precision", "recall",
                          "jaccard", "p_value", "nmi", "ari", "status"]]
        bio_subset = bio_subset.rename(columns={
            "precision":     "bio_precision",
            "recall":        "bio_recall",
            "jaccard":       "bio_jaccard",
            "p_value":       "bio_p_value",
            "nmi":           "bio_nmi",
            "ari":           "bio_ari",
            "overlap_count": "bio_overlap_count",
            "status":        "bio_status",
        })
        merged = comp.merge(bio_subset, on=["algorithm", "dataset"],
                            how="left")
    else:
        merged = comp.copy()
        for c in ("reference", "bio_overlap_count", "bio_precision",
                  "bio_recall", "bio_jaccard", "bio_p_value",
                  "bio_nmi", "bio_ari", "bio_status"):
            merged[c] = ""
    merged.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    sample = None if args.sample_rows == 0 else args.sample_rows
    graph_csr, node_index_map = _load_graph(
        args.raw_path, sample,
        mapping={
            "source": args.source_col,
            "target": args.target_col,
            "weight": args.weight_col,
        },
    )

    ds_name = args.dataset_name or args.raw_path.stem

    rwr_seeds = [int(s) for s in str(args.rwr_seeds).split(",") if s.strip()]
    params_override = {
        "bfs": {"source": int(args.source_node)},
        "rwr": {"seed_nodes": rwr_seeds or [0]},
    }

    if args.algorithms.lower() == "all":
        algos = list(ALGORITHMS)
    else:
        algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]

    # ── 1. Cross-implementation (computational) validation ──
    validator = CrossImplementationValidator(output_dir=args.output_dir)
    validator.add_dataset(
        name=ds_name,
        graph_csr=graph_csr,
        network_type=args.network_type,
        params_override=params_override,
        node_index_map=node_index_map,
    )
    validator.run(algorithms=algos)

    comp_csv  = validator.write_csv()
    comp_plots = validator.write_plots()

    print()
    print("=" * 70)
    print(f"COMPUTATIONAL VALIDATION")
    print(f"CSV  : {comp_csv}")
    for k, v in comp_plots.items():
        print(f"PLOT : {k:<11} {v}")

    failed = [r for r in validator.runs if not r.success]
    if failed:
        print()
        print(f"WARN : {len(failed)} runs failed.  First few:")
        for r in failed[:5]:
            print(f"       {r.algorithm}/{r.dataset}/{r.mode}: {r.error}")

    # ── 2. Biological validation (optional) ──
    if args.biological:
        ref_paths = {}
        if args.trrust_path:     ref_paths["grn"]   = args.trrust_path
        if args.biogrid_path:    ref_paths["ppi"]   = args.biogrid_path
        if args.mirtarbase_path: ref_paths["mirna"] = args.mirtarbase_path

        bio = BiologicalValidator(
            algorithms=tuple(a for a in algos
                             if a in ("pagerank", "hits", "rwr",
                                      "louvain", "mcl")),
            output_dir=args.output_dir,
            reference_paths=ref_paths or None,
        )

        bio_results = _select_results_for_bio(
            validator, ds_name, args.bio_reference_mode,
        )
        bio.add_dataset(
            name=ds_name,
            graph_csr=graph_csr,
            network_type=args.network_type,
            node_index_map=node_index_map,
            results=bio_results,
        )
        bio.run()
        bio_csv   = bio.write_csv()
        bio_plots = bio.write_plots()

        print()
        print("=" * 70)
        print(f"BIOLOGICAL VALIDATION (reference mode: {args.bio_reference_mode})")
        print(f"CSV  : {bio_csv}")
        for k, v in bio_plots.items():
            print(f"PLOT : {k:<22} {v}")

        ok_count      = sum(1 for r in bio.records if r.status == "ok")
        skip_count    = sum(1 for r in bio.records if r.status == "skipped")
        error_count   = sum(1 for r in bio.records if r.status == "error")
        print(f"BIO  : {ok_count} ok | {skip_count} skipped | {error_count} errors")

        merged_csv = args.output_dir / "reports" / \
                     "computational_and_biological_validation.csv"
        _merge_bio_into_comp(comp_csv, bio_csv, merged_csv)
        print(f"MERGE: {merged_csv}")

    # Results shared by the orthogonal / GO validators (same shape as bio).
    ranking_cluster = [a for a in algos
                       if a in ("pagerank", "hits", "rwr", "louvain", "mcl")]

    # ── 3. Orthogonal (topology-independent) validation ──
    if args.orthogonal:
        ref_paths = {}
        if args.disgenet_path: ref_paths["disease"]     = args.disgenet_path
        if args.deg_path:      ref_paths["essential"]   = args.deg_path
        if args.drugbank_path: ref_paths["drug_target"] = args.drugbank_path

        ortho = OrthogonalValidator(
            output_dir=args.output_dir,
            reference_paths=ref_paths or None,
        )
        ortho.add_dataset(
            name=ds_name, graph_csr=graph_csr,
            network_type=args.network_type, node_index_map=node_index_map,
            results=_select_results_for_bio(
                validator, ds_name, args.bio_reference_mode),
        )
        ortho.run(algorithms=tuple(ranking_cluster))
        ortho_csv   = ortho.write_csv()
        ortho_plots = ortho.write_plots()
        print()
        print("=" * 70)
        print("ORTHOGONAL VALIDATION (disease / essential / drug-target)")
        print(f"CSV  : {ortho_csv}")
        for k, v in ortho_plots.items():
            print(f"PLOT : {k:<22} {v}")
        ok = sum(1 for r in ortho.records if r.status == "ok")
        sk = sum(1 for r in ortho.records if r.status == "skipped")
        print(f"ORTH : {ok} ok | {sk} skipped")

    # ── 4. Hold-out (self-contained) validation ──
    if args.holdout:
        hold = HoldoutValidator(
            output_dir=args.output_dir,
            test_fraction=args.holdout_fraction,
        )
        hold.add_dataset(
            name=ds_name, graph_csr=graph_csr,
            network_type=args.network_type,
            params_override=params_override,
        )
        hold.run(algorithms=tuple(ranking_cluster))
        hold_csv   = hold.write_csv()
        hold_plots = hold.write_plots()
        print()
        print("=" * 70)
        print(f"HOLD-OUT VALIDATION (test_fraction={args.holdout_fraction})")
        print(f"CSV  : {hold_csv}")
        for k, v in hold_plots.items():
            print(f"PLOT : {k:<22} {v}")
        for r in hold.records:
            if r.status == "ok":
                metric = (f"auroc={r.auroc:.3f}"
                          if r.auroc == r.auroc else f"lift={r.lift:.3f}")
                print(f"       {r.algorithm:<9} {metric}")

    # ── 5. GO term enrichment ──
    if args.go:
        aspects = None
        if args.go_aspects:
            aspects = {a.strip().upper() for a in args.go_aspects.split(",")
                       if a.strip()}
        go = GOEnrichmentValidator(
            output_dir=args.output_dir,
            reference_path=args.go_path,
            aspects=aspects,
        )
        go.add_dataset(
            name=ds_name, graph_csr=graph_csr,
            network_type=args.network_type, node_index_map=node_index_map,
            results=_select_results_for_bio(
                validator, ds_name, args.bio_reference_mode),
        )
        go.run(algorithms=tuple(ranking_cluster))
        go_csv   = go.write_csv()
        go_plots = go.write_plots()
        print()
        print("=" * 70)
        print("GO TERM ENRICHMENT")
        print(f"CSV  : {go_csv}")
        for k, v in go_plots.items():
            print(f"PLOT : {k:<22} {v}")
        for r in go.records:
            if r.status == "ok":
                print(f"       {r.algorithm:<9} best={r.best_term} "
                      f"p={r.best_term_p:.2e} "
                      f"sig={r.n_groups_significant}/{r.n_groups_tested}")


if __name__ == "__main__":
    main()
