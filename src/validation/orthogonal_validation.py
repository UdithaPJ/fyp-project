"""
src/validation/orthogonal_validation.py
========================================

Independent (orthogonal) biological validation.

The overlap machinery in :mod:`src.validation.overlap` is only meaningful
when the reference carries evidence UNRELATED to the input graph.  The
interaction databases (TRRUST / BioGRID / miRTarBase) share evidence type
with the network, so validating a BioGRID-derived graph against BioGRID is
circular.  This module instead validates algorithm output against gene sets
whose "importance" is defined OUTSIDE the graph topology:

    reference kind   →  database   →  evidence type
    --------------      ----------     -----------------------------------
    disease          →  DisGeNET   →  clinical / genetic association
    essential        →  DEG/OGEE   →  experimental gene knockout
    drug_target      →  DrugBank   →  pharmacological / clinical

Because the evidence is topology-independent, a high overlap between the
nodes an algorithm ranks central and, say, known disease genes is genuine
biological signal — it cannot be an artefact of the graph being built from
the same database.

Validated per (algorithm × dataset × reference kind):
    ranking algos (pagerank / hits / rwr)
        overlap_count, precision, recall, jaccard, Fisher-exact p-value
        of the predicted top-node set vs the gene set.
    clustering algos (louvain / mcl)
        size-weighted community enrichment (precision/recall/jaccard),
        best-community p-value, and NMI / ARI against the binary
        "in gene-set / not" partition.

Outputs
-------
    experiments/outputs/reports/orthogonal_validation.csv
    experiments/outputs/plots/orthogonal_enrichment.png

Rules
-----
* NEVER raises — failures are recorded with ``status='error'``.
* Skips gracefully when a gene-set file is unavailable.
* Network-type agnostic: the same disease/essential/drug references apply
  to GRN, PPI, and miRNA graphs alike.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import scipy.sparse as sp

from src.validation.reference_loader import GeneSetReference, load_gene_set
from src.validation.overlap import compute_overlap, OverlapResult
from src.validation.enrichment import enrich_communities, CommunityEnrichment
from src.validation.metrics import normalized_mutual_info, adjusted_rand_index
# Reuse the result-extraction helpers already written for the interaction
# validator so the two validators stay consistent about result shapes.
from src.validation.biological_validation import (
    _extract_predicted, _extract_topk_predicted, _extract_communities,
    _inner, _graph_universe,
)

_LOG = logging.getLogger(__name__)

# Ranking algorithms produce a top-node set; clustering algorithms produce
# communities.  BFS is a traversal (no ranking) and is not validated here.
_RANKING_ALGOS   = ("pagerank", "hits", "rwr")
_CLUSTER_ALGOS   = ("louvain", "mcl")
DEFAULT_KINDS    = ("disease", "essential", "drug_target")


@dataclass
class OrthogonalRecord:
    algorithm:      str
    dataset:        str
    network_type:   str
    reference:      str            = ""     # gene-set database name
    kind:           str            = ""     # disease | essential | drug_target
    overlap_count:  int            = 0
    predicted_size: int            = 0
    reference_size: int            = 0
    precision:      float          = float("nan")
    recall:         float          = float("nan")
    jaccard:        float          = float("nan")
    p_value:        float          = float("nan")
    nmi:            float          = float("nan")
    ari:            float          = float("nan")
    note:           str            = ""
    status:         str            = "ok"   # ok | skipped | error


@dataclass
class _OrthoDataset:
    name:            str
    graph_csr:       sp.csr_matrix
    network_type:    str
    node_index_map:  dict
    result_by_algo:  dict[str, dict] = field(default_factory=dict)


def _gene_set_partition(node_index_map: dict,
                        ref: GeneSetReference) -> Optional[list[int]]:
    """Binary partition over graph nodes: 1 = gene is in the set, 0 = not.

    Returns None when no graph node maps into the gene set.
    """
    if not node_index_map or ref is None or ref.is_empty():
        return None
    n = max(node_index_map.values()) + 1
    partition = [0] * n
    hits = 0
    for label, idx in node_index_map.items():
        if str(label).strip().upper() in ref.genes:
            partition[int(idx)] = 1
            hits += 1
    return partition if hits else None


class OrthogonalValidator:
    """Validate algorithm output against topology-independent gene sets.

    Workflow mirrors :class:`BiologicalValidator`::

        ov = OrthogonalValidator(output_dir=...)
        ov.add_dataset(name=..., graph_csr=..., network_type=...,
                       node_index_map=..., results={'pagerank': env, ...})
        ov.run()
        ov.write_csv()
        ov.write_plots()

    Never raises.
    """

    def __init__(
        self,
        kinds: tuple[str, ...] = DEFAULT_KINDS,
        output_dir: Optional[Path] = None,
        reference_paths: Optional[dict[str, Path]] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self.kinds = tuple(kinds)
        # When set (> 0), ranking algorithms are evaluated on their top-`top_k`
        # nodes recomputed from the full score vector, rather than the small
        # pre-baked top_nodes list.  Larger k → more statistical power.
        self.top_k = int(top_k) if top_k and int(top_k) > 0 else None

        _root = Path(__file__).resolve().parents[2]
        self.output_dir = Path(output_dir) if output_dir else (
            _root / "experiments" / "outputs"
        )
        (self.output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)

        self._datasets: list[_OrthoDataset] = []
        self._reference_paths: dict[str, Path] = reference_paths or {}
        self._ref_cache: dict[str, Optional[GeneSetReference]] = {}
        self.records: list[OrthogonalRecord] = []

    # ------------------------------------------------------------------
    def add_dataset(
        self,
        name: str,
        graph_csr: sp.csr_matrix,
        network_type: str,
        node_index_map: dict,
        results: dict[str, dict],
    ) -> None:
        self._datasets.append(_OrthoDataset(
            name           = name,
            graph_csr      = graph_csr,
            network_type   = network_type,
            node_index_map = node_index_map or {},
            result_by_algo = results or {},
        ))

    # ------------------------------------------------------------------
    def _get_reference(self, kind: str) -> Optional[GeneSetReference]:
        if kind in self._ref_cache:
            return self._ref_cache[kind]
        try:
            ref = load_gene_set(kind, self._reference_paths.get(kind))
        except Exception as exc:                       # never raise
            _LOG.warning("Gene-set loader %s raised %s — skipping.", kind, exc)
            ref = None
        self._ref_cache[kind] = ref
        return ref

    # ------------------------------------------------------------------
    def run(self, algorithms: Optional[tuple] = None) -> None:
        algos = tuple(algorithms) if algorithms else (
            _RANKING_ALGOS + _CLUSTER_ALGOS
        )
        if not self._datasets:
            _LOG.warning("OrthogonalValidator: no datasets registered.")
            return

        # Pre-load references once (cached).
        available = {k: self._get_reference(k) for k in self.kinds}

        for ds in self._datasets:
            background = ds.graph_csr.shape[0]
            for kind in self.kinds:
                ref = available.get(kind)
                ref_name = {
                    "disease": "DisGeNET", "essential": "DEG/OGEE",
                    "drug_target": "DrugBank",
                }.get(kind, kind)

                for algo in algos:
                    if algo not in ds.result_by_algo:
                        continue  # nothing to validate for this algo
                    if ref is None:
                        self.records.append(OrthogonalRecord(
                            algorithm=algo, dataset=ds.name,
                            network_type=ds.network_type,
                            reference=ref_name, kind=kind,
                            note=f"{ref_name} reference unavailable",
                            status="skipped",
                        ))
                        continue
                    try:
                        rec = self._validate_one(algo, ds, ref, kind,
                                                 ref_name, background)
                    except Exception as exc:            # never raise
                        rec = OrthogonalRecord(
                            algorithm=algo, dataset=ds.name,
                            network_type=ds.network_type,
                            reference=ref_name, kind=kind,
                            note=str(exc)[:200], status="error",
                        )
                    self.records.append(rec)

    # ------------------------------------------------------------------
    def _validate_one(
        self, algo: str, ds: _OrthoDataset, ref: GeneSetReference,
        kind: str, ref_name: str, background: int,
    ) -> OrthogonalRecord:
        result = ds.result_by_algo.get(algo, {})

        if algo in _CLUSTER_ALGOS:
            return self._validate_communities(
                algo, ds, ref, kind, ref_name, background, result,
            )

        if self.top_k:
            predicted = _extract_topk_predicted(
                algo, result, ds.node_index_map, self.top_k,
                network_type=ds.network_type, graph_csr=ds.graph_csr,
                # miRNA references are GENE sets → rank top target genes,
                # not the miRNA source nodes.
                gene_side=(ds.network_type == "mirna"),
            )
        else:
            predicted = _extract_predicted(algo, result, ds.node_index_map)
        if not predicted:
            return OrthogonalRecord(
                algorithm=algo, dataset=ds.name,
                network_type=ds.network_type, reference=ref_name, kind=kind,
                note="no predicted nodes found in result", status="skipped",
            )

        # Restrict the reference to genes present in the graph — a genome-wide
        # gene set (e.g. 30k disease genes) larger than the graph inverts the
        # Fisher table and forces p ≈ 1 even at precision 1.0.
        universe = _graph_universe(ds.node_index_map)
        ref_genes = (ref.genes & universe) if universe else ref.genes
        bg = len(universe) if universe else background

        ov: OverlapResult = compute_overlap(
            predicted, ref_genes, background_size=bg,
        )
        return OrthogonalRecord(
            algorithm=algo, dataset=ds.name, network_type=ds.network_type,
            reference=ref_name, kind=kind,
            overlap_count=ov.overlap_count, predicted_size=ov.predicted_size,
            reference_size=ov.reference_size, precision=ov.precision,
            recall=ov.recall, jaccard=ov.jaccard, p_value=ov.p_value,
            note="ranking-overlap vs in-graph orthogonal gene set", status="ok",
        )

    def _validate_communities(
        self, algo: str, ds: _OrthoDataset, ref: GeneSetReference,
        kind: str, ref_name: str, background: int, result: dict,
    ) -> OrthogonalRecord:
        communities = _extract_communities(result, ds.node_index_map)
        if not communities:
            return OrthogonalRecord(
                algorithm=algo, dataset=ds.name,
                network_type=ds.network_type, reference=ref_name, kind=kind,
                note="no community structure found in result", status="skipped",
            )

        universe = _graph_universe(ds.node_index_map)
        ref_genes = (ref.genes & universe) if universe else ref.genes
        bg = len(universe) if universe else background
        enriched: list[CommunityEnrichment] = enrich_communities(
            communities, ref_genes, background_size=bg,
            top_k=len(communities),
        )
        total_size = sum(e.size for e in enriched) or 1
        weighted = lambda key: sum(getattr(e, key) * e.size
                                   for e in enriched) / total_size
        best_p = min((e.p_value for e in enriched if math.isfinite(e.p_value)),
                     default=float("nan"))

        nmi_val = ari_val = float("nan")
        partition = _gene_set_partition(ds.node_index_map, ref)
        if partition is not None:
            inner = _inner(result)
            assign = (inner.get("community_assignments")
                      or inner.get("cluster_assignments"))
            if isinstance(assign, list) and len(assign) == len(partition):
                try:
                    nmi_val = float(normalized_mutual_info(assign, partition))
                    ari_val = float(adjusted_rand_index(assign, partition))
                except Exception as exc:
                    _LOG.warning("NMI/ARI failed for %s/%s: %s",
                                 algo, ds.name, exc)

        return OrthogonalRecord(
            algorithm=algo, dataset=ds.name, network_type=ds.network_type,
            reference=ref_name, kind=kind,
            overlap_count=sum(e.overlap_count for e in enriched),
            predicted_size=total_size, reference_size=len(ref.genes),
            precision=weighted("precision"), recall=weighted("recall"),
            jaccard=weighted("jaccard"), p_value=best_p,
            nmi=nmi_val, ari=ari_val,
            note=f"community enrichment over {len(enriched)} groups",
            status="ok",
        )

    # ------------------------------------------------------------------
    def write_csv(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.output_dir / "reports" / "orthogonal_validation.csv"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _fmt(v) -> str:
            if v is None:
                return ""
            if isinstance(v, float):
                return "" if math.isnan(v) else f"{v:.6g}"
            return str(v)

        fields = [
            "algorithm", "dataset", "network_type", "reference", "kind",
            "overlap_count", "predicted_size", "reference_size",
            "precision", "recall", "jaccard", "p_value", "nmi", "ari",
            "note", "status",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.records:
                w.writerow({k: _fmt(getattr(r, k)) for k in fields})
        _LOG.info("Orthogonal-validation CSV: %s", path)
        return path

    # ------------------------------------------------------------------
    def write_plots(self) -> dict[str, Path]:
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        return {"orthogonal_enrichment": self._plot_enrichment(plot_dir)}

    def _plot_enrichment(self, plot_dir: Path) -> Path:
        ok = [r for r in self.records if r.status == "ok"
              and math.isfinite(r.p_value) and r.p_value > 0]
        out = plot_dir / "orthogonal_enrichment.png"

        fig, ax = plt.subplots(figsize=(9, 5))
        if not ok:
            ax.text(0.5, 0.5,
                    "No orthogonal validation records\n"
                    "(gene-set files missing or no results)",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            labels = [f"{r.algorithm}/{r.kind}\n{r.dataset[:12]}" for r in ok]
            yvals  = [-math.log10(max(r.p_value, 1e-300)) for r in ok]
            colours = []
            for r in ok:
                if r.p_value < 1e-5:
                    colours.append("#27ae60")
                elif r.p_value < 0.05:
                    colours.append("#f39c12")
                else:
                    colours.append("#bdc3c7")
            ax.bar(range(len(labels)), yvals, color=colours, alpha=0.85)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel(r"$-\log_{10}(p\text{-value})$")
            ax.axhline(-math.log10(0.05), color="#e74c3c", linestyle="--",
                       linewidth=1, label="p = 0.05")
            ax.set_title("Orthogonal Enrichment "
                         "(disease / essential / drug-target genes)")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _LOG.info("Plot: %s", out)
        return out


__all__ = ["OrthogonalValidator", "OrthogonalRecord", "DEFAULT_KINDS"]
