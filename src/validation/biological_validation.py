"""
src/validation/biological_validation.py
========================================

Biological validation layer.  Compares algorithm output against
network-type-appropriate reference databases that are already known to
contain biologically meaningful interactions:

    network_type  →  reference        →  what is validated
    ------------     -------------       -----------------------------
    grn           →  TRRUST           →  PageRank top_regulators,
                                          HITS top_hubs,
                                          RWR top_nodes,
                                          Louvain / MCL communities
    ppi           →  BioGRID          →  PageRank top_nodes,
                                          HITS top_nodes,
                                          RWR top_nodes,
                                          Louvain / MCL communities
    mirna         →  miRTarBase       →  PageRank top_mirnas,
                                          HITS top_hubs,
                                          RWR top_nodes,
                                          Louvain / MCL communities

Metrics computed (all per algorithm × dataset):
    overlap_count   – |predicted ∩ reference|
    precision       – |predicted ∩ reference| / |predicted|
    recall          – |predicted ∩ reference| / |reference|
    jaccard         – |∩| / |∪|
    p_value         – one-sided Fisher exact (enrichment, alternative='greater')
    nmi             – community / reference partition agreement (clustering algos only)
    ari             – adjusted Rand index (clustering algos only)

Outputs
-------
    experiments/outputs/reports/biological_validation.csv
        columns: algorithm, dataset, network_type, reference,
                 overlap_count, predicted_size, reference_size,
                 precision, recall, jaccard, p_value, nmi, ari,
                 note, status

    experiments/outputs/plots/
        enrichment_barplot.png    – -log10(p_value) per algorithm
        enrichment_heatmap.png    – precision/recall/jaccard heatmap
        community_validation.png  – NMI / ARI per (algorithm, dataset)

Rules
-----
* NEVER raises — all failures are recorded with ``status='error'`` and
  ``note=<error string>``.
* Skips gracefully when the reference database is not available.
* Uses ONLY the three project-aligned databases (TRRUST / BioGRID /
  miRTarBase).  No KEGG, no GO, no external network calls.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import scipy.sparse as sp

from src.validation.reference_loader import (
    ReferenceSet, load_reference,
)
from src.validation.overlap import compute_overlap, OverlapResult
from src.validation.enrichment import enrich_communities, CommunityEnrichment
from src.validation.metrics import (
    normalized_mutual_info, adjusted_rand_index,
)

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALGORITHMS: tuple[str, ...] = (
    "pagerank", "hits", "rwr", "louvain", "mcl",
)

_REFERENCE_BY_NETWORK: dict[str, str] = {
    "grn":   "TRRUST",
    "ppi":   "BioGRID",
    "mirna": "miRTarBase",
}

# Result-dict fields containing predicted nodes that should be compared
# against the reference set.  First-match-wins per algorithm.
_PREDICTED_FIELDS: dict[str, tuple[str, ...]] = {
    "pagerank": ("top_regulators", "top_nodes", "top_mirnas"),
    "hits":     ("top_hubs", "top_nodes"),
    "rwr":      ("top_nodes",),
    "louvain":  ("top_communities",),   # special handling
    "mcl":      ("top_clusters", "top_communities", "cluster_assignments"),
}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class BioValidationRecord:
    algorithm:      str
    dataset:        str
    network_type:   str
    reference:      str            = ""
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
class _BioDataset:
    name:            str
    graph_csr:       sp.csr_matrix
    network_type:    str
    node_index_map:  dict
    result_by_algo:  dict[str, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_labels(items: Any) -> list[str]:
    """
    Convert a result list (either ``[idx, idx, ...]`` or
    ``[{"index": int, "label": str}, ...]``) into a list of label strings.

    Falls back to ``"node_<idx>"`` when no label is attached.
    """
    out: list[str] = []
    if not items:
        return out
    for it in items:
        if isinstance(it, dict):
            lbl = it.get("label")
            if lbl is None and "index" in it:
                lbl = f"node_{it['index']}"
            if lbl is not None:
                out.append(str(lbl).strip().upper())
        else:
            # Plain integer index — no label info available
            out.append(f"NODE_{it}")
    return out


def _inner(result: Any) -> dict:
    """Extract the inner result dict from a 7-key envelope or raw dict."""
    if not isinstance(result, dict):
        return {}
    inner = result.get("result")
    if isinstance(inner, dict):
        return inner
    return result


def _extract_predicted(algorithm: str, result: dict) -> list[str]:
    """Return labels for the predicted set of the given algorithm."""
    inner = _inner(result)
    fields = _PREDICTED_FIELDS.get(algorithm, ())
    for f in fields:
        if f in inner and isinstance(inner[f], list) and inner[f]:
            return _safe_labels(inner[f])
    return []


def _extract_communities(
    result: dict, node_index_map: dict,
) -> dict[int, list[str]]:
    """
    Build {community_id: [labels...]} from a louvain / mcl result.

    Prefers ``top_communities`` (already includes label dicts) when
    present; otherwise reconstructs from ``community_assignments`` /
    ``cluster_assignments`` using ``node_index_map``.
    """
    inner = _inner(result)
    out: dict[int, list[str]] = {}

    # Reverse map: int → label
    idx_to_label = {int(v): str(k).strip().upper()
                    for k, v in (node_index_map or {}).items()}

    # Preferred: top_communities[] with member_nodes
    top = inner.get("top_communities") or inner.get("top_clusters")
    if isinstance(top, list) and top:
        for entry in top:
            if not isinstance(entry, dict):
                continue
            cid = int(entry.get("community_id",
                                entry.get("cluster_id",
                                          entry.get("id", len(out)))))
            members = entry.get("member_nodes") or entry.get("members") or []
            labels = _safe_labels(members)
            if not labels and members:
                labels = [idx_to_label.get(int(m), f"NODE_{m}")
                          for m in members if isinstance(m, (int, np.integer))]
            if labels:
                out[cid] = labels

    if out:
        return out

    # Fallback: cluster_assignments / community_assignments per-node
    assign = (inner.get("community_assignments")
              or inner.get("cluster_assignments"))
    if isinstance(assign, list) and assign:
        for i, cid in enumerate(assign):
            try:
                cid_int = int(cid)
            except (TypeError, ValueError):
                continue
            out.setdefault(cid_int, []).append(
                idx_to_label.get(int(i), f"NODE_{i}")
            )
    return out


def _reference_partition(node_index_map: dict,
                         ref: ReferenceSet) -> Optional[list[int]]:
    """
    Build a binary reference partition over graph nodes:
    1 = node is in the reference set, 0 = not.

    Returns None when the overlap between graph nodes and reference
    nodes is empty.
    """
    if not node_index_map or ref is None or ref.is_empty():
        return None

    ref_nodes = ref.all_nodes
    n = max(node_index_map.values()) + 1
    partition = [0] * n
    hits = 0
    for label, idx in node_index_map.items():
        if str(label).strip().upper() in ref_nodes:
            partition[int(idx)] = 1
            hits += 1
    if hits == 0:
        return None
    return partition


# ---------------------------------------------------------------------------
# BiologicalValidator
# ---------------------------------------------------------------------------

class BiologicalValidator:
    """
    Evaluate algorithm outputs against project-aligned biological references.

    Workflow
    --------
        bv = BiologicalValidator(output_dir=...)
        bv.add_dataset(name='string_ppi', graph_csr=..., network_type='ppi',
                       node_index_map=..., results={'pagerank': result, ...})
        bv.run()
        bv.write_csv()
        bv.write_plots()

    Reference lookup is automatic per network_type.  When the relevant
    reference file is missing every record for that dataset is marked
    ``status='skipped'``.

    The validator NEVER raises.  Errors during metric computation are
    captured and recorded with ``status='error'`` + ``note``.
    """

    def __init__(
        self,
        algorithms: tuple[str, ...] = ALGORITHMS,
        output_dir: Optional[Path] = None,
        reference_paths: Optional[dict[str, Path]] = None,
    ) -> None:
        self.algorithms = tuple(algorithms)

        _root = Path(__file__).resolve().parents[2]
        self.output_dir = Path(output_dir) if output_dir else (
            _root / "experiments" / "outputs"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

        self._datasets: list[_BioDataset] = []
        self._reference_paths: dict[str, Path] = reference_paths or {}
        self._ref_cache: dict[str, Optional[ReferenceSet]] = {}
        self.records: list[BioValidationRecord] = []

    # ------------------------------------------------------------------
    # Dataset registration
    # ------------------------------------------------------------------

    def add_dataset(
        self,
        name: str,
        graph_csr: sp.csr_matrix,
        network_type: str,
        node_index_map: dict,
        results: dict[str, dict],
    ) -> None:
        """
        Register a dataset together with its already-computed algorithm
        results.  ``results`` is a mapping ``algorithm_name -> result_dict``.
        """
        self._datasets.append(_BioDataset(
            name           = name,
            graph_csr      = graph_csr,
            network_type   = network_type,
            node_index_map = node_index_map or {},
            result_by_algo = results or {},
        ))

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, algorithms: Optional[tuple] = None) -> None:
        algos = tuple(algorithms) if algorithms else self.algorithms
        if not self._datasets:
            _LOG.warning("BiologicalValidator: no datasets registered.")
            return

        for ds in self._datasets:
            ref = self._get_reference(ds.network_type)
            ref_name = _REFERENCE_BY_NETWORK.get(ds.network_type, "?")
            background_size = ds.graph_csr.shape[0]

            for algo in algos:
                if algo not in ds.result_by_algo:
                    self.records.append(BioValidationRecord(
                        algorithm    = algo,
                        dataset      = ds.name,
                        network_type = ds.network_type,
                        reference    = ref_name,
                        note         = "no algorithm result provided",
                        status       = "skipped",
                    ))
                    continue

                if ref is None:
                    self.records.append(BioValidationRecord(
                        algorithm    = algo,
                        dataset      = ds.name,
                        network_type = ds.network_type,
                        reference    = ref_name,
                        note         = f"{ref_name} reference unavailable",
                        status       = "skipped",
                    ))
                    continue

                try:
                    rec = self._validate_one(
                        algo, ds, ref, background_size,
                    )
                except Exception as exc:  # NEVER raise
                    rec = BioValidationRecord(
                        algorithm    = algo,
                        dataset      = ds.name,
                        network_type = ds.network_type,
                        reference    = ref_name,
                        note         = str(exc)[:200],
                        status       = "error",
                    )
                self.records.append(rec)

    # ------------------------------------------------------------------
    # Per-(algorithm, dataset) validation
    # ------------------------------------------------------------------

    def _validate_one(
        self,
        algo: str,
        ds: _BioDataset,
        ref: ReferenceSet,
        background_size: int,
    ) -> BioValidationRecord:
        ref_name = ref.name
        result   = ds.result_by_algo.get(algo, {})

        if algo in ("louvain", "mcl"):
            return self._validate_communities(
                algo, ds, ref, background_size, result,
            )

        # ── Ranking algorithms (pagerank/hits/rwr) ──
        predicted = _extract_predicted(algo, result)
        if not predicted:
            return BioValidationRecord(
                algorithm    = algo,
                dataset      = ds.name,
                network_type = ds.network_type,
                reference    = ref_name,
                note         = "no predicted nodes found in result",
                status       = "skipped",
            )

        # Choose reference subset matching the algorithm intent
        if algo == "hits" and ds.network_type in ("grn", "mirna"):
            ref_set = ref.sources                 # hubs = TFs / miRNAs
        elif algo == "pagerank" and ds.network_type == "grn":
            ref_set = ref.sources                 # top_regulators ≈ TFs
        elif algo == "pagerank" and ds.network_type == "mirna":
            ref_set = ref.sources                 # top_mirnas ≈ miRNAs
        else:
            ref_set = ref.all_nodes               # generic: any reference node

        ov: OverlapResult = compute_overlap(
            predicted, ref_set, background_size=background_size,
        )
        return BioValidationRecord(
            algorithm      = algo,
            dataset        = ds.name,
            network_type   = ds.network_type,
            reference      = ref_name,
            overlap_count  = ov.overlap_count,
            predicted_size = ov.predicted_size,
            reference_size = ov.reference_size,
            precision      = ov.precision,
            recall         = ov.recall,
            jaccard        = ov.jaccard,
            p_value        = ov.p_value,
            note           = "ranking-overlap vs reference",
            status         = "ok",
        )

    def _validate_communities(
        self,
        algo: str,
        ds: _BioDataset,
        ref: ReferenceSet,
        background_size: int,
        result: dict,
    ) -> BioValidationRecord:
        communities = _extract_communities(result, ds.node_index_map)
        if not communities:
            return BioValidationRecord(
                algorithm    = algo,
                dataset      = ds.name,
                network_type = ds.network_type,
                reference    = ref.name,
                note         = "no community structure found in result",
                status       = "skipped",
            )

        # Enrichment of each community vs reference source set
        ref_set = ref.sources or ref.all_nodes
        enriched: list[CommunityEnrichment] = enrich_communities(
            communities, ref_set, background_size=background_size,
            top_k=len(communities),
        )

        # Aggregate: precision/recall/jaccard weighted by community size
        total_size  = sum(e.size for e in enriched) or 1
        weighted = lambda key: sum(getattr(e, key) * e.size for e in enriched) / total_size
        avg_precision = weighted("precision")
        avg_recall    = weighted("recall")
        avg_jaccard   = weighted("jaccard")
        total_overlap = sum(e.overlap_count for e in enriched)

        # Best community p-value (representative of enrichment significance)
        best_p = min((e.p_value for e in enriched if math.isfinite(e.p_value)),
                     default=float("nan"))

        # NMI / ARI against the binary reference partition
        nmi_val = float("nan")
        ari_val = float("nan")
        ref_partition = _reference_partition(ds.node_index_map, ref)
        if ref_partition is not None:
            inner = _inner(result)
            assign = (inner.get("community_assignments")
                      or inner.get("cluster_assignments"))
            if isinstance(assign, list) and len(assign) == len(ref_partition):
                try:
                    nmi_val = float(normalized_mutual_info(assign, ref_partition))
                    ari_val = float(adjusted_rand_index(assign, ref_partition))
                except Exception as exc:
                    _LOG.warning("NMI/ARI failed for %s/%s: %s",
                                 algo, ds.name, exc)

        return BioValidationRecord(
            algorithm      = algo,
            dataset        = ds.name,
            network_type   = ds.network_type,
            reference      = ref.name,
            overlap_count  = total_overlap,
            predicted_size = total_size,
            reference_size = len(ref_set),
            precision      = avg_precision,
            recall         = avg_recall,
            jaccard        = avg_jaccard,
            p_value        = best_p,
            nmi            = nmi_val,
            ari            = ari_val,
            note           = f"community enrichment over {len(enriched)} groups",
            status         = "ok",
        )

    # ------------------------------------------------------------------
    # Reference loading (cached per network_type)
    # ------------------------------------------------------------------

    def _get_reference(self, network_type: str) -> Optional[ReferenceSet]:
        if network_type in self._ref_cache:
            return self._ref_cache[network_type]
        try:
            ref = load_reference(
                network_type,
                self._reference_paths.get(network_type),
            )
        except Exception as exc:
            _LOG.warning("Reference loader for %s raised %s — skipping.",
                         network_type, exc)
            ref = None
        self._ref_cache[network_type] = ref
        return ref

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------

    def write_csv(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.output_dir / "reports" / "biological_validation.csv"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _fmt(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, float):
                if math.isnan(v):
                    return ""
                return f"{v:.6g}"
            return str(v)

        fields = [
            "algorithm", "dataset", "network_type", "reference",
            "overlap_count", "predicted_size", "reference_size",
            "precision", "recall", "jaccard", "p_value",
            "nmi", "ari", "note", "status",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.records:
                w.writerow({
                    "algorithm":      r.algorithm,
                    "dataset":        r.dataset,
                    "network_type":   r.network_type,
                    "reference":      r.reference,
                    "overlap_count":  r.overlap_count,
                    "predicted_size": r.predicted_size,
                    "reference_size": r.reference_size,
                    "precision":      _fmt(r.precision),
                    "recall":         _fmt(r.recall),
                    "jaccard":        _fmt(r.jaccard),
                    "p_value":        _fmt(r.p_value),
                    "nmi":            _fmt(r.nmi),
                    "ari":            _fmt(r.ari),
                    "note":           r.note,
                    "status":         r.status,
                })
        _LOG.info("Biological-validation CSV: %s", path)
        return path

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def write_plots(self) -> dict[str, Path]:
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        return {
            "enrichment_barplot":   self._plot_barplot(plot_dir),
            "enrichment_heatmap":   self._plot_heatmap(plot_dir),
            "community_validation": self._plot_community(plot_dir),
        }

    # --- enrichment_barplot.png ---------------------------------------

    def _plot_barplot(self, plot_dir: Path) -> Path:
        ok = [r for r in self.records if r.status == "ok"
              and math.isfinite(r.p_value) and r.p_value > 0]
        out = plot_dir / "enrichment_barplot.png"

        fig, ax = plt.subplots(figsize=(8, 5))
        if not ok:
            ax.text(0.5, 0.5, "No biological validation records available\n"
                              "(reference files missing or no algorithm results)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10)
        else:
            labels = [f"{r.algorithm}\n{r.dataset[:14]}" for r in ok]
            yvals  = [-math.log10(max(r.p_value, 1e-300)) for r in ok]
            colors = []
            for r in ok:
                if r.p_value < 1e-5:
                    colors.append("#27ae60")
                elif r.p_value < 0.05:
                    colors.append("#f39c12")
                else:
                    colors.append("#bdc3c7")
            ax.bar(range(len(labels)), yvals, color=colors, alpha=0.85)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(r"$-\log_{10}(p\text{-value})$")
            ax.axhline(-math.log10(0.05),  color="#e74c3c", linestyle="--",
                       linewidth=1, label="p = 0.05")
            ax.axhline(-math.log10(1e-5),  color="#2ecc71", linestyle="--",
                       linewidth=1, label="p = 1e-5")
            ax.set_title("Biological Enrichment (Fisher exact, one-sided)")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _LOG.info("Plot: %s", out)
        return out

    # --- enrichment_heatmap.png ---------------------------------------

    def _plot_heatmap(self, plot_dir: Path) -> Path:
        ok = [r for r in self.records if r.status == "ok"]
        out = plot_dir / "enrichment_heatmap.png"

        fig, ax = plt.subplots(figsize=(8, 5))
        if not ok:
            ax.text(0.5, 0.5, "No biological validation records",
                    ha="center", va="center", transform=ax.transAxes)
            fig.tight_layout()
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return out

        algos    = sorted({r.algorithm for r in ok})
        datasets = sorted({r.dataset   for r in ok})
        metrics  = ["precision", "recall", "jaccard"]

        # Aggregate (mean across metrics): rows = (algo, dataset), cols = metric
        rows = [(a, d) for a in algos for d in datasets]
        grid = np.full((len(rows), len(metrics)), np.nan)
        for i, (a, d) in enumerate(rows):
            for j, m in enumerate(metrics):
                vals = [getattr(r, m) for r in ok
                        if r.algorithm == a and r.dataset == d
                        and math.isfinite(getattr(r, m))]
                if vals:
                    grid[i, j] = float(np.mean(vals))

        # Drop fully-NaN rows
        keep = ~np.all(np.isnan(grid), axis=1)
        grid = grid[keep]
        rows = [r for r, k in zip(rows, keep) if k]
        if grid.size == 0:
            ax.text(0.5, 0.5, "No finite enrichment metrics",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            im = ax.imshow(grid, aspect="auto", cmap="viridis",
                           vmin=0, vmax=max(np.nanmax(grid), 0.01))
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(metrics, fontsize=9)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([f"{a} | {d[:18]}" for a, d in rows],
                               fontsize=8)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if math.isfinite(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j]:.3f}",
                                ha="center", va="center",
                                color="white" if grid[i, j] < 0.5 * np.nanmax(grid)
                                else "black", fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.04)
            ax.set_title("Enrichment Metrics (algorithm × dataset)")

        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _LOG.info("Plot: %s", out)
        return out

    # --- community_validation.png -------------------------------------

    def _plot_community(self, plot_dir: Path) -> Path:
        ok = [r for r in self.records
              if r.status == "ok"
              and r.algorithm in ("louvain", "mcl")
              and (math.isfinite(r.nmi) or math.isfinite(r.ari))]
        out = plot_dir / "community_validation.png"

        fig, ax = plt.subplots(figsize=(8, 5))
        if not ok:
            ax.text(0.5, 0.5, "No community validation data\n"
                              "(louvain / mcl results missing or no reference)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10)
        else:
            labels = [f"{r.algorithm}\n{r.dataset[:14]}" for r in ok]
            x = np.arange(len(labels))
            width = 0.4
            nmi_vals = [r.nmi if math.isfinite(r.nmi) else 0.0 for r in ok]
            ari_vals = [r.ari if math.isfinite(r.ari) else 0.0 for r in ok]
            ax.bar(x - width / 2, nmi_vals, width, label="NMI", color="#4a90e2")
            ax.bar(x + width / 2, ari_vals, width, label="ARI", color="#bd10e0")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Score")
            ax.set_title("Community Validation vs Reference Partition")
            ax.set_ylim(-0.1, 1.05)
            ax.axhline(0.0, color="black", linewidth=0.5)
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _LOG.info("Plot: %s", out)
        return out
