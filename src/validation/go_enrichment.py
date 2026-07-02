"""
src/validation/go_enrichment.py
===============================

Gene Ontology (GO) term enrichment validation.

GO annotations describe what a gene DOES (biological process, molecular
function, cellular component) based on sequence / experimental evidence
that is independent of the interaction network.  If an algorithm's
communities or top-ranked nodes are enriched for coherent GO terms, the
structure it found is functionally meaningful — not a topological artefact.

Per gene set (a community, or a ranking algorithm's top-node set) we test
every GO term annotating the set with a one-sided Fisher-exact test against
the annotated background (graph nodes that carry any GO annotation), then
apply a Bonferroni correction over the number of terms tested.  The best
(smallest corrected p-value) term is reported.

Validated per (algorithm × dataset):
    clustering (louvain / mcl)
        each community enriched separately; report the best term across
        communities and how many communities are significantly enriched.
    ranking (pagerank / hits / rwr)
        the top-node set enriched as a single group.

Outputs
-------
    experiments/outputs/reports/go_enrichment.csv
    experiments/outputs/plots/go_enrichment.png

Requires a local GO annotation file (GAF 2.x, e.g. ``goa_human.gaf``) placed
in ``data/references/`` or pointed to via ``FYP_GO_PATH``.  All processing is
local — no network calls.

Rules
-----
* NEVER raises — failures recorded with ``status='error'``.
* Skips gracefully when the GO file is unavailable.
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
import scipy.sparse as sp

from src.validation.reference_loader import (
    GeneSetReference, load_go_annotations,
)
from src.validation.overlap import fisher_exact_p
from src.validation.biological_validation import (
    _extract_predicted, _extract_communities,
)

_LOG = logging.getLogger(__name__)

_RANKING_ALGOS = ("pagerank", "hits", "rwr")
_CLUSTER_ALGOS = ("louvain", "mcl")

_MIN_TERM_SIZE = 3      # ignore GO terms annotating < 3 background genes
_MAX_TERM_SIZE = 1000   # ignore near-universal terms (uninformative)
_SIG_ALPHA     = 0.05


@dataclass
class GOEnrichmentResult:
    term:       str = ""
    p_corrected: float = float("nan")
    p_raw:      float = float("nan")
    overlap:    int = 0
    term_size:  int = 0
    n_tested:   int = 0


@dataclass
class GORecord:
    algorithm:            str
    dataset:              str
    network_type:         str
    best_term:            str   = ""
    best_term_p:          float = float("nan")   # Bonferroni-corrected
    overlap:              int   = 0
    term_size:            int   = 0
    n_groups_tested:      int   = 0
    n_groups_significant: int   = 0
    note:                 str   = ""
    status:               str   = "ok"           # ok | skipped | error


@dataclass
class _GODataset:
    name:            str
    graph_csr:       sp.csr_matrix
    network_type:    str
    node_index_map:  dict
    result_by_algo:  dict[str, dict] = field(default_factory=dict)


def _enrich_gene_set(
    genes: set[str], go: GeneSetReference, universe: set[str],
) -> Optional[GOEnrichmentResult]:
    """Best Bonferroni-corrected GO term for a single gene set."""
    genes_in = genes & universe
    if not genes_in:
        return None

    # Count how many set genes each candidate term annotates.
    term_hits: dict[str, int] = {}
    for g in genes_in:
        for t in go.gene_terms.get(g, ()):
            term_hits[t] = term_hits.get(t, 0) + 1

    n_universe = len(universe)
    n_draws    = len(genes_in)
    best: Optional[GOEnrichmentResult] = None
    tested = 0
    for term, a in term_hits.items():
        term_size = len(go.term_genes.get(term, set()) & universe)
        if term_size < _MIN_TERM_SIZE or term_size > _MAX_TERM_SIZE:
            continue
        tested += 1
        p = fisher_exact_p(
            a=a, b=n_draws - a,
            c=term_size - a, d=n_universe - n_draws - term_size + a,
        )
        if best is None or p < best.p_raw:
            best = GOEnrichmentResult(term=term, p_raw=p, overlap=a,
                                      term_size=term_size)
    if best is None:
        return None
    best.n_tested = tested
    best.p_corrected = min(1.0, best.p_raw * max(tested, 1))
    return best


class GOEnrichmentValidator:
    """Validate communities / top nodes by GO-term enrichment.

    Usage::

        gv = GOEnrichmentValidator(output_dir=..., aspects={"P"})
        gv.add_dataset(name=..., graph_csr=..., network_type=...,
                       node_index_map=..., results={...})
        gv.run(); gv.write_csv(); gv.write_plots()

    Never raises.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        reference_path: Optional[Path] = None,
        aspects: Optional[set[str]] = None,
    ) -> None:
        _root = Path(__file__).resolve().parents[2]
        self.output_dir = Path(output_dir) if output_dir else (
            _root / "experiments" / "outputs"
        )
        (self.output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)

        self._reference_path = reference_path
        self._aspects = aspects
        self._go: Optional[GeneSetReference] = None
        self._go_loaded = False
        self._datasets: list[_GODataset] = []
        self.records: list[GORecord] = []

    # ------------------------------------------------------------------
    def add_dataset(
        self,
        name: str,
        graph_csr: sp.csr_matrix,
        network_type: str,
        node_index_map: dict,
        results: dict[str, dict],
    ) -> None:
        self._datasets.append(_GODataset(
            name=name, graph_csr=graph_csr,
            network_type=network_type, node_index_map=node_index_map or {},
            result_by_algo=results or {},
        ))

    def _get_go(self) -> Optional[GeneSetReference]:
        if not self._go_loaded:
            try:
                self._go = load_go_annotations(self._reference_path,
                                               self._aspects)
            except Exception as exc:
                _LOG.warning("GO loader raised %s — skipping.", exc)
                self._go = None
            self._go_loaded = True
        return self._go

    def _universe(self, node_index_map: dict,
                  go: GeneSetReference) -> set[str]:
        """Annotated background: graph node labels that carry a GO term."""
        return {str(l).strip().upper() for l in node_index_map
                if str(l).strip().upper() in go.gene_terms}

    # ------------------------------------------------------------------
    def run(self, algorithms: Optional[tuple] = None) -> None:
        algos = tuple(algorithms) if algorithms else (
            _RANKING_ALGOS + _CLUSTER_ALGOS
        )
        if not self._datasets:
            _LOG.warning("GOEnrichmentValidator: no datasets registered.")
            return

        go = self._get_go()
        for ds in self._datasets:
            universe = self._universe(ds.node_index_map, go) if go else set()
            for algo in algos:
                if algo not in ds.result_by_algo:
                    continue
                if go is None or not universe:
                    self.records.append(GORecord(
                        algorithm=algo, dataset=ds.name,
                        network_type=ds.network_type,
                        note=("GO file unavailable" if go is None
                              else "no graph nodes carry GO annotations"),
                        status="skipped",
                    ))
                    continue
                try:
                    rec = self._validate_one(algo, ds, go, universe)
                except Exception as exc:
                    rec = GORecord(
                        algorithm=algo, dataset=ds.name,
                        network_type=ds.network_type,
                        note=str(exc)[:200], status="error",
                    )
                self.records.append(rec)

    def _validate_one(
        self, algo: str, ds: _GODataset, go: GeneSetReference, universe: set,
    ) -> GORecord:
        result = ds.result_by_algo.get(algo, {})

        if algo in _CLUSTER_ALGOS:
            communities = _extract_communities(result, ds.node_index_map)
            groups = [set(v) for v in communities.values() if v]
            note = f"GO enrichment over {len(groups)} communities"
        else:
            predicted = _extract_predicted(algo, result)
            groups = [set(predicted)] if predicted else []
            note = "GO enrichment of top-node set"

        if not groups:
            return GORecord(
                algorithm=algo, dataset=ds.name, network_type=ds.network_type,
                note="no gene set found in result", status="skipped",
            )

        best: Optional[GOEnrichmentResult] = None
        n_tested = 0
        n_sig = 0
        for grp in groups:
            res = _enrich_gene_set(grp, go, universe)
            if res is None:
                continue
            n_tested += 1
            if math.isfinite(res.p_corrected) and res.p_corrected < _SIG_ALPHA:
                n_sig += 1
            if best is None or res.p_corrected < best.p_corrected:
                best = res

        if best is None:
            return GORecord(
                algorithm=algo, dataset=ds.name, network_type=ds.network_type,
                note="no testable GO terms in any group", status="skipped",
            )

        return GORecord(
            algorithm=algo, dataset=ds.name, network_type=ds.network_type,
            best_term=best.term, best_term_p=best.p_corrected,
            overlap=best.overlap, term_size=best.term_size,
            n_groups_tested=n_tested, n_groups_significant=n_sig,
            note=note, status="ok",
        )

    # ------------------------------------------------------------------
    def write_csv(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.output_dir / "reports" / "go_enrichment.csv"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _fmt(v) -> str:
            if v is None:
                return ""
            if isinstance(v, float):
                return "" if math.isnan(v) else f"{v:.6g}"
            return str(v)

        fields = [
            "algorithm", "dataset", "network_type", "best_term",
            "best_term_p", "overlap", "term_size", "n_groups_tested",
            "n_groups_significant", "note", "status",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.records:
                w.writerow({k: _fmt(getattr(r, k)) for k in fields})
        _LOG.info("GO enrichment CSV: %s", path)
        return path

    def write_plots(self) -> dict[str, Path]:
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        return {"go_enrichment": self._plot(plot_dir)}

    def _plot(self, plot_dir: Path) -> Path:
        out = plot_dir / "go_enrichment.png"
        ok = [r for r in self.records if r.status == "ok"
              and math.isfinite(r.best_term_p) and r.best_term_p > 0]

        fig, ax = plt.subplots(figsize=(9, 5))
        if not ok:
            ax.text(0.5, 0.5,
                    "No GO enrichment records\n"
                    "(GO annotation file missing or no annotated nodes)",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            labels = [f"{r.algorithm}\n{r.dataset[:12]}" for r in ok]
            yvals = [-math.log10(max(r.best_term_p, 1e-300)) for r in ok]
            colours = ["#27ae60" if r.best_term_p < _SIG_ALPHA else "#bdc3c7"
                       for r in ok]
            ax.bar(range(len(labels)), yvals, color=colours, alpha=0.85)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.axhline(-math.log10(_SIG_ALPHA), color="#e74c3c",
                       linestyle="--", linewidth=1, label="p = 0.05")
            ax.set_ylabel(r"$-\log_{10}(\text{Bonferroni } p)$")
            ax.set_title("Best GO-term enrichment per algorithm")
            # annotate best term names
            for i, r in enumerate(ok):
                ax.text(i, yvals[i] + 0.05, r.best_term, ha="center",
                        va="bottom", fontsize=6, rotation=90)
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _LOG.info("Plot: %s", out)
        return out


__all__ = ["GOEnrichmentValidator", "GORecord", "GOEnrichmentResult"]
