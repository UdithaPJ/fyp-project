"""
src/validation
==============

Validation utilities for the FYP framework.

Submodules
----------

* :mod:`src.validation.metrics` — pure metric functions comparing two
  algorithm result dicts (Spearman, Pearson, MAE/RMSE, NMI, ARI, etc.)
* :mod:`src.validation.cross_implementation_validation` —
  :class:`CrossImplementationValidator` runs all four implementation
  modes (cpu_single / cpu_multi / gpu_baseline / gpu) on each dataset,
  builds pair-wise comparisons, writes a long-format CSV, and produces
  correlation / error / summary plots.
* :mod:`src.validation.reference_loader` — loaders for the three
  biological reference databases (TRRUST / BioGRID / miRTarBase).
* :mod:`src.validation.overlap` — set-overlap metrics (precision,
  recall, jaccard, Fisher exact p-value).
* :mod:`src.validation.enrichment` — community-vs-reference Fisher
  enrichment.
* :mod:`src.validation.biological_validation` —
  :class:`BiologicalValidator` evaluates algorithm output against
  network-type-appropriate references and produces enrichment plots.
"""

from src.validation.metrics import (
    compute_metrics,
    pagerank_metrics,
    rwr_metrics,
    hits_metrics,
    bfs_metrics,
    louvain_metrics,
    mcl_metrics,
    normalized_mutual_info,
    adjusted_rand_index,
)
from src.validation.cross_implementation_validation import (
    CrossImplementationValidator,
    ValidationDataset,
    RunRecord,
    ALGORITHMS,
    MODES,
    COMPARISON_PAIRS,
)
from src.validation.reference_loader import (
    ReferenceSet,
    load_reference,
    load_trrust,
    load_biogrid,
    load_mirtarbase,
    GeneSetReference,
    load_gene_set,
    load_disgenet,
    load_deg,
    load_drugbank,
    load_go_annotations,
)
from src.validation.overlap import (
    compute_overlap,
    OverlapResult,
    fisher_exact_p,
)
from src.validation.enrichment import (
    enrich_communities,
    CommunityEnrichment,
)
from src.validation.biological_validation import (
    BiologicalValidator,
    BioValidationRecord,
)
from src.validation.orthogonal_validation import (
    OrthogonalValidator,
    OrthogonalRecord,
    DEFAULT_KINDS,
)
from src.validation.holdout_validation import (
    HoldoutValidator,
    HoldoutRecord,
)
from src.validation.go_enrichment import (
    GOEnrichmentValidator,
    GORecord,
    GOEnrichmentResult,
)

__all__ = [
    # metrics
    "compute_metrics",
    "pagerank_metrics",
    "rwr_metrics",
    "hits_metrics",
    "bfs_metrics",
    "louvain_metrics",
    "mcl_metrics",
    "normalized_mutual_info",
    "adjusted_rand_index",
    # cross-implementation
    "CrossImplementationValidator",
    "ValidationDataset",
    "RunRecord",
    "ALGORITHMS",
    "MODES",
    "COMPARISON_PAIRS",
    # biological
    "ReferenceSet",
    "load_reference",
    "load_trrust",
    "load_biogrid",
    "load_mirtarbase",
    "GeneSetReference",
    "load_gene_set",
    "load_disgenet",
    "load_deg",
    "load_drugbank",
    "load_go_annotations",
    "compute_overlap",
    "OverlapResult",
    "fisher_exact_p",
    "enrich_communities",
    "CommunityEnrichment",
    "BiologicalValidator",
    "BioValidationRecord",
    # orthogonal / hold-out / GO validators
    "OrthogonalValidator",
    "OrthogonalRecord",
    "DEFAULT_KINDS",
    "HoldoutValidator",
    "HoldoutRecord",
    "GOEnrichmentValidator",
    "GORecord",
    "GOEnrichmentResult",
]
