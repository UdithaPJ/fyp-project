"""
src/validation/enrichment.py
============================

Helpers for computing community / cluster enrichment against a
reference set of biologically meaningful gene sets.

Given:
    communities      – ``dict[int, list[str]]`` from Louvain / MCL output
                       (community id → list of node labels)
    reference_genes  – ``set[str]`` of known regulators / hubs

For each community, we compute the Fisher exact p-value of the overlap
between that community and the reference, then return the most
significant communities.  NMI / ARI between two partitions (e.g. the
algorithm's communities and an external reference labelling) are also
exposed for completeness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.validation.overlap import compute_overlap, OverlapResult


@dataclass
class CommunityEnrichment:
    community_id:   int
    size:           int
    overlap_count:  int
    precision:      float
    recall:         float
    jaccard:        float
    p_value:        float


def enrich_communities(
    communities: dict[int, list[str]],
    reference_genes: set[str],
    background_size: Optional[int] = None,
    top_k: int = 20,
) -> list[CommunityEnrichment]:
    """
    For each community, compute its Fisher exact enrichment against
    ``reference_genes``.  Returns the top-K communities sorted by
    ascending p-value.

    Parameters
    ----------
    communities:
        Mapping community_id → list of node labels (strings).
    reference_genes:
        Set of known reference genes (e.g. TRRUST TFs, BioGRID hubs).
    background_size:
        Total universe size (e.g. number of nodes in the graph). If
        None, defaults to the sum of all community sizes.
    top_k:
        Return at most this many enriched communities.

    Returns
    -------
    list[CommunityEnrichment]  sorted by p_value ascending.
    """
    if not communities or not reference_genes:
        return []

    if background_size is None or background_size <= 0:
        background_size = sum(len(members) for members in communities.values())

    out: list[CommunityEnrichment] = []
    for cid, members in communities.items():
        if not members:
            continue
        r: OverlapResult = compute_overlap(
            predicted       = members,
            reference       = reference_genes,
            background_size = background_size,
        )
        out.append(CommunityEnrichment(
            community_id  = cid,
            size          = len(members),
            overlap_count = r.overlap_count,
            precision     = r.precision,
            recall        = r.recall,
            jaccard       = r.jaccard,
            p_value       = r.p_value,
        ))

    out.sort(key=lambda e: e.p_value)
    return out[:top_k]
