"""
src/validation/overlap.py
=========================

Set-overlap metrics for biological validation.  Given a predicted set
(e.g. top regulators returned by PageRank) and a reference set (e.g.
TRRUST TFs), compute:

    overlap_count   – |predicted ∩ reference|
    precision       – |predicted ∩ reference| / |predicted|
    recall          – |predicted ∩ reference| / |reference|
    jaccard         – |∩| / |∪|

Plus an enrichment p-value from Fisher's exact test (one-sided, alternative='greater').
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class OverlapResult:
    overlap_count:  int
    precision:      float
    recall:         float
    jaccard:        float
    predicted_size: int
    reference_size: int
    background_size: int
    p_value:        float    # Fisher exact (one-sided, alternative='greater')


def _to_upper_set(xs: Iterable[str]) -> set[str]:
    return {str(x).strip().upper() for x in xs if x is not None and str(x).strip()}


def compute_overlap(predicted: Iterable[str],
                    reference: Iterable[str],
                    background_size: Optional[int] = None) -> OverlapResult:
    """
    Compute precision, recall, jaccard, and a one-sided Fisher exact p-value
    for the overlap between ``predicted`` and ``reference``.

    Parameters
    ----------
    predicted:
        Iterable of node identifiers returned by an algorithm (e.g.
        ``top_regulators`` mapped through ``node_index_map``).
    reference:
        Reference set (e.g. TRRUST transcription factors).
    background_size:
        Total size of the universe (e.g. number of nodes in the graph).
        If None, defaults to ``len(reference) * 10`` as a reasonable
        proxy when no graph context is available.

    Returns
    -------
    OverlapResult
    """
    pred_set = _to_upper_set(predicted)
    ref_set  = _to_upper_set(reference)

    overlap = pred_set & ref_set
    overlap_count = len(overlap)

    precision = overlap_count / len(pred_set) if pred_set else 0.0
    recall    = overlap_count / len(ref_set)  if ref_set  else 0.0
    union     = pred_set | ref_set
    jaccard   = overlap_count / len(union) if union else 0.0

    if background_size is None or background_size <= 0:
        background_size = max(len(ref_set), len(pred_set)) * 10

    p_value = fisher_exact_p(
        a=overlap_count,
        b=len(pred_set) - overlap_count,
        c=len(ref_set) - overlap_count,
        d=max(background_size - len(pred_set) - len(ref_set) + overlap_count, 0),
    )

    return OverlapResult(
        overlap_count   = overlap_count,
        precision       = precision,
        recall          = recall,
        jaccard         = jaccard,
        predicted_size  = len(pred_set),
        reference_size  = len(ref_set),
        background_size = background_size,
        p_value         = p_value,
    )


# ---------------------------------------------------------------------------
# Fisher exact (one-sided, alternative='greater') — log-space hypergeometric
# ---------------------------------------------------------------------------

def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    if k == 0 or k == n:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact_p(a: int, b: int, c: int, d: int) -> float:
    """
    One-sided Fisher exact test p-value (alternative = "greater").

    Contingency table:
        a  b
        c  d

        a = predicted ∩ reference  (success in sample)
        b = predicted - reference  (failure in sample)
        c = reference - predicted  (success in background \\ sample)
        d = background - predicted - reference + overlap (failure in background \\ sample)

    Returns P(X >= a) under the hypergeometric model.
    """
    a, b, c, d = max(int(a), 0), max(int(b), 0), max(int(c), 0), max(int(d), 0)
    n = a + b + c + d
    if n == 0:
        return 1.0

    K = a + c   # total successes in population
    n_draws = a + b
    max_a = min(K, n_draws)

    # log-prob of observing X = k given hypergeometric(N, K, n_draws)
    def _log_p(k: int) -> float:
        return (_log_choose(K, k) +
                _log_choose(n - K, n_draws - k) -
                _log_choose(n, n_draws))

    # Sum P(X >= a) using log-sum-exp
    logs = []
    for k in range(a, max_a + 1):
        lp = _log_p(k)
        if math.isfinite(lp):
            logs.append(lp)
    if not logs:
        return 1.0
    m = max(logs)
    return math.exp(m + math.log(sum(math.exp(x - m) for x in logs)))
