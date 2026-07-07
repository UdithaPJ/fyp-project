"""
tests/test_schema_weight_detection.py
=====================================

Regression tests for ``SchemaDetector`` weight-column detection.

Motivating bug: a TRRUST GRN export (``TF  target  mode  PMID``) had its
numeric **PMID** column auto-selected as the edge weight, because a
purely-numeric column scored 0.45 (0.35 numeric + 0.10 first-numeric) and
cleared the 0.40 auto-select threshold with no weight-like name at all.
Using PMIDs as edge weights silently corrupts weighted algorithms
(PageRank / HITS / RWR).

The fix requires a numeric column to *look like* a weight (fractional
values, or a bounded small-magnitude integer) before its numeric-ness
counts, and treats identifier-named columns (pmid, id, year, …) as never
being weights.
"""

from __future__ import annotations

import pandas as pd

from src.preprocessing.modules import SchemaDetector


def _weight(df: pd.DataFrame) -> str | None:
    return SchemaDetector().detect(df).mapping["weight"]


# --- identifier columns must NOT be picked as weights ----------------------

def test_pmid_column_not_selected_as_weight():
    # Headerless TRRUST is read with the first data row as the header, so the
    # PMID column is literally named after a PMID value.
    df = pd.DataFrame({
        "AATF": ["AATF", "AATF", "ABL1"],
        "BAX": ["CDKN1A", "MYC", "BAX"],
        "Repression": ["Unknown", "Activation", "Activation"],
        "22909821": [17157788, 20549547, 11753601],
    })
    assert _weight(df) is None


def test_named_pmid_column_not_selected():
    df = pd.DataFrame({
        "TF": ["A", "B", "C"],
        "target": ["X", "Y", "Z"],
        "mode": ["Activation", "Repression", "Unknown"],
        "pmid": [22909821, 17157788, 23146908],
    })
    assert _weight(df) is None


def test_year_column_not_selected():
    df = pd.DataFrame({
        "gene1": ["a", "b", "c"],
        "gene2": ["x", "y", "z"],
        "year": [1998, 2005, 2020],
    })
    assert _weight(df) is None


def test_generic_id_suffix_column_not_selected():
    df = pd.DataFrame({
        "source": ["a", "b", "c"],
        "target": ["x", "y", "z"],
        "gene_id": [1001, 2002, 3003],
    })
    assert _weight(df) is None


def test_large_unique_integer_ids_not_selected_even_unnamed():
    # No weight-ish name, large high-cardinality integers → identifier-shaped.
    df = pd.DataFrame({
        "src": ["a", "b", "c", "d"],
        "dst": ["x", "y", "z", "w"],
        "ref": [90210123, 88123456, 77654321, 66111222],
    })
    assert _weight(df) is None


# --- genuine weight columns MUST still be picked ---------------------------

def test_named_weight_column_selected():
    df = pd.DataFrame({
        "source": ["a", "b", "c"],
        "target": ["x", "y", "z"],
        "weight": [0.5, 0.9, 0.1],
    })
    assert _weight(df) == "weight"


def test_string_combined_score_selected():
    df = pd.DataFrame({
        "protein1": ["a", "b", "c"],
        "protein2": ["x", "y", "z"],
        "combined_score": [850, 400, 999],  # bounded integer score, named
    })
    assert _weight(df) == "combined_score"


def test_unnamed_float_weight_selected():
    df = pd.DataFrame({
        "src": ["a", "b", "c"],
        "dst": ["x", "y", "z"],
        "w": [0.5, 0.9, 0.1],  # fractional → weight-shaped
    })
    assert _weight(df) == "w"


def test_small_integer_count_weight_selected():
    df = pd.DataFrame({
        "source": ["a", "b", "c", "a"],
        "target": ["x", "y", "z", "x"],
        "count": [3, 1, 2, 3],  # small, low-cardinality integers → weight-like
    })
    assert _weight(df) == "count"
