"""
tests/test_file_loader_delimiter_sniffing.py
=============================================

Regression tests for ``FileLoader._sniff_delimiter`` in
``src/preprocessing/modules.py``.

Context
-------
``.txt``/``.tab`` uploads with an unknown delimiter are sniffed from an
8KB sample instead of parsing the whole file with pandas' slow
``sep=None, engine="python"`` path (see the upload-time optimisation).
Whitespace (``r"\\s+"``) was added as a first-class candidate delimiter
to support whitespace-only exports (e.g. STRING PPI files:
"protein1 protein2 combined_score").

The risk that motivated these tests: a genuinely tab/comma-delimited
file whose fields contain free-text with spaces (e.g. a description
column) must NOT be misdetected as whitespace-delimited. Safety here
comes from ``_sniff_delimiter``'s consistency check — a candidate is
only accepted if it splits the header AND every sampled data row into
the SAME number of fields; among candidates that pass, the one with
the most columns wins. This is a different (stricter) mechanism than a
simple delimiter-frequency count, so it's worth locking in directly.
"""

from __future__ import annotations

import pytest

from src.preprocessing.modules import FileLoader


# ---------------------------------------------------------------------------
# _sniff_delimiter unit tests
# ---------------------------------------------------------------------------

def test_tab_delimited_with_spaces_in_free_text_field():
    """Free-text descriptions with varying word counts must not confuse
    whitespace detection into beating the true tab delimiter."""

    sample = (
        "gene\tdescription\tscore\n"
        "TP53\ttumor suppressor protein\t0.9\n"
        "BRCA1\tbreast cancer type 1 susceptibility protein\t0.7\n"
        "EGFR\tepidermal growth factor receptor\t0.8\n"
    )
    assert FileLoader._sniff_delimiter(sample) == "\t"


def test_comma_delimited_with_spaces_in_free_text_field():
    sample = (
        "gene,description,score\n"
        "TP53,tumor suppressor protein,0.9\n"
        "BRCA1,breast cancer type 1 susceptibility protein,0.7\n"
        "EGFR,epidermal growth factor receptor,0.8\n"
    )
    assert FileLoader._sniff_delimiter(sample) == ","


def test_string_style_whitespace_delimited_file():
    """STRING PPI export style: no other delimiter present at all.

    Fields are padded with a variable number of spaces (as real
    column-aligned whitespace exports commonly are), so a literal
    single-space delimiter would over-split on the multi-space runs
    and fail the consistency check -- only the ``r"\\s+"`` regex
    candidate collapses runs correctly and is accepted.
    """

    sample = (
        "protein1                protein2                combined_score\n"
        "9606.ENSP00000000233    9606.ENSP00000272298    900\n"
        "9606.ENSP00000000412    9606.ENSP00000359671    650\n"
    )
    assert FileLoader._sniff_delimiter(sample) == r"\s+"


def test_single_space_delimited_file_prefers_whitespace_regex():
    """A file with a single literal space between fields on every row is
    still detected via ``r"\\s+"``, not the literal ``" "`` that
    ``csv.Sniffer`` proposes first.

    Both delimiters parse this particular sample identically, but only
    ``r"\\s+"`` also tolerates variable-width whitespace runs elsewhere
    in the file, so it must win the tie rather than being shadowed by
    the earlier-inserted literal-space candidate.
    """

    sample = "source target weight\nA B 1\nC D 2\n"
    assert FileLoader._sniff_delimiter(sample) == r"\s+"


def test_semicolon_delimited_file():
    sample = "source;target;weight\nA;B;1\nC;D;2\n"
    assert FileLoader._sniff_delimiter(sample) == ";"


def test_pipe_delimited_file():
    sample = "source|target|weight\nA|B|1\nC|D|2\n"
    assert FileLoader._sniff_delimiter(sample) == "|"


def test_colon_delimited_file():
    sample = "source:target:weight\nA:B:1\nC:D:2\n"
    assert FileLoader._sniff_delimiter(sample) == ":"


def test_header_derived_arbitrary_punctuation_candidate():
    """Delimiters outside the fixed shortlist (e.g. '~') should still be
    discovered via header-punctuation derivation."""

    sample = "source~target~weight\nA~B~1\nC~D~2\n"
    assert FileLoader._sniff_delimiter(sample) == "~"


def test_empty_sample_defaults_to_comma():
    assert FileLoader._sniff_delimiter("") == ","
    assert FileLoader._sniff_delimiter("   \n  \n") == ","


def test_single_column_no_delimiter_defaults_to_comma():
    sample = "identifier\nTP53\nBRCA1\nEGFR\n"
    assert FileLoader._sniff_delimiter(sample) == ","


def test_inconsistent_whitespace_is_rejected_in_favour_of_true_delimiter():
    """Direct regression for the reviewed risk: a row whose free-text
    field has MORE words than there are tabs must not let whitespace
    win just because it appears frequently."""

    sample = (
        "id\tnote\n"
        "1\tone two three four five six seven\n"
        "2\ta\n"
    )
    # tab is the only delimiter that splits every line into the same
    # (2-column) count; whitespace gives 8 fields on line 1 and 2 on
    # line 2, so it fails the consistency check and must be rejected.
    assert FileLoader._sniff_delimiter(sample) == "\t"


# ---------------------------------------------------------------------------
# End-to-end regression via FileLoader.load_bytes (.txt), not just the
# delimiter guess in isolation -- confirms the detected separator actually
# parses into the expected shape.
# ---------------------------------------------------------------------------

def test_load_bytes_tab_file_with_spacey_description_parses_correctly():
    content = (
        "gene\tdescription\tscore\n"
        "TP53\ttumor suppressor protein\t0.9\n"
        "BRCA1\tbreast cancer type 1 susceptibility protein\t0.7\n"
    ).encode("utf-8")

    df = FileLoader().load_bytes("regulators.txt", content)

    assert list(df.columns) == ["gene", "description", "score"]
    assert len(df) == 2
    assert df.loc[0, "description"] == "tumor suppressor protein"


def test_load_bytes_string_style_whitespace_file_parses_correctly():
    content = (
        "protein1 protein2 combined_score\n"
        "9606.ENSP00000000233 9606.ENSP00000272298 900\n"
        "9606.ENSP00000000412 9606.ENSP00000359671 650\n"
    ).encode("utf-8")

    df = FileLoader().load_bytes("string_ppi.txt", content)

    assert list(df.columns) == ["protein1", "protein2", "combined_score"]
    assert len(df) == 2
    assert df.loc[0, "protein1"] == "9606.ENSP00000000233"
    assert int(df.loc[0, "combined_score"]) == 900
