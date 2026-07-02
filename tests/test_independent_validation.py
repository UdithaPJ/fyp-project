"""Tests for the independent (circularity-immune) validation suite.

Covers:
    * new gene-set / GO loaders in ``reference_loader``
    * ``OrthogonalValidator``   (disease / essential / drug-target overlap)
    * ``HoldoutValidator``      (edge hold-out link prediction / clustering)
    * ``GOEnrichmentValidator`` (GO term enrichment)
    * circularity guard added to ``BiologicalValidator``

All validators must NEVER raise: missing references / bad inputs are
recorded as ``status='skipped'`` or ``'error'``, never an exception.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _block_graph(n_per_block: int = 30, n_blocks: int = 2,
                 p_in: float = 0.3, seed: int = 0) -> sp.csr_matrix:
    """Undirected graph with ``n_blocks`` dense communities + a few bridges."""
    rng = np.random.default_rng(seed)
    n = n_per_block * n_blocks
    A = sp.lil_matrix((n, n))
    for b in range(n_blocks):
        idx = range(b * n_per_block, (b + 1) * n_per_block)
        for i in idx:
            for j in idx:
                if i < j and rng.random() < p_in:
                    A[i, j] = 1.0
                    A[j, i] = 1.0
    for _ in range(max(5, n // 10)):
        i = int(rng.integers(0, n_per_block))
        j = int(rng.integers(n_per_block, n))
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A.tocsr()


def _labels(n: int) -> dict[str, int]:
    return {f"GENE{i}": i for i in range(n)}


def _pagerank_result(top_indices: list[int], n: int) -> dict:
    return {"result": {
        "scores": [0.0] * n,
        "top_nodes": [{"index": i, "label": f"GENE{i}"} for i in top_indices],
    }}


def _louvain_result(n: int, split: int) -> dict:
    return {"result": {
        "community_assignments": [0 if i < split else 1 for i in range(n)],
        "top_communities": [
            {"community_id": 0,
             "member_nodes": [{"index": i, "label": f"GENE{i}"}
                              for i in range(split)]},
            {"community_id": 1,
             "member_nodes": [{"index": i, "label": f"GENE{i}"}
                              for i in range(split, n)]},
        ],
    }}


# ===========================================================================
# Loaders
# ===========================================================================

def test_load_gene_set_disease(tmp_path: Path):
    from src.validation.reference_loader import load_gene_set
    f = tmp_path / "disgenet_curated.tsv"
    f.write_text("geneSymbol\tdisease\n"
                 + "\n".join(f"GENE{i}\tdz" for i in range(20)))
    ref = load_gene_set("disease", f)
    assert ref is not None
    assert ref.kind == "disease"
    assert "GENE0" in ref.genes
    assert len(ref.genes) == 20


def test_load_gene_set_missing_returns_none(monkeypatch, tmp_path: Path):
    from src.validation import reference_loader as rl
    monkeypatch.setattr(rl, "_find_file", lambda *a, **kw: None)
    assert rl.load_gene_set("disease", tmp_path / "nope.tsv") is None
    assert rl.load_gene_set("unknown_kind") is None


def test_load_go_annotations(tmp_path: Path):
    from src.validation.reference_loader import load_go_annotations
    gaf = tmp_path / "goa_human.gaf"
    rows = []
    for i in range(10):
        rows.append("\t".join([
            "DB", f"P{i}", f"GENE{i}", "", "GO:0000001", "r", "IEA", "",
            "P", "", "", "protein", "taxon:9606", "2020", "GOA",
        ]))
    gaf.write_text("!gaf-version: 2.1\n" + "\n".join(rows))
    go = load_go_annotations(gaf)
    assert go is not None
    assert go.kind == "go"
    assert "GENE0" in go.gene_terms
    assert "GO:0000001" in go.term_genes
    assert len(go.term_genes["GO:0000001"]) == 10


def test_load_go_aspect_filter(tmp_path: Path):
    from src.validation.reference_loader import load_go_annotations
    gaf = tmp_path / "goa_human.gaf"
    rows = [
        "\t".join(["DB", "P0", "GENE0", "", "GO:0000001", "r", "IEA", "",
                   "P", "", "", "protein", "taxon:9606", "2020", "GOA"]),
        "\t".join(["DB", "P1", "GENE1", "", "GO:0000002", "r", "IEA", "",
                   "C", "", "", "protein", "taxon:9606", "2020", "GOA"]),
    ]
    gaf.write_text("\n".join(rows))
    go = load_go_annotations(gaf, aspects={"P"})
    assert go is not None
    assert "GENE0" in go.genes
    assert "GENE1" not in go.genes   # aspect C filtered out


# ===========================================================================
# Loader tolerance for real-world reference file quirks
# ===========================================================================

def test_load_gene_set_ttd_flat_format(tmp_path: Path):
    """TTD ships a long-format record file rather than a table:
    ``<target_id>\\t<field_name>\\t<value>``.  The loader must detect this
    layout and pull symbols from ``GENENAME`` rows only.
    """
    from src.validation.reference_loader import load_gene_set
    ttd = tmp_path / "P1-01-TTD_target_download.txt"
    ttd.write_text(
        "-----\n"
        "TTD - preamble\n"
        "-----\n\n"
        "T47101\tTARGETID\tT47101\n"
        "T47101\tGENENAME\tFGFR1\n"
        "T47101\tBIOCLASS\tKinase\n"
        "T47101\tDRUGINFO\tD0O6UY\tPemigatinib\tApproved\n"
        "\n"
        "T59328\tTARGETID\tT59328\n"
        "T59328\tGENENAME\tEGFR\n"
        "T89515\tGENENAME\tPDF\n"
    )
    r = load_gene_set("drug_target", ttd)
    assert r is not None
    assert r.name == "TTD"
    assert r.genes == {"FGFR1", "EGFR", "PDF"}
    # Junk fields (BIOCLASS/DRUGINFO/preamble) must NOT be treated as genes.
    assert "KINASE" not in r.genes
    assert "PEMIGATINIB" not in r.genes


def test_load_gene_set_disease_headerless(tmp_path: Path):
    """DISEASES (Jensen Lab) has no header row and col 1 is an Ensembl id.
    The loader must skip identifier-only cells and pick col 2.
    """
    from src.validation.reference_loader import load_gene_set
    f = tmp_path / "human_disease_integrated_full.tsv"
    f.write_text(
        "ENSP00000000001\tTP53\tDOID:1612\tbreast cancer\t3.0\n"
        "ENSP00000000002\tBRCA1\tDOID:1612\tbreast cancer\t3.5\n"
        "ENSP00000000003\tMYC\tDOID:0001\tcancer\t2.0\n"
    )
    r = load_gene_set("disease", f)
    assert r is not None
    assert r.genes == {"TP53", "BRCA1", "MYC"}


def test_load_gene_set_depmap_paren_suffix(tmp_path: Path):
    """DepMap Common Essentials uses ``"TP53 (7157)"`` for the Gene column;
    ``_clean_symbol`` must strip the parenthesised Entrez suffix.
    """
    from src.validation.reference_loader import load_gene_set
    f = tmp_path / "depmap_common_essentials.csv"
    f.write_text("gene\nTP53 (7157)\nMYC (4609)\nBRCA1 (672)\n")
    r = load_gene_set("essential", f)
    assert r is not None
    assert r.genes == {"TP53", "MYC", "BRCA1"}


# ===========================================================================
# STRING id → gene-symbol remap
# ===========================================================================

def test_load_string_id_map(tmp_path: Path):
    from src.validation.reference_loader import load_string_id_map
    f = tmp_path / "9606.protein.info.v12.0.txt"
    f.write_text(
        "#string_protein_id\tpreferred_name\tprotein_size\tannotation\n"
        "9606.ENSP00000000233\tARF5\t180\tADP-ribosylation factor 5\n"
        "9606.ENSP00000000412\tM6PR\t277\tMannose-6-phosphate receptor\n"
    )
    m = load_string_id_map(f)
    assert m["9606.ENSP00000000233"] == "ARF5"
    assert m["9606.ENSP00000000412"] == "M6PR"
    # symbols upper-cased
    assert all(v == v.upper() for v in m.values())


def test_load_string_id_map_missing(monkeypatch, tmp_path: Path):
    """When the loader cannot locate any info file, returns ``{}``."""
    from src.validation import reference_loader as rl
    # Force `_find_file` to return None to simulate a machine without the
    # real 9606.protein.info.v*.txt (which is checked into data/raw/).
    monkeypatch.setattr(rl, "_find_file", lambda *a, **kw: None)
    assert rl.load_string_id_map(tmp_path / "absent.txt") == {}


def test_remap_node_index_map_translates_and_keeps_unmapped():
    from src.validation.reference_loader import remap_node_index_map
    nim = {"9606.ENSP00000000233": 0, "9606.ENSP99999999999": 1, "keep": 2}
    id_map = {"9606.ENSP00000000233": "ARF5"}
    out, stats = remap_node_index_map(nim, id_map)
    assert out["ARF5"] == 0            # mapped
    assert out["9606.ENSP99999999999"] == 1  # unchanged (not in id_map)
    assert out["keep"] == 2            # unchanged
    assert stats["mapped"] == 1
    assert stats["unchanged"] == 2
    assert stats["collisions"] == 0


def test_remap_node_index_map_handles_collisions():
    """Two source labels mapping to the same symbol (isoform collision):
    first wins under the symbol; loser stays under its original label so
    its index is still reachable."""
    from src.validation.reference_loader import remap_node_index_map
    nim = {"ENSP_A": 0, "ENSP_B": 1}
    id_map = {"ENSP_A": "TP53", "ENSP_B": "TP53"}
    out, stats = remap_node_index_map(nim, id_map)
    assert out["TP53"] == 0                 # first wins
    assert out["ENSP_B"] == 1               # loser kept under raw label
    assert stats["collisions"] == 1
    assert len(out) == 2                    # no index lost


def test_remap_empty_id_map_is_passthrough():
    from src.validation.reference_loader import remap_node_index_map
    nim = {"a": 0, "b": 1}
    out, stats = remap_node_index_map(nim, {})
    assert out == nim
    assert stats["mapped"] == 0
    assert stats["unchanged"] == 2


# ===========================================================================
# Regression: cpu_single top_nodes are plain int lists, must still map to
# real gene symbols via node_index_map (otherwise all overlaps are zero).
# ===========================================================================

def _pagerank_result_plain_ints(top_indices: list[int], n: int) -> dict:
    """Mirror the CLAUDE.md ``cpu_single`` schema: top_nodes is list[int]."""
    return {"result": {"scores": [0.0] * n, "top_nodes": list(top_indices)}}


def test_extract_predicted_uses_node_index_map():
    """Plain-int top_nodes must resolve to real labels via node_index_map."""
    from src.validation.biological_validation import _extract_predicted
    nim = {"TP53": 3, "MYC": 7, "EGFR": 12}
    result = {"result": {"scores": [0.0] * 20, "top_nodes": [3, 7, 12]}}
    assert _extract_predicted("pagerank", result, nim) == ["TP53", "MYC", "EGFR"]


def test_extract_predicted_without_map_synthesises_labels():
    """Without a map, plain ints collapse to synthetic NODE_<i> tokens."""
    from src.validation.biological_validation import _extract_predicted
    result = {"result": {"scores": [0.0] * 20, "top_nodes": [3, 7]}}
    got = _extract_predicted("pagerank", result, None)
    assert got == ["NODE_3", "NODE_7"]


def test_orthogonal_ranking_with_plain_int_top_nodes(tmp_path: Path):
    """Regression: cpu_single-style result was previously producing zero
    overlap because ``top_nodes: list[int]`` never resolved to gene symbols.
    """
    from src.validation import OrthogonalValidator
    n = 60
    csr = _block_graph()
    nim = _labels(n)
    dis = tmp_path / "disgenet.tsv"
    dis.write_text("geneSymbol\n" + "\n".join(f"GENE{i}" for i in range(20)))
    results = {"pagerank": _pagerank_result_plain_ints(list(range(10)), n)}
    ov = OrthogonalValidator(kinds=("disease",), output_dir=tmp_path,
                             reference_paths={"disease": dis})
    ov.add_dataset("synth", csr, "ppi", nim, results)
    ov.run(algorithms=("pagerank",))
    rec = ov.records[0]
    assert rec.status == "ok"
    assert rec.overlap_count == 10       # ← would have been 0 before the fix
    assert rec.p_value < 0.01


def test_go_ranking_with_plain_int_top_nodes(tmp_path: Path):
    """Same regression for GO enrichment (the CSV symptom the user hit)."""
    from src.validation import GOEnrichmentValidator
    n = 60
    csr = _block_graph()
    nim = _labels(n)
    gaf = tmp_path / "goa_human.gaf"
    _write_gaf(gaf, generic_n=60, specific_n=15)
    results = {"pagerank": _pagerank_result_plain_ints(list(range(10)), n)}
    gv = GOEnrichmentValidator(output_dir=tmp_path, reference_path=gaf)
    gv.add_dataset("synth", csr, "ppi", nim, results)
    gv.run(algorithms=("pagerank",))
    rec = gv.records[0]
    assert rec.status == "ok"           # ← was "skipped" before the fix
    assert rec.best_term == "GO:0042276"
    assert rec.best_term_p < 0.05


# ===========================================================================
# OrthogonalValidator
# ===========================================================================

def test_orthogonal_ranking_enrichment(tmp_path: Path):
    from src.validation import OrthogonalValidator
    n = 60
    csr = _block_graph()
    nim = _labels(n)
    # disease genes = GENE0..19 ; top nodes GENE0..9 are all disease genes
    dis = tmp_path / "disgenet.tsv"
    dis.write_text("geneSymbol\n" + "\n".join(f"GENE{i}" for i in range(20)))
    results = {"pagerank": _pagerank_result(list(range(10)), n)}

    ov = OrthogonalValidator(kinds=("disease",), output_dir=tmp_path,
                             reference_paths={"disease": dis})
    ov.add_dataset("synth", csr, "ppi", nim, results)
    ov.run(algorithms=("pagerank",))
    rec = [r for r in ov.records if r.algorithm == "pagerank"][0]
    assert rec.status == "ok"
    assert rec.overlap_count == 10
    assert rec.precision == pytest.approx(1.0)
    assert rec.p_value < 0.01              # highly enriched


def test_orthogonal_community_nmi(tmp_path: Path):
    from src.validation import OrthogonalValidator
    n = 60
    csr = _block_graph()
    nim = _labels(n)
    dis = tmp_path / "disgenet.tsv"
    # disease genes align with community 0 (GENE0..29)
    dis.write_text("geneSymbol\n" + "\n".join(f"GENE{i}" for i in range(30)))
    results = {"louvain": _louvain_result(n, 30)}

    ov = OrthogonalValidator(kinds=("disease",), output_dir=tmp_path,
                             reference_paths={"disease": dis})
    ov.add_dataset("synth", csr, "ppi", nim, results)
    ov.run(algorithms=("louvain",))
    rec = [r for r in ov.records if r.algorithm == "louvain"][0]
    assert rec.status == "ok"
    assert np.isfinite(rec.nmi)
    assert rec.nmi > 0.5                    # community matches disease partition


def test_orthogonal_missing_reference_skips(monkeypatch, tmp_path: Path):
    from src.validation import OrthogonalValidator
    from src.validation import reference_loader as rl
    monkeypatch.setattr(rl, "_find_file", lambda *a, **kw: None)
    csr = _block_graph()
    results = {"pagerank": _pagerank_result(list(range(10)), 60)}
    ov = OrthogonalValidator(kinds=("disease",), output_dir=tmp_path,
                             reference_paths={"disease": tmp_path / "absent.tsv"})
    ov.add_dataset("synth", csr, "ppi", _labels(60), results)
    ov.run(algorithms=("pagerank",))
    assert all(r.status == "skipped" for r in ov.records)


# ===========================================================================
# HoldoutValidator
# ===========================================================================

def test_holdout_runs_all_algorithms(tmp_path: Path):
    from src.validation import HoldoutValidator
    csr = _block_graph(seed=1)
    hv = HoldoutValidator(output_dir=tmp_path, test_fraction=0.2, seed=1)
    hv.add_dataset("synth", csr, "ppi")
    hv.run(algorithms=("pagerank", "hits", "rwr", "louvain", "mcl"))
    assert len(hv.records) == 5
    assert all(r.status in ("ok", "skipped") for r in hv.records)


def test_holdout_ranking_produces_auroc(tmp_path: Path):
    from src.validation import HoldoutValidator
    csr = _block_graph(seed=2)
    hv = HoldoutValidator(output_dir=tmp_path, test_fraction=0.2, seed=2)
    hv.add_dataset("synth", csr, "ppi")
    hv.run(algorithms=("pagerank",))
    rec = hv.records[0]
    assert rec.status == "ok"
    assert 0.0 <= rec.auroc <= 1.0
    assert rec.train_edges > 0
    assert rec.test_edges > 0


def test_holdout_clustering_lift_positive(tmp_path: Path):
    from src.validation import HoldoutValidator
    # strong block structure → held-out edges recovered within communities
    csr = _block_graph(p_in=0.5, seed=3)
    hv = HoldoutValidator(output_dir=tmp_path, test_fraction=0.2, seed=3)
    hv.add_dataset("synth", csr, "ppi")
    hv.run(algorithms=("louvain",))
    rec = hv.records[0]
    assert rec.status == "ok"
    assert np.isfinite(rec.comembership_heldout)
    assert rec.lift > 1.0            # held-out edges enriched within communities


def test_holdout_auroc_helper_monotone():
    from src.validation.holdout_validation import _auroc
    pos = np.array([0.9, 0.8, 0.7])
    neg = np.array([0.1, 0.2, 0.3])
    assert _auroc(pos, neg) == pytest.approx(1.0)
    assert _auroc(neg, pos) == pytest.approx(0.0)
    # identical distributions → 0.5
    assert _auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(0.5)


def test_holdout_csv_written(tmp_path: Path):
    from src.validation import HoldoutValidator
    csr = _block_graph(seed=4)
    hv = HoldoutValidator(output_dir=tmp_path, seed=4)
    hv.add_dataset("synth", csr, "ppi")
    hv.run(algorithms=("pagerank", "louvain"))
    csv_path = hv.write_csv()
    assert csv_path.exists()
    assert "auroc" in csv_path.read_text().splitlines()[0]


# ===========================================================================
# GOEnrichmentValidator
# ===========================================================================

def _write_gaf(path: Path, generic_n: int, specific_n: int) -> None:
    rows = []
    for i in range(generic_n):
        rows.append("\t".join([
            "DB", f"P{i}", f"GENE{i}", "", "GO:0000001", "r", "IEA", "",
            "P", "", "", "protein", "taxon:9606", "2020", "GOA"]))
    for i in range(specific_n):
        rows.append("\t".join([
            "DB", f"P{i}", f"GENE{i}", "", "GO:0042276", "r", "IEA", "",
            "P", "", "", "protein", "taxon:9606", "2020", "GOA"]))
    path.write_text("!gaf\n" + "\n".join(rows))


def test_go_enrichment_significant(tmp_path: Path):
    from src.validation import GOEnrichmentValidator
    n = 60
    csr = _block_graph()
    nim = _labels(n)
    gaf = tmp_path / "goa_human.gaf"
    _write_gaf(gaf, generic_n=60, specific_n=15)   # GENE0..14 carry specific term
    # top nodes GENE0..9 all carry the specific term
    results = {"pagerank": _pagerank_result(list(range(10)), n)}

    gv = GOEnrichmentValidator(output_dir=tmp_path, reference_path=gaf)
    gv.add_dataset("synth", csr, "ppi", nim, results)
    gv.run(algorithms=("pagerank",))
    rec = gv.records[0]
    assert rec.status == "ok"
    assert rec.best_term == "GO:0042276"
    assert rec.best_term_p < 0.05
    assert rec.n_groups_significant == 1


def test_go_missing_file_skips(monkeypatch, tmp_path: Path):
    from src.validation import GOEnrichmentValidator
    from src.validation import reference_loader as rl
    monkeypatch.setattr(rl, "_find_file", lambda *a, **kw: None)
    csr = _block_graph()
    results = {"pagerank": _pagerank_result(list(range(10)), 60)}
    gv = GOEnrichmentValidator(output_dir=tmp_path,
                               reference_path=tmp_path / "absent.gaf")
    gv.add_dataset("synth", csr, "ppi", _labels(60), results)
    gv.run(algorithms=("pagerank",))
    assert all(r.status == "skipped" for r in gv.records)


# ===========================================================================
# Circularity guard (BiologicalValidator)
# ===========================================================================

def test_circularity_fraction_helper():
    from src.validation.biological_validation import (
        _circularity_fraction, _CIRCULARITY_THRESHOLD,
    )
    from src.validation.reference_loader import ReferenceSet
    ref = ReferenceSet(name="X", network_type="ppi",
                       sources={f"GENE{i}" for i in range(100)})
    ref.n_records = 100
    # all graph nodes are in the reference → circular
    nim = {f"GENE{i}": i for i in range(50)}
    frac = _circularity_fraction(nim, ref)
    assert frac == pytest.approx(1.0)
    assert frac >= _CIRCULARITY_THRESHOLD


def test_circularity_flag_in_record(tmp_path: Path):
    from src.validation import BiologicalValidator
    from src.validation.reference_loader import ReferenceSet
    n = 40
    csr = _block_graph(n_per_block=20)
    nim = _labels(n)
    # Reference contains every graph node → maximal circularity
    ref = ReferenceSet(name="BioGRID", network_type="ppi",
                       sources=set(f"GENE{i}" for i in range(n)),
                       targets=set(f"GENE{i}" for i in range(n)))
    ref.n_records = n

    bv = BiologicalValidator(output_dir=tmp_path)
    bv._ref_cache["ppi"] = ref                         # inject reference
    bv.add_dataset("synth", csr, "ppi", nim,
                   {"pagerank": _pagerank_result(list(range(10)), n)})
    bv.run(algorithms=("pagerank",))
    rec = [r for r in bv.records if r.algorithm == "pagerank"][0]
    assert rec.circularity_risk == pytest.approx(1.0)
    assert "CIRCULAR" in rec.note
