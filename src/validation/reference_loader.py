"""
src/validation/reference_loader.py
==================================

Lightweight loaders for the three biological reference databases used
by :class:`BiologicalValidator`:

    network_type  →  reference database
    --------------  -------------------------
    grn           →  TRRUST    (human TF→target regulatory pairs)
    ppi           →  BioGRID   (protein-protein interactions)
    mirna         →  miRTarBase (miRNA→target gene interactions)

The loaders never raise — when a reference file is missing or
unreadable they return ``None`` and the caller (BiologicalValidator)
treats biological validation as skipped for that algorithm/dataset.

Search order for each reference (first hit wins):
    1. explicit path passed to the loader
    2. environment variable ``FYP_<NAME>_PATH``
    3. ``data/raw/<known filename patterns>``
    4. ``data/processed/<known filename patterns>``
    5. ``data/references/<known filename patterns>``

The loaders return :class:`ReferenceSet` — a dataclass containing a
``set[str]`` of regulators / hubs / sources and a ``set[str]`` of targets,
plus a ``set[frozenset[str]]`` of undirected edges (for PPI / known
interactions).  All identifiers are upper-cased and stripped of
surrounding whitespace.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

_LOG = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIRS: tuple[Path, ...] = (
    _PROJECT_ROOT / "data" / "raw",
    _PROJECT_ROOT / "data" / "processed",
    _PROJECT_ROOT / "data" / "references",
)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ReferenceSet:
    """A loaded reference database, normalised to upper-case symbols."""

    name:        str
    network_type: str            # "grn" | "ppi" | "mirna"
    sources:     set[str] = field(default_factory=set)   # TFs / hubs / miRNAs
    targets:     set[str] = field(default_factory=set)   # target genes
    edges:       set[frozenset[str]] = field(default_factory=set)
    n_records:   int = 0
    source_path: Optional[Path] = None

    @property
    def all_nodes(self) -> set[str]:
        return self.sources | self.targets

    def is_empty(self) -> bool:
        return self.n_records == 0


# ---------------------------------------------------------------------------
# Filename hints (case-insensitive substring match)
# ---------------------------------------------------------------------------

_PATTERNS: dict[str, tuple[str, ...]] = {
    "trrust":     ("trrust", "trrust_rawdata", "trrust.human"),
    "biogrid":    ("biogrid", "biogrid-all", "biogrid_human"),
    "mirtarbase": ("mirtarbase", "mirtar", "hsa_mti"),
    # Orthogonal gene-set references (evidence unrelated to network topology).
    # Multiple filename patterns per kind so the loader auto-discovers any of
    # several equivalent public databases the user might have on hand:
    #   disease     – DisGeNET, DISEASES (Jensen Lab), GWAS Catalog,
    #                 HPO (genes_to_phenotype), ClinVar (gene_condition),
    #                 Orphanet
    #   essential   – DEG, OGEE, DepMap CRISPR common essentials,
    #                 HART lab CEG core-essential gene lists
    #   drug_target – DrugBank, Therapeutic Target Database (TTD),
    #                 Guide to Pharmacology (IUPHAR)
    "disgenet":   ("disgenet", "gene_disease", "curated_gene_disease",
                   "all_gene_disease",
                   "human_disease_integrated", "diseases_integrated",
                   "gwas_catalog", "gwas-associations",
                   "genes_to_phenotype", "phenotype_to_genes", "hpo",
                   "clinvar", "gene_condition", "orphanet"),
    "deg":        ("deg", "ogee", "essential",
                   "common_essentials", "commonessentials",
                   "cegv2", "ceg2", "hart_essential"),
    "drugbank":   ("drugbank", "drug_target", "drug-target",
                   "all_target_polypeptide",
                   "ttd_target", "ttd-target", "target_information",
                   "targets_and_families", "iuphar", "guidetopharmacology"),
    # Gene Ontology annotations (GAF 2.x)
    "go":         ("goa_human", "goa", "gene_association", ".gaf"),
    # STRING protein-info map (Ensembl protein id → gene symbol)
    "string_info": ("protein.info", "protein_info", "string.info"),
}


def _find_file(db_key: str, explicit: Optional[Path] = None) -> Optional[Path]:
    """Locate a reference file using the search order documented above."""
    # 1) explicit
    if explicit is not None:
        p = Path(explicit)
        if p.exists():
            return p
        _LOG.warning("Explicit %s path not found: %s", db_key, p)

    # 2) environment variable
    env_key = f"FYP_{db_key.upper()}_PATH"
    env_val = os.environ.get(env_key)
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p
        _LOG.warning("Env %s set but path missing: %s", env_key, p)

    # 3-5) project data directories
    patterns = _PATTERNS.get(db_key, (db_key,))
    for data_dir in _DATA_DIRS:
        if not data_dir.exists():
            continue
        for child in data_dir.iterdir():
            if not child.is_file():
                continue
            name_l = child.name.lower()
            for pat in patterns:
                if pat in name_l:
                    return child
    return None


# ---------------------------------------------------------------------------
# Generic table reader (TSV / CSV / whitespace-delimited)
# ---------------------------------------------------------------------------

def _iter_table(path: Path, sep_candidates: tuple[str, ...] = ("\t", ",", ";", " ")):
    """Yield row dicts; auto-detects header + delimiter."""
    try:
        # Sniff first 5 non-empty lines
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            sample = ""
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                sample += line
            if not sample:
                return

            # Pick the separator with the highest *consistent* column count
            # across the first few lines.  Tab beats space in ties because
            # TSV is the dominant biological format.
            lines = [ln for ln in sample.split("\n") if ln.strip()]
            chosen_sep   = "\t"
            best_score   = -1
            for sep in sep_candidates:
                counts = [ln.count(sep) for ln in lines]
                if not counts or max(counts) == 0:
                    continue
                # consistent → all rows have the same column count
                same = all(c == counts[0] for c in counts)
                # score: prefer high count + consistency; tab gets a tiny
                # priority bump to outrank space when both are present.
                bias = 0.5 if sep == "\t" else 0.0
                score = counts[0] + (10 if same else 0) + bias
                if score > best_score:
                    best_score = score
                    chosen_sep = sep

        # Re-open and yield
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            # Skip lines that look like comments
            buf_lines = [ln for ln in f if not ln.startswith("#")]
        if not buf_lines:
            return

        # Try DictReader (header) first.
        # Row is treated as a header only when EVERY non-empty cell looks
        # like a column name — i.e. contains letters, is not a plain number,
        # and does NOT start with a biological identifier prefix such as
        # ENSP / ENSG / HGNC:.  Files like DISEASES whose first data row
        # begins with an Ensembl id are correctly recognised as header-less.
        first = buf_lines[0].rstrip("\n\r").split(chosen_sep)
        non_empty = [c for c in first if c.strip()]
        def _looks_like_header_cell(c: str) -> bool:
            c = c.strip()
            if not any(ch.isalpha() for ch in c):
                return False
            if c.replace(".", "").replace("-", "").isdigit():
                return False
            if c.upper().startswith(_IDENTIFIER_PREFIXES):
                return False
            return True
        has_header = bool(non_empty) and all(
            _looks_like_header_cell(c) for c in non_empty
        )

        if has_header:
            reader = csv.DictReader(buf_lines, delimiter=chosen_sep)
            for row in reader:
                yield row
        else:
            for ln in buf_lines:
                parts = ln.rstrip("\n\r").split(chosen_sep)
                yield {f"col{i}": v for i, v in enumerate(parts)}
    except Exception as exc:
        _LOG.warning("Failed to read %s: %s", path, exc)
        return


def _norm(s: str) -> str:
    return s.strip().upper()


# ---------------------------------------------------------------------------
# TRRUST  (GRN)
# ---------------------------------------------------------------------------

def _first_two_values(row: dict) -> tuple[Optional[str], Optional[str]]:
    """Return the first two non-empty string values from a row dict."""
    vals = [str(v).strip() for v in row.values()
            if v is not None and str(v).strip()]
    if len(vals) < 2:
        return None, None
    return vals[0], vals[1]


def load_trrust(path: Optional[Path] = None) -> Optional[ReferenceSet]:
    """
    Load the TRRUST regulatory-pairs file.

    Expected columns (any one row layout):
        TF, Target, Mode, Reference
    or  col0, col1, col2, col3
    """
    p = _find_file("trrust", path)
    if p is None:
        _LOG.info("TRRUST reference not found — biological GRN validation skipped.")
        return None

    rs = ReferenceSet(name="TRRUST", network_type="grn", source_path=p)
    # We need to also consume the header row when there was no real header
    header_consumed = False
    seen_header_row: Optional[dict] = None
    for row in _iter_table(p):
        if not row:
            continue
        # Try named columns first
        tf  = (row.get("TF") or row.get("tf") or row.get("source")
               or row.get("regulator") or row.get("col0"))
        tgt = (row.get("Target") or row.get("target") or row.get("col1"))
        # Positional fallback (handles header-less TRRUST exports)
        if not tf or not tgt:
            tf, tgt = _first_two_values(row)
        if not tf or not tgt:
            continue
        tf, tgt = _norm(str(tf)), _norm(str(tgt))
        if not tf or not tgt:
            continue
        # If the very first row looks like a header (e.g. "TF" / "TARGET"),
        # skip it.
        if not header_consumed:
            header_consumed = True
            if tf in {"TF", "SOURCE", "REGULATOR"} and tgt in {"TARGET", "GENE"}:
                continue
        rs.sources.add(tf)
        rs.targets.add(tgt)
        rs.edges.add(frozenset({tf, tgt}))
        rs.n_records += 1

    _LOG.info("TRRUST loaded: %d rows, %d TFs, %d targets",
              rs.n_records, len(rs.sources), len(rs.targets))
    return rs if rs.n_records > 0 else None


# ---------------------------------------------------------------------------
# BioGRID  (PPI)
# ---------------------------------------------------------------------------

def load_biogrid(path: Optional[Path] = None,
                 organism_id: str = "9606") -> Optional[ReferenceSet]:
    """
    Load a BioGRID tab3 PPI file (streaming, human-filtered by default).

    BioGRID's ``.tab3.txt`` is a 37-column TSV whose **header line begins
    with '#'** (``#BioGRID Interaction ID	...``).  The generic table reader
    discards ``#`` lines and mis-detects the delimiter on this multi-GB
    file, so BioGRID gets a dedicated parser here:

      * streamed line-by-line (never loads the whole 1.5 GB file into RAM),
      * columns located from the (``#``-stripped) header by name, with the
        documented tab3 indices as fallback — symbols at 7 / 8, organism
        ids at 15 / 16,
      * filtered to ``organism_id`` (default human 9606) on BOTH interactors,
      * BioGRID's ``"-"`` missing-symbol placeholder skipped.

    Also accepts a simple pre-extracted two-column symbol edge list (no
    ``#`` header): detected when the header has < 9 columns, in which case
    the first two columns are used.
    """
    p = _find_file("biogrid", path)
    if p is None:
        _LOG.info("BioGRID reference not found — biological PPI validation skipped.")
        return None

    rs = ReferenceSet(name="BioGRID", network_type="ppi", source_path=p)
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().lstrip("#").rstrip("\n\r").split("\t")

            def _col(name: str, default: int) -> int:
                try:
                    return header.index(name)
                except ValueError:
                    return default

            tab3 = len(header) >= 9
            if tab3:
                ia = _col("Official Symbol Interactor A", 7)
                ib = _col("Official Symbol Interactor B", 8)
                oa = _col("Organism ID Interactor A", 15)
                ob = _col("Organism ID Interactor B", 16)
                need = max(ia, ib, oa, ob)
            else:
                ia, ib, oa, ob, need = 0, 1, -1, -1, 1

            for line in f:
                cols = line.rstrip("\n\r").split("\t")
                if len(cols) <= need:
                    continue
                # Human-only (skip when organism columns present and mismatch)
                if oa >= 0 and (cols[oa].strip() != organism_id
                                or cols[ob].strip() != organism_id):
                    continue
                a, b = _norm(cols[ia]), _norm(cols[ib])
                if not a or not b or a == b or a == "-" or b == "-":
                    continue
                rs.sources.add(a)
                rs.sources.add(b)   # PPI undirected — both ends are hubs
                rs.targets.add(a)
                rs.targets.add(b)
                rs.edges.add(frozenset({a, b}))
                rs.n_records += 1
    except Exception as exc:
        _LOG.warning("Failed to read BioGRID %s: %s", p, exc)
        return None

    _LOG.info("BioGRID loaded (organism %s): %d edges, %d proteins",
              organism_id, rs.n_records, len(rs.sources))
    return rs if rs.n_records > 0 else None


# ---------------------------------------------------------------------------
# miRTarBase  (miRNA → target)
# ---------------------------------------------------------------------------

def load_mirtarbase(path: Optional[Path] = None) -> Optional[ReferenceSet]:
    """
    Load a miRTarBase file (miRNA → target gene).

    Expected columns (Excel format converted to TSV / CSV):
        miRTarBase ID, miRNA, Target Gene, Species (Target Gene), ...
    or two-column miRNA, target.
    """
    p = _find_file("mirtarbase", path)
    if p is None:
        _LOG.info("miRTarBase reference not found — biological miRNA validation skipped.")
        return None

    rs = ReferenceSet(name="miRTarBase", network_type="mirna", source_path=p)
    header_consumed = False
    for row in _iter_table(p):
        if not row:
            continue
        mirna = (row.get("miRNA") or row.get("mirna")
                 or row.get("source") or row.get("col0"))
        target = (row.get("Target Gene") or row.get("target_gene")
                  or row.get("target") or row.get("col1"))
        if not mirna or not target:
            mirna, target = _first_two_values(row)
        if not mirna or not target:
            continue
        mirna  = _norm(str(mirna))
        target = _norm(str(target))
        if not mirna or not target:
            continue
        if not header_consumed:
            header_consumed = True
            if mirna in {"MIRNA", "SOURCE"} or target in {"TARGET GENE",
                                                          "TARGET", "GENE"}:
                continue
        rs.sources.add(mirna)
        rs.targets.add(target)
        rs.edges.add(frozenset({mirna, target}))
        rs.n_records += 1

    _LOG.info("miRTarBase loaded: %d pairs, %d miRNAs, %d targets",
              rs.n_records, len(rs.sources), len(rs.targets))
    return rs if rs.n_records > 0 else None


# ===========================================================================
# Orthogonal gene-set references (disease / essential / drug-target / GO)
# ===========================================================================
#
# Unlike TRRUST / BioGRID / miRTarBase (which are *interaction* databases and
# therefore share evidence type with the input network), these references
# carry evidence that is INDEPENDENT of network topology:
#
#     disease genes   – clinical / genetic association  (DisGeNET)
#     essential genes – experimental gene-knockout       (DEG / OGEE)
#     drug targets    – pharmacological evidence          (DrugBank)
#     GO terms        – functional annotation             (Gene Ontology)
#
# A gene-set reference is a flat ``set[str]`` of gene symbols — no edges,
# no source/target split — because "importance" here is defined outside the
# graph.  GO additionally carries a gene → {term} mapping for enrichment.
# ---------------------------------------------------------------------------


@dataclass
class GeneSetReference:
    """A flat gene-set reference, normalised to upper-case symbols.

    ``genes`` is the membership set (e.g. all disease-associated genes).
    ``gene_terms`` / ``term_genes`` are only populated for GO annotations
    and hold the gene ↔ GO-term bipartite mapping used by GO enrichment.
    """

    name:        str
    kind:        str                          # "disease"|"essential"|"drug_target"|"go"
    genes:       set[str] = field(default_factory=set)
    gene_terms:  dict[str, set[str]] = field(default_factory=dict)
    term_genes:  dict[str, set[str]] = field(default_factory=dict)
    n_records:   int = 0
    source_path: Optional[Path] = None

    def is_empty(self) -> bool:
        return not self.genes


# Candidate gene-symbol column names per gene-set kind.  First present
# column wins; positional fallback uses the first alphabetic column.
_GENE_SET_COLUMNS: dict[str, tuple[str, ...]] = {
    # DisGeNET → "geneSymbol"; DISEASES/Jensen → col2 = "Gene symbol";
    # GWAS Catalog → "MAPPED_GENE" (may be multi-valued, best-effort);
    # fall through to the first alphabetic cell for header-less files.
    "disgenet": ("geneSymbol", "gene_symbol", "genesymbol", "symbol",
                 "gene", "gene_name", "Gene symbol", "MAPPED_GENE",
                 "REPORTED GENE(S)",
                 # HPO genes_to_phenotype / ClinVar gene_condition / Orphanet
                 "gene-symbol", "entrez-gene-symbol", "AssociatedGenes",
                 "GeneSymbol", "Gene_Symbol"),
    # DEG/OGEE → "gene_symbol"; DepMap Common Essentials → "gene" or
    # "Gene" (values like "TP53 (7157)" — the loader keeps them, matching
    # trims to the parenthesised suffix — see column-hint doc);
    # HART CEGv2.txt → "GENE" (single column, one gene per line).
    "deg":      ("gene_symbol", "symbol", "gene", "gene_name", "locus",
                 "genename", "Gene", "GENE"),
    # DrugBank → "Gene Name"; TTD → "TARGETID" is a code, "TARGNAME"
    # is descriptive — the useful column is "GENENAME" or "UNIPROID";
    # IUPHAR Targets and Families → "HGNC symbol" or "Human Ensembl Gene".
    "drugbank": ("Gene Name", "gene_name", "gene", "genename", "symbol",
                 "HGNC", "hgnc_symbol", "GENENAME", "TARGET_NAME",
                 "HGNC symbol", "Human Ensembl Gene", "Target Gene Symbol"),
}


_IDENTIFIER_PREFIXES = ("ENSP", "ENSG", "ENST", "HGNC:", "UNIPROT:",
                        "UNIPROTKB:", "NCBI:", "ENTREZ:")


def _first_alpha_value(row: dict) -> Optional[str]:
    """First value that looks like a gene symbol.

    Skips cells that are pure identifiers (Ensembl/UniProt/HGNC) so files
    like DISEASES (col 1 = Ensembl id, col 2 = gene symbol) fall through
    to the correct column.
    """
    for v in row.values():
        if v is None:
            continue
        s = str(v).strip()
        if not s or not any(c.isalpha() for c in s):
            continue
        if s.upper().startswith(_IDENTIFIER_PREFIXES):
            continue
        return s
    return None


def _clean_symbol(raw: str) -> str:
    """Normalise a gene-symbol cell.

    Handles the common source-specific quirks:
      * DepMap  – ``"TP53 (7157)"``  → ``"TP53"``   (strip Entrez suffix)
      * GWAS    – ``"TP53, MYC"``    → ``"TP53"``   (first gene of a list;
                                                     GWAS-style multi-gene
                                                     rows are common)
      * IUPHAR  – ``"HGNC:11998"``   → ``""``       (drop identifier-only)
    """
    s = str(raw).strip()
    # DepMap "SYM (12345)" — cut at the space before the paren
    if " (" in s and s.endswith(")"):
        s = s.split(" (", 1)[0].strip()
    # multi-gene rows: take the first entry
    for sep in (",", " - ", ";", "|"):
        if sep in s:
            s = s.split(sep, 1)[0].strip()
            break
    # drop pure-identifier rows
    if s.upper().startswith(("HGNC:", "UNIPROT:", "ENSG", "ENSP")):
        return ""
    return s


def _is_ttd_flat_format(path: Path) -> bool:
    """Sniff whether ``path`` is TTD's ``<target_id>\\t<field>\\t<value>``
    long-format record file (not a table).  Detects by scanning up to the
    first 200 non-blank lines for ``T<digits>\\tGENENAME\\t*`` rows.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            checked = 0
            for line in f:
                line = line.rstrip("\n\r")
                if not line or line.startswith("-"):
                    continue
                parts = line.split("\t")
                if (len(parts) >= 3
                        and parts[0].startswith("T")
                        and parts[0][1:].isdigit()
                        and parts[1].strip() == "GENENAME"):
                    return True
                checked += 1
                if checked > 200:
                    return False
    except Exception:
        return False
    return False


def _load_ttd_flat(path: Path, name: str) -> GeneSetReference:
    """Parse TTD's flat record file — extract every ``GENENAME`` value."""
    rs = GeneSetReference(name=name, kind="drug_target", source_path=path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 3 or parts[1].strip() != "GENENAME":
                continue
            gene = _clean_symbol(parts[2])
            if not gene:
                continue
            rs.genes.add(_norm(gene))
            rs.n_records += 1
    return rs


def _load_gene_set(kind: str, name: str,
                   path: Optional[Path] = None) -> Optional[GeneSetReference]:
    """Generic loader for a single-column-ish gene-set file.

    Extracts one gene symbol per row using the candidate columns for
    ``kind`` (positional fallback: first alphabetic cell).  Never raises;
    returns ``None`` when the file is missing or yields no symbols.

    Special-cased formats:
      * TTD (drug_target) — detected by ``_is_ttd_flat_format``, parsed
        via :func:`_load_ttd_flat` (long-format records rather than a table).

    Source-specific cell formats (DepMap ``"SYM (id)"``, GWAS multi-gene
    lists, IUPHAR pure identifiers) are normalised by :func:`_clean_symbol`.
    """
    p = _find_file(kind, path)
    if p is None:
        _LOG.info("%s reference not found — orthogonal %s validation skipped.",
                  name, kind)
        return None

    # TTD-specific fast path
    if kind == "drugbank" and _is_ttd_flat_format(p):
        rs = _load_ttd_flat(p, name="TTD")
        _LOG.info("%s (TTD flat) loaded: %d rows, %d unique genes",
                  name, rs.n_records, len(rs.genes))
        return rs if rs.genes else None

    rs = GeneSetReference(name=name, kind={
        "disgenet": "disease",
        "deg":      "essential",
        "drugbank": "drug_target",
    }.get(kind, kind), source_path=p)

    candidates = _GENE_SET_COLUMNS.get(kind, ("gene", "symbol"))
    for row in _iter_table(p):
        if not row:
            continue
        gene = None
        for col in candidates:
            if col in row and row[col] is not None and str(row[col]).strip():
                gene = str(row[col]).strip()
                break
        if gene is None:
            gene = _first_alpha_value(row)
        if not gene:
            continue
        gene = _clean_symbol(gene)
        if not gene:
            continue
        g = _norm(gene)
        # Skip obvious header echoes
        if g in {c.upper() for c in candidates} or g in {"GENE", "SYMBOL"}:
            continue
        if g:
            rs.genes.add(g)
            rs.n_records += 1

    _LOG.info("%s loaded: %d rows, %d unique genes",
              name, rs.n_records, len(rs.genes))
    return rs if rs.genes else None


def load_disgenet(path: Optional[Path] = None) -> Optional[GeneSetReference]:
    """Load DisGeNET disease-associated gene symbols."""
    return _load_gene_set("disgenet", "DisGeNET", path)


def load_deg(path: Optional[Path] = None) -> Optional[GeneSetReference]:
    """Load essential-gene symbols (DEG / OGEE)."""
    return _load_gene_set("deg", "DEG/OGEE", path)


def load_drugbank(path: Optional[Path] = None) -> Optional[GeneSetReference]:
    """Load DrugBank drug-target gene symbols."""
    return _load_gene_set("drugbank", "DrugBank", path)


def load_go_annotations(
    path: Optional[Path] = None,
    aspects: Optional[set[str]] = None,
) -> Optional[GeneSetReference]:
    """Load a GO annotation file (GAF 2.x) into a gene ↔ term mapping.

    GAF columns (1-indexed, tab-separated):
        3  DB Object Symbol   → gene symbol
        5  GO ID              → term (e.g. GO:0008150)
        9  Aspect             → P (process) | F (function) | C (component)

    ``aspects`` filters by the single-letter aspect code; ``None`` keeps
    all three.  Comment lines (starting with ``!``) are skipped.  Never
    raises; returns ``None`` when the file is missing or empty.
    """
    p = _find_file("go", path)
    if p is None:
        _LOG.info("GO annotation file not found — GO enrichment skipped.")
        return None

    rs = GeneSetReference(name="GeneOntology", kind="go", source_path=p)
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line or line.startswith("!"):
                    continue
                cols = line.rstrip("\n\r").split("\t")
                if len(cols) < 9:
                    continue
                symbol = _norm(cols[2])
                term   = cols[4].strip()
                aspect = cols[8].strip().upper() if len(cols) > 8 else ""
                if not symbol or not term:
                    continue
                if aspects and aspect not in aspects:
                    continue
                rs.genes.add(symbol)
                rs.gene_terms.setdefault(symbol, set()).add(term)
                rs.term_genes.setdefault(term, set()).add(symbol)
                rs.n_records += 1
    except Exception as exc:
        _LOG.warning("Failed to read GO annotations %s: %s", p, exc)
        return None

    _LOG.info("GO loaded: %d annotations, %d genes, %d terms",
              rs.n_records, len(rs.genes), len(rs.term_genes))
    return rs if rs.genes else None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_LOADERS = {
    "grn":   load_trrust,
    "ppi":   load_biogrid,
    "mirna": load_mirtarbase,
}

# Orthogonal gene-set loaders, keyed by evidence kind (network-type agnostic).
_GENE_SET_LOADERS = {
    "disease":     load_disgenet,
    "essential":   load_deg,
    "drug_target": load_drugbank,
}


# ---------------------------------------------------------------------------
# STRING protein-id → gene-symbol map + node_index_map remap helper
# ---------------------------------------------------------------------------
#
# The bundled STRING file (``9606.protein.links.v12.0.txt``) uses Ensembl
# protein ids (``9606.ENSP00000000233``).  Every biological reference
# (TRRUST / BioGRID / miRTarBase / DisGeNET / DEG / DrugBank / GO) uses
# HGNC gene symbols (``TP53``).  Without a translation step every overlap
# is zero and validation silently reports ``status=skipped``.
#
# STRING publishes the mapping in ``9606.protein.info.v12.0.txt``:
#
#     #string_protein_id  preferred_name  protein_size  annotation
#     9606.ENSP00000000233  ARF5           180           ADP-ribosylation…
#
# ``load_string_id_map`` returns ``{ensembl_id: gene_symbol}``.
# ``remap_node_index_map`` produces a new ``{gene_symbol: index}`` dict for
# the downstream validators; labels not present in the map are kept
# unchanged so partial mappings still work.

def load_string_id_map(path: Optional[Path] = None) -> dict[str, str]:
    """Load a STRING ``protein.info`` file into ``{protein_id → gene_symbol}``.

    Auto-discovers via the ``string_info`` pattern (``protein.info`` /
    ``string.info``) when ``path`` is not provided.  Returns ``{}`` when the
    file is missing or unreadable — never raises.
    """
    p = _find_file("string_info", path)
    if p is None:
        _LOG.info(
            "STRING protein-info file not found — Ensembl→symbol mapping "
            "unavailable; validators will only match on the raw labels.",
        )
        return {}

    out: dict[str, str] = {}
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                cols = line.rstrip("\n\r").split("\t")
                if len(cols) < 2:
                    continue
                pid = cols[0].strip()
                sym = cols[1].strip()
                if pid and sym:
                    out[pid] = sym.upper()
    except Exception as exc:
        _LOG.warning("Failed to read STRING info %s: %s", p, exc)
        return {}
    _LOG.info("STRING id map loaded: %d entries from %s", len(out), p)
    return out


def remap_node_index_map(
    node_index_map: dict[str, int],
    id_map: dict[str, str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Translate ``node_index_map`` labels through an id → symbol dict.

    Parameters
    ----------
    node_index_map : dict[str, int]
        Original ``{label: node_index}`` produced by ``graphdata_to_csr``.
    id_map : dict[str, str]
        Mapping from raw label (e.g. ``9606.ENSP00000000233``) to gene
        symbol (e.g. ``ARF5``).

    Returns
    -------
    (remapped, stats)
        ``remapped`` is a new ``{symbol_or_original_label: index}`` dict.
        Labels not present in ``id_map`` are kept unchanged so the returned
        map still covers every original node.  When two source labels
        translate to the same symbol only the first wins (isoform
        collisions); the loser stays under its original label so its index
        can still be reached.  ``stats`` reports ``{"mapped", "unchanged",
        "collisions"}`` counts for logging / driver output.
    """
    if not id_map:
        return dict(node_index_map), {"mapped": 0, "unchanged": len(node_index_map),
                                       "collisions": 0}

    remapped: dict[str, int] = {}
    mapped = unchanged = collisions = 0
    for label, idx in node_index_map.items():
        new_label = id_map.get(label)
        if new_label is None:
            new_label = label
            unchanged += 1
        else:
            mapped += 1
        if new_label in remapped:
            collisions += 1
            # keep the loser reachable under its original label
            if label not in remapped:
                remapped[label] = idx
        else:
            remapped[new_label] = idx
    _LOG.info(
        "remap_node_index_map: mapped=%d unchanged=%d collisions=%d",
        mapped, unchanged, collisions,
    )
    return remapped, {"mapped": mapped, "unchanged": unchanged,
                      "collisions": collisions}


def load_gene_set(kind: str,
                  path: Optional[Path] = None) -> Optional[GeneSetReference]:
    """Load an orthogonal gene-set reference by evidence kind.

    ``kind`` ∈ {"disease", "essential", "drug_target"}.  Returns ``None``
    for unknown kinds or missing files (never raises).
    """
    fn = _GENE_SET_LOADERS.get(kind)
    if fn is None:
        _LOG.warning("No gene-set loader for kind=%r", kind)
        return None
    try:
        return fn(path)
    except Exception as exc:
        _LOG.warning("Gene-set loader for %s raised %s — skipping.", kind, exc)
        return None


def load_reference(network_type: str,
                   path: Optional[Path] = None) -> Optional[ReferenceSet]:
    """Load the appropriate reference for the given network type."""
    fn = _LOADERS.get(network_type)
    if fn is None:
        _LOG.warning("No reference loader for network_type=%r", network_type)
        return None
    try:
        return fn(path)
    except Exception as exc:
        _LOG.warning("Reference loader for %s raised %s — skipping.",
                     network_type, exc)
        return None
