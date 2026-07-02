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
    # Orthogonal gene-set references (evidence unrelated to network topology)
    "disgenet":   ("disgenet", "gene_disease", "curated_gene_disease",
                   "all_gene_disease"),
    "deg":        ("deg", "ogee", "essential"),
    "drugbank":   ("drugbank", "drug_target", "drug-target",
                   "all_target_polypeptide"),
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

        # Try DictReader (header) first
        first = buf_lines[0].rstrip("\n\r").split(chosen_sep)
        has_header = any(any(c.isalpha() for c in cell) and not cell.replace(".", "").replace("-", "").isdigit()
                          for cell in first)

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

def load_biogrid(path: Optional[Path] = None) -> Optional[ReferenceSet]:
    """
    Load a BioGRID-style PPI file.

    Accepts both BioGRID-tab2/tab3 and a simple two-column edge list.
    For BioGRID files the symbol columns are
    'Official Symbol Interactor A/B' (tab2/3).
    """
    p = _find_file("biogrid", path)
    if p is None:
        _LOG.info("BioGRID reference not found — biological PPI validation skipped.")
        return None

    rs = ReferenceSet(name="BioGRID", network_type="ppi", source_path=p)
    header_consumed = False
    for row in _iter_table(p):
        if not row:
            continue
        a = (row.get("Official Symbol Interactor A")
             or row.get("Symbol A") or row.get("symbol_a")
             or row.get("interactor_a") or row.get("source")
             or row.get("col0"))
        b = (row.get("Official Symbol Interactor B")
             or row.get("Symbol B") or row.get("symbol_b")
             or row.get("interactor_b") or row.get("target")
             or row.get("col1"))
        if not a or not b:
            a, b = _first_two_values(row)
        if not a or not b:
            continue
        a, b = _norm(str(a)), _norm(str(b))
        if not a or not b or a == b:
            continue
        if not header_consumed:
            header_consumed = True
            # Skip an obvious header row
            if a in {"SOURCE", "PROTEIN1", "SYMBOL A", "INTERACTOR_A"} or \
               b in {"TARGET", "PROTEIN2", "SYMBOL B", "INTERACTOR_B"}:
                continue
        rs.sources.add(a)
        rs.sources.add(b)   # PPI is undirected — both ends are "hubs"
        rs.targets.add(a)
        rs.targets.add(b)
        rs.edges.add(frozenset({a, b}))
        rs.n_records += 1

    _LOG.info("BioGRID loaded: %d edges, %d nodes",
              rs.n_records, len(rs.sources))
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
    "disgenet": ("geneSymbol", "gene_symbol", "genesymbol", "symbol",
                 "gene", "gene_name"),
    "deg":      ("gene_symbol", "symbol", "gene", "gene_name", "locus",
                 "genename"),
    "drugbank": ("Gene Name", "gene_name", "gene", "genename", "symbol",
                 "HGNC", "hgnc_symbol"),
}


def _first_alpha_value(row: dict) -> Optional[str]:
    """First value that contains a letter (a plausible gene symbol)."""
    for v in row.values():
        if v is None:
            continue
        s = str(v).strip()
        if s and any(c.isalpha() for c in s):
            return s
    return None


def _load_gene_set(kind: str, name: str,
                   path: Optional[Path] = None) -> Optional[GeneSetReference]:
    """Generic loader for a single-column-ish gene-set file.

    Extracts one gene symbol per row using the candidate columns for
    ``kind`` (positional fallback: first alphabetic cell).  Never raises;
    returns ``None`` when the file is missing or yields no symbols.
    """
    p = _find_file(kind, path)
    if p is None:
        _LOG.info("%s reference not found — orthogonal %s validation skipped.",
                  name, kind)
        return None

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
