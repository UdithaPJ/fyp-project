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


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_LOADERS = {
    "grn":   load_trrust,
    "ppi":   load_biogrid,
    "mirna": load_mirtarbase,
}


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
