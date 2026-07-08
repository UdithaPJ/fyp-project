"""Modular preprocessing components for tabular-to-graph conversion."""

from __future__ import annotations

import csv
import io
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

from .graph_data import GraphData


@dataclass
class SchemaDetectionResult:
    """Detected graph column mapping with confidence information."""

    mapping: Dict[str, Optional[str]]
    confidence: Dict[str, float]
    column_scores: Dict[str, Dict[str, float]]


class FileLoader:
    """Load supported tabular files into pandas DataFrames."""

    LARGE_DATASET_ROW_THRESHOLD = 1_000_000
    LARGE_DATASET_MEMORY_THRESHOLD_BYTES = 250 * 1024 * 1024
    # MEMORY_FIX (Fix Cat. 7): callers can pass this threshold to
    # ``PreprocessingPipeline.run_adaptive`` to force chunked loading on
    # large files; default switches at 500 MB which keeps peak DataFrame
    # memory bounded by the chunk-aggregated cleaned frame rather than a
    # full single-shot read.
    LARGE_FILE_SIZE_THRESHOLD_BYTES = 500 * 1024 * 1024

    # OPTIMIZED (upload-time): candidate separators considered when
    # sniffing an unknown (.txt/.tab) delimiter from a small sample,
    # instead of parsing the whole file with the slow python engine.
    # ``r"\s+"`` (a regex, matched literally further below rather than
    # escaped) covers whitespace-separated formats such as STRING's
    # PPI export ("protein1 protein2 combined_score").
    _SNIFF_SAMPLE_BYTES = 8192
    _CANDIDATE_DELIMITERS = ["\t", ",", ";", "|", ":", r"\s+"]
    # Characters that legitimately occur inside identifiers/values and
    # must never be guessed as a delimiter just because they appear in
    # the header (gene/protein IDs commonly contain these).
    _DELIMITER_EXCLUDE_CHARS = set("_.-'\"")

    @classmethod
    def _split_line(cls, line: str, delimiter: str) -> List[str]:
        """Split one sample line on ``delimiter`` (plain char or regex)."""

        pattern = delimiter if delimiter == r"\s+" else re.escape(delimiter)
        return re.split(pattern, line.strip())

    @classmethod
    def _header_derived_candidates(cls, header: str) -> List[str]:
        """Discover extra delimiter candidates from punctuation in the header.

        Lets detection handle separators outside the fixed shortlist
        (e.g. ``~``, ``#``, ``@``) instead of only recognising a
        hardcoded set of "known" delimiters.
        """

        found: List[str] = []
        for ch in header:
            if ch.isalnum() or ch.isspace() or ch in cls._DELIMITER_EXCLUDE_CHARS:
                continue
            if ch not in found:
                found.append(ch)
        return found

    @classmethod
    def _sniff_delimiter(cls, sample: str) -> str:
        """Detect a delimiter from a small text sample, not the whole file.

        ``pd.read_csv(sep=None, engine="python")`` auto-detects the
        delimiter but forces the pure-Python parser across the *entire*
        file, which is 10-20x slower than the C engine on large files.
        Sniffing a small sample lets the fast C engine handle the actual
        parse for the common case.

        Every candidate — including csv.Sniffer's guess — is validated
        by checking that it splits the header into the SAME number of
        fields as every sampled data row. A delimiter that "looks
        standard" but disagrees with the header/data-row column count is
        rejected, so this generalises to whatever a given file actually
        uses (whitespace runs, colons, arbitrary punctuation) rather than
        only a hardcoded shortlist. Among candidates that pass, the one
        producing the most columns wins.
        """

        lines = [line for line in sample.splitlines() if line.strip()]
        if not lines:
            return ","
        header, *data_lines = lines
        data_lines = data_lines[:9] or [header]

        def consistent_column_count(delimiter: str) -> Optional[int]:
            try:
                expected = len(cls._split_line(header, delimiter))
            except re.error:
                return None
            if expected < 2:
                return None
            for line in data_lines:
                if len(cls._split_line(line, delimiter)) != expected:
                    return None
            return expected

        candidates = list(cls._CANDIDATE_DELIMITERS)
        try:
            sniffed_delimiter = csv.Sniffer().sniff(sample, delimiters="\t,;|: ").delimiter
            # A literal single space is subsumed by r"\s+" (already a
            # candidate) and strictly worse for whitespace-delimited
            # files: it splits variable-width runs into extra empty
            # fields, so skip it rather than let it win the tie against
            # the regex candidate.
            if sniffed_delimiter not in candidates and sniffed_delimiter != " ":
                candidates.insert(0, sniffed_delimiter)
        except csv.Error:
            pass
        for ch in cls._header_derived_candidates(header):
            if ch not in candidates:
                candidates.append(ch)

        best_delimiter: Optional[str] = None
        best_columns = 1  # must beat a trivial single-column split
        for delimiter in candidates:
            columns = consistent_column_count(delimiter)
            if columns is not None and columns > best_columns:
                best_delimiter, best_columns = delimiter, columns

        return best_delimiter if best_delimiter is not None else ","

    @staticmethod
    def _sep_read_kwargs(detected_sep: str, chunksize: Optional[int] = None) -> dict:
        """Build ``pd.read_csv`` kwargs appropriate for a detected separator.

        Regex separators (currently only ``r"\\s+"``) aren't supported by
        pandas' C engine and reject the ``low_memory`` kwarg under the
        Python engine, so the two paths need slightly different kwargs.
        """

        kwargs: dict = {"sep": detected_sep}
        if chunksize is not None:
            kwargs["chunksize"] = chunksize
        if detected_sep == r"\s+":
            kwargs["engine"] = "python"
        else:
            kwargs["low_memory"] = True
        return kwargs

    def load(self, file_path: str | Path) -> pd.DataFrame:
        """Load a CSV, TSV, TXT, Excel, or JSON file."""

        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            dataframe = self._read_delimited(path, sep=",")
            self._warn_if_large(dataframe, context=str(path))
            return dataframe
        if suffix == ".tsv":
            dataframe = self._read_delimited(path, sep="\t")
            self._warn_if_large(dataframe, context=str(path))
            return dataframe
        if suffix in {".txt", ".tab"}:
            dataframe = self._read_delimited(path, sep=None)
            self._warn_if_large(dataframe, context=str(path))
            return dataframe
        if suffix == ".xlsx":
            dataframe = pd.read_excel(path)
            self._warn_if_large(dataframe, context=str(path))
            return dataframe
        if suffix == ".json":
            dataframe = self._read_json(path)
            self._warn_if_large(dataframe, context=str(path))
            return dataframe

        raise ValueError(
            "Unsupported file type "
            f"'{suffix}'. Supported types are .csv, .tsv, .txt, .tab, .xlsx, and .json."
        )

    def load_bytes(self, filename: str, content: bytes) -> pd.DataFrame:
        """Load a supported file from in-memory bytes."""

        suffix = Path(filename).suffix.lower()

        if suffix == ".csv":
            dataframe = self._read_delimited_buffer(content, sep=",")
            self._warn_if_large(dataframe, context=filename)
            return dataframe
        if suffix == ".tsv":
            dataframe = self._read_delimited_buffer(content, sep="\t")
            self._warn_if_large(dataframe, context=filename)
            return dataframe
        if suffix in {".txt", ".tab"}:
            dataframe = self._read_delimited_buffer(content, sep=None)
            self._warn_if_large(dataframe, context=filename)
            return dataframe
        if suffix == ".xlsx":
            if not content:
                return pd.DataFrame()
            dataframe = pd.read_excel(io.BytesIO(content))
            self._warn_if_large(dataframe, context=filename)
            return dataframe
        if suffix == ".json":
            dataframe = self._read_json_buffer(content)
            self._warn_if_large(dataframe, context=filename)
            return dataframe

        raise ValueError(
            "Unsupported file type "
            f"'{suffix}'. Supported types are .csv, .tsv, .txt, .tab, .xlsx, and .json."
        )

    def load_in_chunks(
        self,
        file_path: str | Path,
        chunksize: int = 10000,
    ) -> Iterator[pd.DataFrame]:
        """Load a delimited text file as an iterator of DataFrame chunks."""

        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(path, sep=",", chunksize=chunksize, low_memory=True)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", chunksize=chunksize, low_memory=True)
        if suffix in {".txt", ".tab"}:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                sample = handle.read(self._SNIFF_SAMPLE_BYTES)
            if not sample.strip():
                return iter(())
            detected_sep = self._sniff_delimiter(sample)
            try:
                return pd.read_csv(
                    path, **self._sep_read_kwargs(detected_sep, chunksize=chunksize)
                )
            except Exception:
                # Fall back to the slow-but-robust auto-detecting parser.
                return pd.read_csv(
                    path,
                    sep=None,
                    engine="python",
                    chunksize=chunksize,
                )

        raise ValueError(
            "Chunk loading is supported for delimited text files only: "
            ".csv, .tsv, .txt, and .tab."
        )

    def _read_delimited(self, path: Path, sep: str | None) -> pd.DataFrame:
        """Read a delimited text file while handling empty inputs."""

        try:
            if sep is None:
                # OPTIMIZED (upload-time): sniff the delimiter from a small
                # sample and parse with the fast C engine, instead of
                # forcing the slow python engine across the whole file.
                with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                    sample = handle.read(self._SNIFF_SAMPLE_BYTES)
                if not sample.strip():
                    return pd.DataFrame()
                detected_sep = self._sniff_delimiter(sample)
                try:
                    return pd.read_csv(path, **self._sep_read_kwargs(detected_sep))
                except Exception:
                    return pd.read_csv(path, sep=None, engine="python")
            # OPTIMIZED: keep the default fast CSV engine for known delimiters.
            return pd.read_csv(path, sep=sep, low_memory=True)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _read_delimited_buffer(self, content: bytes, sep: str | None) -> pd.DataFrame:
        """Read a delimited text payload from bytes."""

        if not content.strip():
            return pd.DataFrame()

        try:
            if sep is None:
                # OPTIMIZED (upload-time): sniff the delimiter from a small
                # sample and parse with the fast C engine, instead of
                # forcing the slow python engine across the whole payload.
                sample = content[: self._SNIFF_SAMPLE_BYTES].decode(
                    "utf-8", errors="ignore"
                )
                detected_sep = self._sniff_delimiter(sample)
                try:
                    return pd.read_csv(
                        io.BytesIO(content), **self._sep_read_kwargs(detected_sep)
                    )
                except Exception:
                    return pd.read_csv(
                        io.BytesIO(content), sep=None, engine="python"
                    )
            # OPTIMIZED: keep the default fast CSV engine for known delimiters.
            return pd.read_csv(io.BytesIO(content), sep=sep, low_memory=True)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _read_json(self, path: Path) -> pd.DataFrame:
        """Read standard JSON records or JSON Lines into a DataFrame."""

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return pd.DataFrame()

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return pd.read_json(path, lines=True)

        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            return pd.json_normalize(payload)

        return pd.DataFrame({"value": [payload]})

    def _read_json_buffer(self, content: bytes) -> pd.DataFrame:
        """Read a JSON payload from bytes."""

        decoded = content.decode("utf-8").strip()
        if not decoded:
            return pd.DataFrame()

        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError:
            return pd.read_json(io.BytesIO(content), lines=True)

        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            return pd.json_normalize(payload)

        return pd.DataFrame({"value": [payload]})

    # OPTIMIZED (upload-time): below this row count, hitting the memory
    # threshold would require ~2.5 KB/row across all columns combined —
    # unrealistic for short gene/protein/TF identifier columns, so the
    # expensive deep memory scan is skipped entirely in that regime.
    _MEMORY_SCAN_ROW_FLOOR = 100_000

    def _warn_if_large(self, dataframe: pd.DataFrame, context: str) -> None:
        """Raise a warning when a loaded dataset is likely to be memory-intensive."""

        if dataframe.empty:
            return

        row_count = len(dataframe)
        if row_count < self._MEMORY_SCAN_ROW_FLOOR < self.LARGE_DATASET_ROW_THRESHOLD:
            return

        # OPTIMIZED: memory_usage(deep=True) walks every object-dtype cell
        # to size its Python string individually — real cost on large
        # frames. Only paid once row_count already suggests it's needed.
        memory_bytes = int(dataframe.memory_usage(index=True, deep=True).sum())
        if (
            row_count >= self.LARGE_DATASET_ROW_THRESHOLD
            or memory_bytes >= self.LARGE_DATASET_MEMORY_THRESHOLD_BYTES
        ):
            warnings.warn(
                (
                    f"Large dataset detected for '{context}' "
                    f"({row_count:,} rows, {memory_bytes / (1024 * 1024):.1f} MB). "
                    "Preprocessing may require substantial memory."
                ),
                ResourceWarning,
                stacklevel=2,
            )


class SchemaDetector:
    """Detect likely source, target, and weight columns."""

    SOURCE_ALIASES = {
        "source",
        "from",
        "src",
        "node1",
        "entity1",
        "protein1",
        "gene1",
        "start",
        "origin",
        "subject",
    }
    TARGET_ALIASES = {
        "target",
        "to",
        "dst",
        "node2",
        "entity2",
        "protein2",
        "gene2",
        "end",
        "destination",
        "object",
    }
    WEIGHT_ALIASES = {
        "weight",
        "score",
        "confidence",
        "probability",
        "value",
        "strength",
        "distance",
        "cost",
    }
    # Columns whose NAME marks them as identifiers, never edge weights.
    # A numeric identifier column (PMID, Entrez id, publication year, row
    # index, …) must not be auto-selected as a weight — using it would scale
    # every edge by an arbitrary id and wreck weighted algorithms.
    IDENTIFIER_ALIASES = {
        "id",
        "identifier",
        "pmid",
        "pubmed",
        "pubmedid",
        "doi",
        "entrez",
        "entrezid",
        "taxon",
        "taxonid",
        "index",
        "idx",
        "key",
        "uuid",
        "accession",
        "refseq",
        "ensembl",
        "year",
        "rowid",
    }
    # Unambiguous identifier tokens: if they appear anywhere in a normalized
    # column name it is an identifier regardless of surrounding characters.
    _IDENTIFIER_TOKENS = ("pmid", "pubmed", "entrez", "accession", "ensembl", "refseq", "taxon")
    # All-integer numeric columns whose magnitude exceeds this look like ids
    # (PMIDs, database accessions), not weights/counts/scores.
    _WEIGHT_MAX_MAGNITUDE = 10_000

    def detect(self, dataframe: pd.DataFrame) -> SchemaDetectionResult:
        """Detect graph-relevant columns and return mapping confidence."""

        columns = list(dataframe.columns)
        if not columns:
            empty_mapping = {"source": None, "target": None, "weight": None}
            return SchemaDetectionResult(
                mapping=empty_mapping,
                confidence={key: 0.0 for key in empty_mapping},
                column_scores={},
            )

        normalized_columns = {
            column: self._normalize_column_name(column) for column in columns
        }
        pair_boosts = self._infer_pair_boosts(columns, normalized_columns)
        # OPTIMIZED: compute numeric detection once per column and reuse it.
        numeric_flags = {
            column: self._is_mostly_numeric(dataframe[column]) for column in columns
        }
        entity_columns = [column for column in columns if not numeric_flags[column]]
        numeric_columns = [column for column in columns if numeric_flags[column]]
        # A numeric column only counts as weight-like if its VALUES resemble a
        # weight/score/count rather than an identifier (see _looks_like_weight).
        weightlike_flags = {
            column: (numeric_flags[column] and self._looks_like_weight(dataframe[column]))
            for column in columns
        }

        source_scores = {
            column: self._score_endpoint_column(
                column=column,
                role="source",
                entity_columns=entity_columns,
                pair_boosts=pair_boosts,
                normalized_column=normalized_columns[column],
            )
            for column in columns
        }
        target_scores = {
            column: self._score_endpoint_column(
                column=column,
                role="target",
                entity_columns=entity_columns,
                pair_boosts=pair_boosts,
                normalized_column=normalized_columns[column],
            )
            for column in columns
        }
        weight_scores = {
            column: self._score_weight_column(
                column=column,
                numeric_columns=numeric_columns,
                total_columns=len(columns),
                normalized_column=normalized_columns[column],
                is_weight_shaped=weightlike_flags[column],
            )
            for column in columns
        }

        source_column, source_confidence = self._pick_best(source_scores)
        target_scores_without_source = {
            column: score
            for column, score in target_scores.items()
            if column != source_column
        }
        target_column, target_confidence = self._pick_best(target_scores_without_source)
        weight_column, weight_confidence = self._pick_best(weight_scores)

        if weight_confidence < 0.40:
            weight_column = None
            weight_confidence = 0.0

        mapping = {
            "source": source_column if source_confidence >= 0.25 else None,
            "target": target_column if target_confidence >= 0.25 else None,
            "weight": weight_column,
        }

        return SchemaDetectionResult(
            mapping=mapping,
            confidence={
                "source": round(source_confidence, 3),
                "target": round(target_confidence, 3),
                "weight": round(weight_confidence, 3),
            },
            column_scores={
                "source": source_scores,
                "target": target_scores,
                "weight": weight_scores,
            },
        )

    def _score_endpoint_column(
        self,
        column: str,
        role: str,
        entity_columns: List[str],
        pair_boosts: Dict[str, Dict[str, float]],
        normalized_column: str,
    ) -> float:
        """Score a column as a likely source or target endpoint."""

        aliases = self.SOURCE_ALIASES if role == "source" else self.TARGET_ALIASES
        score = 0.0

        if normalized_column in aliases:
            score += 0.95
        elif any(alias in normalized_column for alias in aliases):
            score += 0.70

        if column in entity_columns:
            score += 0.15
            if role == "source" and entity_columns and column == entity_columns[0]:
                score += 0.30
            if role == "target" and len(entity_columns) > 1 and column == entity_columns[1]:
                score += 0.30

        score += pair_boosts.get(role, {}).get(column, 0.0)
        return round(min(score, 1.0), 3)

    def _score_weight_column(
        self,
        column: str,
        numeric_columns: List[str],
        total_columns: int,
        normalized_column: str,
        is_weight_shaped: bool,
    ) -> float:
        """Score a column as a likely edge-weight field.

        A weight-like NAME (``weight``, ``score``, ``combined_score``, …) is
        the strongest signal.  Absent that, a numeric column only earns points
        when its VALUES resemble a weight rather than an identifier — this is
        what stops a PMID / database-id / year column from being auto-selected
        as an edge weight (which would silently scale every edge by an
        arbitrary id and corrupt weighted algorithms).
        """

        score = 0.0

        name_has_weight = (
            normalized_column in self.WEIGHT_ALIASES
            or any(alias in normalized_column for alias in self.WEIGHT_ALIASES)
        )
        if normalized_column in self.WEIGHT_ALIASES:
            score += 0.95
        elif name_has_weight:
            score += 0.75

        # Identifier-named columns are never weights (unless the name ALSO
        # carries a weight token, e.g. a hypothetical "score_id").
        if not name_has_weight and self._is_identifier_name(normalized_column):
            return 0.0

        # Value-based evidence only when the column actually looks like a
        # weight — identifier-shaped numerics (large, high-cardinality ints)
        # contribute nothing, so they cannot cross the auto-select threshold
        # on numeric-ness alone.
        if column in numeric_columns and is_weight_shaped:
            score += 0.35
            if total_columns >= 3 and numeric_columns and column == numeric_columns[0]:
                score += 0.10

        return round(min(score, 1.0), 3)

    def _is_identifier_name(self, normalized_column: str) -> bool:
        """Whether a normalized column name denotes an identifier, not a weight."""

        if normalized_column in self.IDENTIFIER_ALIASES:
            return True
        if any(token in normalized_column for token in self._IDENTIFIER_TOKENS):
            return True
        # Names like "geneid", "protein_id", "node_id" (normalized: "...id").
        if len(normalized_column) > 2 and normalized_column.endswith("id"):
            return True
        return False

    def _looks_like_weight(self, series: pd.Series) -> bool:
        """Whether a numeric column's VALUES resemble a weight rather than an id.

        Weight-like: contains fractional values (scores/probabilities), or is
        a bounded small-magnitude integer column (edge counts, 0–1000 style
        scores).  Identifier-like (rejected): large-magnitude integers such as
        PMIDs / Entrez ids / accession numbers, especially when nearly every
        value is distinct.
        """

        numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if numeric.empty:
            return False

        # Any non-integer value is characteristic of a weight / score.
        if (numeric != numeric.round()).any():
            return True

        # All-integer: separate bounded counts/scores from identifiers.
        max_abs = float(numeric.abs().max())
        n = len(numeric)
        unique_ratio = numeric.nunique() / n if n else 0.0

        if max_abs > self._WEIGHT_MAX_MAGNITUDE:
            return False
        # Mid-magnitude but almost entirely unique → id-like (row keys, years).
        if max_abs > 1_000 and unique_ratio > 0.9:
            return False
        return True

    def _infer_pair_boosts(
        self,
        columns: List[str],
        normalized_columns: Dict[str, str],
    ) -> Dict[str, Dict[str, float]]:
        """Infer paired endpoint columns such as ProteinA/ProteinB or node1/node2."""

        source_boosts: Dict[str, float] = {}
        target_boosts: Dict[str, float] = {}

        for first, second in zip(columns, columns[1:]):
            first_match = re.match(r"^(.*?)(1|a)$", normalized_columns[first])
            second_match = re.match(r"^(.*?)(2|b)$", normalized_columns[second])
            if first_match and second_match and first_match.group(1) == second_match.group(1):
                source_boosts[first] = 0.55
                target_boosts[second] = 0.55

        return {"source": source_boosts, "target": target_boosts}

    def _pick_best(self, scores: Dict[str, float]) -> Tuple[Optional[str], float]:
        """Return the best-scoring column and its confidence."""

        if not scores:
            return None, 0.0
        column, score = max(scores.items(), key=lambda item: item[1])
        return column, score

    def _is_mostly_numeric(self, series: pd.Series) -> bool:
        """Check whether a column is mostly numeric after coercion."""

        non_null = series.dropna()
        if non_null.empty:
            return False
        numeric = pd.to_numeric(non_null, errors="coerce")
        return bool((numeric.notna().mean()) >= 0.75)

    def _normalize_column_name(self, column: str) -> str:
        """Normalize a column name for heuristic comparison."""

        return re.sub(r"[^a-z0-9]+", "", str(column).strip().lower())


class ColumnMapper:
    """Apply detected and user-provided column mappings."""

    def apply(
        self,
        dataframe: pd.DataFrame,
        detected_mapping: Dict[str, Optional[str]],
        user_override: Optional[Dict[str, str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
        """Create standard source, target, and weight columns."""

        final_mapping = dict(detected_mapping)
        if user_override:
            final_mapping.update(
                {key: value for key, value in user_override.items() if key in final_mapping}
            )

        missing_required = [
            field for field in ("source", "target") if not final_mapping.get(field)
        ]
        if missing_required:
            raise ValueError(
                "Unable to determine required columns "
                f"{missing_required}. Provide user_override to continue."
            )

        missing_columns = [
            column
            for column in final_mapping.values()
            if column is not None and column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Mapped columns not found in input data: {sorted(set(missing_columns))}."
            )

        # OPTIMIZED: assign builds the mapped view without an explicit full copy call.
        mapped = dataframe.assign(
            source=dataframe[final_mapping["source"]],
            target=dataframe[final_mapping["target"]],
            weight=(
                dataframe[final_mapping["weight"]]
                if final_mapping.get("weight")
                else pd.NA
            ),
        )

        return mapped, final_mapping


class DataCleaner:
    """Clean mapped data before graph construction."""

    def clean(
        self,
        dataframe: pd.DataFrame,
        default_weight: float = 1.0,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Remove invalid rows and normalize key graph columns."""

        cleaned = dataframe.copy(deep=False)

        # MEMORY_FIX (C-3): normalise source/target via a single vectorized
        # string pass then immediately convert to Categorical.  Pandas
        # `astype("string")` stores Python str objects (~50 B/value); on
        # 15 M rows × 2 endpoints this is ~1.5 GB of PyObject overhead.
        # Categorical stores each unique label once + int32 codes.
        for col in ("source", "target"):
            s = cleaned[col].astype("string").str.strip().str.upper()
            # NaN / empty rows are dropped a few lines below; keeping them
            # as nullable string here keeps the .eq("") + .isna() check
            # working before we collapse to Categorical.
            cleaned[col] = s
        cleaned["weight"] = pd.to_numeric(
            cleaned["weight"], errors="coerce"
        ).fillna(default_weight)

        missing_source = cleaned["source"].isna() | cleaned["source"].eq("")
        missing_target = cleaned["target"].isna() | cleaned["target"].eq("")
        invalid_rows = missing_source | missing_target
        dropped_rows = int(invalid_rows.sum())
        if invalid_rows.any():
            cleaned = cleaned.loc[~invalid_rows].reset_index(drop=True)

        # MEMORY_FIX (C-3): collapse to Categorical *after* row-dropping so
        # the category index does not include orphans.  float32 weight is
        # sufficient — biological edge weights rarely need FP64 precision.
        cleaned["source"] = cleaned["source"].astype("category")
        cleaned["target"] = cleaned["target"].astype("category")
        cleaned["weight"] = cleaned["weight"].astype("float32")

        return cleaned, {"dropped_rows_missing_endpoints": dropped_rows}


class DuplicateHandler:
    """Resolve duplicate edges with configurable aggregation."""

    SUPPORTED_STRATEGIES = {"count", "mean", "max", "none"}

    def handle(self, dataframe: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Apply duplicate-edge handling according to the chosen strategy."""

        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported duplicate strategy '{strategy}'. "
                f"Supported values: {sorted(self.SUPPORTED_STRATEGIES)}."
            )

        if dataframe.empty or strategy == "none":
            return dataframe.copy()

        group_columns = ["source", "target"]
        # OPTIMIZED: compute duplicate information once and reuse it.
        duplicate_mask = dataframe.duplicated(subset=group_columns, keep="first")
        duplicate_count = int(duplicate_mask.sum())
        dataframe.attrs["duplicate_edge_count"] = duplicate_count
        if duplicate_count == 0:
            return dataframe

        attribute_columns = [
            column for column in dataframe.columns if column not in {"source", "target", "weight"}
        ]
        # OPTIMIZED: category conversion happens only when duplicates exist.
        working_dataframe = dataframe.copy()
        shared_categories = pd.unique(
            pd.concat(
                [working_dataframe["source"], working_dataframe["target"]],
                ignore_index=True,
            )
        )
        category_dtype = pd.CategoricalDtype(categories=shared_categories, ordered=False)
        working_dataframe["source"] = working_dataframe["source"].astype(category_dtype)
        working_dataframe["target"] = working_dataframe["target"].astype(category_dtype)

        if strategy == "count":
            aggregation = {"weight": "size"}
            for column in attribute_columns:
                aggregation[column] = "first"

            counted = (
                working_dataframe.groupby(
                    group_columns,
                    sort=False,
                    as_index=False,
                    observed=True,
                )
                .agg(aggregation)
            )
            counted["source"] = counted["source"].astype("string")
            counted["target"] = counted["target"].astype("string")
            counted["weight"] = counted["weight"].astype(float)
            counted.attrs["duplicate_edge_count"] = duplicate_count
            return counted

        aggregation = {"weight": strategy}
        for column in attribute_columns:
            aggregation[column] = "first"

        deduplicated = (
            working_dataframe.groupby(
                group_columns,
                sort=False,
                as_index=False,
                observed=True,
            ).agg(aggregation)
        )
        deduplicated["source"] = deduplicated["source"].astype("string")
        deduplicated["target"] = deduplicated["target"].astype("string")
        deduplicated["weight"] = deduplicated["weight"].astype(float)
        deduplicated.attrs["duplicate_edge_count"] = duplicate_count
        return deduplicated


class ValidationModule:
    """Produce validation metrics for the preprocessing run."""

    def build_report(
        self,
        raw_dataframe: pd.DataFrame,
        mapped_dataframe: pd.DataFrame,
        cleaned_dataframe: pd.DataFrame,
        final_dataframe: pd.DataFrame,
        graph_data: GraphData,
        detection_result: SchemaDetectionResult,
        final_mapping: Dict[str, Optional[str]],
        duplicate_strategy: str,
        cleaning_stats: Dict[str, int],
    ) -> Dict[str, object]:
        """Build a structured validation report."""

        # OPTIMIZED: reuse cached duplicate count when available.
        duplicate_edges = cleaned_dataframe.attrs.get("duplicate_edge_count")
        if duplicate_edges is None:
            duplicate_edges = int(
                cleaned_dataframe.duplicated(subset=["source", "target"], keep="first").sum()
            ) if not cleaned_dataframe.empty else 0

        self_loops = int(
            final_dataframe["source"].astype("string").eq(
                final_dataframe["target"].astype("string")
            ).sum()
        ) if not final_dataframe.empty else 0

        report = {
            "number_of_rows": int(len(raw_dataframe)),
            "missing_values": self._count_missing_values(mapped_dataframe),
            "duplicate_edges": duplicate_edges,
            "number_of_nodes": int(len(graph_data.nodes)),
            "number_of_edges": int(len(graph_data.edges)),
            "self_loops": self_loops,
            "duplicate_strategy": duplicate_strategy,
            "detected_mapping": detection_result.mapping,
            "mapping_confidence": detection_result.confidence,
            "applied_mapping": final_mapping,
        }
        report.update(cleaning_stats)
        return report

    def _count_missing_values(self, dataframe: pd.DataFrame) -> Dict[str, int]:
        """Count missing graph-relevant values in the mapped data."""

        counts: Dict[str, int] = {}
        for column in ("source", "target", "weight"):
            if column not in dataframe.columns:
                counts[column] = 0
                continue

            if column in {"source", "target"}:
                # OPTIMIZED: use vectorized string checks instead of row-wise map.
                series = dataframe[column].astype("string")
                blank_count = int(series.str.strip().eq("").fillna(False).sum())
                counts[column] = int(dataframe[column].isna().sum() + blank_count)
            else:
                counts[column] = int(dataframe[column].isna().sum())

        return counts

    def build_chunked_report(
        self,
        number_of_rows: int,
        missing_values: Dict[str, int],
        cleaned_dataframe: pd.DataFrame,
        final_dataframe: pd.DataFrame,
        graph_data: GraphData,
        detection_result: SchemaDetectionResult,
        final_mapping: Dict[str, Optional[str]],
        duplicate_strategy: str,
        cleaning_stats: Dict[str, int],
    ) -> Dict[str, object]:
        """Build a validation report from chunk-aggregated intermediate results."""

        duplicate_edges = cleaned_dataframe.attrs.get("duplicate_edge_count")
        if duplicate_edges is None:
            duplicate_edges = int(
                cleaned_dataframe.duplicated(subset=["source", "target"], keep="first").sum()
            ) if not cleaned_dataframe.empty else 0

        self_loops = int(
            final_dataframe["source"].astype("string").eq(
                final_dataframe["target"].astype("string")
            ).sum()
        ) if not final_dataframe.empty else 0

        report = {
            "number_of_rows": int(number_of_rows),
            "missing_values": missing_values,
            "duplicate_edges": duplicate_edges,
            "number_of_nodes": int(len(graph_data.nodes)),
            "number_of_edges": int(len(graph_data.edges)),
            "self_loops": self_loops,
            "duplicate_strategy": duplicate_strategy,
            "detected_mapping": detection_result.mapping,
            "mapping_confidence": detection_result.confidence,
            "applied_mapping": final_mapping,
        }
        report.update(cleaning_stats)
        return report


class GraphBuilder:
    """Build a GraphData object from cleaned edge records."""

    def build(self, dataframe: pd.DataFrame) -> GraphData:
        """Convert a DataFrame into the GraphData contract."""

        graph = GraphData()
        if dataframe.empty:
            return graph

        # OPTIMIZED: extract unique nodes directly from the 2-column numpy view.
        nodes = pd.unique(dataframe[["source", "target"]].values.ravel())
        for node in nodes:
            graph.nodes[node] = {"id": node}

        attribute_columns = [
            column for column in dataframe.columns if column not in {"source", "target"}
        ]
        source_values = dataframe["source"].to_numpy()
        target_values = dataframe["target"].to_numpy()

        if attribute_columns:
            attribute_values = dataframe[attribute_columns].to_numpy(copy=False)
            for row_index, (source, target) in enumerate(zip(source_values, target_values)):
                attributes = {}
                for column_index, column in enumerate(attribute_columns):
                    value = attribute_values[row_index, column_index]
                    if not pd.isna(value):
                        attributes[column] = value
                graph.edges.append((source, target, attributes))
        else:
            for source, target in zip(source_values, target_values):
                graph.edges.append((source, target, {}))

        return graph