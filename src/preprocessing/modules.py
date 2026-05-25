"""Modular preprocessing components for tabular-to-graph conversion."""

from __future__ import annotations

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
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                chunksize=chunksize,
                low_memory=True,
            )

        raise ValueError(
            "Chunk loading is supported for delimited text files only: "
            ".csv, .tsv, .txt, and .tab."
        )

    def _read_delimited(self, path: Path, sep: str | None) -> pd.DataFrame:
        """Read a delimited text file while handling empty inputs."""

        try:
            if sep is None:
                # OPTIMIZED: delimiter sniffing needs the Python engine here.
                return pd.read_csv(path, sep=None, engine="python", low_memory=True)
            # OPTIMIZED: keep the default fast CSV engine for known delimiters.
            return pd.read_csv(path, sep=sep, low_memory=True)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _read_delimited_buffer(self, content: bytes, sep: str | None) -> pd.DataFrame:
        """Read a delimited text payload from bytes."""

        if not content.strip():
            return pd.DataFrame()

        buffer = io.BytesIO(content)
        try:
            if sep is None:
                # OPTIMIZED: delimiter sniffing needs the Python engine here.
                return pd.read_csv(buffer, sep=None, engine="python", low_memory=True)
            # OPTIMIZED: keep the default fast CSV engine for known delimiters.
            return pd.read_csv(buffer, sep=sep, low_memory=True)
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

    def _warn_if_large(self, dataframe: pd.DataFrame, context: str) -> None:
        """Raise a warning when a loaded dataset is likely to be memory-intensive."""

        if dataframe.empty:
            return

        row_count = len(dataframe)
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
    ) -> float:
        """Score a column as a likely edge-weight field."""

        score = 0.0

        if normalized_column in self.WEIGHT_ALIASES:
            score += 0.95
        elif any(alias in normalized_column for alias in self.WEIGHT_ALIASES):
            score += 0.75

        if column in numeric_columns:
            score += 0.35
            if total_columns >= 3 and numeric_columns and column == numeric_columns[0]:
                score += 0.10

        return round(min(score, 1.0), 3)

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

        # OPTIMIZED: trim string-like columns in a vectorized way.
        string_columns = cleaned.select_dtypes(include=["object", "string"]).columns
        if len(string_columns) > 0:
            cleaned[string_columns] = cleaned[string_columns].apply(
                lambda column: column.str.strip()
            )

        # OPTIMIZED: normalize node identifiers with vectorized string operations.
        cleaned["source"] = (
            cleaned["source"]
            .astype("string")
            .str.strip()
            .str.upper()
        )
        cleaned["target"] = (
            cleaned["target"]
            .astype("string")
            .str.strip()
            .str.upper()
        )
        cleaned["weight"] = pd.to_numeric(
            cleaned["weight"], errors="coerce"
        ).fillna(default_weight)

        missing_source = cleaned["source"].isna() | cleaned["source"].eq("")
        missing_target = cleaned["target"].isna() | cleaned["target"].eq("")
        invalid_rows = missing_source | missing_target
        dropped_rows = int(invalid_rows.sum())
        if invalid_rows.any():
            cleaned = cleaned.loc[~invalid_rows].reset_index(drop=True)
        cleaned["weight"] = cleaned["weight"].astype(float)

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
