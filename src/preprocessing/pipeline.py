"""Pipeline orchestration for tabular-to-graph preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd

from .graph_data import GraphData
from .modules import (
    ColumnMapper,
    DataCleaner,
    DuplicateHandler,
    FileLoader,
    GraphBuilder,
    SchemaDetector,
    SchemaDetectionResult,
    ValidationModule,
)


class PreprocessingPipeline:
    """Coordinate the modular preprocessing steps."""

    CHUNK_MODE_FILE_SIZE_THRESHOLD_BYTES = 100 * 1024 * 1024

    def __init__(self) -> None:
        """Initialize reusable preprocessing modules."""

        self.file_loader = FileLoader()
        self.schema_detector = SchemaDetector()
        self.column_mapper = ColumnMapper()
        self.data_cleaner = DataCleaner()
        self.duplicate_handler = DuplicateHandler()
        self.validation_module = ValidationModule()
        self.graph_builder = GraphBuilder()

    def run(
        self,
        file_path: str | Path,
        user_override: Optional[Dict[str, str]] = None,
        duplicate_strategy: str = "mean",
        default_weight: float = 1.0,
    ) -> Tuple[GraphData, Dict[str, object]]:
        """Preprocess a tabular file into GraphData plus a validation report."""

        raw_dataframe = self.file_loader.load(file_path)
        return self.run_dataframe(
            raw_dataframe=raw_dataframe,
            user_override=user_override,
            duplicate_strategy=duplicate_strategy,
            default_weight=default_weight,
        )

    def run_dataframe(
        self,
        raw_dataframe,
        user_override: Optional[Dict[str, str]] = None,
        duplicate_strategy: str = "mean",
        default_weight: float = 1.0,
    ) -> Tuple[GraphData, Dict[str, object]]:
        """Preprocess an in-memory DataFrame into GraphData plus a validation report."""

        if raw_dataframe.empty and len(raw_dataframe.columns) == 0:
            empty_graph = GraphData()
            empty_detection = SchemaDetectionResult(
                mapping={"source": None, "target": None, "weight": None},
                confidence={"source": 0.0, "target": 0.0, "weight": 0.0},
                column_scores={},
            )
            report = self.validation_module.build_report(
                raw_dataframe=raw_dataframe,
                mapped_dataframe=raw_dataframe.copy(),
                cleaned_dataframe=raw_dataframe.copy(),
                final_dataframe=raw_dataframe.copy(),
                graph_data=empty_graph,
                detection_result=empty_detection,
                final_mapping=empty_detection.mapping,
                duplicate_strategy=duplicate_strategy,
                cleaning_stats={"dropped_rows_missing_endpoints": 0},
            )
            report["issues"] = ["Input file is empty."]
            return empty_graph, report

        detection_result = self.schema_detector.detect(raw_dataframe)
        mapped_dataframe, final_mapping = self.column_mapper.apply(
            dataframe=raw_dataframe,
            detected_mapping=detection_result.mapping,
            user_override=user_override,
        )
        cleaned_dataframe, cleaning_stats = self.data_cleaner.clean(
            mapped_dataframe,
            default_weight=default_weight,
        )
        final_dataframe = self.duplicate_handler.handle(
            cleaned_dataframe,
            strategy=duplicate_strategy,
        )
        graph_data = self.graph_builder.build(final_dataframe)
        report = self.validation_module.build_report(
            raw_dataframe=raw_dataframe,
            mapped_dataframe=mapped_dataframe,
            cleaned_dataframe=cleaned_dataframe,
            final_dataframe=final_dataframe,
            graph_data=graph_data,
            detection_result=detection_result,
            final_mapping=final_mapping,
            duplicate_strategy=duplicate_strategy,
            cleaning_stats=cleaning_stats,
        )
        return graph_data, report

    def run_with_chunks(
        self,
        file_path: str | Path,
        user_override: Optional[Dict[str, str]] = None,
        duplicate_strategy: str = "mean",
        default_weight: float = 1.0,
        chunksize: int = 10000,
    ) -> Tuple[GraphData, Dict[str, object]]:
        """Preprocess a delimited file by loading and cleaning it in chunks."""

        chunk_iterator = self.file_loader.load_in_chunks(file_path, chunksize=chunksize)

        try:
            first_chunk = next(chunk_iterator)
        except StopIteration:
            empty_dataframe = self.file_loader.load(file_path)
            return self.run_dataframe(
                raw_dataframe=empty_dataframe,
                user_override=user_override,
                duplicate_strategy=duplicate_strategy,
                default_weight=default_weight,
            )

        total_rows = int(len(first_chunk))
        detection_result = self.schema_detector.detect(first_chunk)
        final_mapping: Dict[str, Optional[str]]
        cleaned_chunks = []
        missing_values_total = {"source": 0, "target": 0, "weight": 0}
        dropped_rows_total = 0

        mapped_first_chunk, final_mapping = self.column_mapper.apply(
            dataframe=first_chunk,
            detected_mapping=detection_result.mapping,
            user_override=user_override,
        )
        first_missing_values = self.validation_module._count_missing_values(mapped_first_chunk)
        for key, value in first_missing_values.items():
            missing_values_total[key] += int(value)
        cleaned_first_chunk, first_cleaning_stats = self.data_cleaner.clean(
            mapped_first_chunk,
            default_weight=default_weight,
        )
        dropped_rows_total += first_cleaning_stats["dropped_rows_missing_endpoints"]
        if not cleaned_first_chunk.empty:
            cleaned_chunks.append(cleaned_first_chunk)

        for chunk in chunk_iterator:
            total_rows += int(len(chunk))
            mapped_chunk, _ = self.column_mapper.apply(
                dataframe=chunk,
                detected_mapping=final_mapping,
                user_override=None,
            )
            chunk_missing_values = self.validation_module._count_missing_values(mapped_chunk)
            for key, value in chunk_missing_values.items():
                missing_values_total[key] += int(value)

            cleaned_chunk, cleaning_stats = self.data_cleaner.clean(
                mapped_chunk,
                default_weight=default_weight,
            )
            dropped_rows_total += cleaning_stats["dropped_rows_missing_endpoints"]
            if not cleaned_chunk.empty:
                cleaned_chunks.append(cleaned_chunk)

        if cleaned_chunks:
            cleaned_dataframe = pd.concat(cleaned_chunks, ignore_index=True, copy=False)
        else:
            cleaned_dataframe = pd.DataFrame(columns=["source", "target", "weight"])

        final_dataframe = self.duplicate_handler.handle(
            cleaned_dataframe,
            strategy=duplicate_strategy,
        )
        graph_data = self.graph_builder.build(final_dataframe)
        report = self.validation_module.build_chunked_report(
            number_of_rows=total_rows,
            missing_values=missing_values_total,
            cleaned_dataframe=cleaned_dataframe,
            final_dataframe=final_dataframe,
            graph_data=graph_data,
            detection_result=detection_result,
            final_mapping=final_mapping,
            duplicate_strategy=duplicate_strategy,
            cleaning_stats={"dropped_rows_missing_endpoints": dropped_rows_total},
        )
        return graph_data, report

    def run_with_progress(
        self,
        raw_dataframe,
        user_override: Optional[Dict[str, str]] = None,
        duplicate_strategy: str = "mean",
        default_weight: float = 1.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[GraphData, Dict[str, object]]:
        """Preprocess an in-memory DataFrame while reporting coarse-grained progress."""

        self._notify_progress(progress_callback, "starting", 0.0)

        if raw_dataframe.empty and len(raw_dataframe.columns) == 0:
            self._notify_progress(progress_callback, "validation", 0.95)
            empty_graph = GraphData()
            empty_detection = SchemaDetectionResult(
                mapping={"source": None, "target": None, "weight": None},
                confidence={"source": 0.0, "target": 0.0, "weight": 0.0},
                column_scores={},
            )
            report = self.validation_module.build_report(
                raw_dataframe=raw_dataframe,
                mapped_dataframe=raw_dataframe.copy(),
                cleaned_dataframe=raw_dataframe.copy(),
                final_dataframe=raw_dataframe.copy(),
                graph_data=empty_graph,
                detection_result=empty_detection,
                final_mapping=empty_detection.mapping,
                duplicate_strategy=duplicate_strategy,
                cleaning_stats={"dropped_rows_missing_endpoints": 0},
            )
            report["issues"] = ["Input file is empty."]
            self._notify_progress(progress_callback, "completed", 1.0)
            return empty_graph, report

        self._notify_progress(progress_callback, "schema_detection", 0.1)
        detection_result = self.schema_detector.detect(raw_dataframe)

        self._notify_progress(progress_callback, "mapping", 0.25)
        mapped_dataframe, final_mapping = self.column_mapper.apply(
            dataframe=raw_dataframe,
            detected_mapping=detection_result.mapping,
            user_override=user_override,
        )

        self._notify_progress(progress_callback, "cleaning", 0.5)
        cleaned_dataframe, cleaning_stats = self.data_cleaner.clean(
            mapped_dataframe,
            default_weight=default_weight,
        )

        self._notify_progress(progress_callback, "duplicate_handling", 0.7)
        final_dataframe = self.duplicate_handler.handle(
            cleaned_dataframe,
            strategy=duplicate_strategy,
        )

        self._notify_progress(progress_callback, "graph_building", 0.85)
        graph_data = self.graph_builder.build(final_dataframe)

        self._notify_progress(progress_callback, "validation", 0.95)
        report = self.validation_module.build_report(
            raw_dataframe=raw_dataframe,
            mapped_dataframe=mapped_dataframe,
            cleaned_dataframe=cleaned_dataframe,
            final_dataframe=final_dataframe,
            graph_data=graph_data,
            detection_result=detection_result,
            final_mapping=final_mapping,
            duplicate_strategy=duplicate_strategy,
            cleaning_stats=cleaning_stats,
        )
        self._notify_progress(progress_callback, "completed", 1.0)
        return graph_data, report

    def run_adaptive(
        self,
        file_path: str | Path,
        user_override: Optional[Dict[str, str]] = None,
        duplicate_strategy: str = "mean",
        default_weight: float = 1.0,
        chunksize: int = 10000,
        chunk_mode_file_size_threshold_bytes: Optional[int] = None,
    ) -> Tuple[GraphData, Dict[str, object]]:
        """Automatically switch to chunk mode for large delimited files."""

        path = Path(file_path)
        threshold = (
            chunk_mode_file_size_threshold_bytes
            if chunk_mode_file_size_threshold_bytes is not None
            else self.CHUNK_MODE_FILE_SIZE_THRESHOLD_BYTES
        )
        suffix = path.suffix.lower()
        supports_chunk_mode = suffix in {".csv", ".tsv", ".txt", ".tab"}
        use_chunk_mode = supports_chunk_mode and path.exists() and path.stat().st_size >= threshold

        if use_chunk_mode:
            return self.run_with_chunks(
                file_path=file_path,
                user_override=user_override,
                duplicate_strategy=duplicate_strategy,
                default_weight=default_weight,
                chunksize=chunksize,
            )

        return self.run(
            file_path=file_path,
            user_override=user_override,
            duplicate_strategy=duplicate_strategy,
            default_weight=default_weight,
        )

    def _notify_progress(
        self,
        progress_callback: Optional[Callable[[str, float], None]],
        step_name: str,
        percentage: float,
    ) -> None:
        """Invoke the progress callback when one is provided."""

        if progress_callback is not None:
            progress_callback(step_name, percentage)


def preprocess_file_to_graph(
    file_path: str | Path,
    user_override: Optional[Dict[str, str]] = None,
    duplicate_strategy: str = "mean",
    default_weight: float = 1.0,
) -> Tuple[GraphData, Dict[str, object]]:
    """Convenience function for one-off preprocessing runs."""

    pipeline = PreprocessingPipeline()
    return pipeline.run(
        file_path=file_path,
        user_override=user_override,
        duplicate_strategy=duplicate_strategy,
        default_weight=default_weight,
    )


def preprocess_file_to_graph_with_chunks(
    file_path: str | Path,
    user_override: Optional[Dict[str, str]] = None,
    duplicate_strategy: str = "mean",
    default_weight: float = 1.0,
    chunksize: int = 10000,
) -> Tuple[GraphData, Dict[str, object]]:
    """Convenience function for chunk-based preprocessing runs."""

    pipeline = PreprocessingPipeline()
    return pipeline.run_with_chunks(
        file_path=file_path,
        user_override=user_override,
        duplicate_strategy=duplicate_strategy,
        default_weight=default_weight,
        chunksize=chunksize,
    )


def preprocess_file_to_graph_adaptive(
    file_path: str | Path,
    user_override: Optional[Dict[str, str]] = None,
    duplicate_strategy: str = "mean",
    default_weight: float = 1.0,
    chunksize: int = 10000,
    chunk_mode_file_size_threshold_bytes: Optional[int] = None,
) -> Tuple[GraphData, Dict[str, object]]:
    """Convenience function that automatically enables chunk mode for large files."""

    pipeline = PreprocessingPipeline()
    return pipeline.run_adaptive(
        file_path=file_path,
        user_override=user_override,
        duplicate_strategy=duplicate_strategy,
        default_weight=default_weight,
        chunksize=chunksize,
        chunk_mode_file_size_threshold_bytes=chunk_mode_file_size_threshold_bytes,
    )
