"""Services for file ingestion, detection, validation, and graph building."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent          # webapp/backend
WEBAPP_DIR  = BACKEND_DIR.parent          # webapp
PROJECT_ROOT = WEBAPP_DIR.parent          # fyp-project (top of repo)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.converter import graphdata_to_csr
from src.preprocessing.modules import FileLoader, SchemaDetector
from src.preprocessing.pipeline import PreprocessingPipeline

try:
    from ..models.graph import EdgeData, GraphData
    from ..models.responses import DetectResponse, PreprocessResponse, UploadResponse
    from .dataset_store import DatasetRecord, dataset_store
except ImportError:  # pragma: no cover - fallback for running from backend directory
    from models.graph import EdgeData, GraphData
    from models.responses import DetectResponse, PreprocessResponse, UploadResponse
    from services.dataset_store import DatasetRecord, dataset_store


class PreprocessingService:
    """Application service for the preprocessing workflow."""

    GRAPH_PREVIEW_EDGE_LIMIT = 5
    GRAPH_PREVIEW_NODE_LIMIT = 50
    LARGE_DATASET_ROW_THRESHOLD = 1_000_000
    LARGE_DATASET_MEMORY_THRESHOLD_BYTES = 250 * 1024 * 1024

    def __init__(self) -> None:
        """Initialize reusable processing components."""

        self.file_loader = FileLoader()
        self.schema_detector = SchemaDetector()
        self.pipeline = PreprocessingPipeline()

    async def upload_file(self, filename: str, content: bytes) -> UploadResponse:
        """Load, store, and preview an uploaded dataset."""

        dataframe = self.file_loader.load_bytes(filename, content)
        record = dataset_store.save(filename=filename, dataframe=dataframe)
        return UploadResponse(
            upload_id=record.upload_id,
            filename=record.filename,
            columns=[str(column) for column in dataframe.columns.tolist()],
            preview=self._preview_rows(dataframe, limit=20),
            row_count=int(len(dataframe)),
        )

    def detect_columns(
        self,
        columns: List[str],
        sample: List[Dict[str, Any]],
    ) -> DetectResponse:
        """Detect likely graph columns from a sample payload."""

        sample_dataframe = pd.DataFrame(sample)
        if columns and sample_dataframe.empty:
            sample_dataframe = pd.DataFrame(columns=columns)
        elif columns:
            sample_dataframe = sample_dataframe.reindex(columns=columns)

        detection = self.schema_detector.detect(sample_dataframe)
        confidence_values = [
            detection.confidence["source"],
            detection.confidence["target"],
        ]
        if detection.mapping.get("weight") is not None:
            confidence_values.append(detection.confidence["weight"])
        overall_confidence = round(
            sum(confidence_values) / len(confidence_values), 3
        ) if confidence_values else 0.0

        return DetectResponse(
            source=detection.mapping.get("source"),
            target=detection.mapping.get("target"),
            weight=detection.mapping.get("weight"),
            confidence=overall_confidence,
            field_confidence=detection.confidence,
        )

    def preprocess(
        self,
        upload_id: str,
        mapping: Dict[str, str | None],
        duplicate_method: str,
    ) -> PreprocessResponse:
        """Run cleaning, duplicate handling, validation, and graph building.

        After the pipeline completes, the resulting ``GraphData``, the
        derived scipy CSR matrix, and the node-label → index map are all
        attached to the dataset record so subsequent algorithm runs can
        retrieve them directly without re-running the pipeline.
        """

        record = self._get_dataset(upload_id)
        self._warn_if_large(record.dataframe, context=record.filename)
        graph_data, validation = self.pipeline.run_dataframe(
            raw_dataframe=record.dataframe,
            user_override=mapping,
            duplicate_strategy=duplicate_method,
        )
        self._store_graph_artefacts(upload_id, graph_data)
        api_graph = self._to_api_graph(graph_data)
        return PreprocessResponse(
            nodes=len(graph_data.nodes),
            edges=len(graph_data.edges),
            validation=self._make_json_safe(validation),
            graph=api_graph,
        )

    def preprocess_with_progress(
        self,
        upload_id: str,
        mapping: Dict[str, str | None],
        duplicate_method: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> PreprocessResponse:
        """Run preprocessing while reporting progress through a callback.

        Stores the same graph artefacts as :meth:`preprocess` so the
        algorithm layer can fetch them without re-running the pipeline.
        """

        record = self._get_dataset(upload_id)
        self._warn_if_large(record.dataframe, context=record.filename)
        graph_data, validation = self.pipeline.run_with_progress(
            raw_dataframe=record.dataframe,
            user_override=mapping,
            duplicate_strategy=duplicate_method,
            progress_callback=progress_callback,
        )
        self._store_graph_artefacts(upload_id, graph_data)
        api_graph = self._to_api_graph(graph_data)
        return PreprocessResponse(
            nodes=len(graph_data.nodes),
            edges=len(graph_data.edges),
            validation=self._make_json_safe(validation),
            graph=api_graph,
        )

    def _store_graph_artefacts(self, upload_id: str, graph_data) -> None:
        """Attach ``graph_csr`` and ``node_index_map`` to the dataset record.

        The CSR matrix is computed here (once) instead of inside the
        algorithm service so that every algorithm run can reuse it.
        """

        graph_csr, node_index_map = graphdata_to_csr(graph_data)
        # MEMORY_FIX (C-2): drop the raw DataFrame and the GraphData edge
        # list once the CSR is built — algorithm runs only need the CSR
        # plus node_index_map.  Holding all three pinned ~1.9 GB / upload
        # on a 15 M-edge graph (see memory_audit_report.md C-2).
        dataset_store.update(
            upload_id,
            {
                "dataframe":      None,
                "graph_data":     None,
                "graph_csr":      graph_csr,
                "node_index_map": node_index_map,
            },
        )
        import gc  # local import keeps module-level imports unchanged
        del graph_data
        gc.collect()

    def _get_dataset(self, upload_id: str) -> DatasetRecord:
        """Resolve an uploaded dataset or raise a descriptive error."""

        return dataset_store.get(upload_id)

    def _preview_rows(self, dataframe: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
        """Convert the first rows of a DataFrame into JSON-safe records."""

        if dataframe.empty:
            return []
        preview = dataframe.head(limit)
        return self._records_to_json_safe(preview.to_dict(orient="records"))

    def _warn_if_large(self, dataframe: pd.DataFrame, context: str) -> None:
        """Raise a warning when a dataset is likely to be memory-intensive."""

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
                    "Processing may require substantial memory."
                ),
                ResourceWarning,
                stacklevel=2,
            )

    def _to_api_graph(self, graph_data) -> GraphData:
        """Convert the core GraphData object into a lightweight API preview model."""

        # OPTIMIZED: return only a small preview to avoid serializing huge graphs.
        preview_edges = graph_data.edges[: self.GRAPH_PREVIEW_EDGE_LIMIT]
        preview_node_ids = []
        seen_nodes = set()
        for source, target, _attributes in preview_edges:
            if source not in seen_nodes:
                seen_nodes.add(source)
                preview_node_ids.append(source)
            if target not in seen_nodes:
                seen_nodes.add(target)
                preview_node_ids.append(target)
            if len(preview_node_ids) >= self.GRAPH_PREVIEW_NODE_LIMIT:
                break

        preview_nodes = {
            node_id: self._make_json_safe(graph_data.nodes[node_id])
            for node_id in preview_node_ids[: self.GRAPH_PREVIEW_NODE_LIMIT]
            if node_id in graph_data.nodes
        }

        edges = [
            EdgeData(
                source=source,
                target=target,
                attributes=self._make_json_safe(attributes),
            )
            for source, target, attributes in preview_edges
        ]
        return GraphData(
            nodes=preview_nodes,
            edges=edges,
        )

    def _records_to_json_safe(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize preview records for JSON responses."""

        return [self._make_json_safe(record) for record in records]

    def _make_json_safe(self, value: Any) -> Any:
        """Convert pandas and numpy values into plain JSON-friendly objects."""

        if isinstance(value, dict):
            return {str(key): self._make_json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._make_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self._make_json_safe(item) for item in value]
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except TypeError:
                pass
        return value


preprocessing_service = PreprocessingService()
