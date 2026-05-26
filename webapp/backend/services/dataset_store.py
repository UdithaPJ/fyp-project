"""In-memory storage for uploaded datasets.

A single record per ``upload_id`` holds both the raw DataFrame produced by
``/upload`` and the heavy preprocessing artefacts produced by
``/preprocess`` (the ``GraphData`` object, the scipy CSR adjacency matrix,
and the node-label → integer-index map).

The graph fields are populated by ``preprocessing_service.py`` immediately
after the pipeline runs, and are then retrieved directly by
``algorithm_service.py`` for every subsequent algorithm run.  The
preprocessing pipeline therefore runs EXACTLY ONCE per upload session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4

import pandas as pd


@dataclass
class DatasetRecord:
    """Stored dataset metadata, raw DataFrame, and (optional) graph artefacts.

    The ``graph_data``, ``graph_csr``, and ``node_index_map`` fields are
    ``None`` until the user runs the preprocessing step.  After
    ``preprocessing_service`` runs the pipeline once, all three are
    populated and can be read directly by ``algorithm_service`` without
    ever touching the pipeline again.
    """

    upload_id:      str
    filename:       str
    dataframe:      pd.DataFrame
    # Populated after /preprocess (or /preprocess/stream) completes.
    graph_data:     Optional[Any]            = None    # src.preprocessing.GraphData
    graph_csr:      Optional[Any]            = None    # scipy.sparse.csr_matrix
    node_index_map: Optional[Dict[str, int]] = None


class InMemoryDatasetStore:
    """Small in-memory store keyed by upload id."""

    def __init__(self) -> None:
        """Initialize the internal storage dictionary."""

        self._datasets: Dict[str, DatasetRecord] = {}

    def save(self, filename: str, dataframe: pd.DataFrame) -> DatasetRecord:
        """Save a dataset and return its record.

        Graph fields are initialised to ``None`` — they are filled in later
        by ``preprocessing_service`` once the pipeline completes.
        """

        upload_id = uuid4().hex
        record = DatasetRecord(
            upload_id=upload_id,
            filename=filename,
            # OPTIMIZED: keep the loaded DataFrame by reference to avoid an extra full copy.
            dataframe=dataframe,
        )
        self._datasets[upload_id] = record
        return record

    def get(self, upload_id: str) -> DatasetRecord:
        """Retrieve a previously uploaded dataset.

        Raises ``KeyError`` if the ``upload_id`` is unknown.  The graph
        artefacts on the returned record may still be ``None`` if the
        user has not completed the preprocessing step yet — callers that
        require them must check explicitly.
        """

        if upload_id not in self._datasets:
            raise KeyError(f"No uploaded dataset found for id '{upload_id}'.")
        return self._datasets[upload_id]

    def update(self, upload_id: str, fields: Dict[str, Any]) -> DatasetRecord:
        """Update one or more fields of an existing dataset record.

        Used by ``preprocessing_service`` to attach the ``graph_data``,
        ``graph_csr``, and ``node_index_map`` after the pipeline runs.

        Only attributes that already exist on :class:`DatasetRecord` may
        be assigned; unknown keys raise ``AttributeError`` to surface
        typos at development time.
        """

        if upload_id not in self._datasets:
            raise KeyError(f"No uploaded dataset found for id '{upload_id}'.")

        record = self._datasets[upload_id]
        for key, value in fields.items():
            if not hasattr(record, key):
                raise AttributeError(
                    f"DatasetRecord has no field '{key}' — refusing to set it."
                )
            setattr(record, key, value)
        return record


dataset_store = InMemoryDatasetStore()
