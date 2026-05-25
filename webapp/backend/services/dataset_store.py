"""In-memory storage for uploaded datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from uuid import uuid4

import pandas as pd


@dataclass
class DatasetRecord:
    """Stored dataset metadata and contents."""

    upload_id: str
    filename: str
    dataframe: pd.DataFrame


class InMemoryDatasetStore:
    """Small in-memory store keyed by upload id."""

    def __init__(self) -> None:
        """Initialize the internal storage dictionary."""

        self._datasets: Dict[str, DatasetRecord] = {}

    def save(self, filename: str, dataframe: pd.DataFrame) -> DatasetRecord:
        """Save a dataset and return its record."""

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
        """Retrieve a previously uploaded dataset."""

        if upload_id not in self._datasets:
            raise KeyError(f"No uploaded dataset found for id '{upload_id}'.")
        return self._datasets[upload_id]


dataset_store = InMemoryDatasetStore()
