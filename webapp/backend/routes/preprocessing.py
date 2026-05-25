"""Preprocessing API routes."""

from __future__ import annotations

import json
from queue import Queue
from threading import Thread

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

try:
    from ..models.requests import DetectRequest, PreprocessRequest
    from ..models.responses import DetectResponse, PreprocessResponse, UploadResponse
    from ..services.preprocessing_service import preprocessing_service
except ImportError:  # pragma: no cover - fallback for running from backend directory
    from models.requests import DetectRequest, PreprocessRequest
    from models.responses import DetectResponse, PreprocessResponse, UploadResponse
    from services.preprocessing_service import preprocessing_service


router = APIRouter(tags=["preprocessing"])


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a dataset and return a preview."""

    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return await preprocessing_service.upload_file(file.filename, content)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Upload failed: {error}") from error


@router.post("/detect", response_model=DetectResponse)
async def detect_columns(payload: DetectRequest) -> DetectResponse:
    """Detect likely graph columns from sample data."""

    try:
        return preprocessing_service.detect_columns(
            columns=payload.columns,
            sample=payload.sample,
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Detection failed: {error}") from error


@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_dataset(payload: PreprocessRequest) -> PreprocessResponse:
    """Run preprocessing for a previously uploaded dataset."""

    try:
        return preprocessing_service.preprocess(
            upload_id=payload.upload_id,
            mapping=payload.mapping.model_dump(),
            duplicate_method=payload.duplicate_method,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {error}") from error


@router.post("/preprocess/stream")
async def preprocess_dataset_stream(payload: PreprocessRequest) -> StreamingResponse:
    """Stream preprocessing progress updates and the final result as JSON lines."""

    def event_stream():
        event_queue: Queue[object] = Queue()
        stream_complete = object()

        def progress_callback(step_name: str, percentage: float) -> None:
            event_queue.put(
                {
                    "event": "progress",
                    "step": step_name,
                    "percentage": round(float(percentage), 4),
                }
            )

        def worker() -> None:
            try:
                response = preprocessing_service.preprocess_with_progress(
                    upload_id=payload.upload_id,
                    mapping=payload.mapping.model_dump(),
                    duplicate_method=payload.duplicate_method,
                    progress_callback=progress_callback,
                )
                event_queue.put(
                    {
                        "event": "result",
                        "data": response.model_dump(),
                    }
                )
            except KeyError as error:
                event_queue.put(
                    {
                        "event": "error",
                        "detail": str(error),
                        "status": 404,
                    }
                )
            except ValueError as error:
                event_queue.put(
                    {
                        "event": "error",
                        "detail": str(error),
                        "status": 400,
                    }
                )
            except Exception as error:
                event_queue.put(
                    {
                        "event": "error",
                        "detail": f"Preprocessing failed: {error}",
                        "status": 500,
                    }
                )
            finally:
                event_queue.put(stream_complete)

        worker_thread = Thread(target=worker, daemon=True)
        worker_thread.start()

        while True:
            event = event_queue.get()
            if event is stream_complete:
                break
            yield json.dumps(event).encode("utf-8") + b"\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
