"""
backend/routes/system.py — System capability endpoint
======================================================

Endpoint
--------
    GET /system/gpu-status
        Probes the host for a CUDA-capable GPU and returns availability,
        device name, and CUDA version.  The frontend calls this once on
        app load and blocks the workflow with an error if no GPU is found —
        because all analyses run exclusively in GPU mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter

CURRENT_DIR  = Path(__file__).resolve().parent
BACKEND_DIR  = CURRENT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent.parent   # fyp-project root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/gpu-status")
def gpu_status() -> dict:
    """
    Report whether a CUDA GPU is available on this machine.

    Tries PyCUDA first (preferred — same library used by the GPU kernels),
    then falls back to CuPy.  Returns:

        {
          "available":    bool,
          "device_name":  str | null,
          "cuda_version": str | null,
          "driver":       str,       # "pycuda" | "cupy" | "none"
          "error":        str | null
        }
    """
    # ---- Try PyCUDA ----
    try:
        import pycuda.driver as cuda
        cuda.init()
        if cuda.Device.count() == 0:
            return _unavailable("PyCUDA initialised but no CUDA devices found.")
        dev = cuda.Device(0)
        name = dev.name()
        cuda_version = cuda.get_version()
        cuda_str = ".".join(str(x) for x in cuda_version)
        return {
            "available":    True,
            "device_name":  name,
            "cuda_version": cuda_str,
            "driver":       "pycuda",
            "error":        None,
        }
    except Exception as pycuda_err:
        pass

    # ---- Try CuPy as fallback ----
    try:
        import cupy as cp
        _ = cp.zeros(1)   # force context init
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        ver = cp.cuda.runtime.runtimeGetVersion()
        major, rem = divmod(ver, 1000)
        minor = rem // 10
        return {
            "available":    True,
            "device_name":  name,
            "cuda_version": f"{major}.{minor}",
            "driver":       "cupy",
            "error":        None,
        }
    except Exception as cupy_err:
        return _unavailable(
            f"No CUDA GPU detected.  "
            f"PyCUDA: {pycuda_err}.  "
            f"CuPy: {cupy_err}."
        )


def _unavailable(reason: str) -> dict:
    return {
        "available":    False,
        "device_name":  None,
        "cuda_version": None,
        "driver":       "none",
        "error":        reason,
    }
