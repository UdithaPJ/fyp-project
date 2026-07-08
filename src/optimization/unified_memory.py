"""
src/optimization/unified_memory.py — CUDA managed-memory + pinned-memory wrappers
================================================================================

Provides safe allocation helpers for the four memory modes the
``MemoryManager`` may select:

  * ``unified_memory``   — ``cuda.managed_empty`` (CUDA Unified Memory; pages
                           migrate on demand between host and device).
  * ``zero_copy``        — page-locked host memory mapped into the device
                           address space.
  * ``pinned``           — page-locked host memory only (no device mapping);
                           faster H2D/D2H copies than pageable memory.
  * ``device``           — regular ``gpuarray.empty`` device allocation;
                           used as the universal fallback.

Every helper degrades gracefully when PyCUDA or the underlying driver
feature is unavailable — the caller never sees a hard failure from the
allocator itself.  The algorithm path makes the final decision about
whether to proceed or to fail with ``MemoryError``.

Allocators return ``ManagedArray`` exposing both a host-side NumPy view
and a device pointer.  For ``unified_memory`` the host view IS the
device buffer — CUDA migrates pages on access.

``prefetch_to_gpu`` / ``prefetch_to_cpu`` request explicit page
migration on managed allocations (no-op when the driver does not
expose ``cuMemPrefetchAsync``).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Optional PyCUDA
# ---------------------------------------------------------------------------

try:
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    PYCUDA_AVAILABLE = True
except Exception:                                       # noqa: BLE001
    cuda = None                                         # type: ignore[assignment]
    gpuarray = None                                     # type: ignore[assignment]
    PYCUDA_AVAILABLE = False


def _managed_available() -> bool:
    """Probe for ``cuda.managed_empty`` (PyCUDA 2014.1+, CUDA 6.0+)."""
    if not PYCUDA_AVAILABLE:
        return False
    return hasattr(cuda, "managed_empty") and hasattr(cuda, "managed_zeros")


def _pagelocked_available() -> bool:
    return PYCUDA_AVAILABLE and hasattr(cuda, "pagelocked_empty")


# ---------------------------------------------------------------------------
# ManagedArray wrapper
# ---------------------------------------------------------------------------

class ManagedArray:
    """Lightweight wrapper exposing a uniform interface across memory modes.

    Attributes
    ----------
    host_view : numpy.ndarray
        Host-accessible NumPy view.  For ``unified_memory`` and
        ``zero_copy`` the underlying buffer IS the memory the GPU
        kernel reads; for ``device`` mode it is a separate host scratch
        the user must H2D-copy.
    device_ptr : int | gpuarray.GPUArray | None
        Pointer suitable for a kernel launch.
    mode : str
        ``"unified_memory" | "zero_copy" | "pinned" | "device" | "host_only"``.
    """

    __slots__ = ("host_view", "device_ptr", "mode", "_owner")

    def __init__(self, host_view, device_ptr, mode, _owner=None):
        self.host_view  = host_view
        self.device_ptr = device_ptr
        self.mode       = mode
        self._owner     = _owner

    @property
    def shape(self):
        return self.host_view.shape

    @property
    def dtype(self):
        return self.host_view.dtype

    @property
    def nbytes(self):
        return int(self.host_view.nbytes)

    def free(self) -> None:
        """Release the underlying allocation.  Idempotent."""
        try:
            if self.mode == "device" and self.device_ptr is not None:
                try:
                    self.device_ptr.gpudata.free()
                except Exception:                       # noqa: BLE001
                    pass
        finally:
            self.host_view  = None                      # type: ignore[assignment]
            self.device_ptr = None
            self._owner     = None


# ---------------------------------------------------------------------------
# Allocators
# ---------------------------------------------------------------------------

def allocate_managed_array(shape, dtype) -> ManagedArray:
    """Allocate a CUDA Unified Memory array (falls back gracefully)."""
    if not _managed_available():
        return allocate_zero_copy_array(shape, dtype)
    try:
        arr = cuda.managed_empty(
            shape, dtype, mem_flags=cuda.mem_attach_flags.GLOBAL,
        )
        d_ptr = int(arr.base) if getattr(arr, "base", None) is not None else None
        return ManagedArray(
            host_view  = arr,
            device_ptr = d_ptr,
            mode       = "unified_memory",
            _owner     = arr,
        )
    except Exception as exc:                            # noqa: BLE001
        logging.warning(
            "allocate_managed_array: cuda.managed_empty failed (%s); "
            "falling back to zero-copy.", exc,
        )
        return allocate_zero_copy_array(shape, dtype)


def allocate_zero_copy_array(shape, dtype) -> ManagedArray:
    """Allocate page-locked host memory mapped into the device address space."""
    if not _pagelocked_available():
        return allocate_pinned_array(shape, dtype)
    try:
        flags = 0
        if hasattr(cuda, "host_alloc_flags"):
            flags = (cuda.host_alloc_flags.PORTABLE
                     | cuda.host_alloc_flags.DEVICEMAP)
        arr = cuda.pagelocked_empty(shape, dtype, mem_flags=flags)
        try:
            d_ptr = int(arr.base.get_device_pointer())
        except Exception:                               # noqa: BLE001
            d_ptr = None
        mode = "zero_copy" if d_ptr is not None else "pinned"
        return ManagedArray(
            host_view  = arr,
            device_ptr = d_ptr,
            mode       = mode,
            _owner     = arr,
        )
    except Exception as exc:                            # noqa: BLE001
        logging.warning(
            "allocate_zero_copy_array: pagelocked_empty failed (%s); "
            "falling back to pinned host memory.", exc,
        )
        return allocate_pinned_array(shape, dtype)


def allocate_pinned_array(shape, dtype) -> ManagedArray:
    """Allocate page-locked host memory (NOT mapped into device space)."""
    if not _pagelocked_available():
        return allocate_device_array(shape, dtype)
    try:
        arr = cuda.pagelocked_empty(shape, dtype)
        return ManagedArray(
            host_view  = arr,
            device_ptr = None,
            mode       = "pinned",
            _owner     = arr,
        )
    except Exception as exc:                            # noqa: BLE001
        logging.warning(
            "allocate_pinned_array: pagelocked_empty failed (%s); "
            "falling back to device-only allocation.", exc,
        )
        return allocate_device_array(shape, dtype)


def allocate_device_array(shape, dtype) -> ManagedArray:
    """Allocate a plain device array via ``gpuarray.empty``.

    Falls back to a pure-host NumPy array when PyCUDA is unavailable
    OR when allocation fails (e.g. no active CUDA context).
    """
    if not PYCUDA_AVAILABLE:
        host = np.empty(shape, dtype=dtype)
        return ManagedArray(
            host_view  = host,
            device_ptr = None,
            mode       = "host_only",
            _owner     = host,
        )
    try:
        ga = gpuarray.empty(shape, dtype=dtype)
    except Exception as exc:                            # noqa: BLE001
        # Typical reason: no active CUDA context (this allocator was
        # called before the algorithm pushed the primary context).
        logging.warning(
            "allocate_device_array: gpuarray.empty failed (%s); "
            "falling back to host-only allocation.", exc,
        )
        host = np.empty(shape, dtype=dtype)
        return ManagedArray(
            host_view  = host,
            device_ptr = None,
            mode       = "host_only",
            _owner     = host,
        )
    host = np.empty(shape, dtype=dtype)
    return ManagedArray(
        host_view  = host,
        device_ptr = ga,
        mode       = "device",
        _owner     = ga,
    )


# ---------------------------------------------------------------------------
# Copy + prefetch helpers
# ---------------------------------------------------------------------------

def copy_to_managed_array(np_array, target: ManagedArray | None = None,
                          *, dtype=None) -> ManagedArray:
    """Materialise ``np_array`` into a managed / zero-copy / device buffer.

    Picks the best mode when ``target`` is None; otherwise fills the
    target's host view (and, for ``device`` mode, H2D's the device).
    """
    np_array = np.ascontiguousarray(
        np_array, dtype=(dtype or np_array.dtype),
    )
    if target is None:
        target = allocate_managed_array(np_array.shape, np_array.dtype)

    if target.mode in ("unified_memory", "zero_copy", "pinned", "host_only"):
        target.host_view[...] = np_array
    else:  # device
        target.host_view[...] = np_array
        if isinstance(target.device_ptr, gpuarray.GPUArray):
            target.device_ptr.set(np_array)
        else:
            cuda.memcpy_htod(target.device_ptr, np_array)
    return target


def prefetch_to_gpu(managed: ManagedArray, stream=None) -> None:
    """Prefetch a unified-memory allocation to GPU 0 (no-op otherwise)."""
    if managed.mode != "unified_memory" or not PYCUDA_AVAILABLE:
        return
    if not hasattr(cuda, "mem_prefetch_async"):
        return
    try:
        cuda.mem_prefetch_async(
            int(managed.host_view.ctypes.data),
            int(managed.nbytes),
            cuda.Device(0),
            stream,
        )
    except Exception:                                   # noqa: BLE001
        pass


def prefetch_to_cpu(managed: ManagedArray, stream=None) -> None:
    """Prefetch a unified-memory allocation back to the CPU."""
    if managed.mode != "unified_memory" or not PYCUDA_AVAILABLE:
        return
    if not hasattr(cuda, "mem_prefetch_async"):
        return
    try:
        cuda.mem_prefetch_async(
            int(managed.host_view.ctypes.data),
            int(managed.nbytes),
            None,
            stream,
        )
    except Exception:                                   # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------

def capabilities() -> dict:
    """Report which memory modes are usable in the current process."""
    return {
        "pycuda_available":   PYCUDA_AVAILABLE,
        "managed_memory":     _managed_available(),
        "pagelocked_memory":  _pagelocked_available(),
        "mem_prefetch_async": (PYCUDA_AVAILABLE
                               and hasattr(cuda, "mem_prefetch_async")),
    }


__all__ = [
    "ManagedArray",
    "allocate_managed_array",
    "allocate_zero_copy_array",
    "allocate_pinned_array",
    "allocate_device_array",
    "copy_to_managed_array",
    "prefetch_to_gpu",
    "prefetch_to_cpu",
    "capabilities",
]
