"""
src/utils/memory_logger.py
==========================

Diagnostic memory logging helpers used by the benchmarking layer to
trace RAM / VRAM consumption around hot operations.

MEMORY_FIX: introduced as part of the 2026-06 memory audit so we can
verify which fixes actually move the needle on real benchmark runs.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Optional


def _rss_mb() -> float:
    """Return the current process RSS in MB (best-effort, cross-platform)."""
    # Prefer psutil when available — it works on Windows, Linux, macOS.
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:                                              # noqa: BLE001
        pass
    # Linux fallback.
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:                                              # noqa: BLE001
        pass
    # Windows fallback via ctypes (matches scalability_benchmark.py).
    try:
        import ctypes
        import ctypes.wintypes as _W

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", _W.DWORD),
                ("PageFaultCount", _W.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage2", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        k32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        h = k32.OpenProcess(0x0400 | 0x0010, False, k32.GetCurrentProcessId())
        psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        k32.CloseHandle(h)
        return pmc.WorkingSetSize / 1024 ** 2
    except Exception:                                              # noqa: BLE001
        return 0.0


def _vram_used_mb() -> Optional[float]:
    """Return (total - free) VRAM in MB, or None if no CUDA available."""
    # CuPy path (already-active runtime context).
    try:
        import cupy as cp
        free, total = cp.cuda.runtime.memGetInfo()
        return (total - free) / (1024 * 1024)
    except Exception:                                              # noqa: BLE001
        pass
    # PyCUDA fallback — only query if a context is already pushed; we do
    # NOT create one here just for a log message.
    try:
        import pycuda.driver as cuda
        free, total = cuda.mem_get_info()
        return (total - free) / (1024 * 1024)
    except Exception:                                              # noqa: BLE001
        return None


def log_memory(label: str, logger: Optional[logging.Logger] = None) -> None:
    """Log current RAM (and VRAM if available) with a descriptive label."""
    rss_mb = _rss_mb()
    msg = f"[MEMORY] {label}: RAM={rss_mb:.1f} MB"
    vram = _vram_used_mb()
    if vram is not None:
        msg += f", VRAM_used={vram:.1f} MB"
    (logger or logging.getLogger("memory")).info(msg)


def force_gc(label: str = "", logger: Optional[logging.Logger] = None) -> float:
    """Force garbage collection; log freed MB (only when > 1 MB).

    Returns the number of MB reclaimed.
    """
    before = _rss_mb()
    gc.collect()
    after = _rss_mb()
    freed_mb = max(0.0, before - after)
    if freed_mb > 1.0:
        msg = f"[GC] {label or 'force_gc'}: freed {freed_mb:.1f} MB"
        (logger or logging.getLogger("memory")).info(msg)
    return freed_mb


__all__ = ["log_memory", "force_gc"]
