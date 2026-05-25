"""
src/algorithms/base.py — Abstract base class for all algorithm modules
=======================================================================

Every algorithm in ``src/algorithms/`` subclasses :class:`AlgorithmBase`
and exposes three ``@staticmethod`` mode functions:

    cpu_single(graph_csr, params) -> dict
    cpu_multi (graph_csr, params) -> dict
    gpu       (graph_csr, params) -> dict

Each returns a standardised result dict with the keys enforced by
:meth:`AlgorithmBase.validate_result`:

    {
      "algorithm":      str,           # algorithm name (matches cls.NAME)
      "mode":           str,           # "cpu_single" | "cpu_multi" | "gpu"
      "execution_time": float,         # seconds — internal perf_counter timing
      "num_nodes":      int,
      "num_edges":      int,
      "result":         dict,          # algorithm-specific output
    }

The runner ``src/runner/algorithm_runner.py`` wraps each call with its
own :class:`BenchmarkTimer` (CUDA-event accurate for GPU mode) and
**overrides** ``execution_time`` with that more precise measurement.
The internal value is the fallback and a sanity floor.
"""

from __future__ import annotations

import warnings
from abc import ABC
from typing import Any, Dict

import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Required result keys
# ---------------------------------------------------------------------------

REQUIRED_RESULT_KEYS = (
    "algorithm",
    "mode",
    "execution_time",
    "num_nodes",
    "num_edges",
    "result",
)

VALID_MODES = ("cpu_single", "cpu_multi", "gpu")


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class AlgorithmBase(ABC):
    """
    Abstract contract for all algorithm implementations.

    Subclass requirements
    ---------------------
    * Override ``NAME`` (string) and ``PARAM_SCHEMA`` (dict of defaults).
    * Implement ``cpu_single``, ``cpu_multi``, and ``gpu`` as
      ``@staticmethod`` taking ``(graph_csr, params)`` and returning a
      standardised result dict (use :meth:`build_result`).
    """

    # Subclasses MUST override these
    NAME: str = ""
    PARAM_SCHEMA: Dict[str, Any] = {}

    # ---- Mode methods (subclasses override) ----

    @staticmethod
    def cpu_single(graph_csr: sp.csr_matrix, params: dict) -> dict:
        raise NotImplementedError("cpu_single() not implemented")

    @staticmethod
    def cpu_multi(graph_csr: sp.csr_matrix, params: dict) -> dict:
        raise NotImplementedError("cpu_multi() not implemented")

    @staticmethod
    def gpu(graph_csr: sp.csr_matrix, params: dict) -> dict:
        raise NotImplementedError("gpu() not implemented")

    # ---- Parameter handling ----

    @classmethod
    def get_default_params(cls) -> dict:
        """Return a fresh copy of the algorithm's default parameter dict."""
        return dict(cls.PARAM_SCHEMA)

    @classmethod
    def validate_params(cls, params: dict | None) -> dict:
        """
        Merge user params over defaults; warn on unknown keys.

        Unknown keys are kept in the returned dict (so GPU-tuning keys
        like ``block_size`` injected by ``apply_config`` survive), but a
        ``UserWarning`` is emitted so typos in canonical params surface.
        """
        defaults = cls.get_default_params()
        if not params:
            return defaults

        canonical_keys = set(defaults.keys())
        # Reserved keys that the runner / GPU config inject and that every
        # algorithm should tolerate without warning.
        tuning_keys = {
            "block_size", "use_chunking", "use_shared_mem", "use_shared_mem_hash",
            "use_zero_copy", "use_bitmap_frontier", "use_direction_opt",
            "batch_seeds", "community_id_bits", "top_k_per_column",
            "legacy_kernel_mode", "low_sm_mode", "gpu_disabled",
            "chunking_reason", "warn_oom_risk", "precision",
        }

        unknown = [
            k for k in params.keys()
            if k not in canonical_keys and k not in tuning_keys
        ]
        if unknown:
            warnings.warn(
                f"{cls.NAME}: ignoring unknown param keys (kept in dict but "
                f"likely typos): {unknown}",
                UserWarning,
                stacklevel=2,
            )

        merged = dict(defaults)
        merged.update(params)
        return merged

    # ---- Result construction & validation ----

    @classmethod
    def build_result(
        cls,
        mode: str,
        execution_time: float,
        graph_csr: sp.csr_matrix,
        result_data: dict,
    ) -> dict:
        """
        Construct a standardised result dict.

        Subclasses call this at the end of every mode method.  The runner
        may later overwrite ``execution_time`` with its CUDA-event-accurate
        value, but the dict structure remains stable.
        """
        return {
            "algorithm":      cls.NAME,
            "mode":           mode,
            "execution_time": float(execution_time),
            "num_nodes":      int(graph_csr.shape[0]),
            "num_edges":      int(graph_csr.nnz),
            "result":         result_data,
        }

    @staticmethod
    def validate_result(result: dict) -> None:
        """
        Raise ``ValueError`` if the result dict is missing required keys
        or has an invalid ``mode``.

        The runner calls this after every algorithm invocation to catch
        misimplemented subclasses early.
        """
        if not isinstance(result, dict):
            raise ValueError(
                f"Algorithm result must be a dict, got {type(result).__name__}"
            )
        missing = [k for k in REQUIRED_RESULT_KEYS if k not in result]
        if missing:
            raise ValueError(
                f"Algorithm result missing required keys: {missing}.  "
                f"Got keys: {sorted(result.keys())}"
            )
        if result["mode"] not in VALID_MODES:
            raise ValueError(
                f"Invalid result['mode']={result['mode']!r}; "
                f"must be one of {VALID_MODES}"
            )
        if not isinstance(result["result"], dict):
            raise ValueError(
                f"result['result'] must be a dict, "
                f"got {type(result['result']).__name__}"
            )

    @classmethod
    def describe(cls) -> dict:
        """Return a JSON-safe description of this algorithm for the API."""
        return {
            "name":         cls.NAME,
            "param_schema": cls.get_default_params(),
            "description":  (cls.__doc__ or "").strip().split("\n")[0],
        }
