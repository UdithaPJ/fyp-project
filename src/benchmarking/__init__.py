"""
src/benchmarking
================

Benchmarking utilities for the FYP framework.

Submodules
----------

* :mod:`src.benchmarking.benchmark` — original benchmarking runner
  (``run_benchmark``, ``BenchmarkTimer``, ``_ensure_cuda_context``).
  Used by the algorithm files and existing scripts.

* :mod:`src.benchmarking.runtime_benchmark` —
  :class:`RuntimeBenchmarker`: repeated-run timing across all four
  execution modes (cpu_single / cpu_multi / gpu_baseline / gpu).
  Computes mean / std / min / max runtimes and speedup ratios.
  Writes ``runtime_benchmark.csv`` and four plots.

* :mod:`src.benchmarking.scalability_benchmark` —
  :class:`ScalabilityBenchmarker`: benchmarks scalability across
  three synthetic graph types (Barabási–Albert, Erdős–Rényi,
  Watts–Strogatz) at multiple sizes (10K–500K nodes).
  Measures runtime and peak memory delta.
  Writes ``scalability_benchmark.csv`` and three plots.

* :mod:`src.benchmarking.memory_benchmark` —
  :class:`MemoryBenchmarker`: compares GPU-baseline vs optimised GPU
  for memory efficiency (peak VRAM, peak RAM, max graph size).
  Writes ``memory_benchmark.csv`` and four plots.

* :mod:`src.benchmarking.architecture_benchmark` —
  :class:`ArchitectureBenchmarker`: sweeps block_size and chunk_size
  for the optimised GPU implementation ONLY.
  Writes ``architecture_benchmark.csv`` and two plots.
  Exposes ``best_block_size``, ``best_chunk_size``, ``best_runtime``
  per algorithm.

* :mod:`src.benchmarking.convergence_benchmark` —
  :class:`ConvergenceBenchmarker`: benchmarks convergence quality
  (iterations, final error, runtime) for pagerank / hits / rwr
  across gpu_baseline and gpu modes.
  Writes ``convergence_benchmark.csv`` and three plots.

* :mod:`src.benchmarking.metrics` — reserved for benchmarking-specific
  helper metrics (currently empty stub).

* :mod:`src.benchmarking.scaling` — legacy stub (empty).
"""

from src.benchmarking.benchmark import (
    run_benchmark,
    BenchmarkTimer,
    _ensure_cuda_context,
)
from src.benchmarking.runtime_benchmark import (
    RuntimeBenchmarker,
    BenchmarkDataset,
    TimingRecord,
)
from src.benchmarking.scalability_benchmark import (
    ScalabilityBenchmarker,
    ScalabilityRecord,
)
from src.benchmarking.memory_benchmark import (
    MemoryBenchmarker,
    MemoryRecord,
)
from src.benchmarking.architecture_benchmark import (
    ArchitectureBenchmarker,
    ArchRecord,
    ArchDataset,
)
from src.benchmarking.convergence_benchmark import (
    ConvergenceBenchmarker,
    ConvergenceRecord,
)

__all__ = [
    # legacy
    "run_benchmark",
    "BenchmarkTimer",
    "_ensure_cuda_context",
    # runtime
    "RuntimeBenchmarker",
    "BenchmarkDataset",
    "TimingRecord",
    # scalability
    "ScalabilityBenchmarker",
    "ScalabilityRecord",
    # memory
    "MemoryBenchmarker",
    "MemoryRecord",
    # architecture
    "ArchitectureBenchmarker",
    "ArchRecord",
    "ArchDataset",
    # convergence
    "ConvergenceBenchmarker",
    "ConvergenceRecord",
]
