# CLAUDE.md — FYP: GPU-Accelerated Biological Network Analysis Framework

## Project overview

A GPU-accelerated framework for multi-scale biological network analysis.
Supports GRN (Gene Regulatory Networks), PPI (Protein–Protein
Interaction), and miRNA–target interaction networks.

Three main layers:
- `src/`          — core computational engine (shared by webapp + experiments)
- `webapp/`       — user-facing web application (FastAPI + React/Vite)
- `experiments/`  — benchmarking and validation (not user-facing)

## Critical rule: three separate algorithm locations

```
src/algorithms/gpu/cuda_optimized/          ← used by webapp and GPU benchmarking
  Custom CUDA / PyCUDA implementations. All three modes
  (_cpu_single, _cpu_multi, _gpu). These are the primary
  contribution of the project.

src/algorithms/gpu/basic/                   ← used exclusively for benchmarking
  GPU baseline implementations.  Hard-fail policy (no silent fallbacks):
    pagerank, bfs, hits, louvain, rwr  → cuGraph (RAPIDS) only
      ImportError at import time if cuGraph/cuDF absent.
      result["mode"] = "gpu_baseline_cugraph"
    mcl  → CuPy only (cuGraph has no MCL equivalent)
      ImportError at import time if CuPy absent.
      result["mode"] = "gpu_baseline_cupy"
    Never falls back to CPU.  Never falls back from cuGraph to CuPy.
  Naming:  <algo>_gpu_baseline(graph_csr, params)
  Mode:    "gpu_baseline_cugraph" (cuGraph algos) | "gpu_baseline_cupy" (MCL)
  Backend key: result["result"]["backend"] = "cugraph" | "cupy"
  Purpose: comparison baseline for the cuda_optimized implementations.
  Never imported by the web application.

src/algorithms/cpu/single_threaded/         ← used by CPU benchmarking only
  Single-threaded CPU implementations (cpu_single only). No _gpu function.
  Never imported by webapp code.

src/algorithms/cpu/multi_threaded/          ← used by CPU benchmarking only
  Multi-threaded CPU implementations (cpu_multi only). No _gpu function.
  Never imported by webapp code.
  Backend: SuiteSparse:GraphBLAS (python-graphblas) for pagerank, bfs, rwr,
    hits, mcl — OpenMP threads inside SuiteSparse C kernels.
  Exception — louvain: GraphBLAS has no Louvain primitive.  cpu_multi
    Louvain uses NetworKit's PLM / PLMR (Parallel Louvain Method) instead,
    run in an isolated subprocess launched by absolute file path
    (`multi_threaded/_networkit_worker.py`).  Isolation is REQUIRED, not
    optional: importing anything under `src.algorithms` forces a live CUDA
    context (`src/benchmarking/benchmark.py` calls `cupy.zeros(1)` at
    import time), and NetworKit's bundled OpenMP/TBB runtime cannot share
    a process with an already-initialized CUDA context — confirmed to
    abort with `malloc(): mismatching next->prev_size` (glibc heap
    corruption) when run in-process.  `_networkit_worker.py` MUST NOT
    import anything from `src.*`.  If NetworKit is not installed,
    `louvain_cpu_multi` falls back to `louvain_cpu_single` with a warning.
```

Convenience imports from `src/algorithms/cpu/` (re-exports both sub-packages):
    from src.algorithms.cpu import pagerank_cpu_single, pagerank_cpu_multi
    ...and so on for all six algorithms.

Shared helpers used by both single_threaded and multi_threaded variants:
    src/algorithms/common/helpers.py

- **NEVER** import from `src/algorithms/cpu/` in webapp code.
- **NEVER** add `_gpu` functions to any `src/algorithms/cpu/` file.

---

## Supported network types

- **GRN** (Gene Regulatory Networks) — directed, TF → gene edges
- **PPI** (Protein–Protein Interaction networks) — undirected, protein–protein edges
- **miRNA** (miRNA–target interaction networks) — directed bipartite, miRNA → gene edges

## Network-type-aware algorithm behaviour

Algorithms must adapt to the network type passed in `params["network_type"]`:

### `network_type = "grn"`
- All algorithms: graph is **DIRECTED**, preserve edge direction
- PageRank: dangling nodes are pure target genes; redistribute mass only to
  nodes with `out-degree > 0`
- PageRank returns: `top_regulators` (out-degree > 0) and `top_targets`
  (out-degree == 0) separately
- HITS: do **NOT** symmetrize; hubs = TFs, authorities = target genes
- RWR: directed transition matrix normalised by out-degree;
  `seed_nodes` are TF indices
- Louvain: symmetrize internally (`A + A^T`) before running;
  add `"note"` field to result explaining this
- MCL: symmetrize internally; add `"note"` field to result
- BFS: follow out-edges only; `cascade_by_depth` groups genes by regulatory
  distance from source TF

### `network_type = "ppi"`
- All algorithms: graph is **UNDIRECTED** (or treat as undirected)
- PageRank: standard uniform dangling redistribution;
  return `top_nodes` (no regulator/target split)
- HITS: symmetrize `A` before running; hub = authority = connectivity
- RWR: symmetric transition matrix; `seed_nodes` are disease proteins
- Louvain: use graph as-is (already undirected); no symmetrization note
- MCL: use graph as-is; no symmetrization note
- BFS: follow all edges; `cascade_by_depth` shows interaction distance

### `network_type = "mirna"`
- Graph is **DIRECTED BIPARTITE**: miRNA nodes → gene nodes
- PageRank: dangling redistribution to miRNA nodes only;
  return `top_mirnas` and `top_target_genes` separately
- HITS: do **NOT** symmetrize; hubs = miRNAs, authorities = target genes
- RWR: directed; `seed_nodes` are miRNA indices
- Louvain: symmetrize internally; note explains bipartite symmetrization
- MCL: symmetrize internally
- BFS: follow miRNA → gene direction; `max_depth` typically 2

---

## Result dict standard (ALL algorithms, ALL modes must return this)

```python
{
  "algorithm":      str,        # "pagerank", "bfs", ...
  "mode":           str,        # "cpu_single", "cpu_multi", or "gpu"
  "network_type":   str,        # "grn", "ppi", or "mirna"
  "execution_time": float,
  "num_nodes":      int,
  "num_edges":      int,
  "result":         dict        # algorithm-specific output — see specs below
}
```

## Per-algorithm `result["result"]` specs

**PageRank (grn)**
```python
{"scores": list[float], "top_regulators": list[int],
 "top_targets": list[int], "iterations": int, "converged": bool}
```

**PageRank (ppi)**
```python
{"scores": list[float], "top_nodes": list[int],
 "iterations": int, "converged": bool}
```

**PageRank (mirna)**
```python
{"scores": list[float], "top_mirnas": list[int],
 "top_target_genes": list[int], "iterations": int, "converged": bool}
```

**BFS (all network types)**
```python
{"distances": list[int], "visited_order": list[int],
 "num_reachable": int, "cascade_by_depth": dict}
```

**Louvain (all network types)**
```python
{"community_assignments": list[int], "num_communities": int,
 "modularity": float, "top_communities": list[dict],
 "note": str}
```

**RWR (all network types)**
```python
{"scores": list[float], "top_nodes": list[int],
 "top_seeds": list[int], "iterations": int}
```

**HITS (grn / mirna)**
```python
{"hub_scores": list[float], "authority_scores": list[float],
 "top_hubs": list[int], "top_authorities": list[int],
 "hub_authority_overlap": list[int]}
```

**HITS (ppi)**
```python
{"hub_scores": list[float], "authority_scores": list[float],
 "top_nodes": list[int]}
```

**MCL (all network types)**
```python
{"cluster_assignments": list[int], "num_clusters": int,
 "iterations": int, "converged": bool, "note": str}
```

---

## Param schemas (defaults)

```python
pagerank: {"damping": 0.85, "max_iter": 100, "tolerance": 1e-6,
           "network_type": "grn"}
bfs:      {"source": 0, "max_depth": 5, "network_type": "grn"}
louvain:  {"min_delta_q": 1e-4, "max_levels": 10, "resolution": 1.0,
           "network_type": "grn"}
rwr:      {"restart_prob": 0.3, "max_iter": 100, "tolerance": 1e-6,
           "seed_nodes": [], "network_type": "grn"}
hits:     {"max_iter": 100, "tolerance": 1e-6, "network_type": "grn"}
mcl:      {"expansion": 2, "inflation": 2.0, "prune_threshold": 0.001,
           "max_iter": 100, "convergence_tol": 1e-4, "network_type": "grn"}
```

---

## GPU optimization layer (`src/optimization/gpu_config.py`)

Expanded from a hardware-heuristic lookup table into a graph-,
algorithm-, and runtime-aware optimisation system.  Backward
compatible — all six algorithm files consume the merged params dict
via `setdefault` / `.get()`; new keys are additive and unknown keys
are silently ignored.

### Components

`GraphProfiler`
  - `fingerprint(csr)`: fast graph hash (shape + nnz + indptr head/tail
    sample) for cache keying.  Returns a 12-char hex string.
  - `profile(csr)`: CPU-only structural metrics.  Returns
    `n`, `m`, `density`, `avg_degree`, `max_degree`, `min_degree`,
    `std_degree`, `degree_skew`, `hub_fraction`, `isolated_frac`,
    `is_bipartite` (BFS-sampled), `is_symmetric` (`(A−Aᵀ).nnz == 0`),
    `nnz_per_mb`, `vram_estimate_mb`, `sparsity_class`, `degree_class`,
    `format_hint`.
  - `sparsity_class`: `"ultra_sparse"` (density < 1e-5) | `"sparse"`
    (< 1e-3) | `"moderate"` (< 0.1) | `"dense"` (>= 0.1).
  - `degree_class`: `"uniform"` (|skew| < 1) | `"skewed"` (< 3) |
    `"power_law"` (>= 3).
  - `format_hint`: `"csr"` | `"hyb"` | `"sell_c"` | `"coo"`.

`MemoryEstimator`
  - `estimate(algo, profile, gpu_cfg)`: returns `base_mb`,
    `algorithm_mb`, `total_mb`, `available_mb`, `pressure`
    (`"low"` / `"medium"` / `"high"` / `"critical"`),
    `needs_chunking`, `recommended_chunk_size`, `precision_downgrade`.
  - Algorithm-specific multipliers:
    - `pagerank`: 1.5, `bfs`: 1.3, `rwr`: 1.6, `louvain`: 3.0,
      `hits`: 2.5 (A + Aᵀ + 4 score vectors),
      `mcl`: 4.0 baseline, scaled up to `max(4.0, avg_degree × 2)` to
      cover SpGEMM fill-in (capped at 20×).
  - Live `cuda.mem_get_info()` query — chunking decisions reflect the
    current free VRAM rather than a stale snapshot.  Memory estimates
    are NOT cached.

`AlgorithmStrategySelector`
  - `select(algo, profile, gpu_cfg, memory_est)` dispatches to
    `_pagerank`, `_bfs`, `_hits`, `_louvain`, `_rwr`, `_mcl`.  All
    returned keys are already understood by the corresponding
    algorithm file.
  - Decision matrix:
    | Algorithm | Key decisions                                          |
    |-----------|--------------------------------------------------------|
    | pagerank  | `use_pull` (power_law + hub_fraction > 0.05),          |
    |           | `pull_threshold` (3 × avg_degree),                     |
    |           | `ellpack_fraction` (0.02 / 0.05 / 0.10)                |
    | bfs       | `traversal_mode` (push_only / push_pull),              |
    |           | `frontier_threshold` (0.25 / 0.50),                    |
    |           | `use_bitmap_frontier`                                  |
    | hits      | `use_smem_hash`, `reorder_nodes`, `fuse_spmv_norm`     |
    | louvain   | `use_smem_hash`, `use_community_freezing`,             |
    |           | `freeze_threshold` (2 / 4),                            |
    |           | `early_stop_fraction` (0.005 / 0.01),                  |
    |           | `coarsening_backend` ("cupy" / "hybrid_gpu")           |
    | rwr       | `spmv_mode` (csr_fused / edge_parallel),               |
    |           | `reorder_nodes`, `use_zero_copy`                       |
    | mcl       | `spgemm_method` (hash / inner_product),                |
    |           | `prune_threshold` (1e-5 → 1e-2 by sparsity),           |
    |           | `top_k_per_column` (10–100 by density),                |
    |           | `use_bitonic_topk` (max_degree > 256)                  |

`RuntimeProfiler`
  - `record(algo, fp, time, config, meta)`: appends to
    `_history[(algo, fp)]`, capped at 10 entries.
  - `get_recommendation(algo, fp)`: requires ≥ 3 runs.  Relaxes
    `tolerance × 2` if all recent runs converged easily (tol < 1e-4);
    bumps `max_iter × 1.5` if none converged; emits a `_warning` when
    execution time grows by ≥ 1.5× across the window (possible memory
    fragmentation).
  - `get_history(algo=None)`, `clear(algo=None)` for benchmarking.

`apply_config()` (orchestrator)
  - Signature unchanged: `apply_config(algo, csr, params,
    override=False, enable_profiling=False)`.
  - Merge order (highest wins, default):
    `user params > runtime_rec > strategy > hw_recommended`.
    When `override=True`, hw_recommended beats user params on
    overlapping keys (legacy semantics).
  - Adds metadata keys to every result (never overridden):
    `_graph_fingerprint`, `_graph_profile`, `_memory_estimate`,
    `_strategy_selected`, `_hardware_config`.  Optional
    `_runtime_note` when feedback recommends a change.
  - `enable_profiling=True` attaches a full `_profile_report`.
  - Legacy nnz-based chunking heuristics preserved as a final pass for
    backward compatibility.

`generate_profile_report(algo, csr, params, execution_result=None)`
  - Returns a structured dict with sections `hardware`, `graph`,
    `memory`, `strategy`, `runtime_history`, `recommendations`
    (list of human-readable strings), plus `execution` when an
    execution result is supplied.

### Caching

- `_HARDWARE_CONFIG` (= `_CACHED_CONFIG`): detected once per process.
- `_GRAPH_PROFILE_CACHE`: `{fingerprint: profile}`.
- `_STRATEGY_CACHE`: `{(algo, fingerprint): strategy}`.
- `RuntimeProfiler._history`: `{(algo, fingerprint): [records...]}`,
  last 10 per pair.
- Memory estimates are intentionally NOT cached (live free-VRAM query).

### Hardware detection

`get_gpu_config()` returns the same legacy keys plus:
  `cc_major`, `cc_minor`, `shared_mem_per_block_bytes`,
  `l2_cache_size_bytes`, `architecture` (`"Turing"` / `"Ampere"` /
  `"Ada"` / `"Hopper"` / etc.).
Tier classification (`"high"` / `"mid_large"` / `"mid_small"` /
`"cpu_only"`) and CC adjustments unchanged.

---

## Milestone 3.1 — Memory and VRAM Optimisation

The framework can plan and execute on graphs that exceed available
GPU VRAM via three cooperating modules.  All planning is automatic;
algorithm files inherit the plan through the merged params dict that
`apply_config()` already returns.

### Components

`src/optimization/memory_manager.py` — `MemoryManager`
  - `select_execution_mode(algo, csr, params)` returns a plan dict
    with `execution_mode`, `estimated_memory_mb`, `available_vram_mb`,
    `memory_pressure`, `use_chunking`, `chunk_size`,
    `use_unified_memory`, `use_zero_copy`, `use_partitioning`,
    `partition_strategy`, and `mode_reason`.
  - Five execution modes:
    - `normal_gpu` — estimate < 70 % of free VRAM.
    - `chunked_gpu` — algorithm exposes a `_*_gpu_chunked` path.
      Currently supported by `pagerank`, `rwr`, `louvain`.
    - `unified_memory` — CUDA managed memory available (Pascal+).
    - `zero_copy` — DEVICEMAP-mapped pagelocked host memory.
    - `partitioned_gpu` — last resort; caller consumes
      `GraphPartitioner` output (algorithm-side wiring is the next
      milestone for `hits`, `mcl`, `bfs`).
  - Decision order (first match wins): normal_gpu → chunked_gpu →
    unified_memory → zero_copy → partitioned_gpu.
  - `get_available_vram()` — live `cuda.mem_get_info()` query;
    returns `{"free_mb": int, "total_mb": int}`.
  - `compute_chunk_size(csr, available_mb, algo, safety_fraction=0.5)`
    — `budget_bytes // (4 + 8 * avg_degree)`, clamped to `[1, n]`.

`src/optimization/unified_memory.py` — managed / pinned wrappers
  - `ManagedArray(host_view, device_ptr, mode, _owner)` — uniform
    interface across `unified_memory`, `zero_copy`, `pinned`,
    `device`, `host_only`.
  - `allocate_managed_array` → `cuda.managed_empty` (falls back to
    zero-copy → pinned → device → host_only on each failure).
  - `allocate_zero_copy_array` → `cuda.pagelocked_empty` with
    `host_alloc_flags.PORTABLE | DEVICEMAP`.
  - `allocate_pinned_array` → page-locked without device mapping.
  - `allocate_device_array` → `gpuarray.empty`, falling through to
    `host_only` when no CUDA context is active.
  - `prefetch_to_gpu` / `prefetch_to_cpu` — `cuMemPrefetchAsync` on
    managed allocations; no-op for other modes.
  - `capabilities()` returns `{pycuda_available, managed_memory,
    pagelocked_memory, mem_prefetch_async}` for the current process.

`src/optimization/graph_partitioning.py` — `GraphPartitioner`
  - `partition_by_edges(csr, target_edges)` — greedy edge-balanced.
  - `partition_by_degree(csr, target_mb)` — memory-budget balanced.
  - `partition_by_nodes(csr, target_nodes)` — legacy row chunking.
  - `partition(csr, strategy, ...)` — dispatcher with safe defaults.
  - `identify_hub_nodes(csr, hub_threshold=None)` — nodes with degree
    `> max(WARP_SIZE, 3 × avg_degree)` (default).
  - `build_boundary_metadata(csr, partitions)` — counts out-edges
    crossing into every other partition; mutates partitions in place.
  - `Partition` dataclass: `partition_id, start_node, end_node,
    num_nodes, num_edges, estimated_memory_mb, hub_nodes,
    boundary_edges_out`.

### Integration with `apply_config()`

`apply_config()` now calls `MemoryManager.select_execution_mode()`
after building the strategy dict.  The plan keys are merged BEFORE
user params so explicit user overrides still win.  Metadata block
`_memory_plan` is added alongside `_graph_profile`,
`_memory_estimate`, etc.  Backward compatible — all existing keys
preserved.

### Per-algorithm chunked execution status

| Algorithm  | Chunked path  | Notes                                    |
|------------|---------------|------------------------------------------|
| `pagerank` | yes           | `_pagerank_gpu_chunked` + `_ChunkBuffer` |
| `rwr`      | yes           | `_rwr_gpu_chunked` + `_ChunkBuffer`      |
| `louvain`  | yes           | `_louvain_level_chunked` (Phase 1 only)  |
| `hits`     | not yet       | Planner recommends `unified_memory`      |
| `mcl`      | not yet       | Planner recommends `partitioned_gpu`     |
| `bfs`      | not yet       | Planner recommends `unified_memory`      |

For algorithms without a chunked path the planner falls through to
unified memory / zero-copy / partitioned — never to CPU.

### Double-buffered chunked pipeline (existing, shared design)

The three chunk-capable algorithms share the same pattern:
  - `_ChunkBuffer` pair of pre-allocated device arrays (`row_ptr`,
    `col_idx`, `values`, `node_ids`).
  - `stream_transfer` pre-loads chunk i+1 while `stream_compute`
    runs the kernel on chunk i.
  - `cuda.Event.wait_for_event` orders streams without a host sync.

### Tests

  - `tests/test_memory_manager.py` (17 tests) — planner decision matrix,
    edge cases.
  - `tests/test_graph_partitioning.py` (17 tests) — three partition
    strategies, hub detection, boundary metadata.
  - `tests/test_memory_aware_execution.py` (23 tests) — `apply_config`
    integration; verifies no algorithm gets silently downgraded to CPU
    under critical pressure; non-chunk-capable algorithms never receive
    `chunked_gpu`.

### Completion criteria

This milestone is complete when:
  1. Planning succeeds for graphs ≥ VRAM size without crashing.
  2. `execution_mode` is selected automatically and surfaces in
     `apply_config()` output.
  3. Result dicts can include a `memory:{}` block populated from
     `_memory_plan` (algorithm-side metadata wiring is per-algorithm
     opt-in for the next milestone).
  4. Chunking/partitioning correctness is validated against CPU /
     normal-GPU baselines (the three chunk-capable algorithms are
     already verified by their own per-algorithm tests).

### Rules

- `apply_config()` is called by `algorithm_runner.py` before every GPU
  algorithm run.
- Algorithm files themselves never call `gpu_config` directly.
- GPU config is cached after first call — do not re-detect on every run.
- Target hardware: NVIDIA RTX 20-series (compute capability 7.5, 6–8 GB
  VRAM) — specifically RTX 2060 with 6 GB VRAM for testing.
- If CUDA unavailable: fall back to `cpu_single` silently, log warning.

---

## BFS GPU implementation (optimised)

Five-improvement version addressing load imbalance, atomic contention,
memory inefficiency, kernel overhead, and static traversal strategy.

Kernels (all compiled together in one `SourceModule`):
  - `bfs_frontier_tiered` — push kernel with three-tier scheduling
      < 32 neighbours   → 1 thread per node
      32–255 neighbours → 1 warp per node (stride-32 loop)
      ≥ 256 neighbours  → 1 block per node (stride-256 loop)
  - `bfs_pull` — pull kernel for large frontiers, bitmap-based
  - `worklist_to_bitmap`, `bitmap_to_worklist` — direction-switch helpers
  - `bfs_persistent` (optional, cooperative-groups) — level loop inside
      kernel; requires sm ≥ 7.0.  Currently gated behind
      `USE_PERSISTENT_KERNEL = False` until cooperative launches are
      verified on the deployment box.

Key implementation details:
  - Visited array: `uint32` bitmap (32× memory reduction)
  - Atomic pressure: block-local shared-memory buffer + warp-level
    deduplication via `__match_any_sync` (one global `atomicAdd` per
    block per level instead of one per discovered neighbour)
  - Push/pull threshold: `frontier_size > n / (4 · avg_degree)`
  - Compilation target: `-arch=sm_75` (RTX 20-series Turing)
  - Kernel cache: compiled once, reused across calls
  - Network-type handling: PPI symmetrises the adjacency before running
    (transpose == original, so only one CSR is allocated); GRN / mirna
    keep edge direction and allocate the CSR-transpose separately.
  - Fallback chain: optimised GPU → `cpu_single` (logs warning).

Result includes `traversal_modes` list showing the push/pull decision
per level (useful for benchmarking analysis).

Do NOT revert to CuPy SpMV or single-tier kernel.

---

## HITS GPU implementation (optimised)

Custom PyCUDA kernels — no CuPy dependency.  Four-tier degree
classification with block-level partial norms; degree-adaptive
sub-warp SpMV for the warp tier (mirna only, see below); GPU-only
convergence reduction (single 8-byte D2H per iteration).

Key optimisations over the initial implementation:
  1. **Block-level partial norms** (Opt 1): every SpMV kernel writes
     ONE float per block (not one per node) into a shared
     `d_pnorm_blocks` array; `reduce_blocks_to_scalar` reads that
     (~4.7× smaller) array instead of an n-length one.
  2. **GPU-only convergence** (Opt 2): `compute_convergence_partial`
     (FP64 per-block partials) → `reduce_f64_to_scalar` (GPU sqrt(Σ))
     → a single 8-byte D2H read per iteration (`d_conv_scalar.get()`).
  3. **Four-tier degree classification** (Opt 3):
       Warp  (deg < `MED_THRESH`=256)    → `spmv_warp_centric`
       High  (256 ≤ deg < `SUPER_THRESH`=4096) → `spmv_high_block`
       Super (deg ≥ 4096)                → `spmv_super_block`
     (256 = BLOCK_SIZE; 4096 = 16×BLOCK_SIZE, justifying the tier
     boundaries — see in-source comments.)
  4. **Warp-centric warp tier** (Opt 4): `spmv_warp_centric` packs
     `NODES_PER_BLOCK = 8` nodes per 256-thread block, one warp
     (32 lanes) per node — full occupancy vs. the old medium-degree
     kernel's 12 % thread utilisation.
  5. **GPU buffer pool** (Opt 5): `_BUFFER_POOL` reuses `gpuarray`
     allocations across benchmark runs by `(shape, dtype)` key —
     eliminates repeated `cuMemAlloc`/`cuMemFree`.
  6. **Node-reordering validation** (Opt 6): `hits_gpu` always times
     the reorder step (`reorder_cost_ms` in the result); auto-disabled
     for `n < REORDER_MIN_N = 500`.  `apply_config`'s HITS strategy
     sets `reorder_nodes` per graph family
     (`gp["degree_class"] == "power_law"` — see `gpu_config.py`), so
     grn/ppi honour whatever the strategy selector decides.
  7. **Resident prepared-graph cache** (`_HITS_GRAPH_CACHE`): the CPU-
     side symmetrize / transpose / degree-reorder work is a pure
     function of the graph; caching the prepared host CSR arrays
     across calls moves that cost onto the warmup run instead of the
     timed run.

Kernels (compiled once, cached in `_kernel_cache["hits"]`):
  - `spmv_warp`   : `spmv_warp_centric`       — warp-per-node, deg < 256, grn/ppi
  - `spmv_high`   : `spmv_high_block`         — full-block, 256 ≤ deg < 4096
  - `spmv_super`  : `spmv_super_block`        — 1024-thread block, deg ≥ 4096
  - `reduce_blocks`: `reduce_blocks_to_scalar` — block-level norm → scalar
  - `norm_div`    : `normalize_inplace`
  - `conv_partial`: `compute_convergence_partial` — FP64 per-block partials
  - `reduce_f64`  : `reduce_f64_to_scalar`    — GPU FP64 reduce → 8-byte D2H
  - `spmv_warp_tpr2/4/8/16` : `spmv_warp_centric_tpr{2,4,8,16}` — mirna-only
    degree-adaptive warp tier (see below)
  - Deprecated (compiled, not called — kept for A/B reference):
    `spmv_degree_aware`, `compute_partial_norm_sq`, `normalize_vector`,
    `compute_convergence_delta`, `spmv_edge_parallel_low_degree`,
    `spmv_with_norm_sq`, `partial_reduce_to_scalar`

### mirna-only degree-adaptive warp-tier SpMV (throughput fix)

**Problem observed**: same root cause as the RWR mirna fix (see the
RWR section above) — on the RTX 2060 scalability benchmark (avg
degree ≈ 6), `mode=gpu` HITS was slower than both `cpu_multi`
(GraphBLAS) and `gpu_baseline` (cuGraph) across BA/ER/WS.  Unlike RWR,
the internal `apply_config()` call was **not** removed for HITS: its
HITS strategy sets `reorder_nodes` per graph family (power-law → True,
uniform → False) and HITS actually consumes that key, so skipping it
would change behaviour on ER/WS.  HITS is also compute-dominated
(up to 500 iterations × 2 SpMVs), so the fixed `apply_config` overhead
matters far less here than it did for RWR.

The actual bottleneck: `spmv_warp_centric` gives every node a full
32-lane warp regardless of degree.  At avg degree ≈ 6 essentially all
nodes land in the warp tier, so ~26 of 32 lanes sit idle on both the
authority and hub SpMV every iteration.

Fix, gated to `network_type == "mirna"` only (grn/ppi unaffected):
new kernels `spmv_warp_centric_tpr2` / `_tpr4` / `_tpr8` / `_tpr16`
(macro-generated via `HITS_SPMV_TPR_KERNEL`) where `tpr`
threads-per-row cooperate on one node, packing `BLOCK_SIZE / tpr`
nodes per block instead of `NODES_PER_BLOCK = 8`.  Same block-level
partial-norm contract as `spmv_warp_centric` (one float per block,
written to `partial_norm_blocks[block_offset + blockIdx.x]`).
Boundary-safe: sub-groups whose node is out of range stay in the
shuffle with `partial = 0` (keeps the `0xffffffff` mask valid); `tpr`
divides 32 and groups are lane-contiguous, so the sub-group
`__shfl_down_sync` reduction never crosses a group boundary.

`_choose_tpr(avg_degree)` picks `tpr ∈ {2,4,8,16,32}` as the smallest
power of 2 `>= avg_degree` (capped at 32).  `tpr == 32` means
`warp_rows_per_block == BLOCK_SIZE/32 == 8 == NODES_PER_BLOCK` — the
original kernel, grid, and `_pnorm_layout` — so high-average-degree
mirna graphs and non-mirna types are byte-identical to before.
`_pnorm_layout`'s `num_warp_blocks` and the warp-tier grids
(`warp_A_grid` / `warp_AT_grid`) are computed from
`warp_rows_per_block`, so block counts and pnorm offsets stay
consistent whichever kernel is selected.

**Verification note**: since mirna and grn build the identical
adjacency (only `ppi` symmetrises — see Network-type behaviour below),
mirna GPU `top_hubs`/`top_authorities` on a given graph must match the
grn GPU run on the same graph.  Use this as the correctness check
before trusting new benchmark numbers.

**Confirmed on RTX 2060** (`scalability` benchmark, `--network-type
mirna`, edge targets 1M–50M, barabasi_albert / erdos_renyi /
watts_strogatz): GPU Optimised now runs below both CPU GraphBLAS and
GPU Baseline at every tested size across all three graph families.
grn / ppi results are unaffected (unchanged code path).

Per-iteration sequence:
  Authority (a_new = Aᵀ h):
    1. warp tier (`spmv_warp` — `spmv_warp_centric` or mirna-adaptive
       `spmv_warp_centric_tprN`), high tier (`spmv_high_block`), super
       tier (`spmv_super_block`) — only tiers with ≥1 node are launched
    2. `reduce_blocks_to_scalar`  — GPU sqrt(Σ block-partial norms) → scalar
    3. `normalize_inplace`       — divide a_new by GPU scalar
  Hub (h_new = A a_new): same three steps for A
  Convergence:
    4. `compute_convergence_partial` — FP64 partial Σ((Δh)² + (Δa)²)
    5. `reduce_f64_to_scalar`       — GPU sqrt(Σ) → single FP64 scalar
    6. `stream_compute.synchronize()` — the single sync per iteration
       `delta = d_conv_scalar.get()[0]` (8 bytes D2H)

Precision: FP32 for SpMV arithmetic and L2 norms.  FP64 for
  convergence delta only — avoids false early termination near
  `tolerance = 1e-6` without cancellation error.  Convergence threshold
  is scale-invariant: `conv_threshold = tolerance * sqrt(n)` (per-node
  RMS-change criterion, matching cuGraph's n-scaled definition in
  spirit).

Both `A` and `Aᵀ` stored in CSR on GPU throughout iteration.
Pointer swap (`cur_h, nxt_h = nxt_h, cur_h`) avoids data copies.
CUDA streams: `stream_compute` for kernels, `stream_transfer` for
  async H2D transfers during setup.
Context handling: `retain_primary_context().push()` unconditionally;
  coexists with CuPy (avoids `cuModuleLoadDataEx: invalid device context`).
Network-type behaviour:
  - grn / mirna: directed `A` (one shared code branch); returns
    `top_hubs` + `top_authorities` + `hub_authority_overlap`
  - ppi: symmetrised `A + Aᵀ` (binarised); returns `top_nodes` only
Does NOT silently fall back to CPU — raises `RuntimeError` /
  `cuda.LogicError` / `MemoryError`.
Target: `-arch=sm_75` (RTX 20-series Turing, adaptive via
  `_detect_arch_flag`-equivalent probe), `BLOCK_SIZE = 256`,
  `NODES_PER_BLOCK = 8`, `SUPER_BLOCK_SIZE = 1024`,
  `MED_THRESH = 256`, `SUPER_THRESH = 4096`.

---

## Louvain GPU implementation (optimised)

Custom PyCUDA kernels.  CuPy is **optional** — when present it powers
the fully-GPU Phase-2 sort+reduce path; without it the implementation
falls back to a GPU-gather + numpy-sort + GPU-reduce hybrid that still
keeps the large edge arrays on the device.

Key optimisations over the initial implementation:
  1. **GPU-side Phase 2 coarsening**:
     - `_coarsen_gpu_cupy()` (when CuPy available) — `cp.argsort` +
       `cp.add.reduceat` directly on the device.  Only the reduced
       triples cross PCIe.
     - `_coarsen_gpu_fallback()` (no CuPy) — compute compound 64-bit
       sort keys on GPU; only the **keys** are D2H'd, host `np.argsort`
       computes the permutation, the index array is H2D'd back, then
       `gather_by_index` + `segmented_weight_reduce` finish on the GPU.
       ~50 % less PCIe traffic than the original CPU-only path.
  2. **Shared-memory hash table in `compute_proposed_moves`**:
     `SMEM_HASH_SIZE = 64` buckets cache `(community → accumulated
     weight)` during the neighbour scan.  Pass 2 is now a single hash
     walk over unique neighbour communities instead of a per-edge gain
     evaluation followed by a block-wide reduction.  Eliminates
     redundant δQ recomputation for hub nodes with many same-community
     neighbours.  Saturated probe chains drop entries (documented
     non-determinism).
  3. **Community freezing** (`update_freeze_status`):
     A node unchanged for `freeze_threshold = 3` consecutive passes is
     marked frozen.  Frozen nodes are skipped at kernel entry in
     `compute_proposed_moves` (keep their current community).  Any
     move resets the counter.
  4. **Early termination via `move_counter`**:
     `apply_moves` was changed from `improvement_flag` (boolean) to
     `move_counter` (atomic count of nodes that moved).  Phase 1 stops
     when `move_counter < early_stop_fraction * n` (default
     `0.01`, i.e. < 1 % of nodes moved).
  5. **Adaptive compilation** via `_detect_arch_flag()`:
     Returns `(-arch=sm_XY, (cc_major, cc_minor))` from
     `cuda.Device(0).compute_capability()`.  `-use_fast_math` is added
     on Ampere+ (cc ≥ 8); `-DDISABLE_COOPERATIVE_GROUPS` on pre-Volta
     (cc < 7).  Falls back to `sm_75` on probe failure.  This pattern
     is now applied to ALL six GPU algorithm files
     (`pagerank.py`, `bfs.py`, `hits.py`, `louvain.py`, `mcl.py`,
     `rwr.py`).
  6. **Chunked Phase 1** (`_louvain_level_chunked`):
     Fully implemented for graphs that exceed the VRAM safety budget.
     `community / comm_degree_sum / proposed / freeze / frozen` stay
     full-size on the device; CSR rows stream in by chunks.
     `compute_proposed_moves` runs per chunk with `frozen=NULL` (the
     chunked path does not currently use the freeze optimisation —
     prioritises correctness over micro-optimisation in the low-VRAM
     fallback).  `apply_moves` + `update_freeze_status` run once per
     pass over the full community array.  Auto-triggered when
     `est_bytes > VRAM_SAFETY * free_bytes` or `use_chunking=True`.

Preprocessing (CPU, before any GPU work) — unchanged:
  `_symmetrize`, `_remove_self_loops`, `_handle_isolated_nodes`,
  `_normalize_weights`.

Kernels (compiled once, cached in `_kernel_cache["louvain"]`):
  - `compute_proposed_moves` — Phase 1 core.  Three-tier degree-aware
    scheduling.  Pass 1 accumulates `k_self`; Pass 2 builds the SMEM
    hash table; thread 0 walks the hash to pick the winning community.
    Accepts an optional `frozen` mask pointer (NULL allowed) for the
    chunked path.
  - `apply_moves` — Phase 1 batch update.  Atomic `community[]` and
    `comm_degree_sum[]` rebalance; increments `move_counter` per
    moving node.
  - `update_freeze_status` — per-node freeze counter / mask update.
  - `count_community_edges` — Phase 2 step 1, edge→community triple
    emission (unchanged).
  - `gather_by_index` — GPU-side reorder of `(src, dst, wt)` arrays
    by an external permutation (Phase 2 fallback).
  - `segmented_weight_reduce` — segment-head detection + forward-scan
    weight accumulation; atomic counter for output offsets.  Drops
    self-loops only when `drop_self_loops=1` (Louvain keeps them).
  - `compute_modularity_partial` — final Q per level (unchanged).

Phase 2 sort+reduce dispatch (priority order):
  1. `CUPY_SORT_AVAILABLE=True` → `_coarsen_gpu_cupy()`
  2. Fallback → `_coarsen_gpu_fallback()` (GPU gather + CPU argsort +
     GPU segmented reduce)

CUDA streams:
  - `stream_compute`  — kernel execution
  - `stream_transfer` — async H2D for per-level CSR uploads

Context handling: same pattern as HITS / PageRank / MCL / RWR —
  `retain_primary_context()` and unconditional push.  Coexists with
  CuPy in the same process.

Hierarchy bookkeeping (unchanged): `global_community[i]` tracks the
  level-current community of original node `i`; renumbered to `0..K-1`
  via `np.unique` between levels.

Non-determinism: parallel updates + SMEM-hash collisions + frozen
  communities + early termination together can produce a different
  partition than serial Louvain.  Documented in
  `result["result"]["note"]`.  Modularity remains non-decreasing in
  practice.  `max_phase1_passes` caps oscillation.

Does NOT silently fall back to CPU — raises `RuntimeError` /
  `cuda.LogicError` / `MemoryError`.
Target: adaptive (RTX 20-series Turing by default), `BLOCK_SIZE = 256`,
  `SMEM_HASH_SIZE = 64`, `freeze_threshold = 3`,
  `early_stop_fraction = 0.01`.

---

## MCL GPU implementation (optimised)

Custom PyCUDA kernels — no CuPy dependency for compute.

Key optimisations over the initial implementation:
  1. **Hash-based SpGEMM** (`spgemm_hash_row`):
     Gustavson-style row-wise sparse accumulator using a SMEM hash
     table (power-of-2 size, linear probing).  Processes only NONZERO
     entries of A's row, scatters their contributions through B's rows
     into the hash table keyed by output column index.  Eliminates the
     zero × zero work of the inner-product approach and matches the
     power-law degree distribution of biological networks.  Falls back
     to `spgemm_row_chunk` for rows whose estimated output would not
     fit in SMEM hash capacity (heavy-row tier).
  2. **GPU-native pruning compaction**:
     `prefix_sum_block` → `prefix_sum_add_offsets` → `compact_csr_values`
     → `rebuild_row_ptr` replace the previous numpy `np.where` /
     `np.cumsum` round-trip.  Only one int (`new_nnz`) plus a small
     per-block-sums array (ceil(nnz / BLOCK_SIZE) ints) crosses the
     PCIe bus per compaction; the big arrays stay on the device.
  3. **Bitonic-style top-k** for wide columns (`topk_bitonic_column`):
     One block per column, launched with 2 × BLOCK_SIZE = 512 threads.
     A shared-memory buffer of size 2 × BLOCK_SIZE holds the running
     top BLOCK_SIZE in its upper half; each chunk of the column is
     loaded into the lower half and a full Batcher bitonic sort
     ascending re-establishes the order in O((log 2K)²) compare-
     exchange stages.  Threshold = `buf[2·BLOCK_SIZE − K]`.  Routed
     for columns with length > `TOP_K_BITONIC_THRESHOLD = 256`;
     narrower columns continue to use the original count-greater
     kernel.
  4. **Multi-stream pipelining**:
     `stream_compute` runs kernels; `stream_transfer` issues async
     H2D copies of per-iteration matrix arrays.  Inter-stream
     synchronisation via `cuda.Event` + `stream.wait_for_event` —
     no host syncs are needed between upload and kernel launch.
  5. **Degree-aware SpGEMM dispatch** (`_classify_rows_by_degree`):
     A row is classified once per iteration from its row-length:
       heavy   (L ≥ block_size = 256) — inner-product fallback;
       medium  (32 ≤ L < block_size)  — hash SpGEMM, medium hash size;
       light   (0 < L < 32)           — hash SpGEMM, smaller hash size.
     Each tier gets a dedicated kernel launch with the per-tier index
     array passed as `row_list` so a single grid spans only the rows
     of that tier.

Kernels (compiled once, cached in `_kernel_cache["mcl"]`):
  - `spgemm_hash`   : `spgemm_hash_row`        (Gustavson, hash table)
  - `spgemm`        : `spgemm_row_chunk`       (inner-product fallback)
  - `thresh_prune`  : `threshold_prune`        (writes keep_flag)
  - `prefix_sum`    : `prefix_sum_block`       (per-block exclusive scan)
  - `scan_offsets`  : `prefix_sum_add_offsets` (CPU-scanned block sums)
  - `compact`       : `compact_csr_values`     (stream compaction)
  - `rebuild_rptr`  : `rebuild_row_ptr`        (post-compaction row_ptr)
  - `topk_prune`    : `topk_column_prune`      (narrow cols, count-greater)
  - `topk_bitonic`  : `topk_bitonic_column`    (wide cols, bitonic merge)
  - `inflate`       : `inflate_values`
  - `col_sum`       : `column_sum_segmented`
  - `col_norm`      : `normalize_columns`
  - `convergence`   : `convergence_frobenius`

Preprocessing (CPU, before any GPU work):
  - `_symmetrize_mcl(graph_csr, network_type)` — network-type aware
    undirected conversion + self-loops on every node (ensures
    irreducibility of the Markov chain).
      grn / mirna : `binarise(A + Aᵀ)` then add identity
      ppi         : graph used as-is then add identity
  - `_to_column_stochastic` — divides each column by its sum; zero-sum
    columns get a 1.0 placed on their diagonal (absorbing state).
  - `_to_ellpack_r` removed; hash SpGEMM does not benefit from
    ELLPACK padding.

Pruning pipeline (fully GPU-side):
  threshold_prune → prefix_sum_block → prefix_sum_add_offsets →
  compact_csr_values → rebuild_row_ptr → top-k dispatch (warp count-
  greater OR bitonic, based on column length vs.
  `TOP_K_BITONIC_THRESHOLD`).  One CSR D2H + one CSC build remain on
  CPU between threshold and top-k, since the CSR→CSC transpose is
  still numpy-side; reducing this further requires a GPU radix sort
  (future work).

Inflation + normalisation: in-place on CSC.data — pattern unchanged,
  only values updated.

Convergence check:
  - Sparsity pattern must match (host array-equals indptr + indices)
  - On match: GPU FP64 Frobenius reduction → CPU `sqrt(sum(partials))`
  - On mismatch (early iterations): returns `inf`, iteration continues

Precision policy: FP32 for matrix data and arithmetic, FP64 for the
  convergence reduction only.

CUDA streams:
  - `stream_compute`  — all kernel execution
  - `stream_transfer` — async H2D of per-iteration CSR / CSC views
                        (CSR row_ptr / col_idx / values, CSC col_ptr /
                        row_idx, top-k working buffers)

Context handling: same pattern as HITS / Louvain —
  `retain_primary_context()` and unconditional push.

Cluster extraction (CPU, post-convergence): attractor method.  Nodes
  with `M[i, i] > 0` are attractors; every node j is assigned to
  `argmax_i M[i, j]`.  Fallback to weakly-connected components if no
  diagonal entry is positive.  Cluster IDs renumbered to compact
  `0..K-1`.

Compilation: **adaptive** — `_detect_arch_flag()` queries
  `cuda.Device(0).compute_capability()` and builds with
  `-arch=sm_<major><minor> -O3`.  Falls back to `sm_75` (RTX 20-series
  Turing) if the device cannot be probed.

Does NOT silently fall back to CPU — raises `RuntimeError` /
  `cuda.LogicError` / `MemoryError`.
Target: adaptive (RTX 20-series Turing by default), `BLOCK_SIZE = 256`,
  `TOP_K_BITONIC_THRESHOLD = 256`,
  `HEAVY_ROW_THRESH = 256`, `LIGHT_ROW_THRESH = 32`.

### VRAM ceiling and fast-fail policy

MCL is the most memory-intensive of the six algorithms: expansion squares
the matrix (`M @ M`) and the fill-in of the *product* (not the input
graph) is the binding VRAM constraint.  Two adaptive mechanisms push the
ceiling up, then MCL fails cleanly:

- `_adaptive_top_k(n, top_k, free_vram)` caps `top_k` so the pruned
  working set (`~n * top_k` entries, held as CSR + CSC ≈ 16 B/entry) fits
  ~35 % of free VRAM.  Bounds the SpGEMM **input**.  Only binds on graphs
  too large for the user's `top_k`; realistic biological networks pass
  through unchanged.  Logged and surfaced in `result["note"]`.
- `_adaptive_prune_threshold` raises the threshold under pressure to
  shrink the SpGEMM **output**.

- **No silent truncation.**  When the product still exceeds the VRAM
  budget, `_spgemm_gpu` raises a clear `MemoryError` (previously it kept
  the first `capacity` entries and continued with "approximate results" —
  removed as a correctness hazard).  `cuMemAlloc` failures are re-raised
  from `mcl_gpu` with concrete ceiling guidance rather than the cryptic
  driver message.

- **Measured ceiling** on a 6 GB RTX 2060 (grn/undirected, ~6 avg degree):
  ~1M nodes (`barabasi_albert`), ~1.6M (`erdos_renyi`), ~3.3M
  (`watts_strogatz`).  The limit is degree-variance dependent (ER and WS
  at equal n/m differ: ER's Poisson variance yields more fill-in).  Full
  out-of-core MCL (host-streamed SpGEMM+prune+inflate) is future work; the
  SpGEMM output's scatter-heavy access makes naive managed-memory spill
  impractically slow, so it is a substantial undertaking, not a quick
  tiling change.

---

## PageRank GPU implementation

Custom PyCUDA kernels — no CuPy dependency.  Hybrid push/pull dispatch
for extreme-hub networks; single CPU–GPU sync per iteration.

Key optimisations over the initial implementation:
  1. **GPU-side scalar reductions** (`reduce_to_scalar_f32`):
     Strided load + shared-memory tree reduction in one kernel (grid =
     (1,1,1)) writes a single FP32 scalar to a pre-allocated GPU pointer.
     Both the dangling mass sum and the L1 convergence delta stay on the
     device between kernel launches; no intermediate D2H copies.
  2. **Single sync per iteration**: all kernel launches queue on
     `stream_compute`; one `stream_compute.synchronize()` at the end of
     each iteration (after `reduce_to_scalar_f32` for L1) is the only
     host–device sync.  The convergence scalar is then fetched with
     `d_l1_scalar.get()[0]`.
  3. **Fused init + dangling kernel** (`initialize_pr_with_dangling`):
     Computes `PR_new[i] = teleport_val` for every node and — in the same
     kernel — adds `damping * (*d_dangling_sum) / num_eligible` for each
     eligible node, reading `d_dangling_sum` from a GPU pointer.
     Eliminates the separate `distribute_dangling_mass` launch and the
     inter-kernel dependency that previously forced an early sync.
  4. **Adaptive hub threshold** (`_compute_adaptive_hub_threshold`):
     `np.percentile(out_degrees, (1 − fraction) × 100)` at
     `fraction = 0.05` → clamped to `[WARP_SIZE, max_degree]` → snapped
     to nearest power of 2.  Replaces the hardcoded `HUB_THRESHOLD = 32`
     with a graph-dependent split that keeps ~5 % of nodes in ELLPACK.
  5. **Chunked double-buffer pipeline** (`_pagerank_gpu_chunked`):
     `_ChunkBuffer` (pre-allocated GPU ping-pong pair of row_ptr / col_idx
     / values / node_ids arrays) enables `stream_transfer` to pre-load
     chunk i+1 while `stream_compute` scatters chunk i via
     `cuda.Event.wait_for_event`.  Auto-triggered when estimated VRAM >
     80 % of free VRAM or `use_chunking=True` from `apply_config`.
  6. **Pull-mode for extreme in-degree** (`gather_contributions_pull`):
     One block per pull-target node gathers from incoming edges (CSC /
     transposed CSR); no atomicAdd — thread 0 writes the final reduced
     value.  Pull targets are identified by `in_degree ≥ pull_threshold`
     (disabled by default; zero-overhead when off via an all-zero uint8
     mask).  Scatter kernels skip edges to pull targets via a
     `pull_target_mask[v]` check to prevent double-counting.

Kernels (compiled once, cached in `_kernel_cache["pagerank"]`):
  - `initialize_pr` — teleport-only init (legacy path, kept for
    reference; superseded by `initialize_pr_with_dangling` in the GPU
    iteration loop).
  - `initialize_pr_with_dangling` — fused teleport + dangling
    distribution; reads `*dangling_sum_ptr` from GPU memory.
  - `scatter_contributions_csr` — low/medium/high-degree CSR push.
    Three-tier dispatch:
      LOW  (deg < 32)   : thread 0 only, serial atomicAdd
      MED  (32 ≤ deg < hub_threshold) : first warp, stride-32
      HIGH (deg ≥ hub_threshold)      : full block + 256-bucket SMEM hash
    All tiers check `pull_target_mask[v]` and skip pull targets.
    HIGH-tier SMEM hash: linear-probe open-addressing, atomicCAS slot
    claim, atomicAdd value accumulation; saturated chains fall back to
    direct global atomicAdd.
  - `scatter_contributions_ellpack` — extreme-hub push via padded ELLPACK
    (one block per hub, stride-`max_row_len`, `-1` padding skipped);
    also checks `pull_target_mask[v]`.
  - `sum_dangling_pr` — block-partial Σ `PR_old[i]` for dangling nodes;
    partials reduced by `reduce_to_scalar_f32`.
  - `reduce_to_scalar_f32` — strided load + SMEM tree → single FP32
    scalar at `scalar_output[0]`.  Reused for both dangling sum and L1.
  - `gather_contributions_pull` — pull gather for extreme in-degree
    nodes; SMEM tree reduction within block, thread 0 writes
    `PR_new[u] += damping * partial`.
  - `compute_l1_convergence` — block-partial Σ `|PR_new[i] − PR_old[i]|`
    via warp shuffles + 8-lane SMEM reduction; partials reduced by
    `reduce_to_scalar_f32`.

Hybrid storage split at adaptive `hub_threshold`:
  - `0 < out_degree < hub_threshold`  → CSR scatter kernel
  - `out_degree ≥ hub_threshold`      → ELLPACK scatter kernel
  - `out_degree == 0`                 → dangling, handled by dangling sum

Eligible mask (uint8 boolean array, length n):
  - GRN   : `mask[out_degree > 0] = 1` — regulators only.
  - PPI   : `mask[:] = 1` — uniform across all nodes.
  - miRNA : `mask[out_degree > 0] = 1` — miRNA nodes only.
  Degenerate fallback (all zeros): logs warning, falls back to uniform.

Out-degree storage: FP32 weighted sum (`graph_csr.sum(axis=1)`) so
  weighted networks are handled correctly; binary graphs collapse to
  integer degree.

Result keys by network type:
  - GRN   : `scores`, `top_regulators` (top-15, out>0),
            `top_targets` (top-15, out==0)
  - PPI   : `scores`, `top_nodes` (top-20 combined)
  - miRNA : `scores`, `top_mirnas` (top-15, out>0),
            `top_target_genes` (top-15, out==0)

Iteration structure:
  sum_dangling → reduce_to_scalar (dangling) →
  initialize_pr_with_dangling → scatter_csr → scatter_ellpack →
  [gather_pull if enabled] → compute_l1_convergence →
  reduce_to_scalar (L1) → stream_compute.synchronize() →
  d_l1_scalar.get()[0] → pointer swap PR_old/PR_new.

CUDA streams:
  - `stream_compute`  — all kernel launches (ordered implicitly)
  - `stream_transfer` — async H2D during setup and chunked pipeline
                        (CSR / ELLPACK / out_degree / masks / PR_old)

Context handling: same pattern as HITS / Louvain / MCL —
  `retain_primary_context()` and unconditional push.

VRAM check: `_estimate_pagerank_vram` vs `cuda.mem_get_info()`; raises
  `MemoryError` if working set exceeds free VRAM (before allocating).
  Chunked path (`_pagerank_gpu_chunked`) auto-activates to avoid OOM.

Compilation: **adaptive** — `_detect_arch_flag()` queries
  `cuda.Device(0).compute_capability()` and builds with
  `-arch=sm_<major><minor> -O3`.  Falls back to `sm_75` (RTX 20-series
  Turing) if the device cannot be probed.

Does NOT silently fall back to CPU — raises `RuntimeError` /
  `cuda.LogicError` / `MemoryError`.
Target: adaptive (RTX 20-series Turing by default), `BLOCK_SIZE = 256`,
  `SMEM_BUCKETS = 256`, `WARP_SIZE = 32`.

---

## RWR GPU implementation (optimised)

Custom PyCUDA kernels — no CuPy dependency.  Warp-per-node SpMV (default,
grn/ppi/fp16); degree-adaptive sub-warp SpMV (mirna only, see below);
optional FP16 weights; convergence checked every N iterations (single
sync per check); chunked execution for VRAM-bound graphs.

Key optimisations over the initial implementation:
  1. **GPU-side scalar reductions** (`reduce_to_scalar_f32`):
     Block-partial L1 from `l1_convergence_rwr` is reduced to a single
     FP32 scalar on the GPU.  Only that one float crosses PCIe per
     convergence check (vs. the full partial-sum array before).
  2. **Convergence checked every N iterations** (`conv_check_interval`,
     default 10): SpMV kernels queue continuously; the sync + scalar
     read happen only every N iterations (final iteration always
     checks) — eliminates ~90 % of CPU-GPU sync stalls vs. checking
     every iteration.
  3. **Warp-per-node SpMV** (`rwr_spmv_warp_per_node` /
     `_fp16w` / `_batched`): one 256-thread block processes
     `NODES_PER_BLOCK = 8` nodes at a time; each 32-lane warp handles
     exactly one node via warp-shuffle reduction (no SMEM, no
     `__syncthreads`).  8× fewer blocks than one-block-per-node, all
     warps in a block doing useful work simultaneously.
  4. **Persistent GPU working buffers** (`_WorkingBufferCache`):
     `d_p / d_pn / d_p0 / d_partial / d_l1_scalar` allocated once per
     graph size (`n`) and reused across repeated calls — no
     `gpuarray.empty()` / `.free()` churn per run.
  5. **Cached transition matrix + device CSR arrays**: the CPU-side
     column-stochastic `W` (`_W_CACHE`) and the uploaded device CSR
     (`_GPU_CSR_CACHE`) are recomputed / re-uploaded only when the
     graph object, `network_type`, or `precision_mode` changes.
  6. **Optional mixed precision** (`precision_mode = "mixed"`):
     Transition weights stored as FP16 (`unsigned short` half-bit
     pattern); `p / p_new / p₀` remain FP32; accumulation is FP32.
     Halves the w_values bandwidth.  Kernel:
     `rwr_spmv_warp_per_node_fp16w` (`__half2float` decode at use time).
  7. **Batched execution** (`rwr_spmv_warp_per_node_batched`) for
     `1 < B <= MAX_BATCH = 4` seed sets; `p / p₀ / p_new` stored as
     `[n × B]` row-major flat arrays, `gridDim = (blocks, B, 1)`.
  8. **Chunked execution** (`_rwr_gpu_chunked` + `_ChunkBuffer`):
     `p / p_new / p₀ / partial-sum buffers` stay full-size on the
     device throughout.  CSR rows stream in chunks via a pre-allocated
     GPU ping-pong pair; `stream_transfer` pre-loads chunk i+1 while
     `stream_compute` runs chunk i SpMV (`rwr_spmv_warp_per_node_chunk`
     — same warp-per-node layout, reads `chunk_node_ids[local_i]` to
     write back into the global `p_new`).  Auto-triggered only when
     the working set genuinely exceeds `VRAM_BUDGET_FRACTION = 0.80`
     of free VRAM — a `use_chunking=True` param from `apply_config`'s
     coarse estimate is deliberately NOT honoured on its own, since the
     chunked path re-streams the entire `W` CSR host→device every
     iteration (catastrophic for a graph that actually fits).
  9. **CUDA-event profiling** (`enable_profiling=True`): per-run
     breakdown of SpMV / convergence / H2D-transfer / total time,
     attached at `result["result"]["profiling"]`.  Zero overhead when
     disabled.

Removed optimisations (introduced more overhead than benefit on these
graphs — see module docstring): hub-row ELLPACK, SMEM-cached p vector,
node reordering.  Do not reintroduce without re-benchmarking.

### mirna-only degree-adaptive SpMV (throughput fix)

**Problem observed**: on the RTX 2060 scalability benchmark
(barabasi_albert / erdos_renyi / watts_strogatz, avg degree ≈ 6),
`mode=gpu` RWR was slower than both `cpu_multi` (GraphBLAS) and
`gpu_baseline` (cuGraph) at every graph size, for all three network
types.  Root-caused to two issues, both fixed for `network_type ==
"mirna"` only (grn/ppi keep the original path byte-identical, per
existing benchmark expectations for those two types):

  1. **Redundant `apply_config()` call inside the timed region.**
     `algorithm_runner.run_algorithm()` already calls `apply_config()`
     once, before starting the CUDA-event timer, and passes the tuned
     params into `rwr_gpu()`.  `rwr_gpu()` was calling `apply_config()`
     a *second* time internally — and every key that second call
     injects (`spmv_mode`, `reorder_nodes`, `chunk_size`,
     `execution_mode`, …) is ignored by this file's own VRAM/chunking
     logic and param handling (the source of the runner's "ignoring
     unknown param keys" warning).  That second call's CPU graph
     profiling + live `cuda.mem_get_info()` query ran *after* the
     runner's CUDA start-event fired, so its cost was billed to the
     optimised-GPU timer at every graph size — explaining the
     across-the-board slowdown, not just at scale.  For
     `network_type == "mirna"` this second `apply_config()` call is
     now skipped entirely; results are unaffected since nothing
     downstream consumed its output.
  2. **Warp-per-node wastes lanes on low-average-degree graphs.**
     `rwr_spmv_warp_per_node` gives every node a full 32-lane warp.
     At avg degree ≈ 6 (BA/ER/WS test graphs), ~26 of 32 lanes sit
     idle every iteration — cuGraph's load-balanced SpMV does not pay
     this cost.  Fixed with a degree-adaptive sub-warp SpMV.

New kernels: `rwr_spmv_tpr2` / `_tpr4` / `_tpr8` / `_tpr16` — a macro-
generated family (`RWR_SPMV_TPR_KERNEL`) where `tpr` (threads-per-row)
contiguous lanes cooperate on one row, packing `BLOCK_SIZE / tpr` rows
per block instead of `NODES_PER_BLOCK = 8`.  Threads whose row is out
of range do NOT early-return (they participate with `partial = 0` so
the full-warp shuffle mask `0xffffffff` stays valid); only lane 0 of
each `tpr`-wide group writes `p_new`.  `tpr` divides 32 and groups are
lane-contiguous, so the sub-group `__shfl_down_sync` reduction never
crosses a group boundary.

`_choose_tpr(avg_degree)` picks `tpr ∈ {2, 4, 8, 16, 32}` as the
smallest power of 2 that is `>= avg_degree` (capped at 32); `tpr == 32`
is a no-op — it dispatches to the original `rwr_spmv_warp_per_node`
kernel and grid, so high-average-degree mirna graphs are unaffected.
Dispatch (serial single-seed FP32 path only — batched and FP16 keep
warp-per-node):
```python
if fp16_weights:                       # fp16 always uses warp-per-node
    k_spmv = spmv_restart_fp16w
elif network_type == "mirna":
    tpr = _choose_tpr(W.nnz / n)
    k_spmv = spmv_restart if tpr >= 32 else kernels[f"spmv_tpr{tpr}"]
    grid   = (n_spmv_blocks, 1, 1) if tpr >= 32 else \
             (ceil(n / (BLOCK_SIZE // tpr)), 1, 1)
else:                                   # grn / ppi — unchanged
    k_spmv = spmv_restart
```
**Verification note**: since mirna and grn build the identical
transition matrix on these synthetic graphs (`_build_transition_matrix`
has one shared branch for both — see below), the mirna GPU
`top_nodes`/`scores` on a given graph must match the grn GPU run on
the same graph.  Use this as the correctness check for the adaptive
kernel before trusting new benchmark numbers.

**Confirmed on RTX 2060** (`scalability` benchmark, `--network-type
mirna`, edge targets 1M–50M, barabasi_albert / erdos_renyi /
watts_strogatz): GPU Optimised (`mode=gpu`) now runs below both CPU
GraphBLAS (`cpu_multi`) and GPU Baseline (`gpu_baseline`) at every
tested size across all three graph families — the regression described
above is resolved.  grn / ppi results are unaffected (unchanged code
path).

Kernels (compiled once, cached in `_kernel_cache["rwr"]`):
  - `reduce_to_scalar_f32` — strided load + SMEM tree → single FP32
    scalar at `scalar_output[0]`.  Reused for L1 convergence.
  - `rwr_spmv_warp_per_node` — FP32 SpMV + restart, warp-per-node
    (default for grn/ppi).  `__ldg(&p[col_idx[j]])` for read-only
    cache routing; 5 unrolled `__shfl_down_sync` for the warp
    reduction; lane 0 writes `p_new[node_i]`.
  - `rwr_spmv_warp_per_node_fp16w` — same layout, FP16 weight reads
    via `__half2float`.  Selected when `precision_mode == "mixed"`.
  - `rwr_spmv_tpr2` / `_tpr4` / `_tpr8` / `_tpr16` — degree-adaptive
    sub-warp SpMV, mirna-only (see above).
  - `l1_convergence_rwr` — `Σ |p_new[i] − p[i]|` per block via warp
    shuffles + final 8-lane warp reduction in SMEM (FP32).
  - `rwr_spmv_warp_per_node_batched` — multi-seed batched kernel.
    `gridDim = (ceil(n / NODES_PER_BLOCK), B, 1)`; one warp per
    `(node_i, seed_b)` pair; `p / p₀ / p_new` stored as `[n × B]`
    row-major flat arrays.  Used only when `1 < B <= MAX_BATCH = 4`.
  - `rwr_spmv_warp_per_node_chunk` (separate `_CHUNK_KERNEL_SOURCE`
    module) — chunked CSR variant; `chunk_node_ids[local_i]`
    translates chunk-local row → global node id for writeback to
    `p_new`.

Transition matrix W construction (CPU, network-type aware,
`_build_transition_matrix` in `src/algorithms/gpu/cuda_optimized/rwr.py`):
  - GRN / miRNA : directed CSR as-is (one shared code branch — miRNA
            gets no special-case treatment); `W = (D⁻¹ · A).T`.
            Dangling columns (`out_degree == 0`) get a self-loop
            `W[j,j] = 1`.
  - PPI   : `A_sym = A + Aᵀ`, binarised, then column-normalise + the
            dangling-column self-loop fix.

p₀ construction:
  - Non-empty seeds : `p0[seed] = 1 / |seeds|`, zero elsewhere.
  - Empty seeds     : `p0[i] = 1 / n` (uniform — global PageRank-like).

Multi-seed-set handling:
  - `seed_nodes = [1, 2, 3]`        → single seed set (`B = 1`)
  - `seed_nodes = [[1,2], [3,4]]`   → multiple seed sets (batched)
  - `B ≤ MAX_BATCH` (= 4)           → batched kernel
  - `B > MAX_BATCH`                  → serial-per-seed-set loop
  Per-node scores are averaged; `batch_results` field carries
  per-seed-set scores / iterations / converged when `B > 1`.
  Note: chunked + batched is not supported; if both are requested the
  batched path runs on the full graph (may OOM).

Result keys: `scores`, `top_nodes` (top-20), `top_seeds` (top-10
  among the union of seed indices, or `top_nodes[:10]` if no seeds),
  `iterations`, `converged`, `note`, and `batch_results` when `B > 1`.
  `note` also carries the transition-matrix construction note and a
  GPU-pipeline summary (precision, chunked, arch, sync cadence).

Iteration structure (single-seed, serial path):
  spmv (warp_per_node / tpr-adaptive / fp16w) → [every N iters:
  l1_convergence → reduce_to_scalar → stream_compute.synchronize() →
  d_l1_scalar.get()[0]] → pointer swap p/p_new.

CUDA streams:
  - `stream_compute`  — kernel execution (SpMV + L1 + reduction)
  - `stream_transfer` — async H2D during setup + chunked pipeline
  Pointer swap of `p / p_new` each iteration (no data copy).

Context handling: same pattern as HITS / Louvain / MCL / PageRank —
  `retain_primary_context()` and unconditional push.

VRAM check: `_estimate_rwr_vram` (accounts for FP16 weights when
  `precision_mode == "mixed"`).  If estimated working set exceeds
  free VRAM AND chunked is not needed → raises `MemoryError`.  Chunked
  path auto-activates only when the working set exceeds
  `VRAM_BUDGET_FRACTION = 0.80` of free VRAM.

Compilation: **adaptive** — `_detect_arch_flag()` queries
  `cuda.Device(0).compute_capability()` and builds with
  `-arch=sm_<major><minor> -O3`.  Falls back to `sm_75` (RTX 20-series
  Turing) if the device cannot be probed.  Kernel source includes
  `<cuda_fp16.h>` outside the `extern "C"` block to enable `__half`
  decoding in the FP16-weight variant.

Does NOT silently fall back to CPU — raises `RuntimeError` /
  `cuda.LogicError` / `MemoryError`.  CPU mode is provided by a
  separate package (`src.algorithms.cpu.single_threaded.rwr` and
  `src.algorithms.cpu.multi_threaded.rwr`) selected by the runner.
Target: adaptive (RTX 20-series Turing by default), `BLOCK_SIZE = 256`,
  `NODES_PER_BLOCK = 8`, `MAX_BATCH = 4`.

---

## GPU baseline implementations (`src/algorithms/gpu/basic/`)

Deliberately simple GPU reference implementations used as a benchmark
counterweight to the heavily tuned `cuda_optimized` package.  Six files,
one per algorithm, plus a shared `_utils.py`:

```
src/algorithms/gpu/basic/_utils.py
src/algorithms/gpu/basic/pagerank.py
src/algorithms/gpu/basic/bfs.py
src/algorithms/gpu/basic/hits.py
src/algorithms/gpu/basic/louvain.py
src/algorithms/gpu/basic/rwr.py
src/algorithms/gpu/basic/mcl.py
```

### Naming
Each module exposes one top-level function:
```python
<algorithm>_gpu_baseline(graph_csr: sp.csr_matrix, params: dict) -> dict
```
The function returns the standard 7-key envelope with `mode="gpu_baseline"`
and inner-result keys that **exactly match** the corresponding
`cuda_optimized` per-network-type schema (verified by
`tests/test_gpu_baseline.py::test_*_baseline_schema`).  All baselines
additionally include an additive `result["result"]["backend"]` key whose
value is `"cugraph"` or `"cupy"` for benchmarking provenance.

### Backend selection (per-module, in this priority order)
1. **cuGraph** (preferred) — chosen at runtime via
   `cugraph_function(<api_name>)`.  The presence of every cuGraph API is
   re-checked at call time so the baseline never relies on documentation
   assumptions about a specific RAPIDS release.  Signatures are inspected
   via `inspect.signature` and only accepted kwargs are forwarded.
   Return columns are extracted by `cugraph_extract_column(df, candidates)`
   which accepts the first matching name from a candidate list (e.g.
   `["pagerank", "score", "scores"]`) so column renames between releases
   degrade gracefully.
2. **CuPy sparse** (fallback) — power iteration / SpMV / matrix powers
   using `cupy` and `cupyx.scipy.sparse`.  No custom kernels.
3. **RuntimeError** — raised when neither backend is importable; baselines
   never silently fall back to CPU.

### Per-algorithm baseline strategy
| Algorithm | cuGraph API used | CuPy fallback |
|-----------|------------------|---------------|
| pagerank  | `cugraph.pagerank`              | power iteration with dangling redistribution (network-type aware) |
| bfs       | `cugraph.bfs`                   | iterative SpMV frontier expansion on a bitmask |
| hits      | `cugraph.hits`                  | power iteration on A and A^T with L2 normalization |
| louvain   | `cugraph.louvain`               | scipy connected-components labelling (degraded fallback — see note) |
| rwr       | `cugraph.personalized_pagerank` | power iteration with restart vector |
| mcl       | none (cuGraph has no MCL)       | sparse matrix powers + inflation + threshold pruning |

The Louvain fallback is intentionally crude (weakly-connected components
seeded label set).  A real GPU Louvain without custom kernels would
either replicate the optimized implementation or run slower than NetworkX
on CPU.  The schema is preserved so result-shape comparisons against
`cuda_optimized` still work even when cuGraph is absent.

### Result envelope and mode

Mode strings reflect the actual backend used (not the generic `"gpu_baseline"`):

```python
# cuGraph algorithms (pagerank, bfs, hits, louvain, rwr):
{
  "algorithm":      str,                    # "pagerank" | "bfs" | "hits" | "louvain" | "rwr"
  "mode":           "gpu_baseline_cugraph", # strict cuGraph-only
  "network_type":   str,                    # "grn" | "ppi" | "mirna"
  "execution_time": float,
  "num_nodes":      int,
  "num_edges":      int,
  "result":         { ...inner keys identical to cuda_optimized...,
                      "backend": "cugraph" },
}

# MCL (CuPy — cuGraph has no MCL):
{
  "algorithm":      "mcl",
  "mode":           "gpu_baseline_cupy",   # CuPy only
  ...
  "result":         { ..., "backend": "cupy" },
}
```

The runner whitelist accepts `"gpu_baseline"` as the requested mode.  The
algorithm itself sets the specific mode string (`"gpu_baseline_cugraph"` or
`"gpu_baseline_cupy"`), and the runner preserves it (does **not** overwrite
with the generic `"gpu_baseline"`).

The mode string `"gpu_baseline"` (generic) is wired to:
* `src/algorithms/base.py::VALID_MODES` — extended tuple.
* `src/algorithms/base.py::AlgorithmBase.gpu_baseline()` — stub raising
  `NotImplementedError` by default; the adapter overrides it.
* `src/algorithms/__init__.py::_make_adapter` — lazily imports
  `src.algorithms.gpu.basic.<name>` on first call and binds the module
  function as the adapter's `gpu_baseline` staticmethod.
* `src/runner/algorithm_runner.py::run_algorithm` — mode whitelist
  accepts `gpu_baseline`; `BenchmarkTimer` treats it as a GPU-class
  mode (CUDA-event timing when available); `_cuda_context_guard` pushes
  the primary context exactly as it does for `gpu`.  An `ImportError`
  from a missing backend is converted to `RuntimeError` with RAPIDS
  install instructions.

### Benchmarking consistency

**Hard-fail policy (no silent CuPy fallback for cuGraph algorithms):**
Each of pagerank, bfs, hits, louvain, rwr raises `ImportError` at module
import time if cuGraph or cuDF is not installed.  This guarantees every
benchmark result tagged `"gpu_baseline_cugraph"` actually used cuGraph.

MCL uses CuPy (`"gpu_baseline_cupy"`) and does **not** import from
`_utils.py` (which requires cuGraph), so it remains independently
importable on a CuPy-only machine.

### RAPIDS environment probe
A standalone diagnostic script lives at `scripts/rapids_probe.py`.  Run
on the Linux GPU box where benchmarking takes place:
```
python scripts/rapids_probe.py            # human-readable
python scripts/rapids_probe.py --json     # JSON for archiving
```
It reports installed versions, available cuGraph functions, their
parameter names, and the column names each function returns on a tiny
test graph — used to verify that the baseline's runtime API probing is
finding what it expects.

### Tests
`tests/test_gpu_baseline.py` covers:
* Outer envelope structure (backend-independent).
* Inner-key parity with cuda_optimized for every algorithm × network
  type (skipped when cuGraph / CuPy not present).
* `result["mode"]` is `"gpu_baseline_cugraph"` for cuGraph algorithms
  and `"gpu_baseline_cupy"` for MCL.
* `result["result"]["backend"]` is `"cugraph"` or `"cupy"`.
* Hard-fail: importing a cuGraph algorithm without RAPIDS raises
  `ImportError` with an install hint.
* Runner converts `ImportError` to `RuntimeError` with RAPIDS instructions.
* Execution-time recording.
* Adapter sanity: every entry in `ALGORITHM_REGISTRY` exposes a
  `gpu_baseline` staticmethod.

### Rules
- **Never** add custom kernels, PyCUDA, warp primitives, SMEM tricks,
  graph reordering, hybrid push/pull, chunking, or memory-aware
  execution to `src/algorithms/gpu/basic/`.  Those belong exclusively
  to `src/algorithms/gpu/cuda_optimized/`.
- **Never** import the baseline modules from the web application.
  The webapp's `gpu` mode always uses `cuda_optimized`.
- **Never** hardcode a cuGraph API signature from documentation —
  use `cugraph_function(name)` + `inspect.signature` to detect at
  runtime.
- **Never** add a CuPy fallback to a cuGraph algorithm file.  If the
  cuGraph API is absent in the installed RAPIDS version, raise
  `RuntimeError` — do not silently switch to CuPy.

---

## Progress event format (NDJSON)

```json
{"type": "progress", "stage": "<stage>", "percent": 0-100, "message": "<text>"}
{"type": "result",   "data": { ... }}
{"type": "error",    "message": "<text>"}
```

---

## Frontend screens (8 steps, extending the original 4-step flow)

| Step | Component (status) |
|------|--------------------|
| 0 | `Upload.jsx`            (existing — do not modify) |
| 1 | `Mapping.jsx`           (existing — do not modify) |
| 2 | `Validation.jsx`        (existing — do not modify) |
| 3 | `GraphSummary.jsx`      (existing — do not modify) |
| 4 | `AlgorithmSelector.jsx` |
| 5 | `RunAnalysis.jsx`       |
| 6 | `ResultsView.jsx`       (Table / Chart / Graph toggle) |
| 7 | `ExportPanel.jsx`       |

### Visualization toggle (Step 6)

Three views for every algorithm result:
- **Table** — top-k nodes with scores, or cluster table, or cascade table
- **Chart** — bar chart of scores or cluster size distribution
- **Graph** — Cytoscape network with highlighted important nodes,
  nodes coloured by community (louvain/mcl),
  nodes sized by score (pagerank/hits/rwr)

Default view per algorithm:
- `pagerank`, `rwr`, `hits` → Chart
- `louvain`, `mcl` → Graph
- `bfs` → Table

---

## Algorithm parameter UI schema

UI metadata for the algorithm parameter form lives in the `UI_SCHEMA`
dict in `src/runner/algorithm_runner.py`.  It is intentionally
**separate** from the per-algorithm `PARAM_SCHEMA` (which the runner
and the algorithm files use for validation) so algorithm code does
not carry UI concerns.

`UI_SCHEMA[<algo>]` shape:
```python
{
  "display_name": str,
  "description":  str,
  "category":     str,      # "Ranking" | "Traversal" | "Clustering" | "Propagation"
  "params": [
    {"key": str, "label": str, "type": str, "default": Any,
     "tooltip": str, "advanced": bool, ...type-specific keys},
    ...
  ],
}
```

`get_algorithm_info(name)` returns:
```python
{
  "name":         str,
  "display_name": str,
  "description":  str,
  "category":     str,
  "param_schema": dict,   # backend validator (unchanged)
  "ui_schema":    list,   # frontend form builder
}
```

`list_algorithms()` returns the same shape per algorithm — backward
compatible because the old keys (`name`, `param_schema`,
`description`) are preserved.

### Parameter input types

| `type`                  | Renderer                | Notes |
|-------------------------|-------------------------|-------|
| `slider`                | `ParamSlider`           | range input + live value chip; honours `min/max/step` and optional `display_format: "scientific"` |
| `number`                | `ParamNumber`           | numeric text input; `step` may be `None` for free entry |
| `select`                | `ParamSelect`           | dropdown of `options: [{value, label}]`; coerces to number when all option values are numeric |
| `preset_select`         | `ParamPresetSelect`     | pill row of `presets: [{label, value}]`; pills are display affordances — the float `value` is what gets submitted |
| `node_selector`         | `ParamNodeSelector`     | searchable typeahead → single node index |
| `multi_node_selector`   | `ParamMultiNodeSelector`| searchable typeahead + removable pills → list of node indices |

### Advanced settings

Params with `advanced: true` are hidden behind a collapsible
"⚙ Advanced Settings" toggle in `AlgorithmSelector.jsx`.  Default
state is closed; the React component tracks `advancedOpen` per
algorithm-selection (resets to closed when the user picks a
different algorithm).

### Node-name data flow

- Preprocessing populates `node_index_map` in `dataset_store`
  (existing behaviour).
- `GET /graph/nodes?upload_id=&search=&limit=` filters the map
  case-insensitively, sorts by label, caps at `limit` (default 50,
  max 200), and returns `{nodes: [{index, label}], total, truncated}`.
- `getNodes(uploadId, search, limit)` in `api.js` wraps it.
- `ParamNodeSelector` / `ParamMultiNodeSelector` debounce the search
  query at 250 ms.

### Submitted params

`AlgorithmSelector` always submits **raw float / int / list values**
to `POST /algorithms/run`.  Preset labels are UI-only and never
cross the network — when a user clicks "Balanced" on a HITS preset,
`paramValues.tolerance` becomes `1e-6` directly.

### Theme note

The spec's CSS used `var(--accent-cyan)` / `var(--bg-elevated)`
design tokens for a dark theme.  The current `index.css` is a
light theme with literal hex colors (`#244154`, `#2f6f7e`,
`#f4f8fb`).  The new UI components use the existing palette so
they render correctly in this repo; migrating to design tokens is
a separate, codebase-wide change.

---

## Coding conventions

- Python: type hints on all function signatures
- Python: docstrings on all public functions and classes
- Python: every algorithm file must implement three functions:
  `_cpu_single`, `_cpu_multi`, `_gpu`
- Python: try/except around all CuPy / PyCUDA imports with
  `CUPY_AVAILABLE = False` fallback (GPU files only)
- Python: timing uses `time.perf_counter()` for CPU and CUDA events for GPU
- Python: timing covers **ONLY** the algorithm on a loaded dataset, never file I/O
- Python: all benchmark results saved as CSV/JSON to
  `experiments/outputs/reports/`
- Python: call `_ensure_cuda_context()` from
  `src.benchmarking.benchmark` before any CuPy operation in a `_gpu()` function
- React: functional components with hooks only (no class components)
- React: follow existing `App.jsx` state management style
- React: follow existing `index.css` class naming conventions
- No matplotlib in backend — visualization layer returns JSON data only
- No direct algorithm calls from frontend routes —
  always go through `algorithm_service.py` → `algorithm_runner.py`

---

## Integration rules

- `GraphData` (from preprocessing) → `graphdata_to_csr()` → CSR matrix
  This conversion happens in `algorithm_service.py`, not in routes.
- `algorithm_runner.py` is the **only** entry point for running algorithms.
- `src/visualization/` functions are called by `algorithm_service.py`
  **after** the algorithm completes, to attach viz data to the result.
- New algorithms: add file to `src/algorithms/gpu/cuda_optimized/`, register
  in the `ALGORITHM_REGISTRY` dict in `algorithm_runner.py` — frontend
  picks up automatically via `GET /algorithms/catalog`.
- **Data privacy: all processing must be local, no cloud.**

---

## Data flow: preprocessing → algorithm

1. User uploads file → POST /upload
   dataset_store stores: raw DataFrame

2. User maps columns + runs preprocessing → POST /preprocess/stream
   preprocessing_service.py runs pipeline ONCE
   dataset_store stores: graph_data, graph_csr, node_index_map
   Frontend shows Graph Summary (Step 3)

3. User selects algorithm → POST /algorithms/run
   algorithm_service.py retrieves graph_csr + node_index_map
   directly from dataset_store
   Pipeline is NOT re-run
   If graph_csr is None: return HTTP 400 (not preprocessed yet)

The pipeline runs EXACTLY ONCE per upload session.
graph_csr and node_index_map are computed once and reused
for every algorithm run in that session.

---

## Additional rules for the new structure

### Import paths (always use these — never old paths)

```python
# Preprocessing pipeline
from src.preprocessing.pipeline   import PreprocessingPipeline
from src.preprocessing.modules    import GraphBuilder, FileLoader, SchemaDetector
from src.preprocessing.graph_data import GraphData

# Algorithms — GPU / webapp / GPU benchmarking
from src.algorithms.gpu.cuda_optimized.pagerank import _cpu_single, _cpu_multi, _gpu

# Algorithms — CPU benchmarking ONLY (never webapp)
from src.algorithms.cpu.pagerank import _cpu_single, _cpu_multi

# GPU config
from src.optimization.gpu_config import get_gpu_config, apply_config, print_gpu_summary

# Benchmarking
from src.benchmarking.benchmark import run_benchmark, _ensure_cuda_context, BenchmarkTimer

# Runner
from src.runner.algorithm_runner import run_algorithm, list_algorithms
from src.runner.progress         import ProgressReporter

# Visualization
from src.visualization.tables    import make_top_nodes_table, make_cluster_table, make_cascade_table
from src.visualization.charts    import make_score_chart_data, make_cluster_size_chart_data
from src.visualization.graph_viz import make_highlight_data

# Explanation layer (planned)
from src.explanation.explainer   import explain_result
```

### Directory layout

- Backend routes are in:    `webapp/backend/routes/`
- Backend services are in:  `webapp/backend/services/`
- Frontend components:      `webapp/frontend/src/components/`

---

## What is already built (do not modify without instruction)

- `src/preprocessing/`          (pipeline, modules, graph_data)
- `webapp/backend/routes/preprocessing.py`
- `webapp/backend/services/dataset_store.py`
  Stores per upload_id: raw DataFrame, GraphData,
  graph_csr (CSR matrix), node_index_map.
  Graph results are populated by preprocessing_service.py
  after /preprocess completes.
  Retrieved directly by algorithm_service.py —
  pipeline is NEVER re-run after Step 3.
- `webapp/backend/services/preprocessing_service.py`
- `webapp/backend/models/`
- `webapp/frontend/src/components/Upload.jsx`
- `webapp/frontend/src/components/Mapping.jsx`
- `webapp/frontend/src/components/Validation.jsx`
- `webapp/frontend/src/components/GraphSummary.jsx`
- `webapp/frontend/src/components/GraphView.jsx`
- `src/algorithms/gpu/cuda_optimized/` (all algorithm files)
- `src/algorithms/cpu/` (all algorithm files)
- `src/optimization/gpu_config.py`
- `src/benchmarking/benchmark.py`

## What still needs to be built

- `src/graph/converter.py` and `graph_utils.py`
- `src/algorithms/base.py` (`AlgorithmBase`)
- `src/runner/algorithm_runner.py` and `progress.py`
- `src/visualization/tables.py`, `charts.py`, `graph_viz.py`
- `src/explanation/explainer.py`
- `webapp/backend/routes/algorithms.py`, `results.py`, `graph.py`
- `webapp/backend/services/algorithm_service.py`, `result_store.py`
- `webapp/frontend/src/components/AlgorithmSelector.jsx`
- `webapp/frontend/src/components/RunAnalysis.jsx`
- `webapp/frontend/src/components/ResultsView.jsx`
- `webapp/frontend/src/components/viz/TableView.jsx`
- `webapp/frontend/src/components/viz/ChartView.jsx`
- `webapp/frontend/src/components/viz/GraphHighlight.jsx`
- `webapp/frontend/src/components/ExportPanel.jsx`
- `webapp/frontend/src/components/ExplanationPanel.jsx`
- `src/core/pipeline.py`, `config.py`, `runner.py`
- `experiments/benchmark/run_benchmark.py` (full implementation)
- `tests/` (integration tests)

---

## Dependencies

- **Python**: `numpy`, `scipy`, `pandas`, `openpyxl`, `networkx`, `fastapi`,
  `uvicorn`, `pydantic`, `python-multipart`, `matplotlib`, `seaborn`,
  `cupy` (optional), `pycuda` (optional)
- **Node**: `react`, `vite`, `cytoscape`
