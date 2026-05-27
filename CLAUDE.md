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

src/algorithms/cpu/single_threaded/         ← used by CPU benchmarking only
  Single-threaded CPU implementations (cpu_single only). No _gpu function.
  Never imported by webapp code.

src/algorithms/cpu/multi_threaded/          ← used by CPU benchmarking only
  Multi-process CPU implementations (cpu_multi only). No _gpu function.
  Never imported by webapp code.
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

## GPU optimization rules (applied automatically by `algorithm_runner.py`)

- `apply_config()` from `src/optimization/gpu_config.py` is called by
  `algorithm_runner.py` before every GPU algorithm run
- Algorithm files themselves never call `gpu_config` directly
- GPU config is cached after first call — do not re-detect on every run
- Target hardware: NVIDIA RTX 20-series (compute capability 7.5, 6–8 GB VRAM)
  — specifically RTX 2060 with 6 GB VRAM for testing
- If CUDA unavailable: fall back to `cpu_single` silently, log warning

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

Custom PyCUDA kernels — no CuPy dependency.

Key optimisations over the initial implementation:
  1. Fused SpMV + norm: `spmv_with_norm_sq` computes y = M·x AND writes
     per-node y[i]² to `partial_norm` in one kernel pass — eliminates a
     separate `compute_partial_norm_sq` launch.
  2. GPU-side norm reduction: `partial_reduce_to_scalar` reduces the
     `partial_norm` array to a single scalar and writes sqrt(sum) to a
     GPU pointer — `normalize_inplace` reads that pointer in the next
     launch with no CPU round-trip between SpMV and normalisation.
  3. Single CPU sync per iteration: only `d_partial_conv.get()` for the
     convergence delta check ever transfers to the CPU (vs. ~4–5 syncs
     before).
  4. Edge-parallel low-degree kernel: `spmv_edge_parallel_low_degree`
     packs `NODES_PER_BLOCK = 8` nodes per CTA (one warp each), giving
     ~8× better SM occupancy for degree < `WARP_SIZE` nodes, which are
     the majority in biological networks.
  5. Node reordering: nodes sorted by descending out-degree before CSR
     upload (`reorder_nodes=True` by default); similar-degree rows are
     adjacent, improving `x[col_idx[j]]` cache locality.  Original-index
     order restored before building the result dict.

Kernels per iteration: 10 total (vs. ~14 before).
CPU-GPU syncs per iteration: 1 (vs. ~4–5 before).

Per-iteration sequence:
  Authority (a_new = Aᵀ h):
    1. `spmv_edge_parallel_low_degree`  — packed-warp SpMV + norm (deg < 32)
    2. `spmv_with_norm_sq`              — fused SpMV + norm (deg ≥ 32)
    3. `partial_reduce_to_scalar`       — GPU sqrt(Σ partial_norm) → scalar
    4. `normalize_inplace`              — divide a_new by GPU scalar
  Hub (h_new = A a_new):
    5–8. same four kernels for A
  Convergence:
    9. `compute_convergence_partial`    — FP64 partial Σ((Δh)² + (Δa)²)
   10. `stream_compute.synchronize()`  — the single sync
       `delta = sqrt(sum(d_partial_conv.get()))`

Kernel registry (`_kernel_cache["hits"]`):
  - `spmv_low_deg`   : `spmv_edge_parallel_low_degree`
  - `spmv_high`      : `spmv_with_norm_sq`
  - `reduce_scalar`  : `partial_reduce_to_scalar`
  - `norm_div`       : `normalize_inplace`
  - `conv_partial`   : `compute_convergence_partial`
  - Deprecated (compiled, not called): `spmv_degree_aware`,
    `compute_partial_norm_sq`, `normalize_vector`,
    `compute_convergence_delta`

Precision: FP32 for SpMV arithmetic and L2 norms (`partial_norm` is
  float32).  FP64 for convergence delta only — avoids false early
  termination near `tolerance = 1e-6` without cancellation error.

Both `A` and `Aᵀ` stored in CSR on GPU throughout iteration.
Pointer swap (`cur_h, nxt_h = nxt_h, cur_h`) avoids data copies.
CUDA streams: `stream_compute` for kernels, `stream_transfer` for
  async H2D transfers during setup.
Context handling: `retain_primary_context().push()` unconditionally;
  coexists with CuPy (avoids `cuModuleLoadDataEx: invalid device context`).
Network-type behaviour:
  - grn / mirna: directed `A`; returns `top_hubs` + `top_authorities` +
    `hub_authority_overlap`
  - ppi: symmetrised `A + Aᵀ` (binarised); returns `top_nodes` only
Does NOT silently fall back to CPU — raises `RuntimeError` /
  `cuda.LogicError` / `MemoryError`.
Target: `-arch=sm_75` (RTX 20-series Turing), `BLOCK_SIZE = 256`,
  `NODES_PER_BLOCK = 8`.

---

## Louvain GPU implementation

Custom PyCUDA kernels — no CuPy dependency.

Preprocessing (CPU, before any GPU work):
  - `_symmetrize(graph_csr, network_type)` — network-type-aware
    undirected conversion.
      grn / mirna : `A + Aᵀ` (mutual edges weight 2)
      ppi         : graph used as-is (already undirected)
  - `_remove_self_loops` — `setdiag(0)` + `eliminate_zeros()`; self-loops
    are regenerated correctly at higher levels as collapsed
    intra-community weight.
  - `_handle_isolated_nodes` — zero-degree nodes are detected and left
    in singleton communities (the kernel returns `proposed = current`
    for `degree == 0`); the CSR is NOT mutated.
  - `_normalize_weights` — divides by max weight.  Modularity Q is
    invariant under uniform scaling; protects against float32 overflow.

Kernels (compiled once, cached in `_kernel_cache["louvain"]`):
  - `compute_proposed_moves` — Phase 1 core.  Three-tier degree-aware
    scheduling (1 thread / 1 warp / 1 block based on `degree[u]`);
    two-pass design within each block:
      Pass 1 accumulates `k_self` (sum of edge weights to the current
        community) via warp shuffles or shared-mem reduction.
      Pass 2 evaluates per-edge gain for moving u to each neighbour
        community; block-wide reduction finds the winning (gain, comm).
    Writes proposals only; community array is never modified here.
  - `apply_moves` — Phase 1 batch update.  One thread per node, atomic
    update to `community[]` and `comm_degree_sum[]`; increments
    `improvement_flag` so the host can loop Phase 1 to convergence.
    Conflicts (A↔B swap cycles) apply both moves — documented as
    parallel non-determinism, valid partition.
  - `count_community_edges` — Phase 2 step 1.  One thread per edge;
    emits `(comm[src], comm[dst], w)` triples.  Uses precomputed
    `edge_src[e] = np.repeat(arange(n), diff(indptr))`.
  - `compute_modularity_partial` — final Q per level.  One thread per
    edge, partial sums of `w − γ·k_i·k_j·inv_2m` for same-community
    edges, block-reduced; host sums the partials and scales by `inv_2m`.

Phase 2 sort + reduce strategy:
  - GPU `count_community_edges` → D2H copy of `(csrc, cdst, cwt)`
  - CPU `np.lexsort((cdst, csrc))` — correctness-first
  - CPU segment-head detect + `np.add.reduceat` to sum duplicate edges
  - CPU `scipy.sparse.csr_matrix((wt, (src, dst)), shape=(K, K))`
  - Self-loops KEPT (intra-community weight contributes to higher
    levels' modularity)
  - TODO: GPU radix sort (CUB DeviceRadixSort) once thrust / pycuda-cub
    bindings justify the dependency cost.

CUDA streams:
  - `stream_compute`  — kernel execution (Phase 1 loop, modularity,
    count_community_edges)
  - `stream_transfer` — async H2D for per-level CSR uploads

Context handling: same pattern as HITS — `retain_primary_context()`
  and unconditional push.  Coexists with CuPy in the same process.

Hierarchy bookkeeping:
  - `global_community[i]` tracks the level-current community of original
    node i.  After each Phase 1, communities are renumbered to `0..K-1`
    via `np.unique`, then `global_community = renum[global_community]`.
  - `hierarchy` field is one list per level, each containing the
    community ID of every original node at that level.

Non-determinism: parallel move application can produce a different
  partition than serial Louvain.  Documented in
  `result["result"]["note"]`.  Modularity remains non-decreasing in
  practice.  `max_phase1_passes` caps oscillation.

Chunked fallback: `apply_config` may set `use_chunking=True` for low
  VRAM tiers.  Currently treated as advisory (regular path runs); a
  proper chunked Phase 1 is a future optimisation.

Does NOT silently fall back to CPU — raises `RuntimeError` /
  `cuda.LogicError` / `MemoryError`.
Target: `-arch=sm_75` (RTX 20-series Turing), `BLOCK_SIZE = 256`.

---

## MCL GPU implementation

Custom PyCUDA kernels — no CuPy dependency.

Preprocessing (CPU, before any GPU work):
  - `_symmetrize_mcl(graph_csr, network_type)` — network-type aware
    undirected conversion + self-loops on every node (ensures
    irreducibility of the Markov chain).
      grn / mirna : `binarise(A + Aᵀ)` then add identity
      ppi         : graph used as-is then add identity
  - `_to_column_stochastic` — divides each column by its sum; zero-sum
    columns get a 1.0 placed on their diagonal (absorbing state).
  - `_to_ellpack_r` — present but UNUSED; biological networks have high
    row-length variance (hubs vs. leaves) and padding waste outweighs
    coalesced-access benefit.  CSR + CSC are the canonical pair.

Kernels (compiled once, cached in `_kernel_cache["mcl"]`):
  - `spgemm_row_chunk` — inner-product SpGEMM (CSR × CSC → COO).
      One block per output row; threads iterate over output columns;
      sparse dot via two-pointer merge of sorted index lists.
      atomicAdd counter on COO output with capacity guard.
      Chunked by row to bound VRAM growth from fill-in.
  - `threshold_prune` — mark entries < threshold (CPU compaction follows).
  - `topk_column_prune` — exact top-k per column with index tie-break;
      one block per column, O(col_len²) per column (fine because
      threshold prune ran first).
  - `inflate_values` — element-wise `powf(x, r)` in FP32.
  - `column_sum_segmented` — warp-per-column Σ via `__shfl_down_sync`.
  - `normalize_columns` — warp-per-column in-place divide; epsilon guard.
  - `convergence_frobenius` — partial ‖M_new − M_old‖_F² with FP64
      accumulation (FP32 inputs, FP64 partial sums); host sums + sqrt.

SpGEMM chunking strategy:
  - Free VRAM probed via `cuda.mem_get_info()`; 70 % budget for the COO
    output (12 bytes per entry: int + int + float).
  - `chunk_rows = max(1, vram_budget / (density × COO_BYTES))`.
  - Each chunk's COO capacity probed after kernel completion; overflow
    raises `MemoryError` with guidance to tighten pruning.
  - COO triples accumulated CPU-side across chunks → `coo_matrix.tocsr()`.

Pruning order: threshold first (kills bulk of fill-in cheaply), then
  top-k (operates on survivors in CSC view).  CPU compactions between
  steps via cumulative-sum row_ptr / col_ptr rebuild.

Inflation + normalisation: in-place on CSC.data — pattern unchanged,
  only values updated.

Convergence check:
  - Sparsity pattern must match (host array-equals indptr + indices)
  - On match: GPU FP64 Frobenius reduction → CPU `sqrt(sum(partials))`
  - On mismatch (early iterations): returns `inf`, iteration continues

Precision policy: FP32 for matrix data and arithmetic, FP64 for the
  convergence reduction only (detects small Δ near convergence without
  cancellation error).

CUDA streams:
  - `stream_compute`  — kernel execution
  - `stream_transfer` — async H2D for per-iteration uploads
                        (CSC view + matrix arrays)

Context handling: same pattern as HITS / Louvain —
  `retain_primary_context()` and unconditional push.

Cluster extraction (CPU, post-convergence): attractor method.  Nodes
  with `M[i, i] > 0` are attractors; every node j is assigned to
  `argmax_i M[i, j]`.  Fallback to weakly-connected components if no
  diagonal entry is positive.  Cluster IDs renumbered to compact
  `0..K-1`.

Does NOT silently fall back to CPU — raises `RuntimeError` /
  `cuda.LogicError` / `MemoryError`.
Target: `-arch=sm_75` (RTX 20-series Turing), `BLOCK_SIZE = 256`.

---

## PageRank GPU implementation

Custom PyCUDA kernels — no CuPy dependency.  Hybrid CSR + ELLPACK
storage for degree-aware access; scatter-form power iteration over the
original CSR (no transpose / Mᵀ build).

Kernels (compiled once, cached in `_kernel_cache["pagerank"]`):
  - `initialize_pr` — writes the teleport baseline `(1 − d) / N` into
    `PR_new`.  Must precede scatter because scatter atomicAdds INTO
    `PR_new` and assumes it starts at the teleport value.
  - `scatter_contributions_csr` — low/medium-degree source nodes via
    the original CSR arrays.  Three-tier degree-aware dispatch:
      LOW  (deg < 32)         : thread 0 only, serial scatter
      MED  (32 ≤ deg < 256)   : first warp, stride-32 scatter
      HIGH (deg ≥ 256)        : full block + 256-bucket SMEM hash
    HIGH-tier SMEM hash: open-addressing with linear probing, probe-cap
    = SMEM_BUCKETS, atomicCAS for slot claim + atomicAdd for value.
    Saturated chains fall back to direct global atomicAdd (correctness
    guaranteed).  After `__syncthreads()` each thread flushes its
    assigned bucket (SMEM_BUCKETS == BLOCK_SIZE).  Order-of-magnitude
    fewer global atomics per hub block.
  - `scatter_contributions_ellpack` — hub source nodes via padded
    ELLPACK arrays (one block per hub, threads stride through
    `max_row_len`; padding entries are `-1` and skipped).  Win vs. CSR:
    no `row_ptr` indirection, predictable stride.
  - `sum_dangling_pr` — block-partial Σ `PR_old[i]` over dangling nodes
    (host sums the per-block partials).
  - `distribute_dangling_mass` — adds `d · dangling_sum / |eligible|`
    via atomicAdd to every eligible node.  Structure-agnostic: the CPU
    side picks the eligible set per network type.
  - `compute_l1_convergence` — `Σ |PR_new[i] − PR_old[i]|` per block
    via warp shuffle reductions (intra-warp) + final 8-lane warp
    reduction in shared memory (FP32 throughout).

Hybrid storage split at HUB_THRESHOLD = 32:
  - 0 < out_degree < 32  → CSR scatter kernel (filtered `node_ids`)
  - out_degree ≥ 32      → ELLPACK scatter kernel
  - out_degree == 0      → dangling, handled by dangling kernels

Network-type-aware dangling redistribution:
  - GRN   : `eligible = where(out_degree > 0)` — regulators only.
            Preserves TF↔target asymmetry.
  - PPI   : `eligible = arange(n)` — uniform across all nodes.
  - miRNA : `eligible = where(out_degree > 0)` — miRNA nodes only
            (out_degree == 0 ⇒ gene target in a directed bipartite
            graph).  If a `node_index_map` with type labels were
            provided, it would override the out-degree proxy.
  Degenerate fallback (`eligible.size == 0`): logs warning, falls back
  to uniform.

Out-degree storage: FP32 weighted sum (`graph_csr.sum(axis=1)`) so
  weighted networks (PPI confidence scores, weighted GRN) are handled
  correctly; for binary graphs this collapses to the integer degree.

Result keys by network type (matches CLAUDE.md spec):
  - GRN   : `scores`, `top_regulators` (top-15, out>0),
            `top_targets` (top-15, out==0)
  - PPI   : `scores`, `top_nodes` (top-20 combined)
  - miRNA : `scores`, `top_mirnas` (top-15, out>0),
            `top_target_genes` (top-15, out==0)

Iteration structure:
  init → sum_dangling (D2H reduction) → scatter_csr → scatter_ellpack
  → distribute_dangling_mass → compute_l1_convergence (D2H reduction)
  → pointer swap of `PR_old` / `PR_new` (no data copy).  Implicit
  serialization within `stream_compute` orders the kernels correctly
  (scatter sees the post-init `PR_new`; convergence sees the final
  `PR_new` of the iteration).

CUDA streams:
  - `stream_compute`  — all six kernels
  - `stream_transfer` — async H2D during setup (CSR / ELLPACK /
    out_degree / dangling flags / eligible array / initial `PR_old`)

Context handling: same pattern as HITS / Louvain / MCL —
  `retain_primary_context()` and unconditional push.

VRAM check: `_estimate_pagerank_vram` is compared against
  `cuda.mem_get_info()`; raises `MemoryError` if the working set
  exceeds free VRAM.  Chunked fallback (`use_chunking=True` from
  `apply_config`) is currently a logged TODO — the regular path runs
  and `MemoryError` surfaces if allocation fails.

Does NOT silently fall back to CPU — raises `RuntimeError` /
  `cuda.LogicError` / `MemoryError`.
Target: `-arch=sm_75` (RTX 20-series Turing), `BLOCK_SIZE = 256`,
  `SMEM_BUCKETS = 256`, `HUB_THRESHOLD = 32`.

---

## RWR GPU implementation

Custom PyCUDA kernels — no CuPy dependency.  Fused SpMV + restart in a
single kernel; batched variant for multi-seed-set workloads.

Kernels (compiled once, cached in `_kernel_cache["rwr"]`):
  - `rwr_spmv_restart` — fused single-seed-set update
    `p_new[i] = (1 − r) · Σ_j W[i,j] · p[j] + r · p0[i]`.
    Three-tier degree-aware scheduling:
      LOW  (deg < 32)         : thread 0 only, serial scan
      MED  (32 ≤ deg < 256)   : first warp, stride-32 +
                                `__shfl_down_sync` reduction
      HIGH (deg ≥ 256)        : full block, stride-256 + shared-mem
                                tree reduction
    Restart term `r · p0[i]` added by **thread 0 only**, after the
    reduction, in the same write that stores `p_new[i]` — adding it
    from every thread would multiply the contribution.
    Uses `__ldg(&p[col])` on p-neighbour reads to route through the
    read-only texture cache (separate from L1 data; p is read many
    times per iteration but never written by this kernel).
  - `l1_convergence_rwr` — `Σ |p_new[i] − p[i]|` per block via warp
    shuffles + final 8-lane warp reduction in shared memory (same
    pattern as `pagerank.compute_l1_convergence`).
  - `rwr_spmv_restart_batched` — multi-seed batched kernel.
    `gridDim = (n, B, 1)`; one block per `(node_i, seed_b)` pair;
    `p` / `p0` / `p_new` stored as `[n × B]` row-major flat arrays
    (entry `(node, seed) = node * B + seed`).
    Used only when `1 < B <= MAX_BATCH = 4`; larger batches fall back
    to the serial-per-seed-set loop to avoid `O(n · B)` VRAM.

Transition matrix W construction (CPU, network-type aware):
  - GRN   : directed CSR as-is; `W = (D⁻¹ · A).T`.  Dangling columns
            (`out_degree == 0`) get a self-loop `W[j,j] = 1` so the
            absorbing state preserves probability mass during diffusion.
  - PPI   : `A_sym = A + Aᵀ`, binarised, then the same column-normalise
            + dangling-column self-loop fix.
  - miRNA : directed bipartite CSR; same treatment as GRN.  Gene-target
            nodes (out_degree == 0 in the bipartite digraph) get
            self-loops.
  Normalisation runs CPU-side (scipy sparse is well-optimised); the
  resulting FP32 CSR is transferred to GPU.

p₀ construction:
  - Non-empty seeds : `p0[seed] = 1 / |seeds|`, zero elsewhere.
  - Empty seeds     : `p0[i] = 1 / n` (uniform — global PageRank-like).
  Network type does not change the maths, only the semantics of the
  seed set (GRN seeds = TF indices; PPI seeds = disease proteins;
  miRNA seeds = miRNA node indices).

Multi-seed-set handling:
  - `seed_nodes = [1, 2, 3]`        → single seed set (`B = 1`)
  - `seed_nodes = [[1,2], [3,4]]`   → multiple seed sets (batched)
  - `B ≤ MAX_BATCH` (= 4)           → batched kernel
  - `B > MAX_BATCH`                  → serial-per-seed-set loop
  Per-node scores are averaged across all seed sets to form the
  primary score vector; `batch_results` field carries per-seed-set
  scores / iterations / converged when `B > 1`.

Optional node reordering (`reorder_nodes=False` by default):
  CPU permutes W rows/cols by ascending degree → similar-degree nodes
  adjacent in CSR → reduced warp divergence inside the three-tier
  dispatch.  Seed indices are remapped via the inverse permutation
  before GPU work; final scores remapped back via `scores_orig[perm] =
  scores_reord`.  Documented as a perf TODO; default path runs
  unreordered for correctness simplicity.

Result keys (per CLAUDE.md spec):
  `scores`, `top_nodes` (top-20), `top_seeds` (top-10 among the union
  of all seed indices, or `top_nodes[:10]` if no seeds), `iterations`,
  `converged`, `note`, and `batch_results` when `B > 1`.

CUDA streams:
  - `stream_compute`  — kernel execution (`spmv_restart`, `l1_conv`)
  - `stream_transfer` — async H2D of W CSR arrays and p₀ vectors
                        during setup
  Pointer swap of `p` / `p_new` each iteration (no data copy).

Context handling: same pattern as HITS / Louvain / MCL / PageRank —
  `retain_primary_context()` and unconditional push.

VRAM check: `_estimate_rwr_vram` compared against `cuda.mem_get_info()`;
  raises `MemoryError` if the working set exceeds free VRAM.
  `use_chunking=True` and `use_zero_copy=True` (from `apply_config`)
  are currently logged TODOs — the regular path runs and `MemoryError`
  surfaces if allocation fails.

Does NOT silently fall back to CPU — raises `RuntimeError` /
  `cuda.LogicError` / `MemoryError`.
Target: `-arch=sm_75` (RTX 20-series Turing), `BLOCK_SIZE = 256`,
  `MAX_BATCH = 4`.

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
