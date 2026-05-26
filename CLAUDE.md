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
