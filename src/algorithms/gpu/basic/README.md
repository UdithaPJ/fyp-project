# `src/algorithms/gpu/basic/` — GPU Baseline Implementations

## Purpose

Simple GPU reference implementations used as a **benchmarking baseline** against
the heavily tuned implementations in `src/algorithms/gpu/cuda_optimized/`.

Each module exposes one function:

```python
<algorithm>_gpu_baseline(graph_csr: sp.csr_matrix, params: dict) -> dict
```

---

## Backend

### cuGraph algorithms (RAPIDS required)

| Algorithm | cuGraph API used |
|-----------|-----------------|
| `pagerank`| `cugraph.pagerank` |
| `bfs`     | `cugraph.bfs` |
| `hits`    | `cugraph.hits` |
| `louvain` | `cugraph.louvain` |
| `rwr`     | `cugraph.personalized_pagerank` |

**Hard-fail policy:** every one of these modules raises `ImportError` at import
time if cuGraph or cuDF is not installed.  There is no silent CuPy fallback.

Install RAPIDS:
```bash
conda install -c rapidsai -c nvidia -c conda-forge \
    rapids=24.02 python=3.10 cudatoolkit=11.8
```

### CuPy algorithm (cuGraph has no MCL)

| Algorithm | Backend |
|-----------|---------|
| `mcl`     | CuPy sparse matrix powers + element-wise inflation |

MCL does **not** import from `_utils.py` (which requires cuGraph), so it
remains independently importable on a CuPy-only installation.

---

## Benchmarking consistency

### Why hard-fail instead of CuPy fallback?

The previous version silently fell back to CuPy when cuGraph was unavailable.
This caused **benchmarking inconsistency**: two runs on different machines
would use different backends, producing incomparable timings, while both
reporting `"mode": "gpu_baseline"`.

The hard-fail policy guarantees:
- Every `gpu_baseline_cugraph` result came from the cuGraph backend.
- Every `gpu_baseline_cupy` result came from the CuPy backend (MCL only).
- No run silently degrades to a different compute path.

---

## Result mode strings

| Backend  | `result["mode"]`         | `result["result"]["backend"]` |
|----------|--------------------------|-------------------------------|
| cuGraph  | `"gpu_baseline_cugraph"` | `"cugraph"` |
| CuPy     | `"gpu_baseline_cupy"`    | `"cupy"` |

---

## Result schema

Inner result keys **exactly match** the corresponding `cuda_optimized`
per-network-type schema (verified by `tests/test_gpu_baseline.py`).  All
baselines additionally include an additive `result["result"]["backend"]` key
for benchmarking provenance.

---

## Rules

- **Never** add custom kernels, warp primitives, SMEM tricks, graph reordering,
  hybrid push/pull, chunking, or memory-aware execution here.  Those belong
  exclusively to `src/algorithms/gpu/cuda_optimized/`.
- **Never** import the baseline modules from the web application.
- **Never** hardcode a cuGraph API signature — use `cugraph_function(name)` +
  `inspect.signature` to detect at runtime.
- **Never** add a CuPy fallback to a cuGraph algorithm file.  Raise
  `RuntimeError` if the cuGraph API is absent in the installed RAPIDS version.
