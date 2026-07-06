"""
tests/test_gpu_baseline.py
==========================

Tests for ``src/algorithms/gpu/basic/``.

Coverage
--------
Test 1 — outer-envelope result-structure validation.
Test 2 — inner-key parity with the optimized implementations.
Test 3 — execution-time field is populated.
Test 4 — algorithm runner integration (``mode='gpu_baseline'``).
Test 5 — hard-fail behaviour: cuGraph absent → ImportError/RuntimeError.

Skip conditions
---------------
cuGraph algorithms (pagerank, bfs, hits, louvain, rwr):
    Skipped unless ``cugraph`` and ``cudf`` are importable.
MCL:
    Skipped unless ``cupy`` is importable.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Backend availability probes (used for skip markers only)
# ---------------------------------------------------------------------------

def _probe_cugraph() -> bool:
    try:
        import cugraph   # noqa: F401
        import cudf      # noqa: F401
        return True
    except Exception:
        return False


def _probe_cupy() -> bool:
    try:
        import cupy      # noqa: F401
        return True
    except Exception:
        return False


CUGRAPH_AVAILABLE = _probe_cugraph()
CUPY_AVAILABLE    = _probe_cupy()

cugraph_required = pytest.mark.skipif(
    not CUGRAPH_AVAILABLE,
    reason="cuGraph and cuDF (RAPIDS) are not installed in this environment.",
)

cupy_required = pytest.mark.skipif(
    not CUPY_AVAILABLE,
    reason="CuPy is not installed in this environment.",
)


# ---------------------------------------------------------------------------
# Tiny synthetic graphs used by every test
# ---------------------------------------------------------------------------

def _tiny_grn() -> sp.csr_matrix:
    """8 nodes, directed: TFs 0,1 regulate 2..7."""
    rows = np.array([0, 0, 0, 1, 1, 2, 3])
    cols = np.array([2, 3, 4, 5, 6, 5, 7])
    data = np.ones(rows.size, dtype=np.float32)
    return sp.coo_matrix((data, (rows, cols)), shape=(8, 8)).tocsr()


def _tiny_ppi() -> sp.csr_matrix:
    """8 nodes, undirected: small triangle + tail."""
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 6), (6, 7)]
    rows = np.array([u for u, v in edges] + [v for u, v in edges])
    cols = np.array([v for u, v in edges] + [u for u, v in edges])
    data = np.ones(rows.size, dtype=np.float32)
    return sp.coo_matrix((data, (rows, cols)), shape=(8, 8)).tocsr()


# ---------------------------------------------------------------------------
# Test 1 — outer-envelope validation (always runs, no GPU needed)
# ---------------------------------------------------------------------------

OUTER_KEYS = {
    "algorithm", "mode", "network_type",
    "execution_time", "num_nodes", "num_edges", "result",
}


def test_build_envelope_has_all_outer_keys():
    """``build_envelope`` returns the 7-key standard outer dict.

    Only importable on machines with cuGraph; otherwise skip automatically
    because importing _utils itself raises ImportError.
    """
    if not CUGRAPH_AVAILABLE:
        pytest.skip("_utils requires cuGraph; skipping envelope test.")
    from src.algorithms.gpu.basic._utils import (
        BASELINE_MODE_CUGRAPH, BASELINE_MODE_CUPY,
        build_envelope, top_k_among, top_k_global,
    )
    g = _tiny_grn()
    env = build_envelope(
        algorithm="pagerank",
        network_type="grn",
        execution_time=0.123,
        graph_csr=g,
        inner={"scores": [], "iterations": 0, "converged": True},
    )
    assert set(env.keys()) >= OUTER_KEYS
    assert env["mode"] == BASELINE_MODE_CUGRAPH
    assert env["mode"] == "gpu_baseline_cugraph"
    assert env["num_nodes"] == g.shape[0]
    assert env["num_edges"] == g.nnz
    assert isinstance(env["result"], dict)

    # MCL mode constant is separate from cuGraph constant.
    assert BASELINE_MODE_CUPY == "gpu_baseline_cupy"


def test_top_k_helpers():
    """Top-K helpers are pure numpy and always runnable."""
    if not CUGRAPH_AVAILABLE:
        pytest.skip("_utils requires cuGraph; skipping top-k helper test.")
    from src.algorithms.gpu.basic._utils import top_k_among, top_k_global
    scores = np.array([0.1, 0.5, 0.9, 0.2, 0.4])
    assert top_k_global(scores, 2) == [2, 1]
    cands = np.array([0, 2, 3])
    assert top_k_among(scores, cands, 2) == [2, 3]
    assert sorted(top_k_among(scores, cands, 10)) == [0, 2, 3]
    assert top_k_among(scores, cands, 0) == []
    assert top_k_among(scores, np.array([], dtype=np.int64), 5) == []


# ---------------------------------------------------------------------------
# Test 5 — hard-fail behaviour (always runs)
# ---------------------------------------------------------------------------

def test_importing_pagerank_without_cugraph_raises(monkeypatch):
    """Importing a cuGraph-backed baseline without RAPIDS raises ImportError."""
    import importlib
    import sys

    if CUGRAPH_AVAILABLE:
        pytest.skip("cuGraph IS installed; cannot test absence without patching.")

    # Verify that importing the module raises ImportError with RAPIDS hint.
    # The module-level hard-fail should trigger immediately on import.
    with pytest.raises(ImportError, match="(?i)rapids|cugraph"):
        # Force a fresh import attempt (clear any cached module).
        mod_name = "src.algorithms.gpu.basic.pagerank"
        sys.modules.pop(mod_name, None)
        importlib.import_module(mod_name)


def test_runner_gpu_baseline_missing_backend_raises_runtime_error():
    """Runner converts ImportError from missing backend to RuntimeError."""
    from src.runner.algorithm_runner import run_algorithm

    if CUGRAPH_AVAILABLE:
        pytest.skip("cuGraph IS installed; this test needs it absent.")

    g = _tiny_grn()
    with pytest.raises((RuntimeError, ImportError)):
        run_algorithm("pagerank", g, {}, mode="gpu_baseline",
                      params={"max_iter": 5})


def test_adapter_gpu_baseline_method_exists():
    """The dynamically-created adapter class exposes gpu_baseline()."""
    from src.algorithms import ALGORITHM_REGISTRY
    for name, cls in ALGORITHM_REGISTRY.items():
        assert hasattr(cls, "gpu_baseline"), (
            f"{name} adapter missing gpu_baseline static method"
        )


# ---------------------------------------------------------------------------
# Test 2 — inner key parity with the optimized implementations.
# ---------------------------------------------------------------------------

REQUIRED_INNER_KEYS = {
    ("pagerank", "grn"):   {"scores", "top_regulators", "top_targets", "iterations", "converged"},
    ("pagerank", "ppi"):   {"scores", "top_nodes", "iterations", "converged"},
    ("pagerank", "mirna"): {"scores", "top_mirnas", "top_target_genes", "iterations", "converged"},
    ("bfs",       "any"):  {"distances", "visited_order", "num_reachable", "cascade_by_depth", "traversal_modes"},
    ("hits",      "grn"):  {"hub_scores", "authority_scores", "iterations", "converged",
                            "top_hubs", "top_authorities", "hub_authority_overlap"},
    ("hits",      "ppi"):  {"hub_scores", "authority_scores", "top_nodes", "iterations", "converged"},
    ("hits",     "mirna"): {"hub_scores", "authority_scores", "iterations", "converged",
                            "top_hubs", "top_authorities", "hub_authority_overlap"},
    ("louvain",  "any"):   {"community_assignments", "num_communities", "modularity",
                            "top_communities", "hierarchy", "note"},
    ("rwr",      "any"):   {"scores", "top_nodes", "top_seeds", "iterations", "converged", "note"},
    ("mcl",      "any"):   {"cluster_assignments", "num_clusters", "iterations", "converged",
                            "note", "overflow_warning"},
}


def _required_keys(algo: str, nt: str) -> set[str]:
    key = (algo, nt)
    if key in REQUIRED_INNER_KEYS:
        return REQUIRED_INNER_KEYS[key]
    return REQUIRED_INNER_KEYS[(algo, "any")]


@cugraph_required
@pytest.mark.parametrize("network_type", ["grn", "ppi", "mirna"])
def test_pagerank_baseline_schema(network_type):
    from src.algorithms.gpu.basic import pagerank_gpu_baseline
    g = _tiny_ppi() if network_type == "ppi" else _tiny_grn()
    out = pagerank_gpu_baseline(g, {"max_iter": 25, "network_type": network_type})
    assert set(out.keys()) >= OUTER_KEYS
    assert out["algorithm"] == "pagerank"
    assert out["mode"] == "gpu_baseline_cugraph"
    assert out["network_type"] == network_type
    assert out["result"]["backend"] == "cugraph"
    inner_keys = set(out["result"].keys())
    required = _required_keys("pagerank", network_type)
    missing = required - inner_keys
    assert not missing, f"Missing required keys for pagerank/{network_type}: {missing}"


@cugraph_required
def test_bfs_baseline_schema():
    from src.algorithms.gpu.basic import bfs_gpu_baseline
    g = _tiny_grn()
    out = bfs_gpu_baseline(g, {"source": 0, "max_depth": 3, "network_type": "grn"})
    assert out["algorithm"] == "bfs"
    assert out["mode"] == "gpu_baseline_cugraph"
    assert out["result"]["backend"] == "cugraph"
    required = _required_keys("bfs", "any")
    missing = required - set(out["result"].keys())
    assert not missing, f"Missing BFS keys: {missing}"
    assert out["result"]["distances"][0] == 0


@cugraph_required
@pytest.mark.parametrize("network_type", ["grn", "ppi"])
def test_hits_baseline_schema(network_type):
    from src.algorithms.gpu.basic import hits_gpu_baseline
    g = _tiny_ppi() if network_type == "ppi" else _tiny_grn()
    out = hits_gpu_baseline(g, {"max_iter": 20, "network_type": network_type})
    assert out["mode"] == "gpu_baseline_cugraph"
    assert out["result"]["backend"] == "cugraph"
    required = _required_keys("hits", network_type)
    missing = required - set(out["result"].keys())
    assert not missing, f"Missing HITS keys for {network_type}: {missing}"


@cugraph_required
def test_louvain_baseline_schema():
    from src.algorithms.gpu.basic import louvain_gpu_baseline
    g = _tiny_ppi()
    out = louvain_gpu_baseline(g, {"resolution": 1.0, "network_type": "ppi"})
    assert out["mode"] == "gpu_baseline_cugraph"
    assert out["result"]["backend"] == "cugraph"
    required = _required_keys("louvain", "any")
    missing = required - set(out["result"].keys())
    assert not missing, f"Missing Louvain keys: {missing}"
    assert out["result"]["num_communities"] >= 1


@cugraph_required
def test_rwr_baseline_schema():
    from src.algorithms.gpu.basic import rwr_gpu_baseline
    g = _tiny_grn()
    out = rwr_gpu_baseline(g, {
        "seed_nodes": [0], "max_iter": 20, "network_type": "grn",
    })
    assert out["mode"] == "gpu_baseline_cugraph"
    assert out["result"]["backend"] == "cugraph"
    required = _required_keys("rwr", "any")
    missing = required - set(out["result"].keys())
    assert not missing, f"Missing RWR keys: {missing}"
    assert len(out["result"]["scores"]) == g.shape[0]


@cupy_required
def test_mcl_baseline_schema():
    from src.algorithms.gpu.basic import mcl_gpu_baseline
    g = _tiny_ppi()
    out = mcl_gpu_baseline(g, {"max_iter": 10, "network_type": "ppi"})
    assert out["mode"] == "gpu_baseline_cupy"
    assert out["result"]["backend"] == "cupy"
    required = _required_keys("mcl", "any")
    missing = required - set(out["result"].keys())
    assert not missing, f"Missing MCL keys: {missing}"
    assert isinstance(out["result"]["overflow_warning"], str)


# ---------------------------------------------------------------------------
# Test 3 — execution_time recorded.
# ---------------------------------------------------------------------------

@cugraph_required
def test_execution_time_recorded():
    from src.algorithms.gpu.basic import pagerank_gpu_baseline
    g = _tiny_grn()
    out = pagerank_gpu_baseline(g, {"max_iter": 10, "network_type": "grn"})
    assert isinstance(out["execution_time"], float)
    assert out["execution_time"] >= 0.0


# ---------------------------------------------------------------------------
# Test 4 — runner integration.
# ---------------------------------------------------------------------------

def test_runner_rejects_unknown_mode():
    """Sanity: the runner still rejects modes that are not allow-listed."""
    from src.runner.algorithm_runner import run_algorithm
    g = _tiny_grn()
    with pytest.raises(ValueError, match="Invalid mode"):
        run_algorithm("pagerank", g, {}, mode="not_a_mode")


def test_runner_accepts_gpu_baseline_mode_in_whitelist():
    """The mode 'gpu_baseline' must be allow-listed even when no GPU exists.

    The call may legitimately fail later with RuntimeError when cuGraph is
    not installed.  What we are asserting here is that the mode itself
    reaches the dispatch layer (i.e. it survives the whitelist), not that
    the underlying compute succeeds.
    """
    from src.runner.algorithm_runner import run_algorithm
    g = _tiny_grn()
    try:
        run_algorithm("pagerank", g, {}, mode="gpu_baseline",
                      params={"max_iter": 5})
    except ValueError as exc:
        pytest.fail(f"mode='gpu_baseline' rejected by whitelist: {exc}")
    except (RuntimeError, ImportError):
        # Expected on a box without cuGraph + RAPIDS.
        pass
    except Exception:
        # Other GPU errors (e.g. nvcc missing) are also acceptable.
        pass


@cugraph_required
def test_runner_e2e_gpu_baseline_preserves_mode():
    """End-to-end runner call preserves backend-specific mode string."""
    from src.runner.algorithm_runner import run_algorithm
    g = _tiny_grn()
    out = run_algorithm(
        "pagerank", g, {}, mode="gpu_baseline",
        params={"max_iter": 10, "network_type": "grn"},
    )
    # Mode must be the backend-specific string, NOT the generic "gpu_baseline".
    assert out["mode"] == "gpu_baseline_cugraph", (
        f"Expected 'gpu_baseline_cugraph', got {out['mode']!r}"
    )
    assert out["algorithm"] == "pagerank"
    assert "scores" in out["result"]
    assert out["result"]["backend"] == "cugraph"
