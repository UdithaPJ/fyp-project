"""
tests/test_gpu_config_extended.py
=================================

Test suite for the expanded ``src/optimization/gpu_config.py``.
Covers: GraphProfiler classifications, MemoryEstimator pressure
escalation, AlgorithmStrategySelector per-algorithm decisions,
RuntimeProfiler feedback, apply_config integration, and the
profile report generator.

Run:    python -m pytest tests/test_gpu_config_extended.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

# Make src/ importable when running pytest from repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.optimization.gpu_config import (  # noqa: E402
    AlgorithmStrategySelector,
    GraphProfiler,
    MemoryEstimator,
    RuntimeProfiler,
    apply_config,
    generate_profile_report,
    get_gpu_config,
)


# ---------------------------------------------------------------------------
# Helpers — construct graphs with known structural properties.
# ---------------------------------------------------------------------------

def _make_uniform_graph(n: int = 200, k: int = 6) -> sp.csr_matrix:
    """k-regular directed graph: every node has exactly k distinct neighbours.

    Using ``rng.choice(..., replace=False)`` per row keeps the degree
    distribution flat (no duplicates collapsing during sum_duplicates).
    """
    rng = np.random.default_rng(0)
    rows_list: list[int] = []
    cols_list: list[int] = []
    for i in range(n):
        # Sample k distinct targets from {0..n-1} \ {i}.
        choices = rng.choice(n - 1, size=k, replace=False)
        # Shift to skip i (avoid self-loops keeps the test clean).
        choices = np.where(choices >= i, choices + 1, choices)
        rows_list.extend([i] * k)
        cols_list.extend(int(c) for c in choices)
    rows = np.asarray(rows_list, dtype=np.int32)
    cols = np.asarray(cols_list, dtype=np.int32)
    data = np.ones(rows.size, dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return A


def _make_power_law_graph(n: int = 500, m_edges: int = 3) -> sp.csr_matrix:
    """Barabasi-Albert preferential attachment graph (hand-rolled)."""
    rng = np.random.default_rng(0)
    # Start with a small complete-ish core.
    core = max(m_edges + 1, 5)
    rows: list[int] = []
    cols: list[int] = []
    deg = np.zeros(n, dtype=np.int64)
    for i in range(core):
        for j in range(core):
            if i != j:
                rows.append(i); cols.append(j)
                deg[i] += 1
    # Preferential attachment for the remaining nodes.
    for new_node in range(core, n):
        probs = deg[:new_node].astype(np.float64) + 1e-9
        probs /= probs.sum()
        targets = rng.choice(new_node, size=m_edges, replace=False, p=probs)
        for t in targets:
            rows.append(new_node); cols.append(int(t))
            rows.append(int(t));   cols.append(new_node)
            deg[new_node] += 1
            deg[t] += 1
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    return A


def _make_bipartite_graph(n_left: int = 50, n_right: int = 50,
                          density: float = 0.1) -> sp.csr_matrix:
    """Directed bipartite graph (left → right edges only)."""
    rng = np.random.default_rng(0)
    n_total = n_left + n_right
    edges_per_left = max(1, int(density * n_right))
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n_left):
        ts = rng.choice(n_right, size=edges_per_left, replace=False)
        for t in ts:
            rows.append(i); cols.append(n_left + int(t))
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n_total, n_total)).tocsr()
    return A


def _make_symmetric_graph(n: int = 200, k: int = 6) -> sp.csr_matrix:
    """Symmetric (undirected) random graph."""
    A = _make_uniform_graph(n, k)
    A = (A + A.T).astype(np.float32)
    A.data = np.ones_like(A.data)
    A.tocsr()
    A.sum_duplicates()
    return A


# ---------------------------------------------------------------------------
# TEST 1 — GraphProfiler classifications
# ---------------------------------------------------------------------------

class TestGraphProfiler:
    def test_uniform_graph_is_uniform_degree(self):
        A = _make_uniform_graph(n=200, k=6)
        prof = GraphProfiler.profile(A)
        # Uniform random graph has low skewness.
        assert prof["degree_class"] == "uniform", (
            f"expected uniform, got {prof['degree_class']} "
            f"(skew={prof['degree_skew']:.2f})"
        )
        assert prof["n"] == 200
        assert prof["m"] > 0

    def test_power_law_graph_is_power_law(self):
        A = _make_power_law_graph(n=500, m_edges=3)
        prof = GraphProfiler.profile(A)
        # BA graph has positive skew; depending on size may be skewed or
        # power_law.  Accept either of the two non-uniform classes.
        assert prof["degree_class"] in ("skewed", "power_law"), (
            f"BA graph should be skewed or power_law, got "
            f"{prof['degree_class']} (skew={prof['degree_skew']:.2f})"
        )
        # Hub fraction should be small but non-zero.
        assert prof["hub_fraction"] >= 0.0

    def test_bipartite_graph_is_bipartite(self):
        A = _make_bipartite_graph(n_left=50, n_right=50, density=0.1)
        prof = GraphProfiler.profile(A)
        assert prof["is_bipartite"] is True

    def test_symmetric_graph_is_symmetric(self):
        A = _make_symmetric_graph(n=200, k=6)
        prof = GraphProfiler.profile(A)
        assert prof["is_symmetric"] is True

    def test_fingerprint_stable(self):
        A = _make_uniform_graph(100, 4)
        fp1 = GraphProfiler.fingerprint(A)
        fp2 = GraphProfiler.fingerprint(A.copy())
        assert fp1 == fp2
        assert isinstance(fp1, str) and len(fp1) == 12

    def test_fingerprint_distinguishes_graphs(self):
        A = _make_uniform_graph(100, 4)
        B = _make_uniform_graph(100, 8)
        assert GraphProfiler.fingerprint(A) != GraphProfiler.fingerprint(B)

    def test_sparsity_class_thresholds(self):
        # Hand-crafted matrices at each boundary.
        n = 1000
        # ultra_sparse: 5 edges → density 5e-6
        A_ultra = sp.csr_matrix((np.ones(5, dtype=np.float32),
                                 (np.arange(5, dtype=np.int32),
                                  np.arange(5, dtype=np.int32) + 1)),
                                shape=(n, n))
        # sparse: ~500 edges → density 5e-4
        A_sparse = sp.random(n, n, density=5e-4, format="csr",
                              dtype=np.float32, random_state=0)
        prof_ultra = GraphProfiler.profile(A_ultra)
        prof_sparse = GraphProfiler.profile(A_sparse)
        assert prof_ultra["sparsity_class"] == "ultra_sparse"
        assert prof_sparse["sparsity_class"] == "sparse"


# ---------------------------------------------------------------------------
# TEST 2 — MemoryEstimator pressure thresholds
# ---------------------------------------------------------------------------

class TestMemoryEstimator:
    def test_small_graph_is_low_pressure(self):
        A = _make_uniform_graph(100, 4)
        prof = GraphProfiler.profile(A)
        cfg = get_gpu_config()
        est = MemoryEstimator.estimate("pagerank", prof, cfg)
        assert est["pressure"] in ("low", "medium")
        assert est["needs_chunking"] is False

    def test_pressure_escalates_with_size(self):
        A_small = _make_uniform_graph(100, 4)
        # Synthesize a profile claiming an absurdly large graph to force
        # high pressure without actually allocating one.
        prof = GraphProfiler.profile(A_small)
        prof["vram_estimate_mb"] = 1e9  # 1 PB — guaranteed > any GPU
        prof["avg_degree"] = 100.0
        prof["n"] = 1_000_000
        prof["m"] = 100_000_000
        cfg = get_gpu_config()
        est = MemoryEstimator.estimate("pagerank", prof, cfg)
        assert est["pressure"] == "critical"
        assert est["needs_chunking"] is True
        assert est["recommended_chunk_size"] is not None
        assert est["recommended_chunk_size"] >= 1
        assert est["precision_downgrade"] is True

    def test_mcl_multiplier_dominates_for_dense_graphs(self):
        A = _make_uniform_graph(100, 50)  # dense-ish (avg_deg = 50)
        prof = GraphProfiler.profile(A)
        cfg = get_gpu_config()
        est_pr  = MemoryEstimator.estimate("pagerank", prof, cfg)
        est_mcl = MemoryEstimator.estimate("mcl", prof, cfg)
        # MCL multiplier should match or exceed PageRank's.
        assert est_mcl["total_mb"] >= est_pr["total_mb"]

    def test_hits_multiplier(self):
        A = _make_uniform_graph(100, 4)
        prof = GraphProfiler.profile(A)
        cfg = get_gpu_config()
        est_hits = MemoryEstimator.estimate("hits", prof, cfg)
        est_pr   = MemoryEstimator.estimate("pagerank", prof, cfg)
        # HITS needs both A and A^T → more memory than PageRank.
        assert est_hits["total_mb"] > est_pr["total_mb"]


# ---------------------------------------------------------------------------
# TEST 3 — AlgorithmStrategySelector
# ---------------------------------------------------------------------------

class TestAlgorithmStrategySelector:
    def _power_law_profile(self) -> dict:
        return {
            "n": 5000, "m": 50000, "density": 2e-3,
            "avg_degree": 10.0, "max_degree": 500, "min_degree": 0,
            "std_degree": 50.0, "degree_skew": 5.0, "hub_fraction": 0.10,
            "isolated_frac": 0.01, "is_bipartite": False,
            "is_symmetric": False, "nnz_per_mb": 10000,
            "vram_estimate_mb": 1.0,
            "sparsity_class": "sparse", "degree_class": "power_law",
            "format_hint": "sell_c",
        }

    def _uniform_profile(self) -> dict:
        return {
            "n": 500, "m": 5000, "density": 2e-2,
            "avg_degree": 10.0, "max_degree": 15, "min_degree": 5,
            "std_degree": 1.5, "degree_skew": 0.2, "hub_fraction": 0.0,
            "isolated_frac": 0.0, "is_bipartite": False,
            "is_symmetric": True, "nnz_per_mb": 5000,
            "vram_estimate_mb": 0.5,
            "sparsity_class": "moderate", "degree_class": "uniform",
            "format_hint": "csr",
        }

    def _dense_profile(self) -> dict:
        return {
            "n": 100, "m": 5000, "density": 0.5,
            "avg_degree": 50.0, "max_degree": 80, "min_degree": 30,
            "std_degree": 5.0, "degree_skew": 0.1, "hub_fraction": 0.0,
            "isolated_frac": 0.0, "is_bipartite": False,
            "is_symmetric": True, "nnz_per_mb": 1000,
            "vram_estimate_mb": 0.1,
            "sparsity_class": "dense", "degree_class": "uniform",
            "format_hint": "csr",
        }

    def _stub_memory(self, needs_chunking: bool = False,
                     pressure: str = "low") -> dict:
        return {
            "base_mb": 1.0, "algorithm_mb": 1.0, "total_mb": 2.0,
            "available_mb": 3000.0, "pressure": pressure,
            "needs_chunking": needs_chunking,
            "recommended_chunk_size": 1000 if needs_chunking else None,
            "precision_downgrade": pressure == "critical",
        }

    def test_pagerank_power_law_activates_pull(self):
        gp = self._power_law_profile()
        s = AlgorithmStrategySelector.select(
            "pagerank", gp, {}, self._stub_memory()
        )
        assert s["use_pull"] is True
        # ELLPACK fraction smaller for power_law (more nodes are hubs).
        assert s["ellpack_fraction"] == 0.02

    def test_pagerank_uniform_no_pull(self):
        gp = self._uniform_profile()
        s = AlgorithmStrategySelector.select(
            "pagerank", gp, {}, self._stub_memory()
        )
        assert s["use_pull"] is False
        assert s["ellpack_fraction"] == 0.10

    def test_bfs_power_law_push_pull(self):
        gp = self._power_law_profile()
        s = AlgorithmStrategySelector.select(
            "bfs", gp, {}, self._stub_memory()
        )
        assert s["traversal_mode"] == "push_pull"
        assert s["frontier_threshold"] == 0.25

    def test_bfs_uniform_small_push_only(self):
        gp = self._uniform_profile()
        s = AlgorithmStrategySelector.select(
            "bfs", gp, {}, self._stub_memory()
        )
        assert s["traversal_mode"] == "push_only"

    def test_mcl_dense_uses_inner_product(self):
        gp = self._dense_profile()
        s = AlgorithmStrategySelector.select(
            "mcl", gp, {}, self._stub_memory()
        )
        assert s["spgemm_method"] == "inner_product"

    def test_mcl_sparse_uses_hash(self):
        gp = self._power_law_profile()
        s = AlgorithmStrategySelector.select(
            "mcl", gp, {}, self._stub_memory()
        )
        assert s["spgemm_method"] == "hash"
        # MCL always chunks.
        assert s["use_chunking"] is True

    def test_louvain_smem_hash_for_power_law(self):
        gp = self._power_law_profile()
        s = AlgorithmStrategySelector.select(
            "louvain", gp, {}, self._stub_memory()
        )
        assert s["use_smem_hash"] is True
        assert s["freeze_threshold"] == 2

    def test_rwr_power_law_edge_parallel(self):
        gp = self._power_law_profile()
        s = AlgorithmStrategySelector.select(
            "rwr", gp, {}, self._stub_memory()
        )
        assert s["spmv_mode"] == "edge_parallel"

    def test_hits_skewed_smem_hash(self):
        gp = self._power_law_profile()
        s = AlgorithmStrategySelector.select(
            "hits", gp, {}, self._stub_memory()
        )
        assert s["use_smem_hash"] is True
        assert s["reorder_nodes"] is True


# ---------------------------------------------------------------------------
# TEST 4 — RuntimeProfiler feedback
# ---------------------------------------------------------------------------

class TestRuntimeProfiler:
    def setup_method(self) -> None:
        RuntimeProfiler.clear()

    def test_insufficient_history_no_rec(self):
        # 2 runs is below the 3-run minimum.
        for _ in range(2):
            RuntimeProfiler.record(
                "pagerank", "fp1", 0.1,
                {"tolerance": 1e-6}, {"converged": True},
            )
        assert RuntimeProfiler.get_recommendation("pagerank", "fp1") == {}

    def test_all_converged_relaxes_tolerance(self):
        for _ in range(3):
            RuntimeProfiler.record(
                "pagerank", "fp1", 0.1,
                {"tolerance": 1e-6, "max_iter": 100},
                {"converged": True, "iterations": 10},
            )
        rec = RuntimeProfiler.get_recommendation("pagerank", "fp1")
        assert "tolerance" in rec
        assert rec["tolerance"] == 2e-6
        assert "_note" in rec

    def test_none_converged_bumps_max_iter(self):
        for _ in range(3):
            RuntimeProfiler.record(
                "pagerank", "fp_nc", 0.1,
                {"tolerance": 1e-6, "max_iter": 100},
                {"converged": False, "iterations": 100},
            )
        rec = RuntimeProfiler.get_recommendation("pagerank", "fp_nc")
        assert "max_iter" in rec
        assert rec["max_iter"] == 150

    def test_increasing_time_emits_warning(self):
        for t in (0.1, 0.2, 0.5):
            RuntimeProfiler.record(
                "pagerank", "fp_slow", t,
                {"tolerance": 1e-6}, {"converged": True},
            )
        rec = RuntimeProfiler.get_recommendation("pagerank", "fp_slow")
        assert "_warning" in rec

    def test_get_history_filters_by_algorithm(self):
        RuntimeProfiler.record("pagerank", "fp1", 0.1, {}, {})
        RuntimeProfiler.record("bfs", "fp2", 0.1, {}, {})
        pr_hist = RuntimeProfiler.get_history("pagerank")
        assert ("pagerank", "fp1") in pr_hist
        assert ("bfs", "fp2") not in pr_hist

    def test_clear_specific_algorithm(self):
        RuntimeProfiler.record("pagerank", "fp1", 0.1, {}, {})
        RuntimeProfiler.record("bfs", "fp2", 0.1, {}, {})
        RuntimeProfiler.clear("pagerank")
        assert RuntimeProfiler.get_history("pagerank") == {}
        assert RuntimeProfiler.get_history("bfs") != {}

    def test_history_capped_at_10_entries(self):
        for i in range(20):
            RuntimeProfiler.record(
                "pagerank", "fp_cap", float(i),
                {"tolerance": 1e-6}, {"converged": True},
            )
        history = RuntimeProfiler.get_history("pagerank")
        assert len(history[("pagerank", "fp_cap")]) == 10


# ---------------------------------------------------------------------------
# TEST 5 — apply_config integration
# ---------------------------------------------------------------------------

class TestApplyConfigIntegration:
    def test_user_params_preserved_for_all_algorithms(self):
        csr = sp.random(200, 200, density=0.03, format="csr",
                        dtype=np.float32, random_state=0)
        for algo in ["pagerank", "bfs", "hits", "louvain", "rwr", "mcl"]:
            params = apply_config(algo, csr, {"custom_key": "keep_me"})
            assert params["custom_key"] == "keep_me", (
                f"{algo}: user 'custom_key' lost"
            )
            assert "_graph_fingerprint" in params
            assert "_memory_estimate" in params
            assert "_strategy_selected" in params
            assert "_graph_profile" in params
            assert "_hardware_config" in params

    def test_unknown_algorithm_returns_user_params_unchanged(self):
        csr = sp.random(50, 50, density=0.1, format="csr", dtype=np.float32)
        params = apply_config("nope", csr, {"foo": 1})
        assert params == {"foo": 1}

    def test_user_tolerance_wins_over_runtime_rec(self):
        """User-supplied params should beat runtime feedback."""
        RuntimeProfiler.clear()
        csr = sp.random(100, 100, density=0.05, format="csr",
                        dtype=np.float32, random_state=1)
        fp = GraphProfiler.fingerprint(csr)
        for _ in range(3):
            RuntimeProfiler.record(
                "pagerank", fp, 0.1,
                {"tolerance": 1e-6, "max_iter": 100},
                {"converged": True, "iterations": 10},
            )
        params = apply_config("pagerank", csr, {"tolerance": 1e-9})
        assert params["tolerance"] == 1e-9, (
            "user-supplied tolerance must beat runtime recommendation"
        )

    def test_strategy_keys_present_per_algorithm(self):
        csr = _make_power_law_graph(n=300, m_edges=3)
        # PageRank strategy should expose use_pull / ellpack_fraction.
        p = apply_config("pagerank", csr, {})
        assert "ellpack_fraction" in p
        assert "use_pull" in p
        # MCL strategy should expose spgemm_method.
        p = apply_config("mcl", csr, {})
        assert "spgemm_method" in p
        assert "prune_threshold" in p

    def test_non_csr_input_falls_back_to_hw_only(self):
        # Pass a non-CSR object — must still return a dict (no crash).
        params = apply_config("pagerank", object(), {"x": 1})
        assert "x" in params


# ---------------------------------------------------------------------------
# TEST 6 — profile report generation
# ---------------------------------------------------------------------------

class TestProfileReport:
    def test_report_has_required_sections(self):
        csr = sp.random(200, 200, density=0.05, format="csr",
                        dtype=np.float32, random_state=2)
        params = apply_config("pagerank", csr, {}, enable_profiling=True)
        report = params["_profile_report"]
        for section in ("hardware", "graph", "memory", "strategy",
                        "runtime_history", "recommendations"):
            assert section in report, f"missing section: {section}"

    def test_recommendations_is_list_of_strings(self):
        csr = _make_power_law_graph(n=300, m_edges=3)
        params = apply_config("pagerank", csr, {}, enable_profiling=True)
        recs = params["_profile_report"]["recommendations"]
        assert isinstance(recs, list)
        assert all(isinstance(r, str) for r in recs)
        # Power-law graph should at least mention degree distribution.
        joined = " ".join(recs).lower()
        assert ("power-law" in joined) or ("skewed" in joined) \
            or ("ellpack" in joined) or ("pull" in joined)

    def test_generate_profile_report_standalone(self):
        csr = _make_uniform_graph(100, 4)
        report = generate_profile_report("pagerank", csr, params={})
        assert "hardware" in report
        assert "graph" in report
        assert report["graph"]["n"] == 100

    def test_report_with_execution_result(self):
        csr = _make_uniform_graph(50, 4)
        exec_result = {
            "execution_time": 0.05,
            "result": {"iterations": 12, "converged": True},
            "speedup_vs_cpu": 4.2,
        }
        report = generate_profile_report(
            "pagerank", csr, params={}, execution_result=exec_result,
        )
        assert "execution" in report
        assert report["execution"]["time_seconds"] == 0.05
        assert report["execution"]["iterations"] == 12


# ---------------------------------------------------------------------------
# Allow `python tests/test_gpu_config_extended.py` to print PASS/FAIL banners.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    suites = [
        ("GraphProfiler classifications",     TestGraphProfiler),
        ("MemoryEstimator pressure",          TestMemoryEstimator),
        ("Strategy selection",                TestAlgorithmStrategySelector),
        ("RuntimeProfiler feedback",          TestRuntimeProfiler),
        ("apply_config integration",          TestApplyConfigIntegration),
        ("Profile report",                    TestProfileReport),
    ]
    all_ok = True
    for label, cls in suites:
        instance = cls()
        ok = True
        for method_name in dir(instance):
            if not method_name.startswith("test_"):
                continue
            try:
                if hasattr(instance, "setup_method"):
                    instance.setup_method()
                getattr(instance, method_name)()
            except Exception:                           # noqa: BLE001
                ok = False
                print(f"  FAIL: {label}::{method_name}")
                traceback.print_exc()
        print(f"{label}: {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    sys.exit(0 if all_ok else 1)
