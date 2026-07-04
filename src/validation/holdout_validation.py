"""
src/validation/holdout_validation.py
=====================================

Hold-out (link-prediction) validation — fully self-contained.

This is the strongest standalone validation because it needs NO external
database and is immune to the circularity problem: the "ground truth" is a
random subset of the graph's OWN edges that the algorithm never sees.

Protocol
--------
1. Split the graph's edges into a training set (``1 - test_fraction``) and a
   held-out test set (``test_fraction``), with a fixed RNG seed.
2. Build a training CSR from the training edges only.
3. Run the algorithm (``cpu_single`` — deterministic, GPU-independent) on the
   TRAINING graph.
4. Ask whether the algorithm's output recovers the held-out edges it never
   saw:

   ranking algorithms (pagerank / hits / rwr)
       Link-prediction AUROC.  Edge score s(u, v) = score[u] · score[v]
       (preferential-attachment style).  Positives = held-out edges;
       negatives = an equal number of sampled non-edges.  AUROC > 0.5 means
       the centrality ranking carries genuine link-prediction signal.

   clustering algorithms (louvain / mcl)
       Community co-membership enrichment.  Held-out edges should join nodes
       placed in the SAME training-graph community more often than random
       node pairs.  Reported as held-out vs random co-membership rate, a
       lift ratio, and a one-sided Fisher-exact p-value.

BFS is a traversal, not a ranking/clustering, so it is not hold-out
validated.

Outputs
-------
    experiments/outputs/reports/holdout_validation.csv
    experiments/outputs/plots/holdout_validation.png

Rules
-----
* NEVER raises — failures are recorded with ``status='error'``.
* Uses only ``cpu_single`` implementations so it runs anywhere.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import scipy.sparse as sp

from src.validation.overlap import fisher_exact_p

_LOG = logging.getLogger(__name__)

_RANKING_ALGOS = ("pagerank", "hits", "rwr")
_CLUSTER_ALGOS = ("louvain", "mcl")

# Default params mirror cross_implementation_validation._DEFAULT_PARAMS but
# RWR uses empty seeds so its scores are a global (PageRank-like) centrality
# suitable for link prediction rather than a single-seed propagation.
_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "pagerank": {"damping": 0.85, "max_iter": 100, "tolerance": 1e-6},
    "rwr":      {"restart_prob": 0.3, "max_iter": 100,
                 "tolerance": 1e-6, "seed_nodes": []},
    "hits":     {"max_iter": 100, "tolerance": 1e-6},
    "louvain":  {"min_delta_q": 1e-4, "max_levels": 10, "resolution": 1.0},
    "mcl":      {"expansion": 2, "inflation": 2.0, "prune_threshold": 0.001,
                 "max_iter": 100, "convergence_tol": 1e-4},
}


@dataclass
class HoldoutRecord:
    algorithm:    str
    dataset:      str
    network_type: str
    test_edges:   int   = 0
    train_edges:  int   = 0
    # ranking metrics
    auroc:        float = float("nan")
    avg_precision: float = float("nan")
    # clustering metrics
    comembership_heldout: float = float("nan")
    comembership_random:  float = float("nan")
    lift:         float = float("nan")
    p_value:      float = float("nan")
    note:         str   = ""
    status:       str   = "ok"     # ok | skipped | error


@dataclass
class _HoldoutDataset:
    name:            str
    graph_csr:       sp.csr_matrix
    network_type:    str
    params_override: dict[str, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metric helpers (pure numpy)
# ---------------------------------------------------------------------------

def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic (rank-based, tie-aware).

    Equivalent to P(score(positive) > score(negative)) with ties counted
    at 0.5.  Returns NaN when either group is empty.
    """
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, all_scores.size + 1, dtype=np.float64)
    # Average ranks for ties
    _, inv, counts = np.unique(all_scores, return_inverse=True,
                               return_counts=True)
    # sum of ranks per unique value → mean rank
    rank_sum = np.zeros(counts.size, dtype=np.float64)
    np.add.at(rank_sum, inv, ranks)
    mean_rank = rank_sum / counts
    ranks = mean_rank[inv]
    sum_pos_ranks = ranks[:n_pos].sum()
    u_pos = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


def _average_precision(pos: np.ndarray, neg: np.ndarray) -> float:
    """Average precision of ranking positives above negatives by score."""
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores, kind="mergesort")[::-1]
    labels = labels[order]
    cum_tp = np.cumsum(labels)
    precision_at_k = cum_tp / np.arange(1, labels.size + 1)
    ap = float((precision_at_k * labels).sum() / max(n_pos, 1))
    return ap


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class HoldoutValidator:
    """Self-contained link-prediction / community hold-out validation.

    Usage::

        hv = HoldoutValidator(output_dir=..., test_fraction=0.2, seed=42)
        hv.add_dataset("string_ppi", graph_csr, network_type="ppi")
        hv.run(algorithms=["pagerank", "louvain", ...])
        hv.write_csv(); hv.write_plots()
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        test_fraction: float = 0.2,
        seed: int = 42,
        max_eval: int = 5000,
    ) -> None:
        _root = Path(__file__).resolve().parents[2]
        self.output_dir = Path(output_dir) if output_dir else (
            _root / "experiments" / "outputs"
        )
        (self.output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)

        self.test_fraction = float(test_fraction)
        self.seed = int(seed)
        self.max_eval = int(max_eval)
        self._datasets: list[_HoldoutDataset] = []
        self.records: list[HoldoutRecord] = []

    # ------------------------------------------------------------------
    def add_dataset(
        self,
        name: str,
        graph_csr: sp.csr_matrix,
        network_type: str,
        params_override: Optional[dict[str, dict]] = None,
    ) -> None:
        if not sp.isspmatrix_csr(graph_csr):
            graph_csr = graph_csr.tocsr()
        self._datasets.append(_HoldoutDataset(
            name=name, graph_csr=graph_csr,
            network_type=str(network_type).lower(),
            params_override=params_override or {},
        ))

    # ------------------------------------------------------------------
    # Edge splitting
    # ------------------------------------------------------------------
    def _split_edges(
        self, csr: sp.csr_matrix, network_type: str, rng: np.random.Generator,
    ) -> tuple[sp.csr_matrix, np.ndarray, set]:
        """Return (train_csr, test_edges[m,2], all_edge_set).

        For undirected (ppi) graphs edges are deduplicated on the upper
        triangle and the training graph is re-symmetrised; directed graphs
        (grn / mirna) keep edge orientation.
        """
        coo = csr.tocoo()
        rows, cols, data = coo.row, coo.col, coo.data
        n = csr.shape[0]
        undirected = network_type == "ppi"

        if undirected:
            keep = rows < cols        # upper triangle, drops self-loops too
        else:
            keep = rows != cols       # drop self-loops only
        rows, cols, data = rows[keep], cols[keep], data[keep]

        m = rows.size
        all_edge_set = set(zip(rows.tolist(), cols.tolist()))
        if undirected:
            all_edge_set |= set(zip(cols.tolist(), rows.tolist()))

        perm = rng.permutation(m)
        n_test = max(1, int(round(m * self.test_fraction)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

        test_edges = np.column_stack([rows[test_idx], cols[test_idx]])

        tr_r, tr_c, tr_d = rows[train_idx], cols[train_idx], data[train_idx]
        if undirected:
            tr_r2 = np.concatenate([tr_r, tr_c])
            tr_c2 = np.concatenate([tr_c, tr_r])
            tr_d2 = np.concatenate([tr_d, tr_d])
        else:
            tr_r2, tr_c2, tr_d2 = tr_r, tr_c, tr_d
        train_csr = sp.csr_matrix((tr_d2, (tr_r2, tr_c2)), shape=(n, n))
        return train_csr, test_edges, all_edge_set

    def _sample_negatives(
        self, n: int, count: int, all_edge_set: set, rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample ``count`` node pairs (u != v) absent from the full graph."""
        out: list[tuple[int, int]] = []
        attempts = 0
        max_attempts = count * 20
        while len(out) < count and attempts < max_attempts:
            u = int(rng.integers(0, n))
            v = int(rng.integers(0, n))
            attempts += 1
            if u == v or (u, v) in all_edge_set:
                continue
            out.append((u, v))
        return np.array(out, dtype=np.int64) if out else np.empty((0, 2), np.int64)

    # ------------------------------------------------------------------
    # Algorithm execution on the training graph
    # ------------------------------------------------------------------
    def _run_cpu(self, algo: str, csr: sp.csr_matrix,
                 network_type: str, override: dict) -> dict:
        from src.validation.cross_implementation_validation import _cpu_fn
        params = dict(_DEFAULT_PARAMS.get(algo, {}))
        params["network_type"] = network_type
        params.update(override.get(algo, {}))
        fn = _cpu_fn(algo, "cpu_single")
        return fn(csr, params)

    @staticmethod
    def _node_scores(algo: str, inner: dict, n: int) -> Optional[np.ndarray]:
        """Extract a length-n node score vector for a ranking algorithm."""
        if algo == "hits":
            hub = np.asarray(inner.get("hub_scores", []), dtype=np.float64)
            auth = np.asarray(inner.get("authority_scores", []), dtype=np.float64)
            if hub.size == n and auth.size == n:
                return hub + auth
            if hub.size == n:
                return hub
            if auth.size == n:
                return auth
            return None
        scores = np.asarray(inner.get("scores", []), dtype=np.float64)
        return scores if scores.size == n else None

    # ------------------------------------------------------------------
    def run(self, algorithms: Optional[tuple] = None) -> None:
        algos = tuple(algorithms) if algorithms else (
            _RANKING_ALGOS + _CLUSTER_ALGOS
        )
        if not self._datasets:
            _LOG.warning("HoldoutValidator: no datasets registered.")
            return

        for ds in self._datasets:
            rng = np.random.default_rng(self.seed)
            n = ds.graph_csr.shape[0]
            try:
                train_csr, test_edges, all_edges = self._split_edges(
                    ds.graph_csr, ds.network_type, rng,
                )
            except Exception as exc:
                for algo in algos:
                    self.records.append(HoldoutRecord(
                        algorithm=algo, dataset=ds.name,
                        network_type=ds.network_type,
                        note=f"edge split failed: {exc}"[:200], status="error",
                    ))
                continue

            # Cap evaluation size for speed on large graphs.
            if test_edges.shape[0] > self.max_eval:
                sel = rng.choice(test_edges.shape[0], self.max_eval,
                                 replace=False)
                test_edges = test_edges[sel]
            negatives = self._sample_negatives(
                n, test_edges.shape[0], all_edges, rng,
            )
            train_edges = int(train_csr.nnz)

            for algo in algos:
                try:
                    if algo in _CLUSTER_ALGOS:
                        rec = self._eval_clustering(
                            algo, ds, train_csr, test_edges, negatives,
                            train_edges,
                        )
                    elif algo in _RANKING_ALGOS:
                        rec = self._eval_ranking(
                            algo, ds, train_csr, test_edges, negatives,
                            train_edges,
                        )
                    else:
                        rec = HoldoutRecord(
                            algorithm=algo, dataset=ds.name,
                            network_type=ds.network_type,
                            note="not hold-out validated", status="skipped",
                        )
                except Exception as exc:
                    rec = HoldoutRecord(
                        algorithm=algo, dataset=ds.name,
                        network_type=ds.network_type,
                        test_edges=int(test_edges.shape[0]),
                        train_edges=train_edges,
                        note=str(exc)[:200], status="error",
                    )
                self.records.append(rec)

    # ------------------------------------------------------------------
    def _eval_ranking(
        self, algo: str, ds: _HoldoutDataset, train_csr: sp.csr_matrix,
        test_edges: np.ndarray, negatives: np.ndarray, train_edges: int,
    ) -> HoldoutRecord:
        n = train_csr.shape[0]
        inner = self._run_cpu(algo, train_csr, ds.network_type,
                              ds.params_override)
        scores = self._node_scores(algo, inner, n)
        if scores is None:
            return HoldoutRecord(
                algorithm=algo, dataset=ds.name, network_type=ds.network_type,
                test_edges=int(test_edges.shape[0]), train_edges=train_edges,
                note="no usable node score vector", status="skipped",
            )
        if negatives.shape[0] == 0:
            return HoldoutRecord(
                algorithm=algo, dataset=ds.name, network_type=ds.network_type,
                test_edges=int(test_edges.shape[0]), train_edges=train_edges,
                note="no negatives sampled (graph too dense/small)",
                status="skipped",
            )
        pos = scores[test_edges[:, 0]] * scores[test_edges[:, 1]]
        neg = scores[negatives[:, 0]] * scores[negatives[:, 1]]
        return HoldoutRecord(
            algorithm=algo, dataset=ds.name, network_type=ds.network_type,
            test_edges=int(test_edges.shape[0]), train_edges=train_edges,
            auroc=_auroc(pos, neg), avg_precision=_average_precision(pos, neg),
            note="link-prediction AUROC (score-product)", status="ok",
        )

    def _eval_clustering(
        self, algo: str, ds: _HoldoutDataset, train_csr: sp.csr_matrix,
        test_edges: np.ndarray, negatives: np.ndarray, train_edges: int,
    ) -> HoldoutRecord:
        n = train_csr.shape[0]
        inner = self._run_cpu(algo, train_csr, ds.network_type,
                              ds.params_override)
        assign = (inner.get("community_assignments")
                  or inner.get("cluster_assignments"))
        if not isinstance(assign, list) or len(assign) != n:
            return HoldoutRecord(
                algorithm=algo, dataset=ds.name, network_type=ds.network_type,
                test_edges=int(test_edges.shape[0]), train_edges=train_edges,
                note="no usable community assignment vector", status="skipped",
            )
        labels = np.asarray(assign, dtype=np.int64)

        def _same(pairs: np.ndarray) -> tuple[int, int]:
            if pairs.shape[0] == 0:
                return 0, 0
            same = int(np.sum(labels[pairs[:, 0]] == labels[pairs[:, 1]]))
            return same, pairs.shape[0] - same

        a, b = _same(test_edges)          # held-out: same / different comm
        c, d = _same(negatives)           # random:   same / different comm
        co_held = a / (a + b) if (a + b) else float("nan")
        co_rand = c / (c + d) if (c + d) else float("nan")
        lift = (co_held / co_rand) if (co_rand and co_rand > 0) else float("nan")
        # One-sided Fisher: are held-out edges enriched for same-community?
        p_val = fisher_exact_p(a=a, b=b, c=c, d=d)

        return HoldoutRecord(
            algorithm=algo, dataset=ds.name, network_type=ds.network_type,
            test_edges=int(test_edges.shape[0]), train_edges=train_edges,
            comembership_heldout=co_held, comembership_random=co_rand,
            lift=lift, p_value=p_val,
            note="held-out edge community co-membership enrichment",
            status="ok",
        )

    # ------------------------------------------------------------------
    def write_csv(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.output_dir / "reports" / "holdout_validation.csv"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _fmt(v) -> str:
            if v is None:
                return ""
            if isinstance(v, float):
                return "" if math.isnan(v) else f"{v:.6g}"
            return str(v)

        fields = [
            "algorithm", "dataset", "network_type", "test_edges",
            "train_edges", "auroc", "avg_precision",
            "comembership_heldout", "comembership_random", "lift",
            "p_value", "note", "status",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.records:
                w.writerow({k: _fmt(getattr(r, k)) for k in fields})
        _LOG.info("Hold-out validation CSV: %s", path)
        return path

    def write_plots(self) -> dict[str, Path]:
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        return {"holdout_validation": self._plot(plot_dir)}

    def _plot(self, plot_dir: Path) -> Path:
        out = plot_dir / "holdout_validation.png"
        rank_ok = [r for r in self.records
                   if r.status == "ok" and math.isfinite(r.auroc)]
        clust_ok = [r for r in self.records
                    if r.status == "ok" and math.isfinite(r.lift)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Left: ranking AUROC
        if rank_ok:
            labels = [f"{r.algorithm}\n{r.dataset[:12]}" for r in rank_ok]
            ax1.bar(range(len(labels)), [r.auroc for r in rank_ok],
                    color="#4a90e2", alpha=0.85)
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax1.axhline(0.5, color="#e74c3c", linestyle="--", linewidth=1,
                        label="random (0.5)")
            ax1.set_ylim(0, 1.05)
            ax1.set_ylabel("Link-prediction AUROC")
            ax1.set_title("Ranking algorithms (hold-out)")
            ax1.legend(fontsize=8)
            ax1.grid(True, axis="y", alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No ranking hold-out records",
                     ha="center", va="center", transform=ax1.transAxes)

        # Right: clustering co-membership lift
        if clust_ok:
            labels = [f"{r.algorithm}\n{r.dataset[:12]}" for r in clust_ok]
            ax2.bar(range(len(labels)), [r.lift for r in clust_ok],
                    color="#bd10e0", alpha=0.85)
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax2.axhline(1.0, color="#e74c3c", linestyle="--", linewidth=1,
                        label="no enrichment (lift=1)")
            ax2.set_ylabel("Co-membership lift (held-out / random)")
            ax2.set_title("Clustering algorithms (hold-out)")
            ax2.legend(fontsize=8)
            ax2.grid(True, axis="y", alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No clustering hold-out records",
                     ha="center", va="center", transform=ax2.transAxes)

        fig.suptitle("Hold-out validation (edges hidden from the algorithm)",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _LOG.info("Plot: %s", out)
        return out


__all__ = ["HoldoutValidator", "HoldoutRecord"]
