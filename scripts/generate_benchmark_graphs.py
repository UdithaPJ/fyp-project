"""
scripts/generate_benchmark_graphs.py
=====================================

Run this on a **high-end machine** (64+ GB RAM, fast CPU) to pre-generate
large synthetic benchmark graphs and save them as compressed scipy CSR
files.  The benchmark system can then load them instantly without needing
the RAM or time to regenerate.

Output convention
-----------------
Every graph is saved as a pair of files inside ``--output-dir``:

    {graph_type}_n{N}_m{M}.npz         — scipy CSR (float32, no self-loops)
    {graph_type}_n{N}_m{M}_meta.json   — metadata for the benchmarker

where N = actual node count, M = actual edge count (nnz).

The ``ScalabilityBenchmarker`` will auto-discover these files when you
pass ``--graphs-dir`` to ``experiments/benchmark/run_benchmark.py``.

Usage (high-end machine)
------------------------
# Typical biology-scale graphs (BA model, target ~30 M edges)
python scripts/generate_benchmark_graphs.py \\
    --output-dir data/pregenerated \\
    --graph-types barabasi_albert,erdos_renyi,watts_strogatz \\
    --edge-targets 1000000,5000000,10000000,20000000,50000000 \\
    --seed 42

# BA only with custom m parameter
python scripts/generate_benchmark_graphs.py \\
    --output-dir data/pregenerated \\
    --graph-types barabasi_albert \\
    --edge-targets 10000000,50000000 \\
    --ba-m 10 \\
    --seed 42

Memory notes (approximate RAM required during generation)
----------------------------------------------------------
  BA,  10 M edges, m=5  -> n ~ 2.0 M nodes -> ~16 GB RAM via NetworkX
  BA,  20 M edges, m=5  -> n ~ 4.0 M nodes -> ~32 GB RAM via NetworkX
  BA,  50 M edges, m=5  -> n ~10.0 M nodes -> ~80 GB RAM via NetworkX
  ER,  10 M edges       -> n ~ 1.7 M nodes -> ~8  GB RAM (scipy-native)
  ER,  50 M edges       -> n ~ 8.2 M nodes -> ~24 GB RAM (scipy-native)
  WS,  10 M edges, k=6  -> n ~ 1.7 M nodes -> ~8  GB RAM (scipy-native)
  WS,  50 M edges, k=6  -> n ~ 8.2 M nodes -> ~24 GB RAM (scipy-native)

The BA generator falls back to a numpy-native implementation for
n > 200 000 to avoid NetworkX's Python-object overhead.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory-efficient graph generators
# ---------------------------------------------------------------------------

def _ba_numpy(n: int, m: int, seed: int) -> sp.csr_matrix:
    """Barabási–Albert graph — numpy-native, no NetworkX.

    Uses an edge-reservoir-based preferential attachment (the standard
    'linear preferential attachment via repeated sampling from the edge
    list' trick):  pick 2*m random positions from the growing edge list
    to select m attachment targets proportional to degree.

    Memory: ~40 bytes/edge (4 int32 arrays of length ~n*m during build;
    peaks at 2× while converting to COO).  For n=10M, m=5: ~4 GB.
    """
    rng = np.random.default_rng(seed)

    # Start with a complete graph on m+1 nodes.
    seed_edges_src, seed_edges_dst = [], []
    for i in range(m + 1):
        for j in range(i):
            seed_edges_src.extend([i, j])
            seed_edges_dst.extend([j, i])

    # Repeated-endpoints list: every directed edge (u->v) and (v->u)
    # contributes u and v to the endpoint list.  Sampling from this
    # list is equivalent to sampling proportional to degree.
    endpoint_list: list[int] = list(seed_edges_src) + list(seed_edges_dst)

    all_src: list[int] = list(seed_edges_src)
    all_dst: list[int] = list(seed_edges_dst)

    report_step = max(1, n // 20)
    t0 = time.perf_counter()
    for new_node in range(m + 1, n):
        # Pick m distinct targets from the endpoint list.
        ep = np.array(endpoint_list, dtype=np.int32)
        chosen = rng.choice(len(ep), size=2 * m, replace=False)
        targets_raw = ep[chosen]
        # Deduplicate and cap at m.
        targets = list(dict.fromkeys(int(t) for t in targets_raw
                                     if t != new_node))[:m]
        if len(targets) < m:
            # Rare case: fill up from the existing nodes.
            remaining = [i for i in range(new_node) if i not in targets]
            rng.shuffle(remaining)
            targets += remaining[:m - len(targets)]

        for t in targets:
            all_src.extend([new_node, t])
            all_dst.extend([t, new_node])
            endpoint_list.extend([new_node, t])

        if (new_node + 1) % report_step == 0:
            elapsed = time.perf_counter() - t0
            _LOG.info(
                "  BA numpy: node %d / %d  (%.0f s elapsed, "
                "est %.0f s remaining)",
                new_node + 1, n, elapsed,
                elapsed / (new_node - m) * (n - new_node - 1),
            )

    src_arr  = np.array(all_src,  dtype=np.int32)
    dst_arr  = np.array(all_dst,  dtype=np.int32)
    data_arr = np.ones(len(src_arr), dtype=np.float32)
    csr = sp.coo_matrix(
        (data_arr, (src_arr, dst_arr)), shape=(n, n)
    ).tocsr()
    del all_src, all_dst, src_arr, dst_arr, data_arr
    gc.collect()
    return csr


def _ba_networkx(n: int, m: int, seed: int) -> sp.csr_matrix:
    """BA graph via NetworkX.  Fast but needs ~8–10 B/node RAM."""
    import networkx as nx
    _LOG.info("  BA networkx: building n=%d m=%d …", n, m)
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    _LOG.info("  BA networkx: converting to CSR …")
    csr = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float32)
    del G
    gc.collect()
    return csr


def _ba_graph(n: int, m: int = 5, seed: int = 42,
              nx_threshold: int = 200_000) -> sp.csr_matrix:
    """Choose BA implementation based on graph size."""
    if n <= nx_threshold:
        try:
            return _ba_networkx(n, m, seed)
        except Exception as exc:
            _LOG.warning("NetworkX BA failed (%s) — using numpy path", exc)
    return _ba_numpy(n, m, seed)


def _er_graph(n: int, avg_degree: float = 6.0, seed: int = 42) -> sp.csr_matrix:
    """Erdos-Renyi sparse graph — scipy-native, no NetworkX.

    Processes the upper-triangle row-by-row in chunks of rows so that peak
    RAM is bounded by ``chunk_rows * n * 4`` bytes (1 000 rows * 1 M nodes *
    4 bytes = 4 GB; reduce ``chunk_rows`` if you hit limits).
    """
    rng = np.random.default_rng(seed)
    p = min(avg_degree / max(n - 1, 1), 1.0)
    expected_edges = int(n * (n - 1) / 2 * p)
    _LOG.info("  ER: n=%d p=%.3e expected_edges~%d ...", n, p, expected_edges)

    # Process this many source rows at a time.  Each row has at most n int32
    # values; chunk_rows * n * 4 bytes = peak per iteration.
    # At n=1.7M, chunk_rows=1 000 -> ~6.8 GB; keep chunk small.
    chunk_rows = max(1, min(500, n // 1000))

    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []

    t0 = time.perf_counter()
    report_step = max(chunk_rows, n // 20)

    for i_start in range(0, n - 1, chunk_rows):
        i_end = min(i_start + chunk_rows, n - 1)
        batch_rows: list[np.ndarray] = []
        batch_cols: list[np.ndarray] = []
        for i in range(i_start, i_end):
            # Upper triangle: columns are i+1 .. n-1
            j_all = np.arange(i + 1, n, dtype=np.int32)
            keep = rng.random(len(j_all)) < p
            kept_j = j_all[keep]
            if kept_j.size:
                batch_rows.append(np.full(kept_j.size, i, dtype=np.int32))
                batch_cols.append(kept_j)

        if batch_rows:
            all_rows.append(np.concatenate(batch_rows))
            all_cols.append(np.concatenate(batch_cols))
        del batch_rows, batch_cols

        if (i_end - i_start > 0) and (i_end % report_step == 0 or i_end >= n - 1):
            _LOG.info("  ER: %.0f%% rows done (%.0f s)",
                      100.0 * i_end / n,
                      time.perf_counter() - t0)
        gc.collect()

    if not all_rows:
        return sp.csr_matrix((n, n), dtype=np.float32)

    rows = np.concatenate(all_rows).astype(np.int32)
    cols = np.concatenate(all_cols).astype(np.int32)
    del all_rows, all_cols
    gc.collect()

    # Symmetrise: upper + lower triangle.
    sym_rows = np.concatenate([rows, cols])
    sym_cols = np.concatenate([cols, rows])
    sym_data = np.ones(len(sym_rows), dtype=np.float32)
    del rows, cols
    gc.collect()

    csr = sp.coo_matrix(
        (sym_data, (sym_rows, sym_cols)), shape=(n, n)
    ).tocsr()
    del sym_rows, sym_cols, sym_data
    gc.collect()
    csr.sum_duplicates()
    return csr


def _ws_graph(n: int, k: int = 6, p: float = 0.1,
              seed: int = 42) -> sp.csr_matrix:
    """Watts-Strogatz small-world graph.

    Uses NetworkX's fast C implementation (always available when networkx
    is installed).  Falls back to a pure-numpy ring-lattice + rewire if
    NetworkX is absent.
    """
    k_use = k if k % 2 == 0 else k + 1
    k_use = min(k_use, max(2, n - 1))

    # NetworkX path (preferred — uses compiled C, much faster).
    try:
        import networkx as nx
        G = nx.watts_strogatz_graph(n, k_use, p, seed=seed)
        csr = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float32)
        del G; gc.collect()
        return csr
    except Exception as exc:
        _LOG.warning("NetworkX WS failed (%s) — falling back to numpy", exc)

    # Numpy-native fallback: ring lattice + probabilistic rewiring.
    # Memory: O(n * k) which is cheap even for large n.
    rng = np.random.default_rng(seed)
    half_k = k_use // 2

    rows, cols = [], []
    for i in range(n):
        for delta in range(1, half_k + 1):
            j = (i + delta) % n
            rows.extend([i, j])
            cols.extend([j, i])

    edge_set = set(zip(rows[:len(rows) // 2], cols[:len(rows) // 2]))
    final_rows, final_cols = [], []
    for (u, v) in list(zip(rows[:len(rows) // 2], cols[:len(rows) // 2])):
        if rng.random() < p:
            candidates = np.arange(n)
            candidates = candidates[(candidates != u)]
            existing_neighbours = {c for (a, c) in edge_set if a == u}
            candidates = [c for c in candidates if c not in existing_neighbours]
            if candidates:
                w = int(rng.choice(candidates))
                edge_set.discard((u, v))
                edge_set.add((u, w))
                final_rows.extend([u, w])
                final_cols.extend([w, u])
                continue
        final_rows.extend([u, v])
        final_cols.extend([v, u])

    src_arr  = np.array(final_rows, dtype=np.int32)
    dst_arr  = np.array(final_cols, dtype=np.int32)
    data_arr = np.ones(len(src_arr), dtype=np.float32)
    csr = sp.coo_matrix(
        (data_arr, (src_arr, dst_arr)), shape=(n, n)
    ).tocsr()
    del final_rows, final_cols, src_arr, dst_arr, data_arr
    gc.collect()
    return csr


# ---------------------------------------------------------------------------
# Node count -> target edges mapping
# ---------------------------------------------------------------------------

def _nodes_for_edges(graph_type: str, target_m: int,
                     ba_m: int = 5, er_avg_deg: float = 6.0,
                     ws_k: int = 6) -> int:
    """Estimate the node count that produces approximately ``target_m`` edges."""
    if graph_type == "barabasi_albert":
        # m edges per new node (undirected, each edge = 2 nnz).
        # target_m = n * ba_m  (nnz = 2 * number of undirected edges)
        n = max(ba_m + 2, int(target_m / (2 * ba_m)))
    elif graph_type == "erdos_renyi":
        # avg_degree per node -> target_m ~ n * er_avg_deg
        n = max(2, int(target_m / er_avg_deg))
    elif graph_type == "watts_strogatz":
        # WS: each node has exactly k_use edges -> m ~ n * k_use
        k_use = ws_k if ws_k % 2 == 0 else ws_k + 1
        n = max(k_use + 2, int(target_m / k_use))
    else:
        n = int(math.sqrt(target_m))
    return n


# ---------------------------------------------------------------------------
# Save / discover helpers
# ---------------------------------------------------------------------------

def _save_graph(csr: sp.csr_matrix, graph_type: str,
                output_dir: Path, params: dict) -> Path:
    """Save a CSR + metadata.  Returns the .npz path."""
    n = int(csr.shape[0])
    m = int(csr.nnz)
    stem  = f"{graph_type}_n{n}_m{m}"
    npz_path  = output_dir / f"{stem}.npz"
    meta_path = output_dir / f"{stem}_meta.json"

    sp.save_npz(str(npz_path), csr)
    with meta_path.open("w") as fh:
        json.dump(
            {
                "graph_type":  graph_type,
                "n_nodes":     n,
                "n_edges":     m,
                "density":     float(m) / float(n * (n - 1)) if n > 1 else 0.0,
                "avg_degree":  float(m) / max(n, 1),
                "dtype":       str(csr.dtype),
                **params,
            },
            fh,
            indent=2,
        )
    _LOG.info("  Saved %s  (%d nodes, %d edges)", npz_path, n, m)
    return npz_path


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

_GENERATORS = {
    "barabasi_albert": _ba_graph,
    "erdos_renyi":     _er_graph,
    "watts_strogatz":  _ws_graph,
}


def generate(
    output_dir: Path,
    graph_types: list[str],
    edge_targets: list[int],
    seed: int = 42,
    ba_m: int = 5,
    er_avg_deg: float = 6.0,
    ws_k: int = 6,
    skip_existing: bool = True,
) -> list[Path]:
    """Generate graphs for all (type, edge_target) combinations.

    Returns a list of .npz paths that were written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for graph_type in graph_types:
        gen_fn = _GENERATORS.get(graph_type)
        if gen_fn is None:
            _LOG.warning("Unknown graph_type %r — skipping", graph_type)
            continue

        for target_m in sorted(edge_targets):
            n = _nodes_for_edges(
                graph_type, target_m,
                ba_m=ba_m, er_avg_deg=er_avg_deg, ws_k=ws_k,
            )

            # Skip if a matching file already exists (same type + approx n).
            if skip_existing:
                existing = sorted(output_dir.glob(f"{graph_type}_n{n}_m*.npz"))
                if existing:
                    _LOG.info(
                        "Skipping %s n~%d (found %s)",
                        graph_type, n, existing[0].name,
                    )
                    continue

            _LOG.info(
                "Generating %s  target_edges=%d  n=%d …",
                graph_type, target_m, n,
            )
            t0 = time.perf_counter()
            try:
                if graph_type == "barabasi_albert":
                    csr = _ba_graph(n, m=ba_m, seed=seed)
                elif graph_type == "erdos_renyi":
                    csr = _er_graph(n, avg_degree=er_avg_deg, seed=seed)
                elif graph_type == "watts_strogatz":
                    csr = _ws_graph(n, k=ws_k, seed=seed)
                else:
                    csr = gen_fn(n, seed=seed)

                csr.sum_duplicates()
                csr.eliminate_zeros()
                elapsed = time.perf_counter() - t0
                _LOG.info(
                    "  Done: %d nodes, %d edges in %.1f s",
                    csr.shape[0], csr.nnz, elapsed,
                )
                path = _save_graph(
                    csr, graph_type, output_dir,
                    {
                        "seed":       seed,
                        "ba_m":       ba_m,
                        "er_avg_deg": er_avg_deg,
                        "ws_k":       ws_k,
                        "target_edges": target_m,
                        "generation_time_s": round(elapsed, 2),
                    },
                )
                written.append(path)
                del csr
                gc.collect()

            except MemoryError:
                _LOG.error(
                    "OOM generating %s n=%d — reduce --edge-targets "
                    "or increase RAM", graph_type, n,
                )
            except Exception as exc:
                _LOG.error("Failed %s n=%d: %s", graph_type, n, exc)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-generate large benchmark graphs for offline use.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("data") / "pregenerated",
        help="Directory where .npz files will be saved.",
    )
    p.add_argument(
        "--graph-types", default="barabasi_albert,erdos_renyi,watts_strogatz",
        help="Comma-separated list of graph types to generate.",
    )
    p.add_argument(
        "--edge-targets",
        default="1000000,5000000,10000000,20000000,50000000",
        help="Comma-separated list of target edge counts.",
    )
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--ba-m",       type=int, default=5,
                   help="BA model m parameter (edges per new node).")
    p.add_argument("--er-avg-deg", type=float, default=6.0,
                   help="ER model expected average degree.")
    p.add_argument("--ws-k",       type=int, default=6,
                   help="WS model k nearest-neighbours parameter.")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Regenerate even if a matching .npz exists.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    graph_types  = [t.strip() for t in args.graph_types.split(",")  if t.strip()]
    edge_targets = [int(e)    for e in args.edge_targets.split(",")  if e.strip()]

    _LOG.info("Output dir  : %s", args.output_dir)
    _LOG.info("Graph types : %s", graph_types)
    _LOG.info("Edge targets: %s", edge_targets)
    _LOG.info("Seed        : %d", args.seed)

    written = generate(
        output_dir    = args.output_dir,
        graph_types   = graph_types,
        edge_targets  = edge_targets,
        seed          = args.seed,
        ba_m          = args.ba_m,
        er_avg_deg    = args.er_avg_deg,
        ws_k          = args.ws_k,
        skip_existing = not args.no_skip_existing,
    )
    _LOG.info("Done. %d file(s) written.", len(written))
    for p in written:
        _LOG.info("  %s", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
