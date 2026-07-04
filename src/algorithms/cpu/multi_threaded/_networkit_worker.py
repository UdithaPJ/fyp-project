"""
src/algorithms/cpu/multi_threaded/_networkit_worker.py
======================================================

Standalone NetworKit Louvain (PLM / PLMR) worker.

Run as an isolated subprocess — invoked **by absolute file path**, never
via ``-m`` and never imported as a module — so it executes in a fresh
Python interpreter that has NOT initialized CuPy / RAPIDS / PyCUDA.

Why this exists
---------------
``src.algorithms`` builds its registry at import time, which eagerly
imports the ``cuda_optimized`` modules and (through
``src.benchmarking.benchmark``) forces a live CUDA context via
``cupy.zeros(1)``.  NetworKit's pip wheel bundles its own OpenMP/TBB
runtime; when that runs parallel work inside a process that already holds
a CUDA context, glibc aborts with
``malloc(): mismatching next->prev_size`` — a fatal heap corruption from
two native threading/allocator runtimes stepping on shared heap state.

Running the NetworKit step in a child process launched *by file path*
means Python does not import the ``src`` package chain, so no CUDA context
is ever created in this interpreter and the collision cannot happen.

Hard rule
---------
This file MUST NOT import anything from ``src.*`` — doing so would
re-trigger the very CUDA initialization it exists to avoid.  Only
``numpy`` / ``scipy`` / ``networkit`` are allowed.

Protocol
--------
    python _networkit_worker.py <in_npz> <out_npy> <refine> <gamma> <threads>

    <in_npz>  : scipy ``.npz`` of the symmetric CSR adjacency (A_sym)
    <out_npy> : path to write the int64 community-label vector (length n)
    <refine>  : "1" for PLMR (refinement pass), "0" for plain PLM
    <gamma>   : resolution parameter (float)
    <threads> : OpenMP thread count (int; <= 0 means NetworKit default)

Exit code 0 on success; non-zero (with a traceback on stderr) on failure.
"""

import sys

import numpy as np
import scipy.sparse as sp
import networkit as nk


def main() -> int:
    in_npz, out_npy, refine_s, gamma_s, threads_s = sys.argv[1:6]
    refine  = refine_s == "1"
    gamma   = float(gamma_s)
    threads = int(threads_s)

    coo = sp.load_npz(in_npz).tocoo()
    n   = int(coo.shape[0])

    # Upper triangle only (self-loops included) so each undirected edge is
    # added exactly once — the full symmetric matrix would double every
    # off-diagonal edge.
    upper = coo.row <= coo.col
    rows  = coo.row[upper]
    cols  = coo.col[upper]
    data  = coo.data[upper].astype(np.float64, copy=False)

    if threads > 0:
        nk.setNumberOfThreads(threads)

    # Prefer the vectorised bulk constructor; fall back to per-edge addEdge
    # for older NetworKit releases that lack GraphFromCoo.
    graph_from_coo = getattr(nk, "GraphFromCoo", None)
    G = None
    if graph_from_coo is not None:
        try:
            upper_coo = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
            G = graph_from_coo(upper_coo, weighted=True, directed=False)
        except Exception:                                          # noqa: BLE001
            G = None
    if G is None:
        G = nk.Graph(n, weighted=True, directed=False)
        add_edge = G.addEdge
        for a, b, w in zip(rows.tolist(), cols.tolist(), data.tolist()):
            add_edge(int(a), int(b), float(w))

    plm = nk.community.PLM(G, refine=refine, gamma=gamma)
    plm.run()
    labels = np.asarray(plm.getPartition().getVector(), dtype=np.int64)
    np.save(out_npy, labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
