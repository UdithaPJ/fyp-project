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

Graph construction
------------------
Edges are added with ``Graph.addEdge`` in a tight loop.  We deliberately
do **not** use ``nk.GraphFromCoo``: on the deployment NetworKit build it
segfaults (SIGSEGV) on this input, and a C++ segfault cannot be caught by
a Python ``try/except`` — it would kill the whole worker.  ``addEdge`` is
the construction path proven to work on this box.

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
Progress breadcrumbs are written to stderr and flushed, so if a native
segfault occurs the last printed line localises the failing step.
"""

import sys

import numpy as np
import scipy.sparse as sp
import networkit as nk


def _log(msg: str) -> None:
    """Flushed stderr breadcrumb — survives a subsequent native segfault."""
    print(f"[nk_worker] {msg}", file=sys.stderr, flush=True)


def main() -> int:
    in_npz, out_npy, refine_s, gamma_s, threads_s = sys.argv[1:6]
    refine  = refine_s == "1"
    gamma   = float(gamma_s)
    threads = int(threads_s)

    _log(f"start refine={refine} gamma={gamma} threads={threads}")

    coo = sp.load_npz(in_npz).tocoo()
    n   = int(coo.shape[0])

    # Upper triangle only (self-loops included) so each undirected edge is
    # added exactly once — the full symmetric matrix would double every
    # off-diagonal edge.
    upper = coo.row <= coo.col
    rows  = coo.row[upper]
    cols  = coo.col[upper]
    data  = coo.data[upper].astype(np.float64, copy=False)
    _log(f"loaded npz: n={n} edges={rows.size}")

    if threads > 0:
        nk.setNumberOfThreads(threads)
    try:
        _log(f"threads set: max={nk.getMaxNumberOfThreads()}")
    except Exception as exc:                                       # noqa: BLE001
        _log(f"threads set (getMaxNumberOfThreads unavailable: {exc})")

    G = nk.Graph(n, weighted=True, directed=False)
    add_edge = G.addEdge
    for a, b, w in zip(rows.tolist(), cols.tolist(), data.tolist()):
        add_edge(int(a), int(b), float(w))
    _log(f"graph built: nodes={G.numberOfNodes()} edges={G.numberOfEdges()}")

    plm = nk.community.PLM(G, refine=refine, gamma=gamma)
    plm.run()
    _log("PLM run complete")

    labels = np.asarray(plm.getPartition().getVector(), dtype=np.int64)
    np.save(out_npy, labels)
    _log(f"saved labels: {labels.size} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
