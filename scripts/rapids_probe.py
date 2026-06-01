"""
scripts/rapids_probe.py
=======================

Run on the Linux GPU machine to inspect the installed RAPIDS / CuPy
environment.  Reports:

    * cugraph / cudf / cupy version strings
    * which cuGraph algorithm APIs exist
    * each existing function's signature (parameter names)
    * the columns each function returns on a tiny test graph

Usage
-----
    python scripts/rapids_probe.py
    python scripts/rapids_probe.py --json > rapids_env.json

Optional JSON output is suitable for committing to the benchmark output
directory alongside results.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from typing import Any


PROBE_FUNCTIONS = [
    "pagerank",
    "personalized_pagerank",
    "bfs",
    "hits",
    "louvain",
]


def _version(modname: str) -> str:
    try:
        m = importlib.import_module(modname)
        return str(getattr(m, "__version__", "?"))
    except Exception as exc:                                # noqa: BLE001
        return f"NOT INSTALLED ({type(exc).__name__})"


def _probe_cugraph_function(name: str) -> dict:
    """Inspect ``cugraph.<name>``: signature + a small live call."""
    out: dict[str, Any] = {"name": name}
    try:
        import cugraph
        fn = getattr(cugraph, name, None)
    except Exception as exc:                                # noqa: BLE001
        out["exists"] = False
        out["error"] = f"cugraph import failed: {exc}"
        return out

    if fn is None:
        out["exists"] = False
        return out

    out["exists"] = True
    try:
        sig = inspect.signature(fn)
        out["signature"]   = str(sig)
        out["param_names"] = list(sig.parameters.keys())
    except (TypeError, ValueError) as exc:
        out["signature_error"] = str(exc)

    # Live call against a 4-node triangle plus tail edge.
    try:
        import cudf
        import cugraph as cg

        edges = cudf.DataFrame({
            "src": [0, 1, 2, 0, 2],
            "dst": [1, 2, 0, 2, 3],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
        })
        G = cg.Graph(directed=(name != "louvain" and name != "hits"))
        G.from_cudf_edgelist(
            edges, source="src", destination="dst", edge_attr="weight",
        )

        if name == "bfs":
            res = fn(G, 0) if "start" not in out.get("param_names", []) else fn(G, start=0)
        elif name == "louvain":
            res = fn(G)
        else:
            res = fn(G)

        if isinstance(res, tuple):
            out["return_type"]     = f"tuple[{', '.join(type(x).__name__ for x in res)}]"
            out["tuple_length"]    = len(res)
            df_candidate = res[0]
            if hasattr(df_candidate, "columns"):
                out["returned_columns"] = list(df_candidate.columns)
            if len(res) > 1:
                out["second_value"] = repr(res[1])
        else:
            out["return_type"] = type(res).__name__
            if hasattr(res, "columns"):
                out["returned_columns"] = list(res.columns)
    except Exception as exc:                                # noqa: BLE001
        out["live_call_error"] = f"{type(exc).__name__}: {exc}"

    return out


def _device_info() -> dict:
    info: dict[str, Any] = {}
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        info["device_count"] = n
        for i in range(n):
            props = cp.cuda.runtime.getDeviceProperties(i)
            info[f"device_{i}"] = {
                "name":             props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
                "compute_capability": (props["major"], props["minor"]),
                "total_mem_mb":     props["totalGlobalMem"] // (1024 * 1024),
            }
    except Exception as exc:                                # noqa: BLE001
        info["error"] = str(exc)
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON instead of a human-readable report.",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "python":   sys.version.split()[0],
        "versions": {
            "cugraph":  _version("cugraph"),
            "cudf":     _version("cudf"),
            "cupy":     _version("cupy"),
            "numpy":    _version("numpy"),
            "scipy":    _version("scipy"),
            "pycuda":   _version("pycuda"),
        },
        "device":    _device_info(),
        "cugraph_functions": [
            _probe_cugraph_function(name) for name in PROBE_FUNCTIONS
        ],
    }

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    # Human-readable summary
    print("=" * 60)
    print("RAPIDS environment probe")
    print("=" * 60)
    print(f"python: {report['python']}")
    print("Versions:")
    for k, v in report["versions"].items():
        print(f"  {k:8s} = {v}")
    print()
    print("Device info:")
    for k, v in report["device"].items():
        print(f"  {k}: {v}")
    print()
    print("cuGraph functions:")
    for entry in report["cugraph_functions"]:
        print(f"  {entry['name']}: exists={entry.get('exists')}")
        if entry.get("signature"):
            print(f"    signature: {entry['signature']}")
        if entry.get("returned_columns"):
            print(f"    columns:   {entry['returned_columns']}")
        if entry.get("return_type"):
            print(f"    type:      {entry['return_type']}")
        if entry.get("live_call_error"):
            print(f"    live_call_error: {entry['live_call_error']}")
        if entry.get("error"):
            print(f"    error: {entry['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
