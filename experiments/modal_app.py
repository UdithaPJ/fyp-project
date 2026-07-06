"""
experiments/modal_app.py
========================

Run the FYP GPU *scalability* benchmarks on a Modal-hosted A100.

Modal (https://modal.com) bills per-second with no idle charge and gives
$30/month free credit — at ~$2.10/hr for an A100-40GB that is ~14 GPU-hours
free per month, and only actual container run time is billed.

This wrapper reuses the EXISTING entrypoints unchanged:
    scripts/generate_benchmark_graphs.py     (graph pre-generation)
    experiments/benchmark/run_benchmark.py   (scalability | runtime | all)

The container is a CUDA *devel* image so `nvcc` is present — PyCUDA
JIT-compiles the kernels at runtime, and the project's own
`_detect_arch_flag()` compiles for the A100's sm_80 automatically.

Design
------
Two separate functions so you never pay A100 rates for CPU work:

    generate_graphs()  — CPU only, high RAM, writes .npz to the DATA volume.
    run_benchmark()    — A100 GPU, reads pregenerated graphs from that volume.

Pregenerated graphs and outputs live on persistent Modal Volumes, so you
generate once and benchmark many times.

--------------------------------------------------------------------------
Workflow (see the chat for the full copy-paste command list)
--------------------------------------------------------------------------
    pip install modal && modal setup            # one time, links your account

    modal run experiments/modal_app.py::sanity  # ~1 min, confirms the stack

    # Get the graphs onto the data volume — EITHER upload ones you already
    # have, OR generate fresh (deterministic, seed=42):
    modal run experiments/modal_app.py::upload_graphs       # if you have them
    #   ...or...
    modal run experiments/modal_app.py::generate_graphs     # CPU, slow for BA

    # Run the benchmark on the A100:
    modal run experiments/modal_app.py --command \
        "scalability --graphs-dir /data/pregenerated \
         --edge-targets 1000000,5000000,10000000,20000000,50000000 \
         --graph-types barabasi_albert,erdos_renyi,watts_strogatz \
         --algorithms all --modes cpu_single,cpu_multi,gpu_baseline,gpu \
         --n-runs 3 --network-type grn --output-dir /outputs/grn"

    # Pull results back:
    modal volume get fyp-outputs / ./experiments/outputs_from_modal
--------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
LOCAL_ROOT = Path(__file__).resolve().parent.parent   # this file is in experiments/
REMOTE_ROOT = "/root/fyp"
GRAPHS_DIR = "/data/pregenerated"                      # on the data volume

# Default full sweep (matches the project's canonical scalability config).
DEFAULT_EDGE_TARGETS = "1000000,5000000,10000000,20000000,50000000"
DEFAULT_GRAPH_TYPES = "barabasi_albert,erdos_renyi,watts_strogatz"

# MCL host-RAM budget.  Modal's A100 runs on a shared node with ~1 TB physical
# RAM; MCL's psutil-based detection sees ALL of it and never caps top_k, so its
# out-of-core survivor buffers balloon to hundreds of GB (and it runs the slow
# CPU out-of-core path instead of staying GPU-resident).  Pinning this env var
# (documented seam in CLAUDE.md) makes _adaptive_top_k shrink top_k so the
# working set stays bounded and, where it fits, in-core on the GPU.
MCL_HOST_AVAIL_GB = 64
# Container hard memory cap (safety net so a runaway can never bill for 600 GB).
MEM_REQUEST_MB = 16_384
MEM_LIMIT_MB = 98_304

# --------------------------------------------------------------------------
# Container image
# --------------------------------------------------------------------------
# CUDA 12.4 *devel* (ships nvcc, required by PyCUDA's runtime compilation).
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    # PyCUDA compiles a native extension from source at install time — it needs
    # a C/C++ toolchain (the base CUDA image ships nvcc but no host compiler).
    # build-essential provides g++; the env vars point PyCUDA's build at g++
    # (it otherwise defaults to the absent clang++) and let the linker find the
    # libcuda stub that ships in the -devel image.
    .apt_install("build-essential", "python3-dev")
    .env({
        "CC": "gcc",
        "CXX": "g++",
        "CUDA_ROOT": "/usr/local/cuda",
        "LIBRARY_PATH": "/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64",
    })
    .pip_install(
        "numpy", "scipy", "pandas", "openpyxl", "networkx",
        "python-graphblas", "matplotlib", "seaborn",
    )
    # GPU optimised path (your kernels) + CuPy (MCL baseline / Louvain phase-2).
    # cupy is a prebuilt wheel; pycuda builds from source with the g++ above.
    .pip_install("cupy-cuda12x", "pycuda")
    # RAPIDS — needed ONLY for the gpu_baseline mode.  Roughly doubles build
    # time; drop this block if you skip --modes gpu_baseline.
    .pip_install(
        "cudf-cu12", "cugraph-cu12",
        extra_index_url="https://pypi.nvidia.com",
    )
    .add_local_dir(str(LOCAL_ROOT / "src"), f"{REMOTE_ROOT}/src")
    .add_local_dir(str(LOCAL_ROOT / "experiments"), f"{REMOTE_ROOT}/experiments")
    .add_local_dir(str(LOCAL_ROOT / "scripts"), f"{REMOTE_ROOT}/scripts")
)

app = modal.App("fyp-gpu-benchmarks", image=image)

data_volume = modal.Volume.from_name("fyp-data", create_if_missing=True)
output_volume = modal.Volume.from_name("fyp-outputs", create_if_missing=True)


# --------------------------------------------------------------------------
# 0.  Sanity check — cheap, confirms the whole stack imports & GPU works
# --------------------------------------------------------------------------
@app.function(gpu="A100", timeout=60 * 10)
def sanity() -> None:
    """Confirm nvcc, PyCUDA, CuPy, cuGraph, and the benchmarker import on Linux."""
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader"], check=False)
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import ctypes.wintypes  # noqa - proves the Linux import is safe\n"
        "import pycuda.autoinit, cupy\n"
        "print('pycuda + cupy OK, GPU:', cupy.cuda.runtime.getDeviceProperties(0)['name'])\n"
        "from src.benchmarking.scalability_benchmark import ScalabilityBenchmarker\n"
        "print('scalability_benchmark imported OK on Linux')\n"
        "try:\n"
        "    import cugraph, cudf; print('cugraph', cugraph.__version__)\n"
        "except Exception as e:\n"
        "    print('cugraph NOT available:', e)\n"
    ) % REMOTE_ROOT
    r = subprocess.run([sys.executable, "-c", code], cwd=REMOTE_ROOT)
    if r.returncode != 0:
        raise RuntimeError("sanity check failed")


# --------------------------------------------------------------------------
# 1a. Generate graphs fresh (CPU only — cheap, but BA large sizes are slow)
# --------------------------------------------------------------------------
@app.function(
    volumes={"/data": data_volume},
    cpu=8.0,
    memory=98_304,          # request ~96 GB; BA 50M (~10M nodes) needs ~80 GB
    timeout=60 * 60 * 12,   # BA generation can be very slow
)
def generate_graphs(
    edge_targets: str = DEFAULT_EDGE_TARGETS,
    graph_types: str = DEFAULT_GRAPH_TYPES,
    seed: int = 42,
) -> None:
    """Run scripts/generate_benchmark_graphs.py, writing .npz to the data volume."""
    cmd = [
        sys.executable, f"{REMOTE_ROOT}/scripts/generate_benchmark_graphs.py",
        "--output-dir", GRAPHS_DIR,
        "--graph-types", graph_types,
        "--edge-targets", edge_targets,
        "--seed", str(seed),
    ]
    print(f"[modal] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=REMOTE_ROOT)
    data_volume.commit()
    if r.returncode != 0:
        raise RuntimeError(f"generation exited with {r.returncode}")


# --------------------------------------------------------------------------
# 1b. OR upload graphs you already have in data/pregenerated/
# --------------------------------------------------------------------------
@app.local_entrypoint()
def upload_graphs() -> None:
    """Upload local data/pregenerated/*.npz + *_meta.json to the data volume."""
    src = LOCAL_ROOT / "data" / "pregenerated"
    if not src.exists():
        print(f"[modal] not found: {src} — nothing to upload "
              "(use ::generate_graphs instead).")
        return
    files = sorted(src.glob("*.npz")) + sorted(src.glob("*_meta.json"))
    if not files:
        print(f"[modal] no .npz/.json files under {src}")
        return
    with data_volume.batch_upload(force=True) as batch:
        for f in files:
            print(f"[modal] uploading {f.name} ({f.stat().st_size/1e6:.0f} MB)")
            batch.put_file(str(f), f"/pregenerated/{f.name}")
    print(f"[modal] uploaded {len(files)} file(s) to fyp-data:/pregenerated")


# --------------------------------------------------------------------------
# 2.  Benchmark on the A100
# --------------------------------------------------------------------------
@app.function(
    gpu="A100",                       # "A100-80GB" for the 80 GB card
    volumes={"/data": data_volume, "/outputs": output_volume},
    memory=(MEM_REQUEST_MB, MEM_LIMIT_MB),   # (request, hard cap) — caps runaway
    timeout=60 * 60 * 6,              # 6 h ceiling; billed only for actual use
)
def run_benchmark(command: str, mcl_host_avail_gb: int = MCL_HOST_AVAIL_GB) -> None:
    """Execute experiments/benchmark/run_benchmark.py with `command` args."""
    args = command.split()
    if not args:
        raise ValueError("empty command")
    script = f"{REMOTE_ROOT}/experiments/benchmark/run_benchmark.py"
    if "--output-dir" not in args:
        args += ["--output-dir", "/outputs"]

    # Bound MCL's host-RAM budget (see MCL_HOST_AVAIL_GB note above).  The
    # benchmark runs the algorithm in this subprocess, which inherits os.environ,
    # so setting it here reaches mcl_gpu().  Harmless for the other algorithms —
    # only MCL reads it.
    env = os.environ.copy()
    env["MCL_HOST_AVAIL_BYTES"] = str(mcl_host_avail_gb * 1024**3)

    cmd = [sys.executable, script, *args]
    print(f"[modal] running: {' '.join(cmd)}", flush=True)
    print(f"[modal] MCL_HOST_AVAIL_BYTES={env['MCL_HOST_AVAIL_BYTES']} "
          f"(~{mcl_host_avail_gb} GB)", flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader"], check=False)

    result = subprocess.run(cmd, cwd=REMOTE_ROOT, env=env)
    output_volume.commit()            # persist results before the container dies
    if result.returncode != 0:
        raise RuntimeError(f"benchmark exited with code {result.returncode}")


@app.local_entrypoint()
def main(command: str = "") -> None:
    """`modal run experiments/modal_app.py --command "<scalability args>"`."""
    if not command:
        command = (
            f"scalability --graphs-dir {GRAPHS_DIR} "
            f"--edge-targets {DEFAULT_EDGE_TARGETS} "
            f"--graph-types {DEFAULT_GRAPH_TYPES} "
            "--algorithms all --modes cpu_single,cpu_multi,gpu_baseline,gpu "
            "--n-runs 3 --network-type grn --output-dir /outputs/grn"
        )
    run_benchmark.remote(command)
    print("\n[modal] done.  Fetch results with:")
    print("    modal volume get fyp-outputs / ./experiments/outputs_from_modal")
