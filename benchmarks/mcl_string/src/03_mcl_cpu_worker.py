# src/03_mcl_cpu_worker.py
import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import sparse


def pick_latest_csr_npz() -> str:
    data_dir = Path("data_processed")
    mcl_dirs = sorted([d for d in data_dir.glob("ppi_mcl_*") if d.is_dir()])
    if mcl_dirs:
        latest_dir = mcl_dirs[-1]
        csr_files = sorted(latest_dir.glob("ppi_csr_*.npz"))
        return str(csr_files[-1]) if csr_files else "data_processed/ppi_mcl/ppi_csr.npz"
    return "data_processed/ppi_mcl/ppi_csr.npz"


# ---- Runtime knobs ----
R = 2.0
TAU = 1e-6
MAX_ITERS = 60
TOL = 1e-9
# ----------------------

log_lines = []
def log(msg: str):
    print(msg, flush=True)
    log_lines.append(msg)


def col_normalize(M: sparse.csr_matrix) -> sparse.csr_matrix:
    col_sum = np.asarray(M.sum(axis=0)).ravel()
    col_sum[col_sum == 0] = 1.0
    inv = 1.0 / col_sum
    return M @ sparse.diags(inv, format="csr")


def prune(M: sparse.csr_matrix, tau: float) -> sparse.csr_matrix:
    M = M.tocsr(copy=True)
    M.data[M.data < tau] = 0.0
    M.eliminate_zeros()
    return M


def mcl_cpu(A: sparse.csr_matrix, r=2.0, tau=1e-6, max_iters=60, tol=1e-9):
    M = col_normalize(A)

    t_start = time.perf_counter()
    for it in range(1, max_iters + 1):
        t_it = time.perf_counter()
        M_prev = M

        # Expansion
        M = M @ M

        # Inflation
        M.data = np.power(M.data, r, dtype=np.float32)

        # Prune
        M = prune(M, tau)

        # Normalize
        M = col_normalize(M)

        # Convergence
        diff = (M - M_prev).data
        err = float(np.max(np.abs(diff))) if diff.size else 0.0

        log(f"iter={it:02d} nnz={M.nnz:,} err={err:.3e} iter_time={time.perf_counter()-t_it:.2f}s")

        if err < tol:
            break

    return M, (time.perf_counter() - t_start)


def main():
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    mkl_threads = os.environ.get("MKL_NUM_THREADS", str(threads))
    openblas_threads = os.environ.get("OPENBLAS_NUM_THREADS", str(threads))
    blis_threads = os.environ.get("BLIS_NUM_THREADS", str(threads))
    numexpr_threads = os.environ.get("NUMEXPR_NUM_THREADS", str(threads))

    CSR_NPZ = pick_latest_csr_npz()

    # Write into a threads-specific results folder (launcher also makes one)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(os.environ.get("RESULTS_DIR", f"results_{timestamp}"))
    results_dir.mkdir(exist_ok=True, parents=True)

    out_file = results_dir / f"mcl_cpu_threads_{threads}.log"

    log(f"THREADS={threads}")
    log(f"OMP_NUM_THREADS={threads}")
    log(f"MKL_NUM_THREADS={mkl_threads}")
    log(f"OPENBLAS_NUM_THREADS={openblas_threads}")
    log(f"BLIS_NUM_THREADS={blis_threads}")
    log(f"NUMEXPR_NUM_THREADS={numexpr_threads}")
    log(f"CSR_NPZ={CSR_NPZ}")
    log(f"Params: R={R} TAU={TAU} MAX_ITERS={MAX_ITERS} TOL={TOL}")

    t0 = time.perf_counter()
    A = sparse.load_npz(CSR_NPZ).tocsr().astype(np.float32)
    load_t = time.perf_counter() - t0
    log(f"Loaded CSR: n={A.shape[0]:,} nnz={A.nnz:,} load_time={load_t:.2f}s")

    M, total = mcl_cpu(A, r=R, tau=TAU, max_iters=MAX_ITERS, tol=TOL)

    log(f"CPU DONE | final nnz={M.nnz:,} | total_time={total:.2f}s")

    with open(out_file, "w") as f:
        f.write("\n".join(log_lines))

    # machine-readable one-line summary for launcher
    summary_path = results_dir / f"summary_threads_{threads}.txt"
    with open(summary_path, "w") as f:
        f.write(
            f"threads={threads} "
            f"n={A.shape[0]} "
            f"nnz_in={A.nnz} "
            f"nnz_out={M.nnz} "
            f"load_time_s={load_t:.6f} "
            f"total_time_s={total:.6f}\n"
        )


if __name__ == "__main__":
    main()