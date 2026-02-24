import os
import time
from datetime import datetime
import numpy as np
from pathlib import Path
from scipy import sparse

# Optional: find the latest CSR from data_processed/ppi_mcl_*
_data_dir = Path("data_processed")
_mcl_dirs = sorted([d for d in _data_dir.glob("ppi_mcl_*") if d.is_dir()])
if _mcl_dirs:
    _latest_dir = _mcl_dirs[-1]
    _csr_files = sorted(_latest_dir.glob("ppi_csr_*.npz"))
    CSR_NPZ = str(_csr_files[-1]) if _csr_files else "data_processed/ppi_mcl/ppi_csr.npz"
else:
    CSR_NPZ = "data_processed/ppi_mcl/ppi_csr.npz"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path(f"results_{timestamp}")
RESULTS_DIR.mkdir(exist_ok=True)
OUT_FILE = RESULTS_DIR / f"mcl_cpu_multi_{timestamp}.txt"

# Threads (set in terminal ideally; defaults here)
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

# ---- Longer runtime knobs ----
R = 2.0
TAU = 1e-6
MAX_ITERS = 60
TOL = 1e-9
# ------------------------------

log_lines = []
def log(msg: str):
    print(msg)
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

        M = M @ M
        M.data = np.power(M.data, r, dtype=np.float32)
        M = prune(M, tau)
        M = col_normalize(M)

        diff = (M - M_prev).data
        err = float(np.max(np.abs(diff))) if diff.size else 0.0

        log(f"iter={it:02d} nnz={M.nnz:,} err={err:.3e} iter_time={time.perf_counter()-t_it:.2f}s")

        if err < tol:
            break

    return M, (time.perf_counter() - t_start)

def main():
    log(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")

    t0 = time.perf_counter()
    A = sparse.load_npz(CSR_NPZ).tocsr().astype(np.float32)
    log(f"Loaded CSR: n={A.shape[0]:,} nnz={A.nnz:,} load_time={time.perf_counter()-t0:.2f}s")
    log(f"Params: R={R} TAU={TAU} MAX_ITERS={MAX_ITERS} TOL={TOL}")

    M, total = mcl_cpu(A, r=R, tau=TAU, max_iters=MAX_ITERS, tol=TOL)
    log(f"\nCPU MULTI DONE | final nnz={M.nnz:,} | total_time={total:.2f}s")

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(log_lines))

if __name__ == "__main__":
    main()
