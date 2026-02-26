import time
from datetime import datetime
import numpy as np
from pathlib import Path
from scipy import sparse

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
except Exception as e:
    raise SystemExit("CuPy not installed. Install cupy-cuda11x or cupy-cuda12x matching your CUDA.") from e

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
OUT_FILE = RESULTS_DIR / f"mcl_gpu_{timestamp}.txt"

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

def col_normalize_gpu(M: cpsp.csr_matrix) -> cpsp.csr_matrix:
    col_sum = cp.asarray(M.sum(axis=0)).ravel()
    col_sum = cp.where(col_sum == 0, 1.0, col_sum)
    inv = 1.0 / col_sum
    Dinv = cpsp.diags(inv, format="csr")
    return M @ Dinv

def prune_gpu(M: cpsp.csr_matrix, tau: float) -> cpsp.csr_matrix:
    M = M.tocsr(copy=True)
    M.data = cp.where(M.data < tau, 0.0, M.data)
    M.eliminate_zeros()
    return M

def mcl_gpu(A_gpu: cpsp.csr_matrix, r=2.0, tau=1e-6, max_iters=60, tol=1e-9):
    M = col_normalize_gpu(A_gpu)
    cp.cuda.Stream.null.synchronize()

    t_start = time.perf_counter()
    for it in range(1, max_iters + 1):
        t_it = time.perf_counter()
        M_prev = M

        # Expansion
        M = M @ M

        # Inflation
        M.data = cp.power(M.data, r, dtype=cp.float32)

        # Prune
        M = prune_gpu(M, tau)

        # Normalize
        M = col_normalize_gpu(M)

        # Convergence
        diff = (M - M_prev).data
        err = float(cp.max(cp.abs(diff)).get()) if diff.size else 0.0

        cp.cuda.Stream.null.synchronize()
        log(f"iter={it:02d} nnz={M.nnz:,} err={err:.3e} iter_time={time.perf_counter()-t_it:.2f}s")

        if err < tol:
            break

    cp.cuda.Stream.null.synchronize()
    return M, (time.perf_counter() - t_start)

def main():
    t0 = time.perf_counter()
    A = sparse.load_npz(CSR_NPZ).tocsr().astype(np.float32)
    log(f"Loaded CPU CSR: n={A.shape[0]:,} nnz={A.nnz:,} load_time={time.perf_counter()-t0:.2f}s")
    log(f"Params: R={R} TAU={TAU} MAX_ITERS={MAX_ITERS} TOL={TOL}")

    t1 = time.perf_counter()
    A_gpu = cpsp.csr_matrix(A)
    cp.cuda.Stream.null.synchronize()
    log(f"Transferred to GPU | time={time.perf_counter()-t1:.2f}s")

    M_gpu, total = mcl_gpu(A_gpu, r=R, tau=TAU, max_iters=MAX_ITERS, tol=TOL)
    log(f"\nGPU DONE | final nnz={M_gpu.nnz:,} | total_time={total:.2f}s")

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(log_lines))

if __name__ == "__main__":
    main()
