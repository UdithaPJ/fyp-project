import numpy as np
from scipy import sparse

def add_self_loops(A: sparse.csr_matrix, loop_weight: float = 1.0) -> sparse.csr_matrix:
    A = A.tolil(copy=True)
    A.setdiag(loop_weight)
    return A.tocsr()

def normalize_columns(A: sparse.csr_matrix) -> sparse.csr_matrix:
    col_sums = np.asarray(A.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1.0
    inv = 1.0 / col_sums
    Dinv = sparse.diags(inv, offsets=0, format="csr", dtype=A.dtype)
    return A @ Dinv

def inflate(A: sparse.csr_matrix, r: float) -> sparse.csr_matrix:
    A = A.copy()
    A.data = np.power(A.data, r, dtype=A.data.dtype)
    return A

def prune_topk_per_column(A: sparse.csr_matrix, k: int, eps: float) -> sparse.csr_matrix:
    if (k is None or k <= 0) and (eps is None or eps <= 0):
        return A

    Ac = A.tocsc(copy=True)
    n = Ac.shape[1]

    new_data = []
    new_indices = []
    new_indptr = [0]

    for j in range(n):
        start, end = Ac.indptr[j], Ac.indptr[j + 1]
        idx = Ac.indices[start:end]
        val = Ac.data[start:end]

        if val.size == 0:
            new_indptr.append(new_indptr[-1])
            continue

        if eps and eps > 0:
            mask = val >= eps
            idx = idx[mask]
            val = val[mask]

        if val.size == 0:
            new_indptr.append(new_indptr[-1])
            continue

        if k and k > 0 and val.size > k:
            topk_pos = np.argpartition(val, -k)[-k:]
            idx = idx[topk_pos]
            val = val[topk_pos]

        order = np.argsort(idx)
        idx = idx[order]
        val = val[order]

        new_indices.append(idx.astype(np.int32, copy=False))
        new_data.append(val.astype(A.dtype, copy=False))
        new_indptr.append(new_indptr[-1] + val.size)

    if len(new_data) == 0:
        return sparse.csr_matrix(A.shape, dtype=A.dtype)

    new_indices = np.concatenate(new_indices)
    new_data = np.concatenate(new_data)
    new_indptr = np.array(new_indptr, dtype=np.int64)

    Ac2 = sparse.csc_matrix((new_data, new_indices, new_indptr), shape=Ac.shape)
    return Ac2.tocsr()

def mcl(A: sparse.csr_matrix,
        expansion: int = 2,
        inflation_r: float = 2.0,
        loop_weight: float = 1.0,
        max_iter: int = 40,
        tol: float = 1e-3,
        prune_k: int = 100,
        prune_eps: float = 1e-6,
        verbose: bool = True):

    M = add_self_loops(A, loop_weight=loop_weight)
    M = normalize_columns(M)

    last = None

    for it in range(1, max_iter + 1):
        # Expansion (SpGEMM)
        for _ in range(expansion - 1):
            M = (M @ M).tocsr()
            M.sum_duplicates()

        # Inflation
        M = inflate(M, inflation_r)

        # Prune
        M = prune_topk_per_column(M, k=prune_k, eps=prune_eps)

        # Normalize
        M = normalize_columns(M)

        if last is not None:
            diff = (M - last)
            delta = float(np.abs(diff.data).sum() / max(1.0, np.abs(M.data).sum()))
            if verbose:
                print(f"iter {it:02d}: nnz={M.nnz:,}  rel_change={delta:.6f}")
            if delta < tol:
                if verbose:
                    print("Converged.")
                break
        else:
            if verbose:
                print(f"iter {it:02d}: nnz={M.nnz:,}")

        last = M.copy()

    return M

def extract_clusters(M: sparse.csr_matrix):
    Mr = M.tocsr()
    n = Mr.shape[0]
    labels = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        s, e = Mr.indptr[i], Mr.indptr[i + 1]
        if s == e:
            labels[i] = i
            continue
        cols = Mr.indices[s:e]
        vals = Mr.data[s:e]
        labels[i] = cols[int(np.argmax(vals))]

    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)
    return clusters