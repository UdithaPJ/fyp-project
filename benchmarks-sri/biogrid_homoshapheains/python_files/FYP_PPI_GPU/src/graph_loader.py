from scipy import sparse

def load_csr(path):
    return sparse.load_npz(path)
