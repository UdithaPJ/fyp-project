import time
import networkx as nx

def run_cpu_pagerank(csr_matrix):

    # Convert CSR to NetworkX graph
    G = nx.from_scipy_sparse_array(csr_matrix)

    start = time.time()
    pr = nx.pagerank(G, alpha=0.85)
    end = time.time()

    return pr, end - start