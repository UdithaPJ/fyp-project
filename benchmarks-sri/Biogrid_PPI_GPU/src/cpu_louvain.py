import time
import networkx as nx
import community as community_louvain  # python-louvain

def run_cpu_louvain(csr_matrix):

    G = nx.from_scipy_sparse_array(csr_matrix)

    start = time.time()
    partition = community_louvain.best_partition(G)
    modularity = community_louvain.modularity(partition, G)
    end = time.time()

    return partition, modularity, end - start