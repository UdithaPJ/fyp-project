import cudf
import cugraph
import time


def run_gpu_louvain(csr_matrix):

    coo = csr_matrix.tocoo()

    gdf = cudf.DataFrame({
        "src": coo.row,
        "dst": coo.col
    })

    G = cugraph.Graph()
    G.from_cudf_edgelist(gdf, source="src", destination="dst")

    start = time.time()
    parts, modularity = cugraph.louvain(G)
    end = time.time()

    return parts, modularity, end - start
