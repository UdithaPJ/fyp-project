import cudf
import cugraph
import time


def run_gpu_pagerank(csr_matrix):

    coo = csr_matrix.tocoo()

    gdf = cudf.DataFrame({
        "src": coo.row,
        "dst": coo.col
    })

    G = cugraph.Graph()

    G.from_cudf_edgelist(
        gdf,
        source="src",
        destination="dst",
        store_transposed=True
    )

    start = time.time()
    pr = cugraph.pagerank(G)
    end = time.time()

    return pr.to_pandas(), end - start