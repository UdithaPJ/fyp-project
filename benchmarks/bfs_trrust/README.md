Preprocess TRRUST → CSR + stats
    python src/00_preprocess_trrust_to_csr.py

CPU single
    python src/01_bfs_cpu_single.py

CPU multi
    python src/02_bfs_cpu_multi.py

GPU
    python src/03_bfs_gpu_cupy.py