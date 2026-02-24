Preprocess STRING → CSR + stats
    python src/00_preprocess_string_to_csr.py

CPU single
    python src/01_mcl_cpu_single.py

CPU multi
    $env:OMP_NUM_THREADS=8; $env:MKL_NUM_THREADS=8; python src/02_mcl_cpu_multi.py

GPU
    python src/03_mcl_gpu_cupy.py