Recommended environment (important for research reproducibility)
    python -m venv mcl_env
    source mcl_env/bin/activate   # Linux/Mac
    mcl_env\Scripts\activate      # Windows

Install command From project root:
    pip install -r requirements.txt

Preprocess once
    python src/00_preprocess_biogrid_to_csr.py

Single-thread CPU
    python src/01_mcl_cpu_single.py

Multi-thread CPU
    $env:OMP_NUM_THREADS=8; $env:MKL_NUM_THREADS=8; python src/02_mcl_cpu_multi.py

GPU
    python src/03_mcl_gpu_cupy.py