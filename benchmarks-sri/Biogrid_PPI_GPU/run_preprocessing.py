from src.preprocessing import preprocess_biogrid

preprocess_biogrid(
    input_path="data/raw/BIOGRID-ORGANISM-Homo_sapiens-5.0.254.tab3.txt",
    output_dir="data/processed"
)