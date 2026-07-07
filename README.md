# Project Structure and Module Description

This repository contains a GPU-accelerated framework for multi-scale biological network analysis. It is designed to support efficient graph processing, algorithm execution, benchmarking, and biological validation through a modular and extensible architecture.

The system is organized into three main layers:

* Core computational engine (`src/`)
* User-facing web application (`webapp/`)
* Experimental evaluation modules (`experiments/`)

---

## 📁 Root Directory

* **main.py**
  Optional CLI entry point for running the pipeline without the web interface.

* **requirements.txt**
  Lists all dependencies required to run the project.

* **README.md**
  Documentation describing the project, setup, and structure.

---

## 📁 data/

Stores all datasets used in the project.

* **raw/**
  Original datasets downloaded from sources such as BioGRID, TRRUST, and miRTarBase.

* **processed/**
  Cleaned and preprocessed datasets ready for graph construction.

* **scaled/**
  Subsets of datasets (e.g., 10%, 25%, 50%) used for benchmarking and scalability experiments.

---

## 📁 src/ (Core System)

Contains the main computational logic shared across the framework, GUI, and experiments.

---

### 📁 core/

* **pipeline.py**
  Implements the end-to-end workflow: data loading → graph construction → optimization → algorithm execution → result generation.

* **config.py**
  Stores configuration settings such as mode (GUI, benchmark, validation), device (CPU/GPU), and algorithm parameters.

* **runner.py**
  Entry point for executing the pipeline programmatically.

---

### 📁 preprocessing/

* **loader.py**
  Loads datasets from files (CSV/TSV).

* **cleaner.py**
  Cleans and filters raw biological data.

* **graph_builder.py**
  Converts processed data into graph structures.

---

### 📁 graph/

* **csr.py / coo.py**
  Implements sparse graph representations optimized for GPU processing.

* **graph_utils.py**
  Helper functions for graph manipulation and statistics.

---

#### 📁 gpu/

This directory contains GPU-based implementations of graph algorithms, organized into two categories:

---

##### 📁 cuda_optimized/ (Primary Contribution)

- Contains **custom CUDA/PyCUDA implementations** developed in this project  
- Optimized for:
  - mid-range GPUs  
  - memory efficiency (VRAM constraints)  
  - performance tuning  

- These implementations are:
  - used in the web application  
  - the main focus of evaluation and optimization  

---

##### 📁 basic/ (Baseline GPU Implementations)

- Contains **GPU implementations using existing public libraries** (e.g., cuGraph, Gunrock)  
- Used as a **baseline for comparison** against custom optimized versions  

- Purpose:
  - evaluate performance differences  
  - validate correctness of results  
  - compare against standard GPU approaches  

---

#### 📁 cpu/ (Benchmarking Only)

- CPU implementations used for:
  - correctness validation  
  - baseline performance comparison  

---

#### 📁 common/

* Shared helper functions used by both CPU and GPU implementations.

---

### 📁 optimization/

* **memory_manager.py**
  Handles GPU memory allocation and usage.

* **chunking.py**
  Enables processing of graphs larger than available VRAM.

* **unified_memory.py**
  Implements unified memory strategies for CPU-GPU interaction.

* **gpu_config.py**
  Detects GPU specifications and adjusts execution parameters (architecture-aware optimization).

---

### 📁 benchmarking/

* **benchmark.py**
  Runs performance comparisons between CPU and GPU implementations.

* **metrics.py**
  Computes performance metrics such as runtime, speedup, and memory usage.

* **scaling.py**
  Generates experiments for different dataset sizes.

---

### 📁 validation/

* **overlap.py**
  Computes overlap between predicted results and known biological gene sets.

* **enrichment.py**
  Performs pathway enrichment analysis.

* **reference_loader.py**
  Loads reference datasets such as KEGG and MSigDB.

---

### 📁 visualization/

* **plots.py**
  Generates charts (runtime, speedup, rankings).

* **graph_viz.py**
  Visualizes graph structures and highlights important nodes.

* **tables.py**
  Formats results into tabular form.

---

### 📁 utils/

* **logger.py**
  Logging utility for tracking execution.

* **file_utils.py**
  Handles file operations.

---

## 📁 webapp/ (User Interface)

Implements the locally hosted web application for interacting with the framework.

* **app.py**
  Main entry point for the web application (Streamlit or Flask).

---

### 📁 pages/

* **upload.py**
  Handles dataset upload.

* **preprocessing.py**
  Allows users to select node/edge columns and build the graph.

* **analysis.py**
  Provides algorithm selection and execution controls.

* **results.py**
  Displays outputs and visualizations.

---

### 📁 components/

* **sidebar.py**
  Navigation and global controls.

* **controls.py**
  UI elements for parameter selection.

* **charts.py**
  Visualization components used in the UI.

---

## 📁 experiments/ (Non User-Facing)

Used for benchmarking and validation experiments.

---

### 📁 benchmark/

* **run_benchmark.py**
  Executes performance comparisons across CPU and GPU.

* **configs/**
  Stores experiment configurations.

---

### 📁 validation/

* **run_validation.py**
  Executes biological validation experiments.

* **configs/**
  Validation-specific settings.

---

### 📁 outputs/

* **plots/**
  Stores generated graphs and charts.

* **reports/**
  Stores experiment summaries and results.

---

## 📁 notebooks/

Optional Jupyter notebooks for quick experimentation and visualization.

---

## 📁 tests/

Contains test cases to verify correctness of implementations.

---

# Summary

The project is structured to clearly separate:

* Core computational logic (`src/`)
* User-facing interface (`webapp/`)
* Experimental evaluation (`experiments/`)

This modular design ensures scalability, maintainability, and reproducibility of results.
