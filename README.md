# SCITE-RNA

This repository contains the code and data for the paper **Phylogenetic Tree Inference from Single-Cell RNA Sequencing**. 
The code and datasets provided here enable users to replicate the experiments and figures presented in the paper, as well as to run SCITE-RNA on new data.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Cloning the Repository](#cloning-the-repository)
  - [Data Preparation](#data-preparation)
- [Running the Model](#running-the-model)
  - [Set Model Parameters](#set-model-parameters)
  - [Simulated Data](#simulated-data)
  - [Multiple Myeloma Data](#multiple-myeloma-data)
  - [Run on New Data](#run-on-new-data)
- [Generating Figures](#generating-figures)

---

## Overview

We implement a new method for reconstructing phylogenetic trees from single-cell RNA sequencing data. 
SCITE-RNA selects single-nucleotide variants (SNVs), and reconstructs a phylogenetic tree of the sequenced cells. 
We maximize the likelihood of the inferred tree by alternating between the cell lineage and mutation tree spaces until convergence is achieved in both.
This repository provides:
1. Scripts to execute SCITE-RNA. The model is split into C++ `src_cpp` and Python files `src_python`. Especially for large numbers of cells and SNVs it is recommended to use the C++ code, as it is significantly faster. The inferred trees should be comparable between the C++ and Python implementations. 
2. Summaries of the data used in the paper is available in the `data/` directory, which contains all necessary files to reproduce the figures.
3. Visualization scripts to generate plots as presented in the paper.


## Repository Structure

**SCITE-RNA**<br>
├── data/                       # Input data files and results <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── input_data               # Alternative and reference read counts among other files. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── results                  # Inferred trees of the multiple myeloma dataset and figures <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── simulated_data           # Summary statistics of simulated data and respective inferred trees <br>
├── generate_results_cpp/       # C++ scripts for tree inference <br>
├── generate_results_python_r/  # Python and R scripts for simulating data, inferring trees and visualization <br>
├── src_cpp/                    # C++ source files for SCITE-RNA <br>
├── src_python/                 # Python source files for SCITE-RNA <br>
├── configs/                    # Model parameters (Python) <br>
├── CMakeLists.txt              # Primary configuration file for CMake <br>
└── README.md                   # Project overview and setup instructions

## Installation

### Requirements

#### Python Libraries:
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- numba
- math
- jupyter
- pyyaml
- graphviz

#### C++ Requirements

- CMake (>= 3.27)

### Cloning the Repository

To set up the SCITE-RNA project locally:

    git clone https://github.com/cbg-ethz/SCITE-RNA.git
    cd SCITE-RNA


### Data Preparation

To reproduce the figures quickly you can use the files provided in `data`. 
As the size and the number of files was quite large, we produced summary statistics
using 

    generate_results_python_r/generate_summary_statistics.ipynb

## Running the Model

### Set Model Parameters

Adjust model parameters in `configs/config.yaml` for Python 
or adjust them in `src_cpp/mutation_filter.h` for C++.

### Simulated Data

To **generate new simulated data** execute:

    generate_results_python_r/comparison_data_generation.py

It offers the option to set the number of cells and SNVs and the number of clones simulated.
The same file can also be used for tree inference. Alternatively, run the C++ version:

    generate_results_cpp/comparison_num_clones.cpp 

for tree inference (not data generation).

To **compare different optimization strategies** of tree space switching run:

    generate_results_cpp/comparison_tree_spaces_switching.cpp

Otherwise, the model will by default alternate between cell lineage and
mutation tree optimization, starting from a random cell lineage tree. 

All simulated results will be saved in `data/simulated_data/`.

### Multiple Myeloma Data

To run SCITE-RNA on the Multiple Myeloma dataset:

1. Run mutation filtering:

       generate_results_python_r/MM.py

2. Perform tree inference in C++ for faster computation:

       generate_results_cpp/MM.cpp

Results will be saved in `data/results/mm34/`.

### Run on New Data
To use SCITE-RNA on new data:

1. Prepare reference and alternative allele count files in `.txt` format. 
Use the format provided in `data/input_data/new_data` as a reference, 
where columns represent cells and rows represent SNVs.

2. Set the number of bootstrap samples (optional) and run SCITE-RNA tree inference with the following script:

       generate_results_cpp/run_sciterna.cpp

3. The results are saved in `data/results/new_data/`.


## Generating Figures

To reproduce the plots presented in the paper, follow the instructions below:
        
- The plots are generated by default using the summary statistics generated with

        generate_results_python_r/comparison_data_generation.py
 
<br>

- **Figure 3: Comparison of tree optimization strategies**

      generate_results_python_r/comparison_tree_spaces_switching.py

<br>

- **Figure 4: Comparison to SClineager and DENDRO + variable number of clones**

- **Figure A.2: Runtime comparison**

    Optionally rerun DENDRO and SClineager on the simulated data first. 
        
        generate_results_python_r/comparison_clones_sclineager_dendro_sciterna.R
    To generate the figures run
    
        generate_results_python_r/comparison_num_clones.ipynb

<br>

- **Figure 5: Representative tree multiple myeloma**
- **Figure A.1: Gene expression analysis**

        generate_results_python_r/bootstrap_results_mm.ipynb 
All figures will be saved in `data/results/figures/`.