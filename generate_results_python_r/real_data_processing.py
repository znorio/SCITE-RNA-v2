"""
Script used to perform filtering of SNVs and determining the most likely genotypes for the multiple myeloma dataset.
Training is possible, but very slow -> run generate_results_cpp/MM.cpp for tree inference instead.
"""

from src_python.generate_results import generate_sciterna_results

import yaml

with open('../config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

bootstrap_samples = 1000
use_bootstrap = False
n_snps = 3000
n_rounds = 3
sample = "mm34"
flipped_mutation_direction = True
only_preprocessing = False
tree_space = ["c", "m"]

input_path = rf"../data/input_data/{sample}"
output_path = rf"../data/results/{sample}/sciterna"

generate_sciterna_results(path=input_path, pathout=output_path,
                          n_bootstrap=bootstrap_samples, use_bootstrap=use_bootstrap, tree_space=tree_space,
                          flipped_mutation_direction=flipped_mutation_direction, n_keep=n_snps,
                          n_rounds=n_rounds, only_preprocessing=only_preprocessing)
