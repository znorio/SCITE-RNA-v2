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
n_snps = 300
posterior_cutoff = 0.05
method= "threshold"  # "first_k" "threshold" "highest_post"
n_rounds = 2
sample = "mm34"
flipped_mutation_direction = True
only_preprocessing = False
tree_space = ["c", "m"]  # "c" = cell tree space, "m" = mutation tree space
reshuffle_nodes = False # Prune and reinsert individual nodes in the tree
ref_to_alt = False

input_path = rf"../data/input_data/{sample}"
output_path = rf"../data/results/{sample}/sciterna_test"

generate_sciterna_results(path=input_path, pathout=output_path,
                          n_bootstrap=bootstrap_samples, use_bootstrap=use_bootstrap, tree_space=tree_space,
                          flipped_mutation_direction=flipped_mutation_direction, n_keep=n_snps, posterior_cutoff=posterior_cutoff,
                          n_rounds=n_rounds, only_preprocessing=only_preprocessing, method=method, reshuffle_nodes=reshuffle_nodes,
                          ref_to_alt=ref_to_alt)
