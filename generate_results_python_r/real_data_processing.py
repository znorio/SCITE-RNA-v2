"""
Script used to perform filtering of SNVs, tree inference and parameter optimization on cancer data
-> run generate_results_cpp/real_data_processing.cpp for faster results instead.
"""

from src_python.generate_results import generate_sciterna_results

import yaml

with open("../config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

bootstrap_samples = 1000  # number of bootstrap samples if use_bootstrap=True
use_bootstrap = False  # use bootstrapping of mutations or not
n_snps = 300  # number of SNVs if method="first_k"
posterior_cutoff = 0.05  # posterior likelihood of locus being mutated if method="threshold"
method= "threshold"  # "first_k" "threshold" "highest_post"
n_rounds = 2  # Number of rounds of tree optimization and parameter estimation
sample = "mm34"  # sample name
flipped_mutation_direction = True  # if true allow the root genotype to be learned during tree optimization
only_preprocessing = False  # if true only run mutation filtering, no tree inference
tree_space = ["c", "m"]  # "c" = cell tree space, "m" = mutation tree space
reshuffle_nodes = False  # Prune and reinsert individual nodes in the mutation tree
ref_to_alt = False  # if True allow SNVs to only mutate ref -> alt

input_path = rf"../data/input_data/{sample}"
output_path = rf"../data/results/{sample}/sciterna"

generate_sciterna_results(path=input_path, pathout=output_path,
                          n_bootstrap=bootstrap_samples, use_bootstrap=use_bootstrap, tree_space=tree_space,
                          flipped_mutation_direction=flipped_mutation_direction, n_keep=n_snps, posterior_cutoff=posterior_cutoff,
                          n_rounds=n_rounds, only_preprocessing=only_preprocessing, method=method, reshuffle_nodes=reshuffle_nodes,
                          ref_to_alt=ref_to_alt)
