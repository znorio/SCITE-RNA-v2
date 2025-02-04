"""
Script used to perform filtering of SNVs and determining the most likely genotypes for the multiple myeloma dataset.
Training is possible, but very slow -> run generate_results_cpp/MM.cpp for tree inference instead.
"""

from src_python.mutation_filter import MutationFilter
from src_python.swap_optimizer import SwapOptimizer
import numpy as np
import pandas as pd
import json
import os
import yaml

with open('../config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def create_mutation_matrix(parent_vector, mutation_indices, ct):
    n_cells = len(parent_vector)
    n_leaves = int((n_cells+1)/2)
    n_mutations = len(mutation_indices)

    mutation_matrix = np.zeros((n_cells, n_mutations), dtype=int)

    for mutation_idx, cell_idx in enumerate(mutation_indices):
        children = [c for c in ct.dfs(cell_idx)]
        for cell in children:  # Traverse all cells below the mutation cell
            mutation_matrix[cell, mutation_idx] = 1  # Mark cells with the mutation

    return mutation_matrix[:n_leaves].T

def convert_location_to_gene(locations):

    loc_to_gene = []

    df = pd.read_csv("../data/input_data/mm34/gene_positions.csv", index_col=False)
    for location in locations:
        chrom, pos = location.split(":")[0], location.split(":")[1]
        pos = int(pos)
        matching_rows = df[(df['chromosome'] == chrom) & (df['start'] <= pos) & (df['end'] >= pos)]
        matching_genes = matching_rows['gene'].tolist()
        loc_to_gene.append(matching_genes)

    return loc_to_gene

def main():
    bootstrap_samples = 1000
    n_snps = 3000
    sample = "mm34"
    reduced = "" #"_reduced"
    bootstrap_folder = "bootstrap_"
    reverse_mut = False
    only_ref_to_alt = True
    ref_to_alt = ""
    training = False
    if reverse_mut:
        bootstrap_folder = "bootstrap_reverse_mut_"
    if only_ref_to_alt:
        ref_to_alt = "_only_ref_to_alt"

    reference = pd.read_csv(rf'../data/input_data/{sample}/ref{reduced}.csv', index_col=0)
    alternative = pd.read_csv(rf'../data/input_data/{sample}/alt{reduced}.csv', index_col=0)
    ref = np.nan_to_num(np.array(reference), 0)[:,:]
    alt = np.nan_to_num(np.array(alternative), 0)[:,:]

    mf = MutationFilter(f=config["f"], omega=config["omega"], h_factor=config["h_factor"],
                        genotype_freq=config["genotype_freq"],
                        mut_freq=config["mut_freq"])
    optimizer = SwapOptimizer(reverse_mutations=reverse_mut)
    selected, gt1, gt2, not_selected_genotypes = mf.filter_mutations(ref, alt, method='first_k', n_exp=n_snps, only_ref_to_alt=only_ref_to_alt)

    np.random.seed(config["random_seed"])
    indices = np.random.choice(len(selected), (bootstrap_samples, len(selected)), replace=True)

    bootstrap_selected = np.array(selected)[indices]
    bootstrap_gt1 = np.array(gt1)[indices]
    bootstrap_gt2 = np.array(gt2)[indices]
    bootstrap_selected_genes = np.array(reference.columns)[bootstrap_selected]

    base_path =rf"../data/results/{sample}/{bootstrap_folder}{bootstrap_samples}_snvs_{n_snps}{reduced}{ref_to_alt}"
    if not os.path.exists(os.path.join(base_path, f"selected{reduced}.txt")):
        os.makedirs(base_path, exist_ok=True)
        np.savetxt(os.path.join(base_path, f"selected{reduced}.txt"), bootstrap_selected, fmt='%d', delimiter=',')
        np.savetxt(os.path.join(base_path, f"gt1{reduced}.txt"), bootstrap_gt1, fmt='%s', delimiter=',')
        np.savetxt(os.path.join(base_path, f"gt2{reduced}.txt"), bootstrap_gt2, fmt='%s', delimiter=',')
        np.savetxt(os.path.join(base_path, f"not_selected_genotypes{reduced}.txt"), not_selected_genotypes, fmt='%s', delimiter=',')
        np.savetxt(os.path.join(base_path, f"selected_genes{reduced}.txt"), bootstrap_selected_genes, fmt='%s', delimiter=',')
        np.savetxt(os.path.join(base_path, f"top_{n_snps}_loci.txt"), selected, fmt='%i')

    if training:
        os.makedirs(os.path.join(base_path, "sciterna_mut_loc"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "sciterna_parent_vec"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "sciterna_selected_loci"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "sciterna_selected_genes"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "sciterna_mut_indicator"), exist_ok=True)
        for loop in range(bootstrap_samples):

            selected = bootstrap_selected[loop]
            gt1, gt2 = bootstrap_gt1[loop], bootstrap_gt2[loop]

            llh_1, llh_2 = mf.get_llh_mat(ref[:,selected], alt[:,selected], gt1, gt2)
            optimizer.fit_llh(llh_1, llh_2)
            optimizer.optimize()

            selected_loci = reference.columns[selected]
            selected_genes = convert_location_to_gene(selected_loci)

            with open(os.path.join(base_path, "sciterna_selected_genes", f"selected_genes_{loop}.json"), 'w') as file:
                json.dump(selected_genes, file)
            np.savetxt(os.path.join(base_path, "sciterna_selected_loci", f'sciterna_selected_loci_{loop}.txt'), selected, fmt='%i')
            np.savetxt(os.path.join(base_path, "sciterna_parent_vec", f'sciterna_parent_vec_{loop}.txt'), optimizer.ct.parent_vec, fmt='%d')
            np.savetxt(os.path.join(base_path, "sciterna_mut_indicator", f'sciterna_mut_indicator_{loop}.txt'),
                       create_mutation_matrix(optimizer.ct.parent_vec, optimizer.ct.mut_loc, optimizer.ct), fmt='%d')
            np.savetxt(os.path.join(base_path, "sciterna_mut_loc", f'sciterna_mut_loc_{loop}.txt'), optimizer.ct.mut_loc, fmt='%d')

if __name__ == '__main__':
    main()
