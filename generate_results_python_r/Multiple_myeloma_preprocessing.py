"""
Script used to perform filtering of SNVs and determining the most likely genotypes for the multiple myeloma dataset.
Training is possible, but very slow -> run generate_results_cpp/MM.cpp for tree inference instead.
"""

from src_python.noise_mutation_filter import MutationFilter
from src_python.swap_optimizer import SwapOptimizer
from src_python.utils import create_mutation_matrix
import numpy as np
import pandas as pd
import json
import os
import yaml

with open('../config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


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
    reverse_mut = False
    training = True

    reference = pd.read_csv(rf'../data/input_data/{sample}/ref.csv', index_col=0)
    alternative = pd.read_csv(rf'../data/input_data/{sample}/alt.csv', index_col=0)
    ref = np.nan_to_num(np.array(reference), 0)[:, :]
    alt = np.nan_to_num(np.array(alternative), 0)[:, :]

    mf = MutationFilter(error_rate=config["error_rate"], overdispersion=config["overdispersion"],
                        genotype_freq=config["genotype_freq"],
                        mut_freq=config["mut_freq"], alpha_h=config["alpha_h"], beta_h=config["beta_h"],
                        dropout_prob=config["dropout_prob"], dropout_direction_prob=config["dropout_direction_prob"])
    optimizer = SwapOptimizer(reverse_mutations=reverse_mut)
    selected, gt1, gt2, not_selected_genotypes = mf.filter_mutations(ref, alt, method='first_k', n_exp=n_snps)

    np.random.seed(config["random_seed"])
    indices = np.random.choice(len(selected), (bootstrap_samples, len(selected)), replace=True)

    bootstrap_selected = np.array(selected)[indices]
    bootstrap_gt1 = np.array(gt1)[indices]
    bootstrap_gt2 = np.array(gt2)[indices]
    bootstrap_selected_genes = np.array(reference.columns)[bootstrap_selected]

    base_path = rf"../data/results/{sample}/{bootstrap_samples}_snvs_{n_snps}"
    if not os.path.exists(os.path.join(base_path, f"selected.txt")):
        os.makedirs(base_path, exist_ok=True)
        np.savetxt(os.path.join(base_path, f"selected.txt"), bootstrap_selected, fmt='%d', delimiter=',')
        np.savetxt(os.path.join(base_path, f"gt1.txt"), bootstrap_gt1, fmt='%s', delimiter=',')
        np.savetxt(os.path.join(base_path, f"gt2.txt"), bootstrap_gt2, fmt='%s', delimiter=',')
        np.savetxt(os.path.join(base_path, f"not_selected_genotypes.txt"), not_selected_genotypes, fmt='%s',
                   delimiter=',')
        np.savetxt(os.path.join(base_path, f"selected_genes.txt"), bootstrap_selected_genes, fmt='%s', delimiter=',')
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

            llh_1, llh_2 = mf.get_llh_mat(ref[:, selected], alt[:, selected], gt1, gt2)
            optimizer.fit_llh(llh_1, llh_2)
            optimizer.optimize()

            selected_loci = reference.columns[selected]
            selected_genes = convert_location_to_gene(selected_loci)

            with open(os.path.join(base_path, "sciterna_selected_genes", f"selected_genes_{loop}.json"), 'w') as file:
                json.dump(selected_genes, file)
            np.savetxt(os.path.join(base_path, "sciterna_selected_loci", f'sciterna_selected_loci_{loop}.txt'),
                       selected, fmt='%i')
            np.savetxt(os.path.join(base_path, "sciterna_parent_vec", f'sciterna_parent_vec_{loop}.txt'),
                       optimizer.ct.parent_vec, fmt='%d')
            np.savetxt(os.path.join(base_path, "sciterna_mut_indicator", f'sciterna_mut_indicator_{loop}.txt'),
                       create_mutation_matrix(optimizer.ct.parent_vec, optimizer.ct.mut_loc, optimizer.ct), fmt='%d')
            np.savetxt(os.path.join(base_path, "sciterna_mut_loc", f'sciterna_mut_loc_{loop}.txt'),
                       optimizer.ct.mut_loc, fmt='%d')


if __name__ == '__main__':
    main()
