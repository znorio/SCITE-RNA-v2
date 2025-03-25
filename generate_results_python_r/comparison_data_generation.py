"""
Script used to generate simulated datasets. The number of cells and SNVs and the number of clones simulated can be set.
Tree inference is run as well. However, for large datasets you might want to use
generate_results_cpp/comparison_num_clones.cpp for tree inference as it is faster.
"""

from src_python.data_generator import DataGenerator
from src_python.mutation_tree import MutationTree
from src_python.noise_mutation_filter import MutationFilter
from src_python.swap_optimizer import SwapOptimizer
from src_python.utils import create_genotype_matrix, create_mutation_matrix
from src_python.utils import load_config_and_set_random_seed

import os
from tqdm import tqdm
import numpy as np

config = load_config_and_set_random_seed()


def generate_comparison_data(n_cells: int, n_mut: int, size=100, path='./comparison_data/', random_seed=None,
                             n_clones=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    if os.path.exists(path):
        while True:
            ans = input(f'Directory {path} already exists. Existing files will be overwritten. Continue? [Y/N] ')
            match ans:
                case 'Y' | 'y' | 'Yes' | 'yes':
                    break
                case 'N' | 'n' | 'No' | 'no':
                    return
    else:
        os.makedirs(path)

    os.makedirs(os.path.join(path, "ref"), exist_ok=True)
    os.makedirs(os.path.join(path, "alt"), exist_ok=True)
    os.makedirs(os.path.join(path, "parent_vec"), exist_ok=True)
    os.makedirs(os.path.join(path, "mut_indicator"), exist_ok=True)
    os.makedirs(os.path.join(path, "genotype"), exist_ok=True)
    os.makedirs(os.path.join(path, "dropout_probs"), exist_ok=True)
    os.makedirs(os.path.join(path, "dropout_directions"), exist_ok=True)
    os.makedirs(os.path.join(path, "overdispersions_H"), exist_ok=True)

    generator = DataGenerator(n_cells, n_mut)

    for i in tqdm(range(size)):
        ref, alt, dropout_probs, dropout_directions, overdispersions_H = generator.generate_reads(new_tree=True,
                                                                                                  new_mut_type=True,
                                                                                                  num_clones=n_clones)

        mut_indicator = np.zeros((n_mut, n_cells), dtype=bool)
        for j in range(generator.n_mut):
            if generator.gt1[j] == generator.gt2[j]:
                continue  # not mutated
            for leaf in generator.ct.leaves(generator.ct.mut_loc[j]):  # cells below the mutation
                mut_indicator[j, leaf] = True

        mt = MutationTree(n_mut=n_mut, n_cells=n_cells)
        mt.fit_cell_tree(generator.ct)

        np.savetxt(os.path.join(path, f'ref/ref_{i}.txt'), ref, fmt='%i')
        np.savetxt(os.path.join(path, f'alt/alt_{i}.txt'), alt, fmt='%i')
        np.savetxt(os.path.join(path, f'parent_vec/parent_vec_{i}.txt'), generator.ct.parent_vec, fmt='%i')
        np.savetxt(os.path.join(path, f'mut_indicator/mut_indicator_{i}.txt'), mut_indicator, fmt='%i')
        np.savetxt(os.path.join(path, f'genotype/genotype_{i}.txt'), generator.genotype, fmt='%s')
        np.savetxt(os.path.join(path, f'dropout_probs/dropout_probs_{i}.txt'), dropout_probs)
        np.savetxt(os.path.join(path, f'dropout_directions/dropout_directions_{i}.txt'), dropout_directions)
        np.savetxt(os.path.join(path, f'overdispersions_H/overdispersions_H_{i}.txt'), overdispersions_H)


def generate_sciterna_results(path='./comparison_data/', n_tests=100, pathout="./comparison_data/results",
                              tree_space=None, reverse_mutations=True, n_keep=50, n_rounds=3):
    if tree_space is None:
        tree_space = ["c", "m"]

    optimizer = SwapOptimizer(spaces=tree_space, reverse_mutations=reverse_mutations)

    os.makedirs(os.path.join(pathout, "sciterna_selected_loci"), exist_ok=True)
    os.makedirs(os.path.join(pathout, "sciterna_inferred_mut_types"), exist_ok=True)
    os.makedirs(os.path.join(pathout, "sciterna_parent_vec"), exist_ok=True)
    os.makedirs(os.path.join(pathout, "sciterna_genotype"), exist_ok=True)
    os.makedirs(os.path.join(pathout, "sciterna_mut_indicator"), exist_ok=True)
    os.makedirs(os.path.join(pathout, "sciterna_complete_mut_indicator"), exist_ok=True)
    os.makedirs(os.path.join(pathout, "sciterna_individual_dropout_probs"), exist_ok=True)
    os.makedirs(os.path.join(pathout, "sciterna_individual_dropout_directions"), exist_ok=True)
    os.makedirs(os.path.join(pathout, "sciterna_individual_overdispersions_H"), exist_ok=True)
    os.makedirs(os.path.join(pathout, "sciterna_global_parameters"), exist_ok=True)

    print(f'Running inference on data in {path}')

    for i in tqdm(range(0, n_tests)):
        alt = np.loadtxt(os.path.join(path, "alt", 'alt_%i.txt' % i))
        ref = np.loadtxt(os.path.join(path, "ref", 'ref_%i.txt' % i))

        n_cells, n_snvs = alt.shape

        mf = MutationFilter(error_rate=config["error_rate"], overdispersion=config["overdispersion"],
                            genotype_freq=config["genotype_freq"], mut_freq=config["mut_freq"],
                            dropout_alpha=config["dropout_alpha"], dropout_beta=config["dropout_beta"],
                            dropout_dir_alpha=config["dropout_dir_alpha"], dropout_dir_beta=config["dropout_dir_beta"],
                            overdispersion_h=config["overdispersion_h"])

        # prior assumptions for the snv specific dropout and overdispersion parameters
        individual_dropout_probs = (np.ones(n_snvs, dtype=float) * config["dropout_alpha"] /
                                    (config["dropout_alpha"] + config["dropout_beta"]))
        individual_dropout_direction_probs = np.ones(n_snvs, dtype=float) * config["dropout_dir_alpha"] / (
                config["dropout_dir_alpha"] + config["dropout_dir_beta"])
        alpha_h = beta_h = 0.5 * config["overdispersion_h"]
        individual_alphas_h = np.ones(n_snvs, dtype=float) * alpha_h
        individual_betas_h = np.ones(n_snvs, dtype=float) * beta_h

        for r in range(n_rounds):
            selected, gt1, gt2, not_selected_genotypes = mf.filter_mutations(ref, alt, method='first_k', n_exp=n_keep)

            llh_1, llh_2 = mf.get_llh_mat(ref[:, selected], alt[:, selected], gt1, gt2, individual=True,
                                          dropout_probs=individual_dropout_probs,
                                          dropout_direction_probs=individual_dropout_direction_probs,
                                          alphas_h=individual_alphas_h, betas_h=individual_betas_h)
            optimizer.fit_llh(llh_1, llh_2)
            optimizer.optimize()

            flipped = optimizer.ct.flipped
            mutation_matrix = create_mutation_matrix(optimizer.ct.parent_vec, optimizer.ct.mut_loc, optimizer.ct)
            genotype = create_genotype_matrix(not_selected_genotypes, selected, gt1, gt2, mutation_matrix, flipped)
            complete_mut_indicator = np.zeros(genotype.shape, dtype=int)
            for n, sel in enumerate(selected):
                complete_mut_indicator[:, sel] = mutation_matrix[:, n]

            # use the inferred genotypes to learn optimized parameter estimates
            params = mf.update_parameters(np.array(ref), np.array(alt), np.array(genotype))

            (dropout_prob, dropout_direction_prob, overdispersion, error_rate, overdispersion_h,
             individual_dropout_probs, individual_dropout_direction_probs, individual_overdispersions_h) = params

            np.savetxt(os.path.join(pathout, "sciterna_selected_loci", f'sciterna_selected_loci_{r}r{i}.txt'), selected,
                       fmt='%i')
            np.savetxt(os.path.join(pathout, "sciterna_inferred_mut_types", f'sciterna_inferred_mut_types_{r}r{i}.txt'),
                       np.stack((gt1, gt2), axis=0), fmt='%s')
            np.savetxt(os.path.join(pathout, "sciterna_parent_vec", f'sciterna_parent_vec_{r}r{i}.txt'),
                       optimizer.ct.parent_vec, fmt='%i')
            np.savetxt(os.path.join(pathout, "sciterna_mut_indicator", f'sciterna_mut_indicator_{r}r{i}.txt'),
                       mutation_matrix, fmt='%i')
            np.savetxt(os.path.join(pathout, "sciterna_genotype", f'sciterna_genotype_{r}r{i}.txt'), genotype, fmt='%s')
            np.savetxt(os.path.join(pathout, "sciterna_complete_mut_indicator",
                                    f'sciterna_complete_mut_indicator_{r}r{i}.txt'),
                       complete_mut_indicator, fmt='%i')
            np.savetxt(os.path.join(pathout, "sciterna_individual_dropout_probs",
                                    f'sciterna_individual_dropout_probs_{r}r{i}.txt'), individual_dropout_probs)
            np.savetxt(os.path.join(pathout, "sciterna_individual_dropout_directions",
                                    f'sciterna_individual_dropout_directions_{r}r{i}.txt'),
                       individual_dropout_direction_probs)
            np.savetxt(os.path.join(pathout, "sciterna_individual_overdispersions_H",
                                    f'sciterna_individual_overdispersions_H_{r}r{i}.txt'),
                       individual_overdispersions_h)
            np.savetxt(os.path.join(pathout, "sciterna_global_parameters", f'sciterna_global_parameters_{r}r{i}.txt'),
                       [dropout_prob, dropout_direction_prob, overdispersion, error_rate, overdispersion_h])

    print('Done.')


if __name__ == '__main__':
    num_tests = 100
    n_cells_list = [100, 100, 50]
    n_mut_list = [50, 100, 100]
    clones = [5, 10, 20, ""]
    flipped_mutation_direction = False

    for clone in clones:
        for num_cells, num_mut in zip(n_cells_list, n_mut_list):
            data_path = f'../data/simulated_data/{num_cells}c{num_mut}m{clone}c'
            generate_comparison_data(num_cells, num_mut, num_tests, path=data_path, n_clones=clone)
            generate_sciterna_results(data_path, num_tests, data_path, ["c", "m"], flipped_mutation_direction,
                                      num_mut)
