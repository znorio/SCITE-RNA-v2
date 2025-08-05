"""
Script used to generate simulated datasets. The number of cells and SNVs and the number of clones simulated can be set.
Tree inference is run as well. However, for large datasets you might want to use
generate_results_cpp/comparison_num_clones.cpp for tree inference as it is faster.
"""

from src_python.data_generator import DataGenerator
from src_python.mutation_tree import MutationTree
from src_python.utils import load_config_and_set_random_seed
from src_python.generate_results import generate_sciterna_simulation_results

import os
from tqdm import tqdm
import numpy as np

config = load_config_and_set_random_seed()


def generate_comparison_data(n_cells: int, n_mut: int, size=100, path='./comparison_data/', random_seed=None,
                             n_clones=None, coverage_method="zinb"):
    if random_seed is not None:
        np.random.seed(random_seed)

    # if os.path.exists(path):
    #     while True:
    #         ans = input(f'Directory {path} already exists. Existing files will be overwritten. Continue? [Y/N] ')
    #         match ans:
    #             case 'Y' | 'y' | 'Yes' | 'yes':
    #                 break
    #             case 'N' | 'n' | 'No' | 'no':
    #                 return
    # else:
    #     os.makedirs(path)

    os.makedirs(os.path.join(path, "ref"), exist_ok=True)
    os.makedirs(os.path.join(path, "alt"), exist_ok=True)
    os.makedirs(os.path.join(path, "parent_vec"), exist_ok=True)
    os.makedirs(os.path.join(path, "mut_indicator"), exist_ok=True)
    os.makedirs(os.path.join(path, "genotype"), exist_ok=True)
    os.makedirs(os.path.join(path, "dropout_probs"), exist_ok=True)
    os.makedirs(os.path.join(path, "dropout_directions"), exist_ok=True)
    os.makedirs(os.path.join(path, "overdispersions_H"), exist_ok=True)
    os.makedirs(os.path.join(path, "mutation_location"), exist_ok=True)

    generator = DataGenerator(n_cells, n_mut, coverage_method=coverage_method)

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
        np.savetxt(os.path.join(path, f'mutation_location/mutation_location_{i}.txt'), generator.ct.mut_loc, fmt='%i')

num_tests = 100  # Number of simulated samples
n_rounds = 1  # Number of rounds of SCITE-RNA to optimize the SNV specific parameters like dropout probabilities
n_cells_list = [50]
n_mut_list = [500]
clones = ["", 5, 10, 20]
flipped_mutation_direction = True
tree_space = ["c", "m"]
coverage_method = "zinb"

for clone in clones:
    for num_cells, num_mut in zip(n_cells_list, n_mut_list):
        data_path = f'../data/simulated_data/{num_cells}c{num_mut}m{clone}'
        generate_comparison_data(num_cells, num_mut, num_tests, path=data_path, n_clones=clone, coverage_method=coverage_method)
        # path_results = os.path.join(data_path, 'sciterna')
        # generate_sciterna_simulation_results(path=data_path, pathout=path_results, n_tests=num_tests,
        #                                      tree_space=tree_space,
        #                                      flipped_mutation_direction=flipped_mutation_direction,
        #                                      n_keep=num_mut, n_rounds=n_rounds)
