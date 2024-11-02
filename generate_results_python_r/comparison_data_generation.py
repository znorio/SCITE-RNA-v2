"""
Script used to generate simulated datasets. The number of cells and SNVs and the number of clones simulated can be set.
Tree inference is run as well. However, for large datasets you might want to use generate_results_cpp/comparison_num_clones.cpp
for tree inference as it is faster.
"""

import sys, os
from tqdm import tqdm
import numpy as np
import yaml

sys.path.append('../')

from src_python.data_generator import DataGenerator
from src_python.mutation_tree import MutationTree
from src_python.mutation_filter import MutationFilter
from src_python.swap_optimizer import SwapOptimizer

with open('../config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def generate_comparison_data(n_cells: int, n_mut: int, size=100, path='./comparison_data/', seed=None, n_clones= None):
    if seed is not None:
        np.random.seed(seed)
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
    
    generator = DataGenerator(n_cells, n_mut)
    
    for i in tqdm(range(size)):
        ref, alt = generator.generate_reads(new_tree=True, new_mut_type=True, num_clones=n_clones)

        mut_indicator = np.zeros((n_cells, n_mut), dtype=bool)
        for j in range(generator.n_mut):
            if generator.gt1[j] == generator.gt2[j]:
                continue # not mutated
            for leaf in generator.ct.leaves(generator.ct.mut_loc[j]):
                mut_indicator[leaf, j] = True

        mt = MutationTree(n_mut=n_mut, n_cells=n_cells)
        mt.fit_cell_tree(generator.ct)

        np.savetxt(os.path.join(path, f'ref/ref_{i}.txt'), ref.T, fmt='%i')
        np.savetxt(os.path.join(path, f'alt/alt_{i}.txt'), alt.T, fmt='%i')
        np.savetxt(os.path.join(path, f'parent_vec/parent_vec_{i}.txt'), generator.ct.parent_vec, fmt='%i')
        np.savetxt(os.path.join(path, f'mut_indicator/mut_indicator_{i}.txt'), mut_indicator.T, fmt='%i')
        np.savetxt(os.path.join(path, f'genotype/genotype_{i}.txt'), generator.genotype.T, fmt='%s')

def create_genotype_matrix(not_selected_genotypes, selected, gt1, gt2, mutation_matrix, flipped):
    n_cells = mutation_matrix.shape[1]
    n_loci = len(selected) + len(not_selected_genotypes)
    genotype_matrix = np.full((n_loci, n_cells),"", dtype="str")
    not_selected = [i for i in range(n_loci) if i not in selected]

    # those not selected have a gt independent of the tree learning
    for n, locus in enumerate(not_selected):
        genotype_matrix[locus] = [not_selected_genotypes[n] for _ in range(n_cells)]
    # these are the genes selected for the tree learning
    for n, locus in enumerate(selected):
        if flipped[n]:
            genotype_matrix[locus] = np.where(mutation_matrix[n] == 0, gt2[n], np.where(mutation_matrix[n] == 1, gt1[n], mutation_matrix[n]))
        else:
            genotype_matrix[locus] = np.where(mutation_matrix[n] == 0, gt1[n], np.where(mutation_matrix[n] == 1, gt2[n], mutation_matrix[n]))
    return genotype_matrix

def create_mutation_matrix(parent_vector, mutation_indices, ct):
    n_cells = len(parent_vector)
    n_leaves = int((n_cells+1)/2)
    n_mutations = len(mutation_indices)

    # Initialize mutation matrix with zeros
    mutation_matrix = np.zeros((n_cells, n_mutations), dtype=int)

    # Mark cells with mutations
    for mutation_idx, cell_idx in enumerate(mutation_indices):
        children = [c for c in ct.dfs(cell_idx)]
        for cell in children:  # Traverse all cells below the mutation cell
            mutation_matrix[cell, mutation_idx] = 1  # Mark cells with the mutation

    return mutation_matrix[:n_leaves].T

def generate_sciterna_results(path='./comparison_data/', n_tests=100, pathout="./comparison_data/results",
                              tree_space=["c"], reverse_mutations=True, n_keep=50):
    optimizer = SwapOptimizer(spaces = tree_space, reverse_mutations=reverse_mutations)

    os.makedirs(os.path.join(pathout, "sciterna_selected_loci"), exist_ok=True)
    os.makedirs(os.path.join(pathout, "sciterna_inferred_mut_types"), exist_ok=True)
    os.makedirs(os.path.join(pathout, "sciterna_parent_vec"), exist_ok=True)
    os.makedirs(os.path.join(pathout, "sciterna_genotype"), exist_ok=True)
    os.makedirs(os.path.join(pathout, "sciterna_mut_indicator"), exist_ok=True)
    os.makedirs(os.path.join(pathout, "sciterna_complete_mut_indicator"), exist_ok=True)

    print(f'Running inference on data in {path}')
    for i in tqdm(range(0, n_tests)):
        alt = np.loadtxt(os.path.join(path, "alt", 'alt_%i.txt' % i)).T
        ref = np.loadtxt(os.path.join(path, "ref", 'ref_%i.txt' % i)).T
        mut_indicator = np.loadtxt(os.path.join(path, f'mut_indicator/mut_indicator_{i}.txt'))

        mf = MutationFilter(f=config["f"], omega=config["omega"], h_factor=config["h_factor"], genotype_freq=config["genotype_freq"],
                            mut_freq=config["mut_freq"])
        selected, gt1, gt2, not_selected_genotypes = mf.filter_mutations(ref, alt, method='first_k', n_exp=n_keep, t=0.5) #  method='threshold', t=0.5,
        llh_1, llh_2 = mf.get_llh_mat(ref[:, selected], alt[:, selected], gt1, gt2)

        np.savetxt(os.path.join(pathout, "sciterna_selected_loci", f'sciterna_selected_loci_{i}.txt'), selected, fmt='%i')
        np.savetxt(os.path.join(pathout, "sciterna_inferred_mut_types", f'sciterna_inferred_mut_types_{i}.txt'),
                   np.stack((gt1, gt2), axis=0), fmt='%s')

        optimizer.fit_llh(llh_1, llh_2)
        optimizer.optimize()

        np.savetxt(os.path.join(pathout, "sciterna_parent_vec", f'sciterna_parent_vec_{i}.txt'), optimizer.ct.parent_vec,
                   fmt='%i')
        mutation_matrix = create_mutation_matrix(optimizer.ct.parent_vec, optimizer.ct.mut_loc, optimizer.ct)
        np.savetxt(os.path.join(pathout, "sciterna_mut_indicator", f'sciterna_mut_indicator_{i}.txt'),
                   mutation_matrix, fmt='%i')
        flipped = optimizer.ct.flipped
        genotype = create_genotype_matrix(not_selected_genotypes, selected, gt1, gt2, mutation_matrix, flipped)
        np.savetxt(os.path.join(pathout, "sciterna_genotype", f'sciterna_genotype_{i}.txt'), genotype, fmt='%s')
        complete_mut_indicator = np.zeros(genotype.shape, dtype=int)
        for n, sel in enumerate(selected):
            complete_mut_indicator[sel] = mutation_matrix[n]

        np.savetxt(os.path.join(pathout, "sciterna_complete_mut_indicator", f'sciterna_complete_mut_indicator_{i}.txt'),
                   complete_mut_indicator, fmt='%i')
    print('Done.')


if __name__ == '__main__':
    n_tests = 100
    n_cells_list = [200, 300]
    n_mut_list = [200, 300]
    clones = [5, 10, 20, ""]
    flipped = False

    for clone in clones:
        for n_cells, n_mut in zip(n_cells_list, n_mut_list):
            path = f'../data/simulated_data/{n_cells}c{n_mut}m{clone}'
            generate_comparison_data(n_cells, n_mut, n_tests, path=path, n_clones = clone)
            generate_sciterna_results(path, n_tests, path,["c", "m"], flipped, n_mut)