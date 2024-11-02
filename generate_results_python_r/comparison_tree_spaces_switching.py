"""
Script used to compare different modes of tree optimization. These are optimizing only in the mutation tree space (m),
optimizing only cell lineage trees (c) and alternating between the two spaces, starting either in the cell lineage space (c,m)
or in the mutation tree space (m,c).
"""

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import yaml

from src_python.mutation_filter import MutationFilter
from src_python.swap_optimizer import SwapOptimizer
from src_python.cell_tree import CellTree

with open('../config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

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


n_tests = 100
n_cells_list = [500, 100, 500]
n_mut_list = [100, 500, 500]

tree_spaces = [["m"], ["c"], ["c", "m"], ["m", "c"]]

generate_results = False # set to True to rerun the SCITE-RNA tree inference
flipped = False # flip mutations or not
cpp = "_cpp"


if generate_results:
    for space in tree_spaces:
        for n_cells, n_mut in zip(n_cells_list, n_mut_list):
            path = f'../data/simulated_data/{n_cells}c{n_mut}m'
            path_results = f'../data/simulated_data/{n_cells}c{n_mut}m/sciterna_tree_space_comparison{cpp}_{"_".join(space)}'
            generate_sciterna_results(path, n_tests, path_results, space, flipped, n_mut)

fig, axes = plt.subplots(2, 3, figsize=(22, 12))

for s, (n_cells, n_mut) in enumerate(zip(n_cells_list, n_mut_list)):
    optimal_tree_llh = {}
    file_path = rf"../data/results/figures/optimal_tree_llh_comparison_{n_cells}c{n_mut}m.json"
    if not os.path.exists(file_path):
        for space in tree_spaces:
            optimal_tree_llh["_".join(space)] = {}
            path = f"../data/simulated_data/{n_cells}c{n_mut}m"
            path_results = f'../data/simulated_data/{n_cells}c{n_mut}m/sciterna_tree_space_comparison{cpp}_{"_".join(space)}'
            optimal_tree_llh["_".join(space)][f"{n_cells}_{n_mut}"] = []
            mf = MutationFilter(f=config["f"], omega=config["omega"], h_factor=config["h_factor"], genotype_freq=config["genotype_freq"],
                            mut_freq=config["mut_freq"])
            for i in tqdm(range(n_tests)):
                sciterna_parent_vec = np.loadtxt(os.path.join(path_results, f'sciterna_parent_vec/sciterna_parent_vec_{i}.txt'), dtype=int)
                true_parent_vec = np.loadtxt(os.path.join(path, f'parent_vec/parent_vec_{i}.txt'), dtype=int)
                ref = np.loadtxt(os.path.join(path, f'ref/ref_{i}.txt')).T
                alt = np.loadtxt(os.path.join(path, f'alt/alt_{i}.txt')).T
                mut_indicator = np.loadtxt(os.path.join(path, f'mut_indicator/mut_indicator_{i}.txt'))
                rows_to_zero = np.all(mut_indicator == 1, axis=1)  # if all cells are mutated = no mutations
                mut_indicator[rows_to_zero] = 0
                selected = np.loadtxt(os.path.join(path_results, f'sciterna_selected_loci/sciterna_selected_loci_{i}.txt'), dtype=int)
                gt1, gt2 = np.loadtxt(os.path.join(path_results, f'sciterna_inferred_mut_types/sciterna_inferred_mut_types_{i}.txt'), dtype=str)
                # sciterna_mut_indicator = np.loadtxt(os.path.join(path_results, f"sciterna_complete_mut_indicator/sciterna_complete_mut_indicator_{i}.txt"), dtype=int)
                llh_1, llh_2 = mf.get_llh_mat(ref[:, selected], alt[:, selected], gt1, gt2)

                # prepare for joint calculation
                n_cells = int((len(true_parent_vec) + 1) / 2)
                ct = CellTree(n_cells)
                ct.fit_llh(llh_1, llh_2)

                # calculate differences in joint likelihood
                ct.use_parent_vec(true_parent_vec)
                ct.update_all()
                true_joint = ct.joint

                ct.use_parent_vec(sciterna_parent_vec)
                ct.update_all()
                optimal_tree_llh["_".join(space)][f"{n_cells}_{n_mut}"].append((ct.joint-true_joint)/(ct.n_cells * ct.n_mut))
                # optimal_tree_llh["_".join(space)][f"{n_cells}_{n_mut}"].append(np.mean(np.abs(sciterna_mut_indicator-mut_indicator)))

        with open(file_path, "w") as f:
            json.dump(optimal_tree_llh, f)

    with open(file_path, "r") as f:
        optimal_tree_llh = json.load(f)

    all_data = []
    labels = []
    for outer_key, inner_dict in optimal_tree_llh.items():
        labels.append(outer_key)
        all_data.append(np.array(inner_dict[f"{n_cells}_{n_mut}"]))

    colors = ['lightblue', 'yellow', 'turquoise', 'lightgreen']
    meanprops = dict(color='black', linewidth=3)
    box = axes[0, s].boxplot(np.array(all_data).T, showmeans=True, meanline=True, widths=0.8,
                             patch_artist=True, meanprops=meanprops, showfliers=False)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    for median, color in zip(box['medians'], colors):
        median.set_color(color)

    labels = ["m", "c", "cm", "mc"]
    means = np.mean(all_data, axis=1)
    legend_entries = []
    for j, mean in enumerate(means):
        legend_entries.append(f'Mean Log Likelihood {labels[j]}: {mean:.5f}')

    handles = [plt.Line2D([0], [0], color=colors[j], linestyle='--', linewidth=1, label=legend_entries[j])
               for j in range(len(colors))]

    axes[0, s].legend(handles=handles, loc='lower right', fontsize=16)
    axes[0, s].axhline(0, color='orange', linestyle='--', linewidth=2)
    axes[0, s].set_xticks(range(1, len(labels) + 1))
    axes[0, s].set_xticklabels(labels, rotation=90, fontsize=24)
    axes[0, s].set_title(f'{n_cells} Cells, {n_mut} SNVs', fontsize=28, pad=15)

    if s == 0:
        axes[0, s].set_ylabel('Normalized Log Likelihood vs. True Tree', fontsize=19)

    # Lower plots: subtract cell_trees

    all_data = []
    labels = []
    cell_trees = np.array(optimal_tree_llh["c"][f"{n_cells}_{n_mut}"])
    for outer_key, inner_dict in optimal_tree_llh.items():
        labels.append(outer_key)
        all_data.append(np.array(inner_dict[f"{n_cells}_{n_mut}"]) - cell_trees)

    labels = ["m", "c", "cm", "mc"]
    colors = ['lightblue', 'yellow', 'turquoise', 'lightgreen']
    meanprops = dict(color='black', linewidth=3)
    box = axes[1, s].boxplot(np.array(all_data).T, showmeans=True, meanline=True, widths=0.8,
                             patch_artist=True, meanprops=meanprops, showfliers=False)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    for median, color in zip(box['medians'], colors):
        median.set_color(color)

    axes[1, s].axhline(0, color='orange', linestyle='--', linewidth=2)
    axes[1, s].set_xticks(range(1, len(labels) + 1))
    axes[1, s].set_xticklabels(labels, rotation=90, fontsize=24)

    if s == 0:
        axes[1, s].set_ylabel('Normalized Log Likelihood vs. Cell Tree', fontsize=19)

plt.tight_layout()
plt.savefig("../data/results/figures/space_switching.png")
plt.show()