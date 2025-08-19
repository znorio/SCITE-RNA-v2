"""
Script used to compare different modes of tree optimization. These are optimizing only in the mutation tree space (m),
optimizing only cell lineage trees (c) and alternating between the two spaces, starting either in the cell lineage
space (c,m) or in the mutation tree space (m,c).
"""

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json

from src_python.mutation_filter import MutationFilter
from src_python.cell_tree import CellTree
from src_python.utils import load_config_and_set_random_seed, path_len_dist, mut_count_distance
from src_python.generate_results import generate_sciterna_simulation_results

config = load_config_and_set_random_seed()


def run_sciterna_tree_inference(spaces, num_cells_list, num_mut_list, n_tests, flipped, compare_cpp, num_rounds):
    for space in spaces:
        for n_cells, n_mut in zip(num_cells_list, num_mut_list):
            path = f'../data/simulated_data/{n_cells}c{n_mut}m'
            path_results = os.path.join(path, f'sciterna_tree_space_comparison{compare_cpp}_{"_".join(space)}')
            generate_sciterna_simulation_results(path=path, pathout=path_results, n_tests=n_tests, tree_space=space,
                                                 flipped_mutation_direction=flipped, n_keep=n_mut, n_rounds=num_rounds)


def plot_results(num_cells_list, num_mut_list, optimal_tree_llh, n_rounds, title):
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    # each column corresponds to a different number of cells and mutations
    for s, (n_cells, n_mut) in enumerate(zip(num_cells_list, num_mut_list)): # s are columns
        #plot the upper row
        all_data, labels = prepare_plot_data(optimal_tree_llh, n_cells, n_mut, n_rounds)
        # all data has format {'m': [1,...], 'c': [1,...], 'c_m': [1,...], 'm_c': [1,...]} labels = ['m', 'c', 'c_m', 'm_c']
        plot_boxplot(axes[0, s], all_data, labels, n_cells, n_mut, f'{title} Predicted vs. True Tree')

        #plot the lower row
        cell_tree_likelihoods = np.array(all_data["c"])
        relative_data = {}
        for key in all_data.keys():
            relative_data[key] = np.array(all_data[key]) - cell_tree_likelihoods
        plot_boxplot(axes[1, s], relative_data, labels, n_cells, n_mut, f'{title} Predicted vs. Cell Tree')

    plt.tight_layout()
    plt.savefig(f"../data/results/figures/space_switching_{title}.pdf")
    plt.show()


def prepare_plot_data(optimal_tree_llh, n_cells, n_mut, num_rounds):
    all_data = {}
    labels = []
    for outer_key, inner_dict in optimal_tree_llh.items():
        all_data[outer_key] = []
        labels.append(outer_key)
        round_data = []
        for r in range(num_rounds):
            round_data.extend(inner_dict[f"{n_cells}_{n_mut}_{r}"])
        all_data[outer_key].extend(round_data)
    return all_data, labels


def plot_boxplot(ax, all_data, labels, n_cells, n_mut, ylabel):
    colors = ['lightblue', 'yellow', 'turquoise', 'lightgreen']
    meanprops = dict(color='black', linewidth=3)

    # Ensure all_data is a list of lists with consistent lengths
    all_data_list = [np.array(all_data[label]) for label in labels]

    box = ax.boxplot(all_data_list, showmeans=True, meanline=True, widths=0.8,
                     patch_artist=True, meanprops=meanprops, showfliers=False)

    for patch, color in zip(box['boxes'], colors[:len(box['boxes'])]):
        patch.set_facecolor(color)

    labels = [label.replace("_", "") for label in labels]
    ax.axhline(0, color='orange', linestyle='--', linewidth=2)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=24)
    ax.set_title(f'{n_cells} Cells, {n_mut} SNVs', fontsize=28, pad=15)
    ax.set_ylabel(ylabel, fontsize=19)

def load_and_plot_results(num_cells_list, num_mut_list, spaces, n_tests, compare_cpp, n_rounds, flipped_mutation_direction):
    mapping_dict = {'A': 1.0, 'H': 0.5, 'R': 0}
    vectorized_map = np.vectorize(lambda x: float(mapping_dict[x]))

    optimal_tree_llhs = {}
    path_len_distances = {}
    vaf_distances = {}
    mut_count_distances = {}
    for s, (n_cells, n_mut) in enumerate(zip(num_cells_list, num_mut_list)):
        file_path = rf"../data/results/figures/{n_cells}c{n_mut}m/optimal_tree_llh_comparison_{n_cells}c{n_mut}m.json"
        file_path_dist = rf"../data/results/figures/{n_cells}c{n_mut}m/path_len_distance_comparison_{n_cells}c{n_mut}m.json"
        file_path_vaf = rf"../data/results/figures/{n_cells}c{n_mut}m/vaf_comparison_{n_cells}c{n_mut}m.json"
        file_path_mut_dist = rf"../data/results/figures/{n_cells}c{n_mut}m/mut_count_distance_comparison_{n_cells}c{n_mut}m.json"
        if not os.path.exists(file_path_vaf):
            os.makedirs(rf"../data/results/figures/{n_cells}c{n_mut}m", exist_ok=True)
            optimal_tree_llh = {}
            path_len_distance = {}
            vaf_distance = {}
            mut_count_dist = {}
            for space in spaces:
                optimal_tree_llh["_".join(space)] = {}
                path_len_distance["_".join(space)] = {}
                vaf_distance["_".join(space)] = {}
                mut_count_dist["_".join(space)] = {}
                path = f"../data/simulated_data/{n_cells}c{n_mut}m"
                path_results = os.path.join(path, f'sciterna_tree_space_comparison{compare_cpp}_{"_".join(space)}')

                mf = MutationFilter(error_rate=config["error_rate"], overdispersion=config["overdispersion"],
                                    genotype_freq=config["genotype_freq"], mut_freq=config["mut_freq"],
                                    dropout_alpha=config["dropout_alpha"], dropout_beta=config["dropout_beta"],
                                    dropout_direction_prob=config["dropout_direction"],
                                    overdispersion_h=config["overdispersion_h"])

                for r in range(n_rounds):
                    optimal_tree_llh["_".join(space)][f"{n_cells}_{n_mut}_{r}"] = []
                    path_len_distance["_".join(space)][f"{n_cells}_{n_mut}_{r}"] = []
                    vaf_distance["_".join(space)][f"{n_cells}_{n_mut}_{r}"] = []
                    mut_count_dist["_".join(space)][f"{n_cells}_{n_mut}_{r}"] = []

                    for i in tqdm(range(n_tests)):
                        sciterna_parent_vec = np.loadtxt(os.path.join(path_results, f'sciterna_parent_vec/sciterna_parent_vec_{r}r{i}.txt'), dtype=int)
                        true_parent_vec = np.loadtxt(os.path.join(path, f'parent_vec/parent_vec_{i}.txt'), dtype=int)
                        ref = np.loadtxt(os.path.join(path, f'ref/ref_{i}.txt'))
                        alt = np.loadtxt(os.path.join(path, f'alt/alt_{i}.txt'))
                        genotype_pred = np.loadtxt(os.path.join(path_results, "sciterna_genotype", f"sciterna_genotype_{r}r{i}.txt"), dtype=str)
                        gt = np.loadtxt(os.path.join(path, "genotype", f"genotype_{i}.txt"), dtype=str)

                        genotype_predicted = vectorized_map(genotype_pred)
                        genotype = vectorized_map(gt)

                        # mut_indicator = np.loadtxt(os.path.join(path, f'mut_indicator/mut_indicator_{i}.txt'))
                        # rows_to_zero = np.all(mut_indicator == 1, axis=1)  # if all cells are mutated = no mutations
                        # mut_indicator[rows_to_zero] = 0
                        selected = np.loadtxt(os.path.join(path_results, f'sciterna_selected_loci/sciterna_selected_loci_{r}r{i}.txt'), dtype=int)
                        gt1, gt2 = np.loadtxt(os.path.join(path_results, f'sciterna_inferred_mut_types/sciterna_inferred_mut_types_{r}r{i}.txt'), dtype=str)
                        llh_1, llh_2 = mf.get_llh_mat(ref[:, selected], alt[:, selected], gt1, gt2)

                        # prepare for joint calculation
                        n_cells = int((len(true_parent_vec) + 1) / 2)
                        ct_gt = CellTree(n_cells, n_mut, flipped_mutation_direction)
                        ct_gt.fit_llh(llh_1, llh_2)
                        ct_sciterna = CellTree(n_cells, n_mut, flipped_mutation_direction)
                        ct_sciterna.fit_llh(llh_1, llh_2)

                        # calculate differences in joint likelihood
                        ct_gt.use_parent_vec(true_parent_vec)
                        ct_gt.update_all()
                        true_joint = ct_gt.joint

                        ct_sciterna.use_parent_vec(sciterna_parent_vec)
                        ct_sciterna.update_all()
                        optimal_tree_llh["_".join(space)][f"{n_cells}_{n_mut}_{r}"].append(
                            (ct_sciterna.joint - true_joint) / (ct_sciterna.n_cells * ct_sciterna.n_mut))

                        path_len_distance["_".join(space)][f"{n_cells}_{n_mut}_{r}"].append(
                            path_len_dist(ct_gt, ct_sciterna))
                        vaf_distance["_".join(space)][f"{n_cells}_{n_mut}_{r}"].append(np.mean(np.abs(genotype_predicted - genotype)))
                        mut_count_dist["_".join(space)][f"{n_cells}_{n_mut}_{r}"].append(mut_count_distance(genotype, genotype_predicted))

            with open(file_path, "w") as f:
                json.dump(optimal_tree_llh, f)
            with open(file_path_dist, "w") as f:
                json.dump(path_len_distance, f)
            with open(file_path_vaf, "w") as f:
                json.dump(vaf_distance, f)
            with open(file_path_mut_dist, "w") as f:
                json.dump(mut_count_dist, f)

        with open(file_path, "r") as f:
            optimal_tree_llh = json.load(f)
        with open(file_path_dist, "r") as f:
            path_len_distance = json.load(f)
        with open(file_path_vaf, "r") as f:
            vaf_distance = json.load(f)
        with open(file_path_mut_dist, "r") as f:
            mut_count_dist = json.load(f)

        def merge_dicts(source, target):
            for key in source.keys():
                if key not in target:
                    target[key] = source[key]
                else:
                    for inner_key in source[key].keys():
                        target[key][inner_key] = source[key][inner_key]

        merge_dicts(optimal_tree_llh, optimal_tree_llhs)
        merge_dicts(path_len_distance, path_len_distances)
        merge_dicts(vaf_distance, vaf_distances)
        merge_dicts(mut_count_dist, mut_count_distances)

    plot_results(num_cells_list, num_mut_list, optimal_tree_llhs, n_rounds, "LL")
    plot_results(num_cells_list, num_mut_list, path_len_distances, n_rounds, "Pathlength")
    plot_results(num_cells_list, num_mut_list, vaf_distances, n_rounds, "VAF")
    plot_results(num_cells_list, num_mut_list, mut_count_distances, n_rounds, "MutationCountDist")


num_tests = 100  # Number of simulated samples
n_rounds = 3  # Number of rounds of SCITE-RNA to optimize the SNV specific parameters like dropout probabilities
n_cells_list = [100, 500, 500]
n_mut_list = [500, 500, 100]

tree_spaces = [["m"], ["c"], ["c", "m"], ["m", "c"]]

generate_results = True  # set to True to rerun the SCITE-RNA tree inference
flipped_mutation_direction = False  # flip mutations or not (change root genotype) -> only done in cell lineage trees
cpp = "_cpp"

if generate_results:
    run_sciterna_tree_inference(tree_spaces, n_cells_list, n_mut_list, num_tests, flipped_mutation_direction, cpp,
                                n_rounds)

load_and_plot_results(n_cells_list, n_mut_list, tree_spaces, num_tests, cpp, n_rounds, flipped_mutation_direction)