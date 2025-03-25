import numpy as np
import os
import yaml


def load_config_and_set_random_seed():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config", "config.yaml")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    seed = config["random_seed"]
    np.random.seed(seed)

    return config


def create_genotype_matrix(not_selected_genotypes, selected, gt1, gt2, mutation_matrix, flipped):
    n_cells = mutation_matrix.shape[0]
    n_loci = len(selected) + len(not_selected_genotypes)
    genotype_matrix = np.full((n_cells, n_loci), "", dtype="str")
    not_selected = [i for i in range(n_loci) if i not in selected]

    # those not selected have a gt independent of the tree learning
    for n, locus in enumerate(not_selected):
        genotype_matrix[:, locus] = [not_selected_genotypes[n] for _ in range(n_cells)]

    # these are the genes selected for the tree learning
    for n, locus in enumerate(selected):
        if flipped[n]:
            genotype_matrix[:, locus] = np.where(mutation_matrix[:, n] == 0, gt2[n],
                                                 np.where(mutation_matrix[:, n] == 1, gt1[n], mutation_matrix[:, n]))
        else:
            genotype_matrix[:, locus] = np.where(mutation_matrix[:, n] == 0, gt1[n],
                                                 np.where(mutation_matrix[:, n] == 1, gt2[n], mutation_matrix[:, n]))
    return genotype_matrix


def create_mutation_matrix(parent_vector, mutation_indices, ct):
    n_nodes = len(parent_vector)
    n_cells = int((n_nodes + 1) / 2)
    n_mutations = len(mutation_indices)

    mutation_matrix = np.zeros((n_nodes, n_mutations), dtype=int)

    # Mark cells with mutations
    for mutation_idx, cell_idx in enumerate(mutation_indices):
        children = [c for c in ct.dfs(cell_idx)]
        for cell in children:  # Traverse all cells below the mutation cell
            mutation_matrix[cell, mutation_idx] = 1  # Mark cells with the mutation

    return mutation_matrix[:n_cells]
