"""
This module provides utility functions for loading configuration, setting random seeds,
and creating genotype and mutation matrices for tree learning algorithms.
"""

import numpy as np
import os
import yaml
from scipy.spatial.distance import pdist, squareform


def leaf_dist_mat(ct, unrooted=False):
    ''' Distance matrix for all leaves in a cell lineage tree '''
    result = - np.ones((ct.n_cells, ct.n_vtx), dtype=int)
    np.fill_diagonal(result, 0)

    for vtx in ct.rdfs_experimental(ct.main_root):
        if ct.isleaf(vtx):
            continue

        dist_growth = 1 if unrooted and vtx == ct.main_root else 2

        children = ct.children(vtx)
        for child in children:
            for leaf in ct.leaves(child):
                result[leaf, vtx] = result[leaf, child] + 1

        if len(children) < 2:
            continue

        for leaf1 in ct.leaves(children[0]):
            for leaf2 in ct.leaves(children[1]):
                dist = result[leaf1, children[0]] + result[leaf2, children[1]] + dist_growth
                result[leaf1, leaf2] = dist
                result[leaf2, leaf1] = dist

    return result[:, :ct.n_cells]


def path_len_dist(ct1, ct2, unrooted=False):
    '''
    MSE between the distance matrices of two cell/mutation trees
    The MSE (excluding the diagonal), only distances between leaf cells
    '''

    dist_mat1, dist_mat2 = leaf_dist_mat(ct1, unrooted), leaf_dist_mat(ct2, unrooted)
    denominator = (dist_mat1.size - dist_mat1.shape[0])
    return np.sum(np.abs(dist_mat1 - dist_mat2)) / denominator
    # return np.sum((dist_mat1 - dist_mat2)**2) / denominator

def mut_count_distance(genotype_matrix1, genotype_matrix2):
    # Compute the pairwise Hamming distances
    hamming_distances = pdist(genotype_matrix1, metric='hamming')
    distance_matrix1 = squareform(hamming_distances)
    distance_matrix1 *= genotype_matrix1.shape[1]

    hamming_distances = pdist(genotype_matrix2, metric='hamming')
    distance_matrix2 = squareform(hamming_distances)
    distance_matrix2 *=  genotype_matrix2.shape[1]

    denominator = (distance_matrix1.size - distance_matrix1.shape[0])
    return np.sum(np.abs(distance_matrix1 - distance_matrix2)) / denominator
    # return np.sum((distance_matrix1 - distance_matrix2)**2) / denominator


def load_config_and_set_random_seed():
    """
    Loads configuration from a YAML file and sets a random seed in numpy.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config", "config.yaml")

    with open(config_path, 'r') as file:
        configuration = yaml.safe_load(file)

    seed = configuration["random_seed"]
    np.random.seed(seed)

    return configuration


def create_genotype_matrix(not_selected_genotypes, selected, gt1, gt2, mutation_matrix, flipped):
    """
    Creates a genotype matrix based on selected and not selected genotypes.

    Arguments:
        not_selected_genotypes (list): SNVs not selected for tree learning.
        selected (list): Indices of selected SNVs.
        gt1 (list): Genotype 1 values.
        gt2 (list): Genotype 2 values.
        mutation_matrix (np.ndarray): Matrix indicating mutations.
        flipped (np.ndarray): Array indicating flipped mutations.
    """
    n_cells = mutation_matrix.shape[0]
    n_loci = len(selected) + len(not_selected_genotypes)
    genotype_matrix = np.full((n_cells, n_loci), "", dtype="str")
    not_selected = [i for i in range(n_loci) if i not in selected]

    # those not selected have a gt independent of the tree learning
    if len(not_selected_genotypes) != len(not_selected):
        for locus in not_selected:
            genotype_matrix[:, locus] = 'X'
    else:
        for n, locus in enumerate(not_selected):
            genotype_matrix[:, locus] = not_selected_genotypes[n]

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
    """
    Creates a mutation matrix based on a cell lineage tree with parent vector and mutation indices.
    """
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
