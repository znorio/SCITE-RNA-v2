import numpy as np
import os
from tqdm import tqdm

from src_python.swap_optimizer import SwapOptimizer
from src_python.noise_mutation_filter import MutationFilter
from src_python.utils import load_config_and_set_random_seed, create_genotype_matrix, create_mutation_matrix

config = load_config_and_set_random_seed()


def generate_sciterna_results(path="./comparison_data/", n_tests=100, pathout="./comparison_data/results",
                              tree_space=None, reverse_mutations=True, n_keep=50, n_rounds=3):

    """
    Runs SCITE-RNA on the data in the input path and saves the results in the output path.

    Arguments:
        path: input path where the data is stored
        n_tests: number of simulated samples
        pathout: output path where the results will be stored
        tree_space: tree spaces that will be searched and the order of the search
        reverse_mutations: whether thr root genotype can be changed during optimization
        n_keep: number of mutations to keep in the preprocessing step
        n_rounds: number of rounds of SCITE-RNA to optimize the SNV specific parameters like dropout probabilities
    """

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

    print(f"Running inference on data in {path}")

    for i in tqdm(range(0, n_tests)):
        alt = np.loadtxt(os.path.join(path, "alt", "alt_%i.txt" % i))
        ref = np.loadtxt(os.path.join(path, "ref", "ref_%i.txt" % i))

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
            selected, gt1, gt2, not_selected_genotypes = mf.filter_mutations(ref, alt, method="first_k", n_exp=n_keep)

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

            np.savetxt(
                os.path.join(pathout, "sciterna_selected_loci", f"sciterna_selected_loci_{r}r{i}.txt"),
                selected,
                fmt="%i"
            )
            np.savetxt(
                os.path.join(pathout, "sciterna_inferred_mut_types", f"sciterna_inferred_mut_types_{r}r{i}.txt"),
                np.stack((gt1, gt2), axis=0),
                fmt="%s"
            )
            np.savetxt(
                os.path.join(pathout, "sciterna_parent_vec", f"sciterna_parent_vec_{r}r{i}.txt"),
                optimizer.ct.parent_vec,
                fmt="%i"
            )
            np.savetxt(
                os.path.join(pathout, "sciterna_mut_indicator", f"sciterna_mut_indicator_{r}r{i}.txt"),
                mutation_matrix,
                fmt="%i"
            )
            np.savetxt(
                os.path.join(pathout, "sciterna_genotype", f"sciterna_genotype_{r}r{i}.txt"),
                genotype,
                fmt="%s"
            )
            np.savetxt(
                os.path.join(pathout, "sciterna_complete_mut_indicator",
                             f"sciterna_complete_mut_indicator_{r}r{i}.txt"),
                complete_mut_indicator,
                fmt="%i"
            )
            np.savetxt(
                os.path.join(pathout, "sciterna_individual_dropout_probs",
                             f"sciterna_individual_dropout_probs_{r}r{i}.txt"),
                individual_dropout_probs
            )
            np.savetxt(
                os.path.join(pathout, "sciterna_individual_dropout_directions",
                             f"sciterna_individual_dropout_directions_{r}r{i}.txt"),
                individual_dropout_direction_probs
            )
            np.savetxt(
                os.path.join(pathout, "sciterna_individual_overdispersions_H",
                             f"sciterna_individual_overdispersions_H_{r}r{i}.txt"),
                individual_overdispersions_h
            )
            np.savetxt(
                os.path.join(pathout, "sciterna_global_parameters", f"sciterna_global_parameters_{r}r{i}.txt"),
                [dropout_prob, dropout_direction_prob, overdispersion, error_rate, overdispersion_h]
            )

    print("Done.")
