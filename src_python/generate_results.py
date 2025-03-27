import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import json

from src_python.swap_optimizer import SwapOptimizer
from src_python.noise_mutation_filter import MutationFilter
from src_python.utils import load_config_and_set_random_seed, create_genotype_matrix, create_mutation_matrix

config = load_config_and_set_random_seed()


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


def create_directories(pathout):
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


def process_rounds(mf, ref, alt, n_snvs, n_rounds, optimizer, pathout, i, selected, gt1, gt2, not_selected_genotypes):
    individual_dropout_probs = (np.ones(n_snvs, dtype=float) * config["dropout_alpha"] /
                                (config["dropout_alpha"] + config["dropout_beta"]))
    individual_dropout_direction_probs = np.ones(n_snvs, dtype=float) * config["dropout_dir_alpha"] / (
            config["dropout_dir_alpha"] + config["dropout_dir_beta"])
    alpha_h = beta_h = 0.5 * config["overdispersion_h"]
    individual_alphas_h = np.ones(n_snvs, dtype=float) * alpha_h
    individual_betas_h = np.ones(n_snvs, dtype=float) * beta_h

    for r in range(n_rounds):
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


def generate_sciterna_simulation_results(path="./comparison_data/", pathout="./comparison_data/results", n_tests=100,
                                         tree_space=None, flipped_mutation_direction=True, n_keep=50, n_rounds=3):
    """
    Runs SCITE-RNA on the data in the input path and saves the results in the output path.

    Arguments:
        path: input path where the data is stored
        pathout: output path where the results will be stored
        n_tests: number of simulated samples
        tree_space: tree spaces that will be searched and the order of the search
        flipped_mutation_direction: whether the root genotype can be changed during optimization
        n_keep: number of mutations to keep in the preprocessing step
        n_rounds: number of rounds of SCITE-RNA to optimize the SNV specific parameters like dropout probabilities
    """

    if tree_space is None:
        tree_space = ["c", "m"]

    optimizer = SwapOptimizer(spaces=tree_space, flipped_mutation_direction=flipped_mutation_direction)
    create_directories(pathout)

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

        selected, gt1, gt2, not_selected_genotypes = mf.filter_mutations(ref, alt, method="first_k", n_exp=n_keep)
        process_rounds(mf, ref, alt, n_snvs, n_rounds, optimizer, pathout, i, selected, gt1, gt2,
                       not_selected_genotypes)

    print("Done.")


def generate_sciterna_results(path="./comparison_data/", pathout="./comparison_data/results",
                              n_bootstrap=100, use_bootstrap=True, tree_space=None,
                              flipped_mutation_direction=True, n_keep=50, n_rounds=3,
                              only_preprocessing=False):
    """
    Runs SCITE-RNA on the data in the input path and saves the results in the output path.

    Arguments:
        path: input path where the data is stored
        pathout: output path where the results will be stored
        n_bootstrap: number of bootstrap sample trees to generate
        use_bootstrap: whether to use bootstrap sampling
        tree_space: tree spaces that will be searched and the order of the search
        flipped_mutation_direction: whether thr root genotype can be changed during optimization
        n_keep: number of mutations to keep in the preprocessing step
        n_rounds: number of rounds of SCITE-RNA to optimize the SNV specific parameters like dropout probabilities
        only_preprocessing: whether to only run the mutation filtering step
    """

    reference = pd.read_csv(os.path.join(path, "ref.csv"), index_col=0)
    alternative = pd.read_csv(os.path.join(path, "alt.csv"), index_col=0)
    ref = np.nan_to_num(np.array(reference), nan=0)[:, :]
    alt = np.nan_to_num(np.array(alternative), nan=0)[:, :]

    mf = MutationFilter(error_rate=config["error_rate"], overdispersion=config["overdispersion"],
                        genotype_freq=config["genotype_freq"], mut_freq=config["mut_freq"],
                        dropout_alpha=config["dropout_alpha"], dropout_beta=config["dropout_beta"],
                        dropout_dir_alpha=config["dropout_dir_alpha"],
                        dropout_dir_beta=config["dropout_dir_beta"],
                        overdispersion_h=config["overdispersion_h"])

    n_cells, n_snvs = ref.shape

    np.random.seed(config["random_seed"])

    print("Preprocessing data...")
    selected, gt1, gt2, not_selected_genotypes = mf.filter_mutations(ref, alt, method="first_k", n_exp=n_keep)

    if use_bootstrap:
        indices = np.random.choice(len(selected), (n_bootstrap, len(selected)), replace=True)

        selected = np.array(selected)[indices]
        gt1 = np.array(gt1)[indices]
        gt2 = np.array(gt2)[indices]
        not_selected_genotypes = np.array(not_selected_genotypes)
        b = "bootstrap_"
    else:
        b = ""

    selected_genes = np.array(reference.columns)[selected]

    os.makedirs(pathout, exist_ok=True)
    np.savetxt(os.path.join(pathout, f"{b}selected.txt"), selected, fmt='%d', delimiter=',')
    np.savetxt(os.path.join(pathout, f"{b}gt1.txt"), gt1, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(pathout, f"{b}gt2.txt"), gt2, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(pathout, f"{b}not_selected_genotypes.txt"), not_selected_genotypes, fmt='%s',
               delimiter=',')
    np.savetxt(os.path.join(pathout, f"{b}selected_genes.txt"), selected_genes, fmt='%s', delimiter=',')

    if not only_preprocessing:
        optimizer = SwapOptimizer(spaces=tree_space, flipped_mutation_direction=flipped_mutation_direction)

        os.makedirs(os.path.join(pathout, "sciterna_selected_genes"), exist_ok=True)
        create_directories(pathout)

        print(f"Running inference on data in {path}")

        if use_bootstrap:
            for i in tqdm(range(0, n_bootstrap)):
                b_selected = selected[i]
                b_gt1, b_gt2 = gt1[i], gt2[i]
                # b_not_selected_genotypes = not_selected_genotypes[i]

                b_selected_loci = reference.columns[b_selected]
                b_selected_genes = convert_location_to_gene(b_selected_loci)

                with open(os.path.join(pathout, "sciterna_selected_genes", f"selected_genes_{i}.json"), 'w') as file:
                    json.dump(b_selected_genes, file)

                # TODO To be able to update the individual dropout probabilities, dropout direction probabilities and
                # overdispersions, the process_rounds function needs to be be able to dtermine the genotype matrix.
                # even for SNVs not selected in the bootstrap sample.
                # process_rounds(mf, ref, alt, n_snvs, n_rounds, optimizer, pathout, i, b_selected, b_gt1, b_gt2,
                #                b_not_selected_genotypes)

        else:
            selected_genes = convert_location_to_gene(selected)
            with open(os.path.join(pathout, "sciterna_selected_genes", f"selected_genes.json"), 'w') as file:
                json.dump(selected_genes, file)

            process_rounds(mf, ref, alt, n_snvs, n_rounds, optimizer, pathout, "", selected, gt1, gt2,
                           not_selected_genotypes)

    print("Done.")
