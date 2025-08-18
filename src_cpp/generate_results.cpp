//
// Created by Norio on 05.04.2025.
//

#include <string>
#include <vector>
#include <filesystem>
#include <iostream>
#include <fstream>
#include "generate_results.h"
#include <algorithm>

#include <mutation_filter.h>
#include <swap_optimizer.h>
#include "utils.h"


std::vector<std::vector<std::string>> convert_location_to_gene(const std::vector<std::string>& locations, const std::string& gene_file_path) {
    std::vector<std::vector<std::string>> loc_to_gene;
    std::ifstream file(gene_file_path);
    std::string line;

    struct GeneRow {
        std::string chromosome;
        int start;
        int end;
        std::string gene;
    };

    std::vector<GeneRow> gene_rows;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string chrom, start, end, gene;
        std::getline(ss, chrom, ',');
        std::getline(ss, start, ',');
        std::getline(ss, end, ',');
        std::getline(ss, gene, ',');

        gene_rows.push_back({ chrom, std::stoi(start), std::stoi(end), gene });
    }

    for (const auto& loc : locations) {
        auto delim_pos = loc.find(':');
        std::string chrom = loc.substr(0, delim_pos);
        int pos = std::stoi(loc.substr(delim_pos + 1));

        std::vector<std::string> matching_genes;
        for (const auto& row : gene_rows) {
            if (row.chromosome == chrom && pos >= row.start && pos <= row.end) {
                matching_genes.push_back(row.gene);
            }
        }
        loc_to_gene.push_back(matching_genes);
    }

    return loc_to_gene;
}

void create_directories(const std::string& pathout, bool reduced_ouput) {
    std::vector<std::string> dirs = {};
    if (reduced_ouput) {
        dirs = {
            "sciterna_selected_loci",
            "sciterna_inferred_mut_types",
            "sciterna_parent_vec",
            "sciterna_individual_dropout_probs",
            "sciterna_individual_overdispersions_H",
            "sciterna_global_parameters",
            "sciterna_flipped",
            "sciterna_mutation_location"
        };
    }
    else{
        dirs = {
            "sciterna_selected_loci",
            "sciterna_inferred_mut_types",
            "sciterna_parent_vec",
            "sciterna_genotype",
            "sciterna_mut_indicator",
            "sciterna_complete_mut_indicator",
            "sciterna_individual_dropout_probs",
            "sciterna_individual_overdispersions_H",
            "sciterna_global_parameters",
            "sciterna_flipped",
            "sciterna_mutation_location",
        };
    }

    for (const auto& d : dirs) {
        std::filesystem::create_directories(std::filesystem::path(pathout) / d);
    }
}

std::vector<int> load_selected(const std::string& path) {
    std::vector<int> selected;
    std::ifstream file(path);
    int val;
    while (file >> val) {
        selected.push_back(val);
    }
    return selected;
}


std::vector<char> load_genotypes(const std::string& path) {
    std::vector<char> genotypes;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            genotypes.push_back(line[0]);  // Only take the first character
        }
    }
    return genotypes;
}

void process_rounds(MutationFilter &mf, SwapOptimizer &optimizer, const std::vector<std::vector<int>> &ref,
                    const std::vector<std::vector<int>> &alt, int n_snvs, int n_rounds, const std::string &pathout,
                    int i, std::vector<int> selected, const std::vector<char> &gt1, const std::vector<char> &gt2,
                    const std::vector<char> &not_selected_genotypes, int max_loops = 100, bool insert_nodes = true,
                    bool reduced_output = false) {

    load_config("../config/config.yaml");

    double dropout_alpha = std::stod(config_variables["dropout_alpha"]);
    double dropout_beta = std::stod(config_variables["dropout_beta"]);
    std::vector<double> individual_dropout_probabilities_(n_snvs, dropout_alpha / (dropout_alpha + dropout_beta));

    double overdispersion_het = std::stod(config_variables["overdispersion_h"]);

    std::vector<double> individual_overdispersions_h_(n_snvs, overdispersion_het);

    std::vector<double> dropout_probs_round = individual_dropout_probabilities_;
    std::vector<double> overdispersion_h_round = individual_overdispersions_h_;

    for (int r = 0; r < n_rounds; ++r) {

        auto [llh_1, llh_2] = mf.get_llh_mat(slice_columns(ref, selected),
                                             slice_columns(alt, selected), gt1, gt2, true,
                                             dropout_probs_round, overdispersion_h_round);


        optimizer.fit_llh(llh_1, llh_2);
        optimizer.optimize(max_loops, insert_nodes);

        const auto& flipped = optimizer.ct.flipped;
        auto mutation_matrix = create_mutation_matrix(optimizer.ct.parent_vector_ct, optimizer.ct.mut_loc, optimizer.ct);
        auto genotype = create_genotype_matrix(not_selected_genotypes, selected, gt1, gt2, mutation_matrix, flipped);
        std::vector<std::vector<int>> complete_mut_indicator(genotype.size(), std::vector<int>(genotype[0].size(), 0));

        for (size_t n = 0; n < selected.size(); ++n) {
            for (size_t j = 0; j < complete_mut_indicator.size(); ++j) {
                complete_mut_indicator[j][selected[n]] = mutation_matrix[j][n];
            }
        }

        auto params = mf.update_parameters(slice_columns(ref, selected), slice_columns(alt, selected), slice_columns_char(genotype, selected));
//        auto params = mf.update_parameters(ref, alt, genotype);
        auto [dropout_prob, overdispersion, error_rate, overdispersion_h,
                individual_dropouts, individual_overdispersions] = params;


        mf.update_alpha_beta(error_rate, overdispersion);
        dropout_probs_round = individual_dropouts;
        overdispersion_h_round = individual_overdispersions;

        save_vector_to_file(pathout + "/sciterna_selected_loci/sciterna_selected_loci_" + std::to_string(r) + "r" + std::to_string(i) + ".txt", selected);
        save_char_matrix_to_file(pathout + "/sciterna_inferred_mut_types/sciterna_inferred_mut_types_" + std::to_string(r) + "r" + std::to_string(i) + ".txt", {gt1, gt2});
        save_vector_to_file(pathout + "/sciterna_parent_vec/sciterna_parent_vec_" + std::to_string(r) + "r" + std::to_string(i) + ".txt", optimizer.ct.parent_vector_ct);
        save_vector_to_file(pathout + "/sciterna_mutation_location/sciterna_mutation_location_" + std::to_string(r) + "r" + std::to_string(i) + ".txt", optimizer.ct.mut_loc);
        save_double_vector_to_file(pathout + "/sciterna_individual_dropout_probs/sciterna_individual_dropout_probs_" + std::to_string(r) + "r" + std::to_string(i) + ".txt", individual_dropouts);
        save_double_vector_to_file(pathout + "/sciterna_individual_overdispersions_H/sciterna_individual_overdispersions_H_" + std::to_string(r) + "r" + std::to_string(i) + ".txt", individual_overdispersions);
        save_double_vector_to_file(pathout + "/sciterna_global_parameters/sciterna_global_parameters_" + std::to_string(r) + "r" +
        std::to_string(i) + ".txt", {dropout_prob, overdispersion, error_rate, overdispersion_h});
        save_vector_to_file(pathout + "/sciterna_flipped/sciterna_flipped_" + std::to_string(r) + "r" + std::to_string(i) + ".txt", std::vector<int>(flipped.begin(), flipped.end()));

        if (!reduced_output){
            save_matrix_to_file(pathout + "/sciterna_mut_indicator/sciterna_mut_indicator_" + std::to_string(r) + "r" + std::to_string(i) + ".txt", mutation_matrix);
            save_char_matrix_to_file(pathout + "/sciterna_genotype/sciterna_genotype_" + std::to_string(r) + "r" + std::to_string(i) + ".txt", genotype);
            save_matrix_to_file(pathout + "/sciterna_complete_mut_indicator/sciterna_complete_mut_indicator_" + std::to_string(r) + "r" + std::to_string(i) + ".txt", complete_mut_indicator);
        }
    }
}


void generate_sciterna_simulation_results(
        const std::string& path = "./comparison_data/",
        const std::string& pathout = "./comparison_data/results",
        int n_tests = 100,
        const std::vector<std::string>& tree_space = {"c", "m"},
        bool flipped_mutation_direction = true,
        int n_keep = 50,
        int n_rounds = 3) {

    load_config("../config/config.yaml");

    create_directories(pathout, false);

    std::cout << "Running inference on data in " << path << std::endl;
    std::vector<double> runtimes;

    for (int i = 0; i < n_tests; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<int>> alt = load_txt(path + "/alt/alt_" + std::to_string(i) + ".txt");
        std::vector<std::vector<int>> ref = load_txt(path + "/ref/ref_" + std::to_string(i) + ".txt");

        int n_snvs = static_cast<int>(alt[0].size());
        int n_cells = static_cast<int>(alt.size());
        SwapOptimizer optimizer(tree_space, flipped_mutation_direction, n_snvs, n_cells);


        std::map<std::string, double> genotype_freq = {
                {"A", std::stod(config_variables["genotype_freq.  A"])},
                {"H", std::stod(config_variables["genotype_freq.  H"])},
                {"R", std::stod(config_variables["genotype_freq.  R"])},
        };

        MutationFilter mf(std::stod(config_variables["error_rate"]), std::stod(config_variables["overdispersion"]),
                          genotype_freq, std::stod(config_variables["mut_freq"]),
                          std::stod(config_variables["dropout_alpha"]), std::stod(config_variables["dropout_beta"]),
                          std::stod(config_variables["dropout_direction"]), std::stod(config_variables["overdispersion_h"]));

        auto [selected, gt1, gt2, not_selected_genotypes] = mf.filter_mutations(ref, alt, "first_k", 0.5, n_keep);
        process_rounds(mf, optimizer, ref, alt, n_snvs, n_rounds, pathout, i, selected, gt1, gt2,
                       not_selected_genotypes);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        runtimes.push_back(elapsed.count());
    }

    save_double_vector_to_file(pathout + "/sciterna_runtimes.txt", runtimes);
    std::cout << "Done." << std::endl;
}


void generate_sciterna_results(
        const std::vector<std::vector<int>>& ref,
        const std::vector<std::vector<int>>& alt,
        const std::string& path = "./comparison_data/",
        std::string pathout = "./comparison_data/results",
        int n_bootstrap = 100,
        bool use_bootstrap = true,
        const std::vector<std::string>& tree_space = {"c", "m"},
        bool flipped_mutation_direction = true,
        int n_keep = 50,
        double posterior_threshold = 0.9,
        int n_rounds = 3,
        bool only_preprocessing = false,
        const std::string& method = "threshold",
        bool insert_nodes = true,
        bool load_from_file = true,
        bool reduced_output = false) {

    load_config("../config/config.yaml");

    int n_cells = ref.size();

    std::map<std::string, double> genotype_freq = {
            {"A", std::stod(config_variables["genotype_freq.  A"])},
            {"H", std::stod(config_variables["genotype_freq.  H"])},
            {"R", std::stod(config_variables["genotype_freq.  R"])},
    };

    MutationFilter mf(std::stod(config_variables["error_rate"]), std::stod(config_variables["overdispersion"]),
                      genotype_freq, std::stod(config_variables["mut_freq"]),
                      std::stod(config_variables["dropout_alpha"]), std::stod(config_variables["dropout_beta"]),
                      std::stod(config_variables["dropout_direction"]), std::stod(config_variables["overdispersion_h"]));

    std::vector<int> selected;
    std::vector<char> gt1, gt2;
    std::vector<char> not_selected_genotypes;

    std::string b = use_bootstrap ? "_bootstrap" : "";
    pathout = pathout + b;

    if (load_from_file) {
        std::cout << "Loading selected mutations from file..." << std::endl;
        selected = load_selected(pathout + "/selected_by_distribution.txt");
        gt1 = load_genotypes(pathout + "/gt1_by_distribution.txt");
        gt2 = load_genotypes(pathout + "/gt2_by_distribution.txt");
        not_selected_genotypes = load_genotypes(pathout + "/not_selected_genotypes_by_distribution.txt");
    } else {
        std::cout << "Preprocessing data..." << std::endl;
        std::tie(selected, gt1, gt2, not_selected_genotypes)  =
                mf.filter_mutations(ref, alt, method, posterior_threshold, n_keep);

        save_vector_to_file(pathout + "/" + "selected.txt", selected);
        save_char_vector_to_file(pathout + "/" + "gt1.txt", gt1);
        save_char_vector_to_file(pathout + "/" + "gt2.txt", gt2);
        save_char_vector_to_file(pathout + "/" + "not_selected_genotypes.txt", not_selected_genotypes);
    }

    create_directories(pathout, reduced_output);

    if (!only_preprocessing) {
        SwapOptimizer optimizer(tree_space, flipped_mutation_direction, static_cast<int>(selected.size()), n_cells);

        create_directories(pathout + "/sciterna_selected_genes", reduced_output);

        std::cout << "Running inference on data in " << path << std::endl;

        if (use_bootstrap) {
            std::cout << "Running bootstrap..." << std::endl;
            for (int i = 0; i < n_bootstrap; ++i) {
                std::vector<int> b_selected;
                std::vector<char> b_gt1, b_gt2;

                static std::mt19937 rng(std::stoi(config_variables["random_seed"])); // Use a random seed

                std::uniform_int_distribution<int> dist(0, static_cast<int>(selected.size() - 1));

                for (int j = 0; j < selected.size(); ++j) {
                    int index = dist(rng);
                    b_selected.push_back(selected[index]);
                    b_gt1.push_back(gt1[index]);
                    b_gt2.push_back(gt2[index]);
                }

                process_rounds(mf, optimizer, ref, alt, static_cast<int>(b_selected.size()), n_rounds, pathout, i,
                               b_selected, b_gt1, b_gt2, not_selected_genotypes, 100, insert_nodes, reduced_output);
            }
        } else {
            process_rounds(mf, optimizer, ref, alt, static_cast<int>(selected.size()), n_rounds, pathout, 0, selected,
                           gt1, gt2, not_selected_genotypes, 100, insert_nodes, reduced_output);
        }
    }
    std::cout << "Done." << std::endl;
}
