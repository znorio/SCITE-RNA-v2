//
// Created by Norio on 05.04.2025.
//

#ifndef SCITE_RNA_GENERATE_RESULTS_H
#define SCITE_RNA_GENERATE_RESULTS_H


#include "mutation_filter.h"
#include "swap_optimizer.h"

std::vector<std::vector<std::string>> convert_location_to_gene(const std::vector<std::string>& locations, const std::string& gene_file_path);
void create_directories(const std::string& pathout);
void process_rounds(
        MutationFilter& mf,
        SwapOptimizer& optimizer,
        const std::vector<std::vector<int>>& ref,
        const std::vector<std::vector<int>>& alt,
        int n_snvs,
        int n_rounds,
        const std::string& pathout,
        int i,
        std::vector<int> selected,
        const std::vector<char>& gt1,
        const std::vector<char>& gt2,
        const std::vector<char>& not_selected_genotypes
);
void generate_sciterna_simulation_results(const std::string& path,
                                          const std::string& pathout,
                                          int n_tests,
                                          const std::vector<std::string>& tree_space,
                                          bool flipped_mutation_direction,
                                          int n_keep,
                                          int n_rounds,
                                          bool save_runtimes);

#endif //SCITE_RNA_GENERATE_RESULTS_H
