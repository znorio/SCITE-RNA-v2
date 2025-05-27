/*
Script used to run SCITE-RNA for different modes of tree space optimization. These are optimizing only in the mutation tree space (m),
optimizing only cell lineage trees (c) and alternating between the two spaces, starting either in the cell lineage space (c,m)
or in the mutation tree space (m,c).
*/

#include <vector>
#include <string>
#include <mutation_filter.h>

#include "generate_results.h"
//
// run inference with SCITE-RNA
int main() {
    int n_tests = 100; //number of runs
    int n_rounds = 3; //number of optimization rounds
    std::vector<int> n_cells_list = {500};
    std::vector<int> n_mut_list = {500};
    std::vector<std::vector<std::string>> tree_spaces = {{"m"}, {"c"}, {"c", "m"}, {"m", "c"}}; // {"m"}, {"c"}, {"c", "m"},  {"m", "c"}
    bool flipped_mutation_direction = false;

    for (const auto& space : tree_spaces) {
        for (int i = 0; i < n_cells_list.size(); ++i) {
            int n_cells = n_cells_list[i];
            int n_mut = n_mut_list[i];
            std::string path = "../data/simulated_data/" + std::to_string(n_cells) + "c" + std::to_string(n_mut) + "m";
//            std::string path = "/cluster/work/bewi/members/znorio/data/simulated_data/" + std::to_string(n_cells) + "c" + std::to_string(n_mut) + "m";
            std::string path_results = path + "/sciterna_tree_space_comparison_cpp_";

            for (auto it = space.begin(); it != space.end(); ++it) {
                path_results += *it;
                if (std::next(it) != space.end()) {
                    path_results += "_";
                }
                else{
                    path_results += "/";
                }
            }
            generate_sciterna_simulation_results(path, path_results, n_tests, space, flipped_mutation_direction, n_mut, n_rounds);
        }
    }
    return 0;
}