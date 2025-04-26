/*
Script used to run SCITE-RNA on simulated data with a variable number of cells, SNVs and clones.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

#include "utils.h"
#include "generate_results.h"


// run inference with SCITE-RNA
int main() {
    int n_tests = 100; //number of runs
    int n_rounds = 3; //number of optimization rounds
    std::vector<int> n_cells_list = {500, 100, 500};
    std::vector<int> n_mut_list = {100, 500, 500};
    std::vector<std::string> tree_space = {"c", "m"};
    std::vector<std::string> clones =  {""}; // {"5", "10", "20", ""};
    bool flipped_mutation_direction = true;

    load_config("../config/config.yaml");
    std::cout << "Random seed: " << config_variables["random_seed"] << std::endl;

    for (const auto& clone : clones) {
        for (int i = 0; i < n_cells_list.size(); ++i) {
            int n_cells = n_cells_list[i];
            int n_mut = n_mut_list[i];
            std::string path = "../data/simulated_data/" + std::to_string(n_cells) + "c" + std::to_string(n_mut) + "m" + clone +"/";
//            std::string path = "/cluster/work/bewi/members/znorio/data/simulated_data/" + std::to_string(n_cells) + "c" + std::to_string(n_mut) + "m" + clone +"/";
            std::string pathout = path +  "sciterna/";
            generate_sciterna_simulation_results(path, pathout, n_tests, tree_space, flipped_mutation_direction, n_mut, n_rounds);
        }
    }
    return 0;
}