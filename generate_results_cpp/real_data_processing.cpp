/*
Script used to run SCITE-RNA on the multiple myeloma dataset MM34.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <mutation_filter.h>

#include "utils.h"
#include <mutation_filter.h>
#include "generate_results.h"


int main() {
    int bootstrap_samples = 1000;
    bool use_bootstrap = true;
    int n_snps = 3000;
    double posterior_threshold = 0.05;
    std::string method = "threshold";
    int n_rounds = 3;
    std::string sample = "mm16";
    bool flipped_mutation_direction = true;
    bool only_preprocessing = false;
    std::vector<std::string> tree_space = {"c", "m"};
    bool reshuffle_nodes = false; // false makes optimization faster for large numbers of mutations
    bool load_from_file = false;
    bool reduced_output = false;

    std::string input_path = "../data/input_data/" + sample;
//    std::string input_path = "/cluster/work/bewi/members/znorio/data/input_data/" + sample;
    std::string output_path = "../data/results/" + sample + "/sciterna";
//    std::string output_path = "/cluster/work/bewi/members/znorio/data/results/" + sample + "/sciterna";

    std::vector<std::vector<int>> ref = read_csv(input_path + "/ref.csv");
    std::vector<std::vector<int>> alt = read_csv(input_path + "/alt.csv");

    generate_sciterna_results(ref, alt, input_path, output_path,
                            bootstrap_samples, use_bootstrap, tree_space,
                            flipped_mutation_direction, n_snps, posterior_threshold,
                            n_rounds, only_preprocessing, method, reshuffle_nodes, load_from_file, reduced_output);
    return 0;
}
