#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <mutation_filter.h>

#include "utils.h"
#include <mutation_filter.h>
#include "generate_results.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_dir> <index>" << std::endl;
        return 1;
    }

    std::string sample_dir = argv[1]; // e.g., "100c100m"
    int idx = std::stoi(argv[2]);

    std::string input_path = "/cluster/work/bewi/members/znorio/data/simulated_data/" + sample_dir + "/";
//    std::string input_path = "../data/simulated_data/" + sample_dir + "/";
    std::string output_path = "/cluster/work/bewi/members/znorio/data/results/" + sample_dir + "/sciterna_" + std::to_string(idx);
//    std::string output_path = "../data/results/" + sample_dir + "/sciterna_" + std::to_string(idx);

    std::string alt_file = input_path + "alt/alt_" + std::to_string(idx) + ".txt";
    std::string ref_file = input_path + "ref/ref_" + std::to_string(idx) + ".txt";

    std::vector<std::vector<int>> alt = load_txt(alt_file);
    std::vector<std::vector<int>> ref = load_txt(ref_file);

    int bootstrap_samples = 1000;
    bool use_bootstrap = true;
    int n_snps = 100;
    double posterior_threshold = 0.05;
    std::string method = "threshold";
    int n_rounds = 3;
    bool flipped_mutation_direction = true;
    bool only_preprocessing = false;
    std::vector<std::string> tree_space = {"c", "m"};
    bool reshuffle_nodes = true;
    bool load_from_file = false;
    bool reduced_output = true;

    generate_sciterna_results(ref, alt, input_path, output_path,
                              bootstrap_samples, use_bootstrap, tree_space,
                              flipped_mutation_direction, n_snps, posterior_threshold,
                              n_rounds, only_preprocessing, method, reshuffle_nodes, load_from_file, reduced_output);

    return 0;
}