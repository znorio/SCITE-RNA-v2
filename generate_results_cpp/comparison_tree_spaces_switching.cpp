/*
Script used to run SCITE-RNA for different modes of tree space optimization. These are optimizing only in the mutation tree space (m),
optimizing only cell lineage trees (c) and alternating between the two spaces, starting either in the cell lineage space (c,m)
or in the mutation tree space (m,c).
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <mutation_filter.h>
#include <swap_optimizer.h>

// using indices to extract columns of matrix
std::vector<std::vector<int>> slice_columns(const std::vector<std::vector<int>>& matrix, const std::vector<int>& indices) {
    std::vector<std::vector<int>> sliced_matrix;
    size_t num_rows = matrix.size();

    for (size_t i = 0; i < num_rows; ++i) {
        std::vector<int> row;
        for (int idx : indices) {
            row.push_back(matrix[i][idx]);
        }
        sliced_matrix.push_back(row);
    }

    return sliced_matrix;
}

// load alternative and reference read counts
std::vector<std::vector<int>> load_txt(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<int>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<int> row;
        int value;

        while (iss >> value) {
            row.push_back(value);
        }
        data.push_back(row);
    }
    file.close();

    // TODO just generate the data, so that the transposing is no longer necessary
    std::vector<std::vector<int>> transposed(data[0].size(), std::vector<int>(data.size()));

    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            transposed[j][i] = data[i][j];
        }
    }

    return transposed;
}

// load selected loci, which is used if the mutation filtering step was done beforehand
std::vector<int> loadSelectedVector(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<int> data;
    int value;

    while (file >> value) {
        data.push_back(value);
    }

    file.close();
    return data;
}

// load genotypes, which is used if the mutation filtering step was done beforehand
void loadGenotypes(const std::string& filename, std::vector<char>& gt1, std::vector<char>& gt2) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::string line;
    std::vector<std::vector<char>> genotypes;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        char value;
        std::vector<char> row;

        while (ss >> value) {
            row.push_back(value);
        }

        genotypes.push_back(row);
    }

    file.close();

    if (genotypes.size() != 2) {
        throw std::runtime_error("Unexpected number of genotype rows in file " + filename);
    }

    gt1 = genotypes[0];
    gt2 = genotypes[1];
}

// run tree inference and save results
void generate_sciterna_results(std::string const& path = "", int n_tests = 100, const std::string& pathout = "",
                               const std::vector<std::string>& tree_space = {"c"}, bool reverse_mutations = true, int n_cells = 50) {

    // Create necessary directories
    std::filesystem::create_directories(pathout + "sciterna_selected_loci");
    std::filesystem::create_directories(pathout + "sciterna_inferred_mut_types");
    std::filesystem::create_directories(pathout + "sciterna_parent_vec");

    std::cout << "Running inference on data in " << path << std::endl;

    for (int i = 0; i < n_tests; ++i) {
        std::cout << i << std::endl;
        std::vector<std::vector<int>> alt, ref;

        // Load data
        std::basic_string<char> alt_file(path + "/alt/alt_" + std::to_string(i) + ".txt");
        std::basic_string<char> ref_file(path + "/ref/ref_" + std::to_string(i) + ".txt");

        alt = load_txt(alt_file);
        ref = load_txt(ref_file);

        MutationFilter mf;
        std::basic_string<char> selected_file(pathout + "/sciterna_selected_loci/sciterna_selected_loci_" + std::to_string(i) + ".txt");
        std::basic_string<char> path_g(pathout + "/sciterna_inferred_mut_types/sciterna_inferred_mut_types_" + std::to_string(i) + ".txt");

        std::vector<char> gt1, gt2;
        std::vector<int> selected;
        if (!std::filesystem::exists(selected_file) || !std::filesystem::exists(path_g)) {
            auto [new_selected, new_gt1, new_gt2, not_selected_genotypes] = mf.filter_mutations(ref, alt,
                                                                        "first_k", 0.5, n_cells, true);
            selected = new_selected;
            gt1 = new_gt1;
            gt2 = new_gt2;
        }
        else{
            selected = loadSelectedVector(selected_file);
            loadGenotypes(path_g, gt1, gt2);
        }

        auto [llh_1, llh_2] = mf.get_llh_mat(slice_columns(ref, selected), slice_columns(alt, selected), gt1, gt2);

        // save preprocessed data
        std::ofstream select_file(pathout + "/sciterna_selected_loci/sciterna_selected_loci_" + std::to_string(i) + ".txt");
        for (const auto& sel : selected) {
            select_file << sel << "\n";
        }
        select_file.close();

        std::ofstream mut_types_file(pathout + "/sciterna_inferred_mut_types/sciterna_inferred_mut_types_" + std::to_string(i) + ".txt");

        for (char j : gt1) {
            mut_types_file << j << " ";
        }
        mut_types_file << "\n";

        for (char j : gt2) {
            mut_types_file << j << " ";
        }
        mut_types_file << "\n";

        mut_types_file.close();

        int num_cells = static_cast<int>(llh_1.size());
        int num_mut = static_cast<int>(llh_1[0].size());

        // run optimization
        SwapOptimizer optimizer(tree_space, reverse_mutations, num_mut, num_cells);
        optimizer.fit_llh(llh_1, llh_2);
        optimizer.optimize();

        // save parent vector
        std::ofstream parent_vec_file(pathout + "/sciterna_parent_vec/sciterna_parent_vec_" + std::to_string(i) + ".txt");
        for (const auto& parent : optimizer.ct.parent_vector_ct) {
            parent_vec_file << parent << "\n";
        }
        parent_vec_file.close();
    }

    std::cout << "Done." << std::endl;
}


// run inference with SCITE-RNA
int main() {
    int n_tests = 100; //number of runs
    std::vector<int> n_cells_list = {500, 100, 500};
    std::vector<int> n_mut_list = {100, 500, 500};
    std::vector<std::vector<std::string>> tree_spaces = {{"m"}, {"c"}, {"c", "m"},  {"m", "c"}}; // {"m"}, {"c"}, {"c", "m"},  {"m", "c"}
    bool flipped = false;

    for (const auto& space : tree_spaces) {
        for (int i = 0; i < n_cells_list.size(); ++i) {
            int n_cells = n_cells_list[i];
            int n_mut = n_mut_list[i];
            std::string path = "../data/simulated_data/" + std::to_string(n_cells) + "c" + std::to_string(n_mut) + "m";
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
            generate_sciterna_results(path, n_tests, path_results, space, flipped, n_mut);
        }
    }
    return 0;
}