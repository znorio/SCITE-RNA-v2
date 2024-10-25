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
void generate_sciterna_results(std::string const& path = "", int n_tests = 100,
                               const std::vector<std::string>& tree_space = {"c"}, bool reverse_mutations = true, int select_n = 50) {

    // Create necessary directories
    std::filesystem::create_directories(path + "sciterna_selected_loci");
    std::filesystem::create_directories(path + "sciterna_inferred_mut_types");
    std::filesystem::create_directories(path + "sciterna_parent_vec");
    std::filesystem::create_directories(path + "sciterna_mut_loc");
    std::filesystem::create_directories(path + "sciterna_not_selected_genotypes");

    std::cout << "Running inference on data in " << path << std::endl;

    std::vector<double> timings;

    for (int i = 0; i < n_tests; ++i) {
        std::cout << i << std::endl;
        std::vector<std::vector<int>> alt, ref;

        // Load data
        std::basic_string<char> alt_file(path + "/alt/alt_" + std::to_string(i) + ".txt");
        std::basic_string<char> ref_file(path + "/ref/ref_" + std::to_string(i) + ".txt");

        alt = load_txt(alt_file);
        ref = load_txt(ref_file);

        auto start = std::chrono::high_resolution_clock::now();

        MutationFilter mf;
//        std::basic_string<char> selected_file(path + "/sciterna_selected_loci/sciterna_selected_loci_" + std::to_string(i) + ".txt");
//        std::basic_string<char> path_g(path + "/sciterna_inferred_mut_types/sciterna_inferred_mut_types_" + std::to_string(i) + ".txt");

        auto [selected, gt1, gt2, gt_not_selected] = mf.filter_mutations(ref, alt,
                                                                    "first_k", 0.5, select_n, true);


        auto [llh_1, llh_2] = mf.get_llh_mat(slice_columns(ref, selected), slice_columns(alt, selected), gt1, gt2);

        // save preprocessed data
        std::ofstream select_file(path + "/sciterna_selected_loci/sciterna_selected_loci_" + std::to_string(i) + ".txt");
        for (const auto& sel : selected) {
            select_file << sel << "\n";
        }
        select_file.close();

        std::ofstream not_selected_file(path + "/sciterna_not_selected_genotypes/sciterna_not_selected_genotypes_" + std::to_string(i) + ".txt");
        for (const auto& sel : gt_not_selected) {
            not_selected_file << sel << "\n";
        }
        select_file.close();

        std::ofstream mut_types_file(path + "/sciterna_inferred_mut_types/sciterna_inferred_mut_types_" + std::to_string(i) + ".txt");
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

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        timings.push_back(elapsed.count());

        // save parent vector
        std::ofstream parent_vec_file(path + "/sciterna_parent_vec/sciterna_parent_vec_" + std::to_string(i) + ".txt");
        for (const auto& parent : optimizer.ct.parent_vector_ct) {
            parent_vec_file << parent << "\n";
        }
        parent_vec_file.close();

        std::ofstream mut_loc_file(path + "/sciterna_mut_loc/sciterna_mut_loc_" + std::to_string(i) + ".txt");
        for (const auto& parent : optimizer.ct.mut_loc) {
            mut_loc_file << parent << "\n";
        }
        parent_vec_file.close();
    }

    std::string filename = path + "execution_times_sciterna.txt";
    std::ofstream outFile(filename);
    for (const auto& time : timings) {
        outFile << time << std::endl;
    }
    outFile.close();

    std::cout << "Done." << std::endl;
}


// run inference with SCITE-RNA
int main() {
    int n_tests = 100; //number of runs
    std::vector<int> n_cells_list = {50, 100, 100};
    std::vector<int> n_mut_list = {100, 100, 50};
    std::vector<std::string> tree_space = {"c", "m"};
    std::vector<std::string> clones = {"5", "10", "20", ""};
    bool flipped = false;

    for (const auto& clone : clones) {
        for (int i = 0; i < n_cells_list.size(); ++i) {
            int n_cells = n_cells_list[i];
            int n_mut = n_mut_list[i];
            std::string path = "./" + std::to_string(n_cells) + "c" + std::to_string(n_mut) + "m" + clone +"/";
            generate_sciterna_results(path, n_tests, tree_space, flipped, n_mut);
        }
    }
    return 0;
}