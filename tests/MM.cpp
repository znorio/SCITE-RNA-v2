#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>
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

// load integer matrix
std::vector<std::vector<int>> loadIntMatrix(const std::string& filename) {
    std::vector<std::vector<int>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::vector<int> row;
            std::istringstream stream(line);
            std::string cell;
            while (std::getline(stream, cell, ',')) {
                row.push_back(std::stoi(cell));
            }
            data.push_back(row);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
    return data;
}

// load char matrix
std::vector<std::vector<char>> loadStringMatrix(const std::string& filename) {
    std::vector<std::vector<char>> data;
    std::ifstream file(filename);

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::vector<char> row;
            std::istringstream stream(line);
            std::string cell;
            while (std::getline(stream, cell, ',')) {
                for (char c : cell) {
                    row.push_back(c);
                }
            }
            data.push_back(row);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
    return data;
}

std::vector<std::vector<int>> read_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<std::vector<int>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<int> row;
        bool has_valid_value = false;

        while (std::getline(lineStream, cell, ',')) {
            if (cell.empty()) {
                // Handle empty cell by adding 0
                row.push_back(0);
            } else {
                try {
                    row.push_back(std::stoi(cell));
                    has_valid_value = true;
                } catch (const std::invalid_argument&) {

                } catch (const std::out_of_range&) {
                    row.push_back(0);
                    has_valid_value = true;
                }
            }
        }

        if (has_valid_value) {
            data.push_back(row);
        }
    }

    file.close();
    return data;
}

int main() {
    int bootstrap_samples = 1000;
    int n_snps = 3000;
    std::string sample = "34";
    std::string reduced =  ""; // "_reduced";
    std::string ref_to_alt = "_only_ref_to_alt";
    std::vector<std::string> tree_space = {"c", "m"};
    bool reverse_mutations = false;
    std::string bootstrap_folder = "/bootstrap_";
    if (reverse_mutations){
        bootstrap_folder = "/bootstrap_reverse_mut_";
    }
    std::string base_path = "."; // Local path
    if (!std::filesystem::exists(base_path + sample)) {
        base_path = ".";  // alternative path
    }

    std::string path = base_path + sample + "/results";
    std::string reference_file = base_path + sample + "/data/ref" + reduced + ".csv";
    std::string alternative_file = base_path + sample + "/data/alt" + reduced + ".csv";

    std::string pathout = base_path + sample + "/data" + bootstrap_folder +  std::to_string(n_snps) +
            reduced + std::to_string(bootstrap_samples) + ref_to_alt;
    std::filesystem::create_directories(pathout + "/sciterna_parent_vec");
    std::filesystem::create_directories(pathout + "/sciterna_mut_loc");
    std::filesystem::create_directories(pathout + "/sciterna_selected_loci");
    std::filesystem::create_directories(pathout + "/sciterna_flipped_gt");
    std::string selected_file = pathout + "/selected" + reduced + ".txt";
    std::string gt1_file = pathout + "/gt1" + reduced + ".txt";
    std::string gt2_file = pathout + "/gt2" + reduced + ".txt";

    auto ref = read_csv(reference_file);
    auto alt = read_csv(alternative_file);

    std::vector<std::vector<int>> bootstrap_selected = loadIntMatrix(selected_file);
    std::vector<std::vector<char>> bootstrap_gt1 = loadStringMatrix(gt1_file);
    std::vector<std::vector<char>> bootstrap_gt2 = loadStringMatrix(gt2_file);

    int n_loops = bootstrap_samples;

    std::vector<int> selected;
    std::vector<char> gt1;
    std::vector<char> gt2;

    for (int loop = 354; loop < 370; ++loop) {
        std::cout << std::to_string(loop) << ". bootstrap sample" << std::endl;
        selected = bootstrap_selected[loop];
        gt1 = bootstrap_gt1[loop];
        gt2 = bootstrap_gt2[loop];

        MutationFilter mf;
        auto [llh_1, llh_2] = mf.get_llh_mat(slice_columns(ref, selected), slice_columns(alt, selected), gt1, gt2);

        SwapOptimizer optimizer(tree_space, reverse_mutations, n_snps,
                                static_cast<int> (llh_1.size()));
        optimizer.fit_llh(llh_1, llh_2);
        optimizer.optimize();
        // save parent vector
        std::ofstream parent_vec_file(pathout + "/sciterna_parent_vec/sciterna_parent_vec_" + std::to_string(loop) + ".txt");
        for (const auto& parent : optimizer.ct.parent_vector_ct) {
            parent_vec_file << parent << "\n";
        }
        parent_vec_file.close();

        std::ofstream mut_loc_file(pathout + "/sciterna_mut_loc/sciterna_mut_loc_" + std::to_string(loop) + ".txt");
        for (const auto& mutLoc : optimizer.ct.mut_loc) {
            mut_loc_file << mutLoc << "\n";
        }
        mut_loc_file.close();

        std::ofstream selected_loci_file(pathout + "/sciterna_selected_loci/sciterna_selected_loci_" + std::to_string(loop) + ".txt");
        for (const auto& select : selected) {
            selected_loci_file << select << "\n";
        }
        selected_loci_file.close();

        std::ofstream flipped_loci_file(pathout + "/sciterna_flipped_gt/sciterna_flipped_gt_" + std::to_string(loop) + ".txt");
        for (const auto& flip : optimizer.ct.flipped) {
            flipped_loci_file << flip << "\n";
        }
        flipped_loci_file.close();
    }

    std::cout << "Done." << std::endl;
    return 0;
}
