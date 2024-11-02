/*
Script can be used to run SCITE-RNA on new data. For bootstrapping set the number of bootstrap_samples
to the desired value.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <mutation_filter.h>
#include <swap_optimizer.h>
#include <random>
#include <ctime>

std::vector<std::vector<int>> bootstrap_snvs(const std::vector<std::vector<int>>& original, const std::vector<int>& indices) {
    int n_cells = original.size();
    int n_snvs = indices.size();

    // Initialize a bootstrapped matrix with the same number of rows but bootstrapped columns
    std::vector<std::vector<int>> bootstrapped(n_cells, std::vector<int>(n_snvs));

    for (int i = 0; i < n_cells; ++i) {
        for (int j = 0; j < n_snvs; ++j) {
            bootstrapped[i][j] = original[i][indices[j]];
        }
    }
    return bootstrapped;
}

void save_to_file(const std::vector<std::vector<int>>& matrix, const std::string& filename) {
    std::ofstream file(filename);

    size_t n_rows = matrix.size();
    size_t n_cols = matrix[0].size();

    for (size_t col = 0; col < n_cols; ++col) {
        for (size_t row = 0; row < n_rows; ++row) {
            file << matrix[row][col];
            if (row < n_rows - 1) {
                file << " ";
            }
        }
        file << "\n";
    }
    file.close();
}

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

    std::vector<std::vector<int>> transposed(data[0].size(), std::vector<int>(data.size()));

    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            transposed[j][i] = data[i][j];
        }
    }

    return transposed;
}


// run tree inference and save results
void generate_sciterna_results(std::string const& path = "", int loop = 0,
                               const std::vector<std::string>& tree_space = {"c"},
                               bool reverse_mutations = true,
                               int n_bootstrap = 1) {

    std::cout << "Running inference on data in " << path << std::endl;

    std::cout << loop << std::endl;
    std::vector<std::vector<int>> alt, ref;

    // Load data
    std::basic_string<char> alt_file("../data/input_data/new_data/alt.txt");
    std::basic_string<char> ref_file("../data/input_data/new_data/ref.txt");

    alt = load_txt(alt_file);
    ref = load_txt(ref_file);

    int n_cells = alt.size();
    int n_snvs = alt[0].size();

    std::cout << "Cells: " << n_cells << std::endl;
    std::cout << "SNVs: " << n_snvs << std::endl;

    std::mt19937 gen(static_cast<unsigned int>(std::time(0)));
    std::uniform_int_distribution<> dis(0, n_snvs - 1);

    // Generate bootstrapped indices for SNVs
    std::vector<int> indices;
    for (int i = 0; i < n_snvs; ++i) {
        indices.push_back(dis(gen));
    }

    // Create bootstrapped ref and alt matrices using the same indices
    std::vector<std::vector<int>> ref_bootstrapped = bootstrap_snvs(ref, indices);
    std::vector<std::vector<int>> alt_bootstrapped = bootstrap_snvs(alt, indices);

    if (n_bootstrap == 1){
        ref_bootstrapped = ref;
        alt_bootstrapped = alt;
    }
    else{
        save_to_file(ref_bootstrapped, path + "/bootstrapped_ref.txt");
        save_to_file(alt_bootstrapped, path + "/bootstrapped_alt.txt");
    }

    MutationFilter mf;

    auto [selected, gt1, gt2, gt_not_selected] = mf.filter_mutations(ref_bootstrapped, alt_bootstrapped, "threshold", 0.5, n_snvs, true);

    auto [llh_1, llh_2] = mf.get_llh_mat(slice_columns(ref_bootstrapped, selected), slice_columns(alt_bootstrapped, selected), gt1, gt2);

    // save preprocessed data
    std::ofstream select_file(path + "/sciterna_selected_loci.txt");
    for (const auto& sel : selected) {
        select_file << sel << "\n";
    }
    select_file.close();

    std::ofstream not_selected_file(path + "/sciterna_not_selected_genotypes.txt");
    for (const auto& sel : gt_not_selected) {
        not_selected_file << sel << "\n";
    }
    select_file.close();

    std::ofstream mut_types_file(path + "/sciterna_inferred_mut_types.txt");
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
    std::ofstream parent_vec_file(path + "/sciterna_parent_vec.txt");
    for (const auto& parent : optimizer.ct.parent_vector_ct) {
        parent_vec_file << parent << "\n";
    }
    parent_vec_file.close();

    std::ofstream mut_loc_file(path + "/sciterna_mut_loc.txt");
    for (const auto& parent : optimizer.ct.mut_loc) {
        mut_loc_file << parent << "\n";
    }
    parent_vec_file.close();

    std::cout << "Done." << std::endl;
}


// run inference with SCITE-RNA
int main() {
    int bootstrap_samples = 1; // 1 for no bootstrapping / larger 1 (e.g. 100) for bootstrapping of SNVs
    std::vector<std::string> tree_space = {"c", "m"};
    bool flipped = false;

    std::string path;
    for (int i = 0; i < bootstrap_samples; ++i) {
        if (bootstrap_samples <= 1){
            path = "../data/results/new_data/sample";
        }
        else{
            path = "../data/results/new_data/bootstrap_sample_" + std::to_string(i);
        }
        std::filesystem::create_directories(path);
        generate_sciterna_results(path, i, tree_space, flipped, bootstrap_samples);
    }
    return 0;
}