/*
Defines helper functions for loading configurations, saving matrices, and manipulating data structures.
*/

#include "utils.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

std::map<std::string, std::string> config_variables;

// Used to trim trailing whitespace from a string
std::string rtrim(const std::string& str) {
    size_t last = str.find_last_not_of(" \t");
    return (last == std::string::npos) ? "" : str.substr(0, last + 1);
}

// Load configuration from a yaml file
void load_config(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    std::string line;
    std::string current_section; // To track the current section for nested keys

    while (std::getline(file, line)) {
        // Remove comments
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        line = rtrim(line); // Only trim trailing whitespaces

        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        size_t leading_spaces = line.find_first_not_of(" \t");

        if (line.back() == ':') {
            // New section header
            current_section = rtrim(line.substr(0, line.size() - 1)); // Trim trailing ':'
        } else if (line.find(':') != std::string::npos) {
            // Key-value pair
            size_t colon_pos = line.find(':');
            std::string key = rtrim(line.substr(0, colon_pos));
            std::string value = rtrim(line.substr(colon_pos + 1));

            if (leading_spaces > 0 && !current_section.empty()) {
                // Nested key
                config_variables[current_section + "." + key] = value;
            } else {
                // Reset to a top-level key
                current_section.clear();
                config_variables[key] = value;
            }
        }
    }

    // Debug: Print all key-value pairs
//    std::cout << "Loaded configuration variables:" << std::endl;
//    for (const auto& [key, value] : config_variables) {
//        std::cout << key << ": " << value << std::endl;
//    }
}

// Implementation of save functions
void save_char_matrix_to_file(const std::string& filepath, const std::vector<std::vector<char>>& matrix) {
    std::ofstream file(filepath);
    for (const auto& row : matrix) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << " ";
        }
        file << "\n";
    }
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

    return data;
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


// using indices to extract columns of matrix
std::vector<std::vector<char>> slice_columns_char(const std::vector<std::vector<char>>& matrix, const std::vector<int>& indices) {
    std::vector<std::vector<char>> sliced_matrix;
    size_t num_rows = matrix.size();

    for (size_t i = 0; i < num_rows; ++i) {
        std::vector<char> row;
        for (int idx : indices) {
            row.push_back(matrix[i][idx]);
        }
        sliced_matrix.push_back(row);
    }

    return sliced_matrix;
}

// add two 1d vectors
std::vector<double> addVectors(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for element-wise addition.");
    }
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// returns maximum of indexed columns of a 2d matrix
std::vector<double> getMaxValues(const std::vector<std::vector<double>>& matrix, const std::vector<int>& indices) {
    std::vector<double> max_values(matrix.size(), std::numeric_limits<double>::lowest());
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (int idx : indices) {
            if (idx < matrix[i].size()) {
                max_values[i] = std::max(max_values[i], matrix[i][idx]);
            }
        }
    }
    return max_values;
}

// get column of a 2D matrix
std::vector<int> get_column(const std::vector<std::vector<int>>& matrix, size_t col_index) {
    std::vector<int> column;
    column.reserve(matrix.size());  // Reserve space to avoid multiple allocations

    for (const auto& row : matrix) {
        if (col_index < row.size()) {
            column.push_back(row[col_index]);
        }
    }
    return column;
}

// add scalar to 1D vector
std::vector<double> add_scalar_to_vector(double scalar, const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = scalar + vec[i];
    }
    return result;
}


// Save functions for different data types
void save_matrix_to_file(const std::string& filepath, const std::vector<std::vector<int>>& matrix) {
    std::ofstream file(filepath);
    for (const auto& row : matrix) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << " ";
        }
        file << "\n";
    }
}

void save_vector_to_file(const std::string& filepath, const std::vector<int>& vector) {
    std::ofstream file(filepath);
    for (size_t i = 0; i < vector.size(); ++i) {
        file << vector[i];
        if (i < vector.size() - 1) file << "\n";
    }
}

void save_char_vector_to_file(const std::string& filepath, const std::vector<char>& vector) {
    std::ofstream file(filepath);
    for (size_t i = 0; i < vector.size(); ++i) {
        file << vector[i];
        if (i < vector.size() - 1) {
            file << "\n";
        }
    }
}

void save_double_vector_to_file(const std::string& filepath, const std::vector<double>& vector) {
    std::ofstream file(filepath);
    for (size_t i = 0; i < vector.size(); ++i) {
        file << vector[i];
        if (i < vector.size() - 1) file << "\n";
    }
}

// Function to create a mutation matrix indicating which cells have which mutations
std::vector<std::vector<int>> create_mutation_matrix(
        const std::vector<int>& parent_vector,
        const std::vector<int>& mutation_indices,
        CellTree& ct
) {
    int n_nodes = static_cast<int>(parent_vector.size());
    int n_cells = (n_nodes + 1) / 2;
    int n_mutations = static_cast<int>(mutation_indices.size());

    std::vector<std::vector<int>> mutation_matrix(n_nodes, std::vector<int>(n_mutations, 0));

    // Mark cells with mutations
    for (int mutation_idx = 0; mutation_idx < n_mutations; ++mutation_idx) {
        int cell_idx = mutation_indices[mutation_idx];
        std::vector<int> children = ct.dfs(cell_idx);

        for (int cell: children) {  // Traverse all cells below the mutation cell
            mutation_matrix[cell][mutation_idx] = 1;  // Mark cells with the mutation
        }
    }

    // Return only the relevant part of the matrix
    std::vector<std::vector<int>> result(n_cells);
    for (int i = 0; i < n_cells; ++i) {
        result[i] = mutation_matrix[i];
    }

    return result;// using indices to extract columns of matrix
}


// Function to create a genotype matrix for each cell and SNV locus
std::vector<std::vector<char>> create_genotype_matrix(
        const std::vector<char>& not_selected_genotypes,
        const std::vector<int>& selected,
        const std::vector<char>& gt1,
        const std::vector<char>& gt2,
        const std::vector<std::vector<int>>& mutation_matrix,
        const std::vector<bool>& flipped
) {
    int n_cells = static_cast<int>(mutation_matrix.size());
    int n_loci = static_cast<int>(selected.size()) + static_cast<int>(not_selected_genotypes.size());
    std::vector<std::vector<char>> genotype_matrix(n_cells, std::vector<char>(n_loci, ' '));

    // Determine the indices of not selected genotypes
    std::vector<int> not_selected;
    for (int i = 0; i < n_loci; ++i) {
        if (std::find(selected.begin(), selected.end(), i) == selected.end()) {
            not_selected.push_back(i);
        }
    }

    // Assign genotypes for not selected loci
    if (not_selected_genotypes.size() != not_selected.size()) {
        for (int locus : not_selected) {
            for (int i = 0; i < n_cells; ++i) {
                genotype_matrix[i][locus] = 'X';
            }
        }
    }
    else {
        for (size_t n = 0; n < not_selected.size(); ++n) {
            int locus = not_selected[n];
            for (int i = 0; i < n_cells; ++i) {
                genotype_matrix[i][locus] = not_selected_genotypes[n];
            }
        }
    }

    // Assign genotypes for selected loci based on mutation matrix and flipped status
    for (size_t n = 0; n < selected.size(); ++n) {
        int locus = selected[n];
        for (int i = 0; i < n_cells; ++i) {
            if (flipped[n]) {
                genotype_matrix[i][locus] = (mutation_matrix[i][n] == 0) ? gt2[n] :
                                            (mutation_matrix[i][n] == 1) ? gt1[n] :
                                            '?';  // Placeholder for unexpected values
            } else {
                genotype_matrix[i][locus] = (mutation_matrix[i][n] == 0) ? gt1[n] :
                                            (mutation_matrix[i][n] == 1) ? gt2[n] :
                                            '?';  // Placeholder for unexpected values
            }
        }
    }

    return genotype_matrix;
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

// Function to read a CSV file and return a 2D vector of integers
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


// Load selected mutations from a file
std::vector<int> load_selected(const std::string& path) {
    std::vector<int> selected;
    std::ifstream file(path);
    int val;
    while (file >> val) {
        selected.push_back(val);
    }
    return selected;
}


// Load genotype matrix from file
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


// Load gt1/gt2 genotypes from a file
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
