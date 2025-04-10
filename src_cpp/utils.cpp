//
// Created by Norio on 09.04.2025.
//

#include "utils.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <stack>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

std::map<std::string, std::string> config_variables;

std::string rtrim(const std::string& str) {
    size_t last = str.find_last_not_of(" \t");
    return (last == std::string::npos) ? "" : str.substr(0, last + 1);
}

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
        } else if (line.find(":") != std::string::npos) {
            // Key-value pair
            size_t colon_pos = line.find(":");
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

//    std::vector<std::vector<int>> transposed(data[0].size(), std::vector<int>(data.size()));
//
//    for (size_t i = 0; i < data.size(); ++i) {
//        for (size_t j = 0; j < data[i].size(); ++j) {
//            transposed[j][i] = data[i][j];
//        }
//    }

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

void save_double_vector_to_file(const std::string& filepath, const std::vector<double>& vector) {
    std::ofstream file(filepath);
    for (size_t i = 0; i < vector.size(); ++i) {
        file << vector[i];
        if (i < vector.size() - 1) file << "\n";
    }
}


std::vector<std::vector<int>> create_mutation_matrix(
        const std::vector<int>& parent_vector,
        const std::vector<int>& mutation_indices,
        CellTree& ct
) {
    int n_nodes = parent_vector.size();
    int n_cells = (n_nodes + 1) / 2;
    int n_mutations = mutation_indices.size();

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


// Function to create a genotype matrix
std::vector<std::vector<char>> create_genotype_matrix(
        const std::vector<char>& not_selected_genotypes,
        const std::vector<int>& selected,
        const std::vector<char>& gt1,
        const std::vector<char>& gt2,
        const std::vector<std::vector<int>>& mutation_matrix,
        const std::vector<bool>& flipped
) {
    int n_cells = mutation_matrix.size();
    int n_loci = selected.size() + not_selected_genotypes.size();
    std::vector<std::vector<char>> genotype_matrix(n_cells, std::vector<char>(n_loci, ' '));

    // Determine the indices of not selected genotypes
    std::vector<int> not_selected;
    for (int i = 0; i < n_loci; ++i) {
        if (std::find(selected.begin(), selected.end(), i) == selected.end()) {
            not_selected.push_back(i);
        }
    }

    // Assign genotypes for not selected loci
    for (size_t n = 0; n < not_selected.size(); ++n) {
        int locus = not_selected[n];
        for (int i = 0; i < n_cells; ++i) {
            genotype_matrix[i][locus] = not_selected_genotypes[n];
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