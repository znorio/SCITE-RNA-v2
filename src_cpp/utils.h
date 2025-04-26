//
// Created by Norio on 09.04.2025.
//

#ifndef SCITE_RNA_UTILS_H
#define SCITE_RNA_UTILS_H

#include <map>
#include <string>
#include <vector>
#include "cell_tree.h"

extern std::map<std::string, std::string> config_variables;
void load_config(const std::string& file_path);
std::vector<std::vector<char>> create_genotype_matrix(
        const std::vector<char>& not_selected_genotypes,
        const std::vector<int>& selected,
        const std::vector<char>& gt1,
        const std::vector<char>& gt2,
        const std::vector<std::vector<int>>& mutation_matrix,
        const std::vector<bool>& flipped
);
void save_char_matrix_to_file(const std::string& filepath, const std::vector<std::vector<char>>& matrix);
void save_matrix_to_file(const std::string& filepath, const std::vector<std::vector<int>>& matrix);
void save_vector_to_file(const std::string& filepath, const std::vector<int>& vector);
void save_char_vector_to_file(const std::string& filepath, const std::vector<char>& vector);

std::vector<std::vector<int>> load_txt(const std::string& filename);
std::vector<std::vector<int>> create_mutation_matrix(
        const std::vector<int>& parent_vector,
        const std::vector<int>& mutation_indices,
        CellTree& ct);
void save_double_vector_to_file(const std::string& filepath, const std::vector<double>& vector);
std::vector<std::vector<int>> slice_columns(const std::vector<std::vector<int>>& matrix, const std::vector<int>& indices);
std::vector<std::vector<char>> slice_columns_char(const std::vector<std::vector<char>>& matrix, const std::vector<int>& indices);
std::vector<std::vector<int>> read_csv(const std::string& filename);
void loadGenotypes(const std::string& filename, std::vector<char>& gt1, std::vector<char>& gt2);
std::vector<int> loadSelectedVector(const std::string& filename);

#endif //SCITE_RNA_UTILS_H
