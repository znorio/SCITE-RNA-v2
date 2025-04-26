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


//// load integer matrix
//std::vector<std::vector<int>> loadIntMatrix(const std::string& filename) {
//    std::vector<std::vector<int>> data;
//    std::ifstream file(filename);
//    if (!file.is_open()) {
//        throw std::runtime_error("Could not open file: " + filename);
//    }
//
//    if (file.is_open()) {
//        std::string line;
//        while (std::getline(file, line)) {
//            std::vector<int> row;
//            std::istringstream stream(line);
//            std::string cell;
//            while (std::getline(stream, cell, ',')) {
//                row.push_back(std::stoi(cell));
//            }
//            data.push_back(row);
//        }
//        file.close();
//    } else {
//        std::cerr << "Unable to open file: " << filename << std::endl;
//    }
//    return data;
//}
//
//// load char matrix
//std::vector<std::vector<char>> loadStringMatrix(const std::string& filename) {
//    std::vector<std::vector<char>> data;
//    std::ifstream file(filename);
//
//    if (file.is_open()) {
//        std::string line;
//        while (std::getline(file, line)) {
//            std::vector<char> row;
//            std::istringstream stream(line);
//            std::string cell;
//            while (std::getline(stream, cell, ',')) {
//                for (char c : cell) {
//                    row.push_back(c);
//                }
//            }
//            data.push_back(row);
//        }
//        file.close();
//    } else {
//        std::cerr << "Unable to open file: " << filename << std::endl;
//    }
//    return data;
//}


int main() {

    int bootstrap_samples = 1000;
    bool use_bootstrap = true;
    int n_snps = 3000;
    double posterior_threshold = 0.95;
    std::string method = "threshold";
    int n_rounds = 3;
    std::string sample = "mm34";
    bool flipped_mutation_direction = true;
    bool only_preprocessing = false;
    std::vector<std::string> tree_space = {"c", "m"};
    bool reshuffle_nodes = false; // false makes optimization faster for large numbers of mutations

//    std::string input_path = "../data/input_data/" + sample;
    std::string input_path = "/cluster/work/bewi/members/znorio/data/input_data/" + sample;
//    std::string output_path = "../data/results/" + sample + "/sciterna";
    std::string output_path = "/cluster/work/bewi/members/znorio/data/results/" + sample + "/sciterna";

    generate_sciterna_results(input_path, output_path,
                            bootstrap_samples, use_bootstrap, tree_space,
                            flipped_mutation_direction, n_snps, posterior_threshold,
                            n_rounds, only_preprocessing, method, reshuffle_nodes);

    return 0;
}
