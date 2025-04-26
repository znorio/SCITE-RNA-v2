#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include "mutation_filter.h"
#include "cell_tree.h"
#include "utils.h"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Vectorized mapping function
float vectorizedMap(char x, const unordered_map<char, float>& mappingDict) {
    return mappingDict.at(x);
}

// Function to write a vector of vectors to a file
void writeToFile(const string& filename, const vector<vector<double>>& data) {
    ofstream file(filename);
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << ",";
        }
        file << "\n";
    }
}

// Template function to load data from a file into a vector
template <typename T>
vector<T> loadtxt(const string& filename) {
    vector<T> data;
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }

    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        T value;
        if (!(iss >> value)) {
            throw runtime_error("Error parsing file: " + filename);
        }
        data.push_back(value);
    }

    file.close();
    return data;
}

// Template function to load data from a file into a 2D vector
template <typename T>
vector<vector<T>> loadtxt2D(const string& filename) {
    vector<vector<T>> data;
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }

    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        vector<T> row;
        T value;
        while (iss >> value) {
            row.push_back(value);
        }
        data.push_back(row);
    }

    file.close();
    return data;
}

// Function to compute the leaf distance matrix
MatrixXi leaf_dist_mat(CellTree ct, bool unrooted = false) {
    MatrixXi result = MatrixXi::Constant(ct.n_cells, ct.n_vtx, -1);
    result.diagonal().setConstant(0);

    for (int vtx : ct.rdfs(ct.main_root_ct)) {
        if (ct.isLeaf(vtx)) {
            continue;
        }

        int dist_growth = (unrooted && vtx == ct.main_root_ct) ? 1 : 2;

        vector<int> children = ct.children_list_ct[vtx];
        for (int child : children) {
            for (int leaf : ct.leaves(child)) {
                result(leaf, vtx) = result(leaf, child) + 1;
            }
        }

        if (children.size() < 2) {
            continue;
        }

        for (int leaf1 : ct.leaves(children[0])) {
            for (int leaf2 : ct.leaves(children[1])) {
                int dist = result(leaf1, children[0]) + result(leaf2, children[1]) + dist_growth;
                result(leaf1, leaf2) = dist;
                result(leaf2, leaf1) = dist;
            }
        }
    }

    return result.leftCols(ct.n_cells);
}

// Function to compute the path length distance
double path_len_dist(const CellTree& ct1, const CellTree& ct2, bool unrooted = false) {
    MatrixXi dist_mat1 = leaf_dist_mat(ct1, unrooted);
    MatrixXi dist_mat2 = leaf_dist_mat(ct2, unrooted);

    double denominator = static_cast<double>(dist_mat1.size()) - static_cast<double>(dist_mat1.rows());
    MatrixXi diff = dist_mat1 - dist_mat2;
    double mse = diff.array().square().sum() / denominator;

    return mse;
}

// Function to compute the pairwise Hamming distances
vector<vector<double>> hamming_distance_matrix(const vector<vector<double>>& genotype_matrix) {
    int n = static_cast<int>(genotype_matrix.size());
    int m = static_cast<int>(genotype_matrix[0].size());

    vector<vector<double>> distance_matrix(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            int diff_count = 0;
            for (int k = 0; k < m; ++k) {
                if (genotype_matrix[i][k] != genotype_matrix[j][k]) {
                    ++diff_count;
                }
            }
            auto dist = static_cast<double>(diff_count);
            distance_matrix[i][j] = dist;
            distance_matrix[j][i] = dist;
        }
    }

    return distance_matrix;
}

// Function to compute the mutation count distance
double mut_count_distance(const vector<vector<double>>& genotype_matrix1, const vector<vector<double>>& genotype_matrix2) {
    auto distance_matrix1 = hamming_distance_matrix(genotype_matrix1);
    auto distance_matrix2 = hamming_distance_matrix(genotype_matrix2);

    int n = static_cast<int>(distance_matrix1.size());
    if (n == 0) return 0.0;

    double sum_squared_diff = 0.0;
    int count = 0;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {  // Only upper triangle (excluding diagonal)
            double diff = distance_matrix1[i][j] - distance_matrix2[i][j];
            sum_squared_diff += diff * diff;
            ++count;
        }
    }

    if (count == 0) return 0.0;
    return sum_squared_diff / static_cast<double>(count);
}


void writeToFile(const string& filename, const vector<pair<string, map<string, vector<double>>>>& data) {

    ofstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }
    file << fixed << setprecision(10);

    file << "{" << endl;

    for (size_t i = 0; i < data.size(); ++i) {
        const auto& [spaceKey, records] = data[i];
        file << "  \"" << spaceKey << "\": {" << endl;

        size_t count = 0;
        for (const auto& [key, values] : records) {
            file << "    \"" << key << "\": [";
            for (size_t j = 0; j < values.size(); ++j) {
                file << values[j];
                if (j < values.size() - 1) file << ", ";
            }
            file << "]";
            if (count++ < records.size() - 1) file << ",";
            file << endl;
        }

        file << "  }";
        if (i < data.size() - 1) file << ",";
        file << endl;
    }

    file << "}" << endl;

    file.close();
}


double meanAbsDiff(const std::vector<std::vector<double>>& genotypePredicted, const std::vector<std::vector<double>>& genotype) {
    assert(genotypePredicted.size() == genotype.size());
    double sum = 0.0;
    size_t totalElements = 0;

    for (size_t i = 0; i < genotype.size(); ++i) {
        assert(genotypePredicted[i].size() == genotype[i].size());
        for (size_t j = 0; j < genotype[i].size(); ++j) {
            sum += std::abs(genotypePredicted[i][j] - genotype[i][j]);
            ++totalElements;
        }
    }
    return sum / static_cast<double>(totalElements);
}
std::vector<std::vector<double>> mapGenotype(
        const std::vector<std::vector<char>>& genotype,
        const std::unordered_map<char, float>& mappingDict) {

    std::vector<std::vector<double>> mappedGenotype(genotype.size());

    for (size_t i = 0; i < genotype.size(); ++i) {
        mappedGenotype[i].resize(genotype[i].size());
        for (size_t j = 0; j < genotype[i].size(); ++j) {
            auto it = mappingDict.find(genotype[i][j]);
            assert(it != mappingDict.end());  // Ensure mapping exists
            mappedGenotype[i][j] = static_cast<double>(it->second);
        }
    }

    return mappedGenotype;
}


int main() {
    unordered_map<char, float> mappingDict = {{'A', 1.0}, {'H', 0.5}, {'R', 0.0}};

    vector<int> numCellsList = {100,500,500};
    vector<int> numMutList = {500,100,500};
    vector<vector<string>> spaces = {{"m"}, {"c"}, {"c", "m"}, {"m", "c"}};
    int nRounds = 3;
    int nTests = 100;
    bool flipped_mutation_direction = false;

    load_config("../config/config.yaml");

    std::map<std::string, double> genotype_freq = {
            {"A", std::stod(config_variables["genotype_freq.  A"])},
            {"H", std::stod(config_variables["genotype_freq.  H"])},
            {"R", std::stod(config_variables["genotype_freq.  R"])},
    };

    for (size_t s = 0; s < numCellsList.size(); ++s) {
        int nCells = numCellsList[s];
        int nMut = numMutList[s];

        string ncm = to_string(nCells) + "c" + to_string(nMut) + "m";

        string basePath = "../data/results/figures/";
        basePath.append(ncm).append("/");

        string filePath(basePath), filePathDist(basePath), filePathVaf(basePath), filePathMutCountDist(basePath);

        filePath.append("optimal_tree_llh_comparison_").append(ncm).append(".json");
        filePathDist.append("path_len_distance_comparison_").append(ncm).append(".json");
        filePathVaf.append("vaf_comparison_").append(ncm).append(".json");
        filePathMutCountDist.append("mut_count_distance_comparison_").append(ncm).append(".json");

        ifstream vafFile(filePathVaf);
        if (!vafFile.good()) {
            // Create directories if they do not exist
            filesystem::create_directories(basePath);

            vector<pair<string, map<string, vector<double>>>> optimalTreeLlh;
            vector<pair<string, map<string, vector<double>>>> pathLenDistance;
            vector<pair<string, map<string, vector<double>>>> vafDistance;
            vector<pair<string, map<string, vector<double>>>> mutCountDistance;

            for (const auto& space : spaces) {
                string spaceKey = accumulate(space.begin(), space.end(), string(),
                                             [](const string& a, const string& b) { return a.empty() ? b : a + "_" + b; });

                optimalTreeLlh.emplace_back(spaceKey, map<string, vector<double>>{});
                pathLenDistance.emplace_back(spaceKey, map<string, vector<double>>{});
                vafDistance.emplace_back(spaceKey, map<string, vector<double>>{});
                mutCountDistance.emplace_back(spaceKey, map<string, vector<double>>{});

                string path = "../data/simulated_data/" + ncm;

                string pathResults = path;
                pathResults.append("/sciterna_tree_space_comparison_cpp_" + spaceKey);

                MutationFilter mf(std::stod(config_variables["error_rate"]), std::stod(config_variables["overdispersion"]),
                                  genotype_freq, std::stod(config_variables["mut_freq"]),
                                  std::stod(config_variables["dropout_alpha"]), std::stod(config_variables["dropout_beta"]),
                                  std::stod(config_variables["dropout_direction"]), std::stod(config_variables["overdispersion_h"]));

                for (int r = 0; r < nRounds; ++r) {
                    string key = to_string(nCells) + "_" + to_string(nMut) + "_" + to_string(r);

                    optimalTreeLlh.back().second[key] = vector<double>();
                    pathLenDistance.back().second[key] = vector<double>();
                    vafDistance.back().second[key] = vector<double>();
                    mutCountDistance.back().second[key] = vector<double>();

                    for (int i = 0; i < nTests; ++i) {
                        cout << "Processing: " << r << " " << i << endl;
                        // Load data
                        vector<int> sciternaParentVec = loadtxt<int>(pathResults + "/sciterna_parent_vec/sciterna_parent_vec_" + to_string(r) + "r" + to_string(i) + ".txt");
                        vector<int> trueParentVec = loadtxt<int>(path + "/parent_vec/parent_vec_" + to_string(i) + ".txt");
                        vector<vector<int>> ref = loadtxt2D<int>(path + "/ref/ref_" + to_string(i) + ".txt");
                        vector<vector<int>> alt = loadtxt2D<int>(path + "/alt/alt_" + to_string(i) + ".txt");
                        vector<vector<char>> genotypePred = loadtxt2D<char>(pathResults + "/sciterna_genotype/sciterna_genotype_" + to_string(r) + "r" + to_string(i) + ".txt");
                        vector<vector<char>> genotype = loadtxt2D<char>(path + "/genotype/genotype_" + to_string(i) + ".txt");

                        // Vectorize mapping
                        vector<vector<double>> genotypePredicted = mapGenotype(genotypePred, mappingDict);
                        vector<vector<double>> genotypeGt = mapGenotype(genotype, mappingDict);

                        vector<int> selected = loadtxt<int>(pathResults + "/sciterna_selected_loci/sciterna_selected_loci_" + to_string(r) + "r" + to_string(i) + ".txt");
                        vector<char> gt1, gt2;
                        loadGenotypes(pathResults + "/sciterna_inferred_mut_types/sciterna_inferred_mut_types_" + to_string(r) + "r" + to_string(i) + ".txt", gt1, gt2);

                        auto [llh_1, llh_2] = mf.get_llh_mat(slice_columns(ref, selected),
                                                             slice_columns(alt, selected), gt1, gt2, false);

                        // Prepare for joint calculation
                        int nCellsCalc = (static_cast<int>(trueParentVec.size()) + 1) / 2;
                        CellTree ctGt(nCellsCalc, nMut, flipped_mutation_direction);
                        CellTree ctSciterna(nCellsCalc, nMut, flipped_mutation_direction);
                        ctGt.fitLlh(llh_1, llh_2);
                        ctSciterna.fitLlh(llh_1, llh_2);

                        // Calculate differences in joint likelihood
                        ctGt.useParentVec(trueParentVec);
                        ctGt.updateAll();
                        double trueJoint = ctGt.joint;

                        ctSciterna.useParentVec(sciternaParentVec);
                        ctSciterna.updateAll();

                        optimalTreeLlh.back().second[key].push_back((ctSciterna.joint - trueJoint) / (ctSciterna.n_cells * ctSciterna.n_mut));
                        pathLenDistance.back().second[key].push_back(path_len_dist(ctGt, ctSciterna));
                        vafDistance.back().second[key].push_back(meanAbsDiff(genotypePredicted, genotypeGt));
                        mutCountDistance.back().second[key].push_back(mut_count_distance(genotypePredicted, genotypeGt));
                    }
                }
            }

            // Write results to files
            writeToFile(filePath, optimalTreeLlh);
            writeToFile(filePathDist, pathLenDistance);
            writeToFile(filePathVaf, vafDistance);
            writeToFile(filePathMutCountDist, mutCountDistance);
        }
    }

    return 0;
}
