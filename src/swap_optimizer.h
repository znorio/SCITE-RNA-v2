#ifndef SCITE_RNA_CPP_SWAP_OPTIMIZER_H
#define SCITE_RNA_CPP_SWAP_OPTIMIZER_H

#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include "cell_tree.h"
#include "mutation_tree.h"

class SwapOptimizer {
private:
    int sig_digits;
    std::vector<std::string> spaces;
    bool reverse_mutations;
    int n_decimals{};
    int n_mut_;
    int n_cells_;

public:
    explicit SwapOptimizer(std::vector<std::string> spaces = {"c", "m"}, bool reverse_mutations = true,
                           int n_mut=2, int n_cells=2, int sig_digits = 10);
    [[nodiscard]] double getCurrentJoint() const;
    [[nodiscard]] double getMtJoint() const;
    void fit_llh(const std::vector<std::vector<double>>& llh_1, const std::vector<std::vector<double>>& llh_2);
    void optimize(int max_loops = 100);
    static double round_to_n_decimals(double value, int n_decimals);
    MutationTree mt;
    CellTree ct;
};

#endif
