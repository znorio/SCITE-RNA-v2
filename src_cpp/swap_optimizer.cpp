/*
To optimize the trees, SCITE-RNA alternates between mutation and cell lineage tree spaces.
*/

#include <algorithm>

#include "swap_optimizer.h"
#include <utility>
#include "cell_tree.h"

// initialize tree space swap optimizer
SwapOptimizer::SwapOptimizer(std::vector<std::string> spaces, bool reverse_mutations, int n_mut, int n_cells, int sig_digits)
        : sig_digits(sig_digits), spaces(std::move(spaces)), mt(n_mut, n_cells), n_cells_(n_cells), n_mut_(n_mut),
        ct(n_cells, n_mut, reverse_mutations){}

// get rounded cell lineage tree joint LLH
double SwapOptimizer::getCtJoint() const {
    return round_to_n_decimals(ct.joint, n_decimals);
}

// get rounded mutation tree joint LLH
double SwapOptimizer::getMtJoint() const {
    return round_to_n_decimals(mt.joint, n_decimals);
}

// initialize the mutation and cell lineage trees
void SwapOptimizer::fit_llh(const std::vector<std::vector<double>>& llh_1, const std::vector<std::vector<double>>& llh_2) {
    ct.fitLlh(llh_1, llh_2);
    mt.fitLlh(llh_1, llh_2);

    double mean_abs = 0.0;
    for (int i = 0; i < n_cells_; ++i) {
        for (int j = 0; j < n_mut_; ++j) {
            mean_abs += std::abs(llh_1[i][j] + llh_2[i][j]);
        }
    }
    mean_abs /= 2;
    n_decimals = static_cast<int>(sig_digits - std::log10(mean_abs));
}

// optimize mutation and cell lineage tree spaces
void SwapOptimizer::optimize(int max_loops, bool insert_nodes) {
    std::vector<bool> converged = {spaces.end() == std::find(spaces.begin(), spaces.end(), "c"),
                                   spaces.end() == std::find(spaces.begin(), spaces.end(), "m")};

    int current_space = spaces[0] == "c" ? 0 : 1;
    int loop_count = 0;
    double start_joint = -std::numeric_limits<double>::infinity();

    while (!std::all_of(converged.begin(), converged.end(), [](bool v) { return v; })) {
        if (loop_count >= max_loops) {
            std::cerr << "Maximal loop number exceeded." << std::endl;
            break;
        }
        ++loop_count;

        if (current_space == 0) {
            std::cout << "Optimizing cell lineage tree ..." << std::endl;
            ct.exhaustiveOptimize();
            mt.fitCellTree(ct);
            mt.updateAll();
        } else {
            std::cout << "Optimizing mutation tree ..." << std::endl;
            mt.exhaustiveOptimize(insert_nodes);
            ct.fitMutationTree(mt);
            ct.updateAll();
        }

        std::cout << ct.joint << " " << mt.joint << std::endl;

        double current_joint = (spaces.size() == 1 && spaces[0] == "m") ? getMtJoint() :
                               (spaces.size() == 1 && spaces[0] == "c") ? getCtJoint() :
                               (getCtJoint()); // get the likelihood of the current space

        if (start_joint < current_joint) {
            converged[current_space] = false;
        } else if (start_joint == current_joint) {
            converged[current_space] = true;
        } else {
            throw std::runtime_error("Observed decrease in joint likelihood.");
        }

        start_joint = (spaces.size() == 1 && spaces[0] == "m") ? getMtJoint() :
                      (spaces.size() == 1 && spaces[0] == "c") ? getCtJoint() :
                      (getCtJoint()); // get the starting likelihood of the next space


        if (std::find(spaces.begin(), spaces.end(), "c") != spaces.end() &&
            std::find(spaces.begin(), spaces.end(), "m") != spaces.end()) {
            current_space = 1 - current_space;
        }
    }
}

// round to make sure, that smaller rounding errors don't affect the optimization
double SwapOptimizer::round_to_n_decimals(double value, int n_decimals) {
    double scale = std::pow(10.0, n_decimals);
    return std::round(value * scale) / scale;
}