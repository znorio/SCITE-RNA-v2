#ifndef CELL_TREE_H
#define CELL_TREE_H

#include <iostream>
#include <vector>
#include <limits>

#include <mutation_tree.h>

class MutationTree;

class CellTree {
private:
    int n_cells;
    double joint_1_sum;
    bool reversible;
    std::vector<std::vector<double>> llr;
    std::vector<double> loc_joint_1;
    std::vector<double> loc_joint_2;
    std::vector<std::vector<int>> children_list_ct;

public:
    // cell tree functions
    explicit CellTree(int n_cells = 3, int n_mut = 0, bool reversible_mut = true);
    void fitMutationTree(MutationTree mt);
    void randSubtree(bool initialTree=false, std::vector<int>* leaves = nullptr, std::vector<int>* internals = nullptr);
    void useParentVec(const std::vector<int>& new_parent_vec, int main_r=-1);
    void fitLlh(const std::vector<std::vector<double>>& llh_1, const std::vector<std::vector<double>>& llh_2);
    void updateAll();
    void updateLlr();
    void updateMutLoc();
    void exhaustiveOptimize(bool leaf_only=false);
    void greedyInsert(int anchor, int subroot);
    std::pair<int, double> searchInsertionLoc(int target, int anchor, int subroot);
    std::tuple<double, bool, std::vector<double>> jointCalculationNotFlipped(bool was_leaf, int anchor,
                                                                             std::vector<double> current_llr_max_without_anchor,
                                                                             int currentTarget,
                                                                             std::vector<double> originalLlr);
    std::tuple <double, bool, std::vector<double>, std::vector<double>> jointCalculationFlipped(bool was_leaf, int anchor,
                                                                                     std::vector<double> current_llr_max_without_anchor,
                                                                                     std::vector<double> current_llr_min_without_anchor,
                                                                                     int currentTarget,
                                                                                     std::vector<double> originalLlr);

    // basic tree functions
    void assignParent(int child, int parent);
    void reroot(int root);
    void removeChild(int parent, int child);
    void addChild(int parent, int child);
    bool isLeaf(int node);
    bool isRoot(int vtx);
    void prune(int subroot);
    void binaryPrune(int subroot);
    void splice(int subroot);
    void insert(int anchor, int target);
    int sibling(int node);
    std::vector<int> roots();
    std::vector<int> dfs(int subroot);
    std::vector<int> rdfs(int subroot);

    // helper functions
    static std::vector<double> addVectors(const std::vector<double>& a, const std::vector<double>& b);

    // public variables
    double joint = -std::numeric_limits<double>::infinity();
    int n_mut;
    int n_vtx;
    int main_root_ct = 0;
    std::vector<int> mut_loc;
    std::vector<int> parent_vector_ct;
    std::vector<bool> flipped;
};

#endif