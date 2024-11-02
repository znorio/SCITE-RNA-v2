#ifndef SCITE_RNA_CPP_MUTATION_TREE_H
#define SCITE_RNA_CPP_MUTATION_TREE_H

#include <iostream>
#include <vector>
#include <limits>

#include <cell_tree.h>

class CellTree;

class MutationTree {
private:
    int n_mut;
    double wt_llh = -std::numeric_limits<double>::infinity();
    std::vector<bool> flipped;
    std::vector<int> parent_vector_mt;
    std::vector<std::vector<double>> llr;
    std::vector<std::vector<double>> cumul_llr;
    std::vector<double> loc_joint_1;
    std::vector<double> loc_joint_2;

public:
    explicit MutationTree(int n_mut = 2, int n_cells = 2);
    void fitCellTree(CellTree ct);
    void randomMutationTree();
    void useParentVec(const std::vector<int>& new_parent_vec, int main_r=-1);
    void fitLlh(const std::vector<std::vector<double>>& llh_1, const std::vector<std::vector<double>>& llh_2);
    void updateAll();
    void updateCumulLlr();
    void updateCellLoc();
    void exhaustiveOptimize(bool insert_nodes);
    void greedyAttach(int subroot);
    void greedyAttachNode(int subroot);

    // basic tree functions
    void assignParent(int child, int parent);
    void reroot(int root);
    bool isRoot(int vtx);
    void removeChild(int parent, int child);
    void addChild(int parent, int child);
    bool isLeaf(int node);
    void prune(int subroot);
    void pruneNode(int vtx);
    void insert(int subroot, int target);
    std::vector<int> roots();
    std::vector<int> dfs(int subroot);
    std::vector<int> rdfs(int subroot);

    // helper functions
    static std::vector<double> getMaxValues(const std::vector<std::vector<double>>& matrix, const std::vector<int>& indices);
    static std::vector<double> addVectors(const std::vector<double>& a, const std::vector<double>& b);

    // public variables
    double joint = -std::numeric_limits<double>::infinity();
    int wt;
    int main_root_mt;
    int n_vtx;
    int n_cells;
    std::vector<int> cell_loc;
    std::vector<std::vector<int>> children_list_mt;
};


#endif //SCITE_RNA_CPP_MUTATION_TREE_H
