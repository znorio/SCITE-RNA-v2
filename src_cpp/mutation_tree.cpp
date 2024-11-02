/*
Defines the mutation tree and how it is optimized.
*/

#include <iostream>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <cassert>
#include <set>
#include <deque>
#include <vector>
#include <chrono>
//#include <Eigen/Dense>

#include "mutation_tree.h"

// initialize mutation tree
MutationTree::MutationTree(int n_mut, int n_cells)
        : n_mut(n_mut), n_cells(n_cells) {
    if (n_mut < 2) {
        std::cerr << "Mutation tree too small, nothing to explore." << std::endl;
    }

    n_vtx = n_mut + 1;
    flipped = std::vector<bool>(n_vtx, false);
    cell_loc = std::vector<int>(n_cells, -1);
    wt = n_mut;
    parent_vector_mt = std::vector<int>(n_vtx, -1);
    children_list_mt.resize(n_vtx + 1);
    main_root_mt = wt;

    reroot(wt);
    randomMutationTree();
//    std::vector<int> new_parent_vector = {2,2,51,2,2,2,2,5,5,7,6,10,6,11,11,8,2,7,2,13,9,20,2,18,21,9,17,
//                                          18,6,2,29,4,11,51,27,51,14,0,32,31,30,25,30,9,36,5,18,10,37,36,31,-1};
//    useParentVec(new_parent_vector);
}

// convert cell tree to mutation tree
void MutationTree::fitCellTree(CellTree ct) {
    assert(n_mut == ct.n_mut);
    assert(ct.roots().size() == 1);
    parent_vector_mt = std::vector<int>(n_vtx, -1);
    children_list_mt = std::vector<std::vector<int>>(n_vtx + 1);
    addChild(-1, wt); // maybe reinitialization can be avoided?

    std::vector<int> mrm(ct.n_vtx + 1, -1); // mrm for "most recent mutation"
    mrm.back() = wt; // put wildtype at the end

    for (int cvtx : ct.dfs(ct.main_root_ct)) {  // cvtx for "cell vertex"
        std::vector<int> mut_list;
        for (int i = 0; i < ct.mut_loc.size(); ++i) {
            if (ct.mut_loc[i] == cvtx) {
                mut_list.push_back(i);
            }
        }
        int parent = ct.parent_vector_ct[cvtx];
        if (parent == -1){
            parent = static_cast<int>(mrm.size()) - 1;
        }
        int parent_mut = mrm[parent];
        if (!mut_list.empty()) {
//            std::sort(mut_list.begin(), mut_list.end());
            std::shuffle(mut_list.begin(), mut_list.end(), std::mt19937{std::random_device{}()}); // randomize the order of mutations at the same edge
            assignParent(mut_list[0], parent_mut); // assigns the first mutation to the parent_mut
            for (size_t i = 1; i < mut_list.size(); ++i) {
                assignParent(mut_list[i], mut_list[i-1]);
            }
            mrm[cvtx] = mut_list.back();
        } else {
            mrm[cvtx] = mrm[parent];
        }
    }

    std::copy(ct.flipped.begin(), ct.flipped.end(), flipped.begin());
    wt_llh = 0.0;
    for (int j = 0; j < n_mut; ++j) {
        if (flipped[j]) {
            wt_llh += loc_joint_2[j];
        } else {
            wt_llh += loc_joint_1[j];
        }
    }
}

// initialize random mutation tree
void MutationTree::randomMutationTree() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, n_mut - 1);
    int root_assigned = dist(gen);
    assignParent(root_assigned, wt); // Randomly choose one mutation to have self.wt as its parent
    assignParent(wt, -1); // assign wt to root

    // Assign each mutation a random parent ensuring it's a valid tree
    for (int vtx = 0; vtx < n_mut; ++vtx) {
        if (vtx == root_assigned) {
            continue; // Skip the already assigned root mutation
        }

        // Assign a random parent from the set of already assigned mutations or self.wt
        std::vector<int> potential_parents;
        for (int i = 0; i < n_mut; ++i) {
            if (i != vtx && parent_vector_mt[i] != -1) {
                potential_parents.push_back(i);
            }
        }
        potential_parents.push_back(wt);
        std::uniform_int_distribution<> parent_dist(0, static_cast<int>(potential_parents.size()) - 1);
        int parent = potential_parents[parent_dist(gen)];
        assignParent(vtx, parent);
    }
}

// initialize mutation tree with a new parent vector
void MutationTree::useParentVec(const std::vector<int>& new_parent_vec, int main_r) {
    if (new_parent_vec.size() != static_cast<size_t>(n_vtx)) {
        throw std::invalid_argument("Parent vector must have the same length as number of vertices.");
    }

    parent_vector_mt = new_parent_vec;
    children_list_mt = std::vector<std::vector<int>>(n_vtx + 1); // Reinitialize the children_list

    for (int vtx = 0; vtx < n_vtx; ++vtx) {
        addChild(new_parent_vec[vtx],vtx);
    }

    if (main_r == -1) {
        main_root_mt = roots()[0];
    } else if (parent_vector_mt[main_root_mt] != -1) {
        throw std::invalid_argument("Provided main root is not a root.");
    } else {
        main_root_mt = main_r;
    }
}

// initialize LLR, joint LL, cumulative LLR, cell locations
void MutationTree::fitLlh(const std::vector<std::vector<double>>& llh_1, const std::vector<std::vector<double>>& llh_2) {
    assert(llh_1.size() == llh_2.size() && llh_1[0].size() == llh_2[0].size());

    llr.resize(n_cells, std::vector<double>(n_vtx));
    cumul_llr.resize(n_cells, std::vector<double>(n_vtx));
    loc_joint_1.resize(n_mut);
    loc_joint_2.resize(n_mut);

    for (int i = 0; i < n_cells; ++i) {
        for (int j = 0; j < n_mut; ++j) {
            llr[i][j] = llh_2[i][j] - llh_1[i][j];
        }
        llr[i][n_mut] = 0; // stands for wildtype
    }

    for (int j = 0; j < n_mut; ++j) {
        loc_joint_1[j] = 0;
        loc_joint_2[j] = 0;
        for (int i = 0; i < n_cells; ++i) {
            loc_joint_1[j] += llh_1[i][j];
            loc_joint_2[j] += llh_2[i][j];
        }
    }

    wt_llh = 0.0; // as flipped is not optimized in the mutation tree space wt_llh is a constant
    for (int j = 0; j < n_mut; ++j) {
        if (flipped[j]) {
            wt_llh += loc_joint_2[j];
        } else {
            wt_llh += loc_joint_1[j];
        }
    }
    updateAll();
}

// update cumulative LLR and cell locations
void MutationTree::updateAll() {
    updateCumulLlr();
    updateCellLoc();
}

// update the cumulative LLR, which is the LLR of cells having all the mutations above the attachment point in the mutation tree and no others
void MutationTree::updateCumulLlr() {

    for (int rt : roots()) {
        for (int vtx : dfs(rt)) {
            std::vector<double> llr_summand(n_cells);

            // Compute llr_summand for the current vertex
            if (flipped[vtx]){
                for (int i = 0; i < n_cells; ++i) {
                    llr_summand[i] = -llr[i][vtx];
                }
            } else {
                for (int i = 0; i < n_cells; ++i) {
                    llr_summand[i] = llr[i][vtx];
                }
            }

            if (isRoot(vtx)) {
                for (int i = 0; i < n_cells; ++i) {
                    cumul_llr[i][vtx] = llr_summand[i];
                }
            } else {
                int parent_vtx = parent_vector_mt[vtx];
                for (int i = 0; i < n_cells; ++i) {
                    cumul_llr[i][vtx] = cumul_llr[i][parent_vtx] + llr_summand[i];
                }
            }
        }
    }
}

// insert cells at their optimal location
void MutationTree::updateCellLoc() {
    joint = 0.0;

    for (int i = 0; i < n_cells; ++i) {
        double max_value = -std::numeric_limits<double>::infinity();
        int max_index = -1;

        for (int j = 0; j < n_vtx; ++j) {
            if (cumul_llr[i][j] > max_value) {
                max_value = cumul_llr[i][j];
                max_index = j;
            }
        }
        joint += max_value;

        cell_loc[i] = max_index;
    }
    joint += wt_llh;
}

// prune individual mutations and subtrees, then reattach them at their optimal location
void MutationTree::exhaustiveOptimize(bool insert_nodes) {
    std::vector<int> mut_random_order(n_mut);
    std::iota(mut_random_order.begin(), mut_random_order.end(), 0);

    if (insert_nodes) {
        for (int subroot: mut_random_order) {
            if (children_list_mt[subroot].size() > 1) {
                continue;
            }
            if (subroot == main_root_mt) {
                continue;
            }
            pruneNode(subroot);
            updateAll();
            greedyAttachNode(subroot);
        }
        updateAll();
        std::cout << "After Node reattachment " << joint << std::endl;
    }

    std::shuffle(mut_random_order.begin(), mut_random_order.end(), std::mt19937{std::random_device{}()});
    for (int subroot : mut_random_order) {
        if (subroot == main_root_mt){
            continue;
        }
        prune(subroot);
        updateAll();
        greedyAttach(subroot);
    }
}

// attach subtree at the optimal location of the mutation tree
// this can be done efficiently by pre-calculating the maxima of the subtree and main tree.
// while reattaching the maxima of the subtree changes by the attachment point cumul_llr,
// but as it is the same for every node in the subtree it can be added to the precalculated maxima directly.
void MutationTree::greedyAttach(int subroot) {
    std::vector<int> main_root_nodes = dfs(main_root_mt);
    std::vector<int> subroot_nodes = dfs(subroot);

    std::vector<double> main_tree_max = getMaxValues(cumul_llr, main_root_nodes);
    std::vector<double> subtree_max = getMaxValues(cumul_llr, subroot_nodes);

    double best_llr = -std::numeric_limits<double>::infinity();
    int best_loc = -1;

    for (int vtx : main_root_nodes) {
        std::vector<double> total_llr_vec(n_cells);
        for (int i = 0; i < n_cells; ++i) {
            total_llr_vec[i] = std::max(main_tree_max[i], subtree_max[i] + cumul_llr[i][vtx]);
        }
        double total_llr = std::accumulate(total_llr_vec.begin(), total_llr_vec.end(), 0.0);

        if (total_llr > best_llr) {
            best_llr = total_llr;
            best_loc = vtx;
        }
    }

    assignParent(subroot, best_loc);
    updateAll();
//    std::cout << best_loc << " " << joint << std::endl;
}


// insert a node at the optimal location of the mutation tree
void MutationTree::greedyAttachNode(int subroot) {
    double best_llr_append = -std::numeric_limits<double>::infinity();
    double best_llr_insert = -std::numeric_limits<double>::infinity();
    int best_loc = -1;
    int best_child = -1;

    // Append the pruned mutation to some other mutation
    // this changes the cumul_llr only at subroot, so that the maxima for the rest of the matrix don't need to be
    // computed every time
    // pre-calculate summand_llr at subroot
    std::vector<double> summand_llr_subroot(n_cells);
    if (flipped[subroot]){
        for (int i = 0; i < n_cells; ++i) {
            summand_llr_subroot[i] = -llr[i][subroot];
        }
    } else {
        for (int i = 0; i < n_cells; ++i) {
            summand_llr_subroot[i] = llr[i][subroot];
        }
    }

    // calculate the max of the cumulative llr for the whole matrix except at subroot
    std::vector<double> cumul_llr_rest_max(n_cells);
    for (int i = 0; i < n_cells; ++i) {
        double max_value = -std::numeric_limits<double>::infinity();

        for (int j = 0; j < n_vtx; ++j) {
            if (j != subroot) {
                max_value = std::max(max_value, cumul_llr[i][j]);
            }
        }
        cumul_llr_rest_max[i] = max_value;
    }
    for (int vtx = 0; vtx < n_vtx; ++vtx){ // equivalent to: for (int vtx : dfs(main_root))
        if (vtx == subroot){
            continue;
        }
        std::vector<double> cumul_llr_subroot(n_cells);
        for (int i = 0; i < n_cells; ++i) {
            cumul_llr_subroot[i] = summand_llr_subroot[i] + cumul_llr[i][vtx];
        }

        double total_llr = wt_llh;

        for (int i = 0; i < n_cells; ++i) {
            if (cumul_llr_rest_max[i] > cumul_llr_subroot[i]) {
                total_llr += cumul_llr_rest_max[i];
            } else {
                total_llr += cumul_llr_subroot[i];
            }
        }
        // This approach is more intuitive, but slower...
//        assignParent(subroot, vtx);
//        updateAll();
//        double total_llr = joint;
//        pruneNode(subroot);
        if (total_llr > best_llr_append) {
            best_llr_append = total_llr;
            best_loc = vtx;
        }
    }

    // Insert mutation and append subtree
    // This action changes the cumul_llr of the whole subtree below the attachment point. To make the calculation of
    // the maxima (= best cell locations) more efficient, we start at the leaves and work our way though the tree. We keep track of the maxima
    // of the subtrees. So the maxima of the parent is the maxima of its children and the updated cumul_llr of the parent
    // node.
    // To find the global maximum we can pre-calculate the maxima of the main tree and their location. If the maxima are
    // not inside the subtree, we can use the precalculated maxima.

    // calculate the maxima and cell locations of the cumulative llr for the original cumulative llr matrix
    //    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> best_inserts; // there might be several equally likely locations
    std::vector<double> cumul_llr_max(n_cells);

    for (int i = 0; i < n_cells; ++i) {
        double max_value = -std::numeric_limits<double>::infinity();
        int max_index = -1;

        for (int j = 0; j < n_vtx; ++j) {
            if (cumul_llr[i][j] > max_value) {
                max_value = cumul_llr[i][j];
                max_index = j;
            }
        }
        cumul_llr_max[i] = max_value;
        cell_loc[i] = max_index;
    }

    // initialize a queue - we can only calculate a node once we have processed all it's children
    std::deque<int> queue;
    for (int i = 0; i < n_vtx; ++i) {
        if (i != subroot) {
            queue.push_back(i);
        }
    }

    // this matrix is used to store the maxima of the subtrees of each node
    std::vector<std::vector<double>> maxima(n_vtx, std::vector<double>(n_cells,-std::numeric_limits<double>::infinity()));
    std::set<int> all_columns;
    for (int i = 0; i < n_vtx; ++i) {
        all_columns.insert(i);
    }
    std::vector<double> max_subroot(n_cells);
    std::vector<double> maxima_node(n_cells);

    while (!queue.empty()) {
        int node = queue.front();
        queue.pop_front();
        std::vector<int> children = children_list_mt[node];

        bool all_children_in_maxima = true; // determines whether we already have processed all children, also true for leaves
        for (int child : children) {
            if (maxima[child][0] == -std::numeric_limits<double>::infinity()) {
                all_children_in_maxima = false;
                break;
            }
        }

        if (all_children_in_maxima) { // only process a node once all it's children have been processed or if it's a leaf

            for (int i = 0; i < n_cells; ++i) {
                max_subroot[i] = cumul_llr[i][parent_vector_mt[node]] + summand_llr_subroot[i]; // calculate the new llr of the mutation to be inserted
                maxima_node[i] = cumul_llr[i][node] + summand_llr_subroot[i]; // calculate the new llr of the node the mutation is inserted above
            }

            if (children.empty()) {
                maxima[node] = maxima_node; // if the node is a leaf the maxima of the subtree are the maxima of the node
            }
            else { // if the node has children, the maxima of the subtree are the maxima of its children and the node itself
                for (int i = 0; i < n_cells; ++i) {
                    double max_val = maxima_node[i];
                    for (int child : children) {
                            max_val = std::max(max_val, maxima[child][i]);
                        }
                    maxima[node][i] = max_val;
                }
            }

            if (node == main_root_mt) { // the main root is processed last, as it is only processed after all it's children, no mutation can be inserted above wt
                break;
            }

            double Joint = 0;
            std::vector<int> subtree = dfs(node);

            subtree.push_back(subroot);

            for (int i = 0; i < n_cells; ++i) {
                // if the maxima (= cell locations) are not within the subtree
                if (std::find(subtree.begin(), subtree.end(), cell_loc[i]) == subtree.end()) {
                    Joint += std::max({max_subroot[i], maxima[node][i], cumul_llr_max[i]});
                }
                else {
                    // if the summand is larger than 0 and the cell location is not the mutation inserted, adding the
                    // summand will not change the cell location (= location of the maxima is still within the subtree)
                    if (summand_llr_subroot[i] > 0 && cell_loc[i] != subroot) {
                        Joint += std::max(max_subroot[i], maxima[node][i]);
                    } else { // otherwise there is the chance that the new maximum lies outside the subtree, so we have
                        // to recalculate the maximum
                        double new_max = -std::numeric_limits<double>::infinity();
                        std::vector <double> copy_cumul_llr = cumul_llr[i];
                        for (int sn : subtree){
                            copy_cumul_llr[sn] += summand_llr_subroot[i]; // summand is added to the subtree
                        }
                        copy_cumul_llr[subroot] = max_subroot[i];
                        for (int col = 0; col < n_vtx; ++col) {
                            new_max = std::max(new_max, copy_cumul_llr[col]);
                        }
                        Joint += new_max;
                    }
                }
            }

            double total_llr = Joint + wt_llh;

            if (total_llr == best_llr_insert) {
                best_inserts.push_back(node);
            }
            if (total_llr > best_llr_insert) {
                best_llr_insert = total_llr;
                best_child = node;
                best_inserts.clear();
                best_inserts.push_back(node);
            }
        } else {
            queue.push_back(node);  // Re-add the node for later processing
        }
    }
    // more intuitive node insertion code - however slower...
//    for (int vtx : dfs(main_root)) {
//        std::vector<int> children_vec = children_list[vtx];
//        for (int child : children_vec) {
//            insert(subroot, child);
//            updateAll();
//            double total_llr = joint;
//            pruneNode(subroot);
//            if (total_llr > best_llr_insert) {
//                best_llr_insert = total_llr;
//                best_child = child;
//            }
//        }
//    }
//    auto end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double, std::milli> duration = end - start;
//    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    if (best_llr_append > best_llr_insert) {
        assignParent(subroot, best_loc);
    } else {
        std::random_device rd;   // Seed generator
        std::mt19937 gen(rd());  // Mersenne Twister RNG
        std::uniform_int_distribution<std::vector<int>::size_type> dis(0, best_inserts.size() - 1); // Uniform distribution over the vector indices

        // Get the random element in one line
        int random_element = best_inserts[dis(gen)];
//        int chosen_node = *std::min_element(best_inserts.begin(), best_inserts.end());
        insert(subroot, random_element);
//        std::cout << min_node << " insert" << std::endl;
    }

//    std::cout << best_loc << " append" << std::endl;
//    std::cout << best_llr_append << " " << best_llr_insert << " " << std::endl;
}

// BASIC TREE FUNCTIONS

// assign child to a new parent
void MutationTree::assignParent(int child, int new_parent) {
    removeChild(parent_vector_mt[child], child);
    parent_vector_mt[child] = new_parent;
    addChild(new_parent, child);
}

// define new main root
void MutationTree::reroot(int new_main_root) {
    if (parent_vector_mt[new_main_root] != -1) {
        prune(new_main_root);
    }
    main_root_mt = new_main_root;
}

// returns whether vtx is a root
bool MutationTree::isRoot(int vtx) {
    return parent_vector_mt[vtx] == -1;
}

// remove child from children list of parent
void MutationTree::removeChild(int parent, int child) {
    if (parent == -1) {
        parent = n_vtx;
    }

    auto& childList = children_list_mt[parent];
    auto it = std::find(childList.begin(), childList.end(), child);

    if (it != childList.end()) {
        childList.erase(it);
    }
}

// add new child to children list of parent
void MutationTree::addChild(int parent, int child) {
    if (parent == -1) {
        parent = n_vtx;
    }
    children_list_mt[parent].push_back(child);
}

// check if node is a leaf
bool MutationTree::isLeaf(int node) {
    return children_list_mt[node].empty();
}

// assign subroot to wildtype
void MutationTree::prune(int subroot){
    assignParent(subroot, -1);
}

// Prune the node. Attach the children to the parent of the node.
void MutationTree::pruneNode(int vtx) {
    int new_parent = parent_vector_mt[vtx];
    std::vector<int> children = children_list_mt[vtx]; // make a copy of children list since children_list can change during assignParent
    for (int child : children) {
        assignParent(child, new_parent);
    }
    assignParent(vtx, -1); // Detach the pruned node
}

// Insert anchor into the tree above target and assign target to anchor
void MutationTree::insert(int subroot, int target) {
    int parent = parent_vector_mt[target];
    assignParent(target, subroot);    // target becomes the child of subroot
    assignParent(subroot, parent); // subroot becomes the parent of target's current parent

    if (target == main_root_mt) {
        std::cout << "Careful mutation root has changed!" << std::endl;
        main_root_mt = subroot;                // Update main_root if necessary
    }
}

// return roots
std::vector<int> MutationTree::roots() {
    return children_list_mt.back();
}

// depth first search
std::vector<int> MutationTree::dfs(int subroot) {
    std::vector<int> stack = {subroot};
    std::vector<int> output;
    output.reserve(n_vtx);
    int check = 0;

    while (!stack.empty()) {
        check += 1;
        if (check > n_vtx){
            throw std::runtime_error("Problem with dfs");
        }

        int node = stack.back();
        stack.pop_back();
        output.push_back(node);

        // Add children of the current node to the stack
        for (int child : children_list_mt[node]) {
            stack.push_back(child);
        }
    }
    return output;
}

// reverse depth first search
std::vector<int> MutationTree::rdfs(int subroot) {
    std::vector<int> stack = {subroot};
    std::vector<int> output;
    output.reserve(n_vtx);
    int check = 0;

    while (!stack.empty()) {
        check += 1;
        if (check > n_vtx){
            throw std::runtime_error("Problem with rdfs");
        }

        int node = stack.back();
        stack.pop_back();
        output.push_back(node);

        // Add children of the current node to the stack
        for (int child : children_list_mt[node]) {
            stack.push_back(child);
        }
    }

    // Reverse the output to get nodes in reversed DFS order
    std::reverse(output.begin(), output.end());

    return output;
}

// HELPER FUNCTIONS

// returns maximum of indexed columns of a 2d matrix
std::vector<double> MutationTree::getMaxValues(const std::vector<std::vector<double>>& matrix, const std::vector<int>& indices) {
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

// add two 1d vectors
std::vector<double> MutationTree::addVectors(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for element-wise addition.");
    }
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}