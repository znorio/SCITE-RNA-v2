"""
Defines the cell lineage tree and how it is optimized.
"""

import numpy as np
import graphviz
import warnings

from pywin.Demos.dyndlg import test1
from scipy.special import logsumexp

from src_python.tree_base import PruneTree
from src_python.utils import load_config_and_set_random_seed

config = load_config_and_set_random_seed()


class CellTree(PruneTree):
    def __init__(self, n_cells=3, n_mut=0, flipped_mutation_direction=False):
        """
        [Arguments]
            reversible_mut: a boolean value indicating whether the opposite direction should be considered.
                If the mutations are not reversible, the direction is assumed to be from gt1 to gt2.
        """
        if n_cells < 3:
            warnings.warn("Cell tree too small, nothing to explore", RuntimeWarning)

        super().__init__(2 * n_cells - 1)
        self.n_cells = n_cells
        self.n_mut = n_mut
        self.flipped_mutation_direction = flipped_mutation_direction
        self.llr = np.empty((self.n_vtx, self.n_mut))
        self.loc_joint_1 = None
        self.loc_joint_2 = None
        self.joint = None

        # initialize with random structure
        self.rand_subtree()
        self.current_llr_max_without_anchor = None
        self.attachment_probs = None
        self.expected_mutations_per_edge = None

        # self.use_parent_vec([53,56,88,73,63,53,55,57,64,79,69,58,57,68,75,67,62,51,64,72,74,65,66,58,60,54,72,74,61,61,
        #                      77,56,81,73,52,54,50,69,52,51,76,62,59,65,63,86,55,97,50,78,60,70,67,77,59,71,70,80,88,82,
        #                      66,91,71,68,86,93,80,85,84,79,83,85,83,90,75,76,78,92,81,89,82,84,87,96,87,94,90,89,95,91,
        #                      95,92,93,94,96,97,98,98,-1])

    @property
    def n_mut(self):
        return self.mut_loc.size

    @n_mut.setter
    def n_mut(self, n_mut):
        self.mut_loc = np.ones(n_mut, dtype=int) * -1
        self.flipped = np.zeros(n_mut, dtype=bool)

    def rand_subtree(self, leaves=None, internals=None):
        """
        Construct a random subtree. If no subtree is specified, randomize the entire tree.
        """
        # determine the leaf and internal vertices
        if leaves is None or internals is None:
            leaves = [i for i in range(self.n_cells)]
            internals = [i for i in range(self.n_cells, self.n_vtx)]
        else:
            if len(leaves) != len(internals) + 1:
                raise ValueError("There must be exactly one more leaf than internals.")
        leaves = list(np.sort(leaves))
        internals = list(np.sort(internals))

        # repeatedly assign two children to an internal vertex
        for parent in internals:
            children = np.random.choice(leaves, size=2, replace=False)  # choose two random children
            for child in children:
                self.assign_parent(child, parent)
                leaves.remove(child)  # the children cannot be assigned to other parents
            leaves.append(parent)  # the parent can now serve as a child

        self.reroot(internals[-1])

    def rand_mut_loc(self, num_clones, leaf_mut_prob=0):
        """
        This function is used by the data generator to create simulated data with a specified number of clones.
        """
        if num_clones == "":
            self.mut_loc = np.random.randint(self.n_vtx, size=self.n_mut)
        else:
            if num_clones > self.n_mut:
                raise ValueError("number of clones must be less than or equal to number of mutations")
            weights = np.ones(self.n_vtx)
            weights[:self.n_cells] = leaf_mut_prob  # make it less likely that clones consist of just a single cell
            # weights[self.main_root] = 10 # higher probability of having mutations placed at the root than elsewhere
            weights /= weights.sum()
            sample = list(np.random.choice(self.n_vtx, size=num_clones, replace=False, p=weights))
            if self.main_root not in sample:
                num_clones = num_clones - 1  # vertexes without any mutations make up another clone
                sample.remove(np.random.choice(sample))

            all_muts = list(range(self.n_mut))
            np.random.shuffle(all_muts)

            # first assign 1 random mutation per clone
            sample_copy = sample.copy()
            for c in range(num_clones):
                clone = np.random.choice(sample_copy)
                self.mut_loc[all_muts[c]] = clone
                sample_copy.remove(clone)

            # assign other mutations randomly to the clones
            for mut in all_muts[num_clones:]:
                self.mut_loc[mut] = np.random.choice(sample)

    def fit_llh(self, llh_1, llh_2):
        """
        Gets ready to run optimization using provided log-likelihoods.
        The two genotypes involved in the mutation, gt1 and gt2, are not required for optimization.

        [Arguments]
            llh1: 2D array in which entry [i,j] is the log-likelihood of cell i having gt1 at locus j
            llh2: 2D array in which entry [i,j] is the log-likelihood of cell i having gt2 at locus j
        """
        assert (llh_1.shape == llh_2.shape)

        # adjust tree size if needed
        if llh_1.shape[0] != self.n_cells:
            self.__init__(*llh_1.shape, self.flipped_mutation_direction)
            warnings.warn(
                "Reinitialized cell tree since the number of cells does not match the row count of the llh matrix.")
        elif llh_1.shape[1] != self.n_mut:
            self.n_mut = llh_1.shape[1]

        # data to be used directly in optimization
        self.llr = np.empty((self.n_vtx, self.n_mut))
        self.llr[:self.n_cells, :] = llh_2 - llh_1

        # joint likelihood of each locus when all cells have genotype 1 or 2
        self.loc_joint_1 = llh_1.sum(axis=0)
        self.loc_joint_2 = llh_2.sum(axis=0)

        # assign mutations to optimal locations
        self.update_all()

    def fit_mutation_tree(self, mt):
        """
        Fits the cell tree to a given mutation tree.
        """
        assert (self.n_cells == mt.n_cells)

        # most recent common ancestor of cells below a mutation node
        mrca = np.ones(mt.n_vtx, dtype=int) * -1  # -1 means no cell has been found below this mutation

        next_internal = self.n_cells
        mutation = -1
        for mvtx in mt.rdfs_experimental(mt.main_root):  # mvtx for "mutation vertex"
            leaves = [mrca[child] for child in mt.children(mvtx) if mrca[child] != -1]
            leaves += np.where(mt.cell_loc == mvtx)[0].tolist()
            if len(leaves) == 0:  # no cell below, nothing to do
                continue
            elif len(leaves) == 1:  # one cell below, no internal node added
                mrca[mvtx] = leaves[0]
                self.mut_loc[mutation] = leaves[0]
            elif len(leaves) > 1:  # more than one cell below, add new internal node(s)
                internals = [i for i in range(next_internal, next_internal + len(leaves) - 1)]
                self.rand_subtree(leaves, internals)
                mrca[mvtx] = internals[-1]
                next_internal += len(internals)
            mutation = mvtx

    def update_llr(self):
        """
        Updates the log-likelihood ratios between the two genotypes.
        """
        for rt in self.roots:
            for vtx in self.rdfs_experimental(rt):
                if self.isleaf(vtx):  # nothing to be done for leaves
                    continue
                # LLR at internal vertex is the sum of LLR of both children
                self.llr[vtx, :] = self.llr[self.children(vtx), :].sum(axis=0)


    def update_tree_llh_mut_loc(self):
        """
        Updates the optimal mutation locations and the joint likelihood of the tree.
        """
        # if mutation directions are unknown, test both directions
        if self.flipped_mutation_direction:
            locs_neg = self.llr.argmin(axis=0)
            locs_pos = self.llr.argmax(axis=0)
            llhs_neg = self.loc_joint_2 - self.llr[locs_neg, np.arange(self.llr.shape[1])]
            llhs_pos = self.loc_joint_1 + self.llr[locs_pos, np.arange(self.llr.shape[1])]

            neg_larger = llhs_neg > llhs_pos
            self.mut_loc = np.where(neg_larger, locs_neg, locs_pos)
            self.flipped = neg_larger
            loc_joint = np.where(neg_larger, llhs_neg, llhs_pos)

        else:
            self.mut_loc = self.llr.argmax(axis=0)
            loc_joint = np.array([self.llr[self.mut_loc[j], j] for j in range(self.n_mut)]) + self.loc_joint_1
            # self.loc_joint_1 = Likelihood of all cells having genotype 1 -> no mutation
            # self.mut_loc choose the mutation location with the highest LLR of being mutated. For hidden nodes the
            # LLR is the sum of the children LLRs

        self.joint = loc_joint.sum()

    def update_tree_llh_marginalized(self, mut_loc=False):
        """
        Updates the mutation placement probabilities in the mutation tree
        by marginalizing over all possible locations, and computes the joint likelihood.
        """
        if self.flipped_mutation_direction:

            llhs_pos = self.loc_joint_1 + self.llr
            llhs_neg = self.loc_joint_2 - self.llr

            joint_pos = logsumexp(llhs_pos, axis=0)
            joint_neg = logsumexp(llhs_neg, axis=0)

            loc_joint = logsumexp(np.vstack([joint_pos, joint_neg]), axis=0)
            self.joint = np.sum(loc_joint)

            if mut_loc:
                neg_larger = joint_neg > joint_pos
                self.flipped = neg_larger

                self.mut_loc = np.where(
                    neg_larger,
                    np.argmax(llhs_neg, axis=0),
                    np.argmax(llhs_pos, axis=0)
                )

                # Compute full attachment probabilities
                p_pos = np.exp(llhs_pos - loc_joint)
                p_neg = np.exp(llhs_neg - loc_joint)
                self.attachment_probs = p_pos + p_neg
        else:
            loc_joint = self.loc_joint_1 + logsumexp(self.llr, axis=0)
            self.mut_loc = self.llr.argmax(axis=0)
            self.joint = np.sum(loc_joint)

    def update_tree_llh_marginalized_not(self, min_clones=4, min_mutations_per_edge=2, cost_per_edge=50,
                                                  mut_loc=False):
        if self.flipped_mutation_direction:
            # Full LLR-based likelihood matrices for both directions
            llhs_pos = self.loc_joint_1 + self.llr
            llhs_neg = self.loc_joint_2 - self.llr

            # Softmax-normalized probabilities for each mutation-location pair
            joint_pos = logsumexp(llhs_pos, axis=0)
            joint_neg = logsumexp(llhs_neg, axis=0)
            loc_joint = logsumexp(np.vstack([joint_pos, joint_neg]), axis=0)

            # Probabilities under each direction
            p_pos = np.exp(llhs_pos - loc_joint)
            p_neg = np.exp(llhs_neg - loc_joint)

            # Expected mutations per edge
            self.expected_mutations_per_edge = p_pos.sum(axis=1) + p_neg.sum(axis=1)

            # Identify top candidate attachment nodes (clones)
            above_threshold = np.where(self.expected_mutations_per_edge >= min_mutations_per_edge)[0]
            n_clones = max(min_clones, len(above_threshold))
            top_attachment_points = np.argsort(self.expected_mutations_per_edge)[-min_clones:]

            # Restrict the log-likelihoods to only top nodes
            llhs_pos_top = llhs_pos[top_attachment_points]
            llhs_neg_top = llhs_neg[top_attachment_points]

            # For each mutation, determine best node & direction among top edges
            best_pos_llh = np.max(llhs_pos_top, axis=0)
            best_neg_llh = np.max(llhs_neg_top, axis=0)
            use_neg = best_neg_llh > best_pos_llh
            self.flipped = use_neg

            best_nodes_pos = np.argmax(llhs_pos_top, axis=0)
            best_nodes_neg = np.argmax(llhs_neg_top, axis=0)
            selected_nodes = np.where(use_neg, best_nodes_neg, best_nodes_pos)

            self.mut_loc = top_attachment_points[selected_nodes]

            # Compute joint likelihood using selected values
            best_llhs = np.where(use_neg, best_neg_llh, best_pos_llh)
            self.joint = np.sum(best_llhs) - n_clones * cost_per_edge

            if mut_loc:
                print(np.unique(self.mut_loc))

        else:
            # Direction is fixed (positive only)
            llhs = self.loc_joint_1[:, None] + self.llr

            joint_pos = logsumexp(llhs, axis=0)
            p_pos = np.exp(llhs - joint_pos)
            self.expected_mutations_per_edge = p_pos.sum(axis=1)

            above_threshold = np.where(self.expected_mutations_per_edge >= min_mutations_per_edge)[0]
            n_clones = max(min_clones, len(above_threshold))
            top_attachment_points = np.argsort(self.expected_mutations_per_edge)[-n_clones:]

            llhs_top = llhs[top_attachment_points]
            selected_nodes = np.argmax(llhs_top, axis=0)
            self.mut_loc = top_attachment_points[selected_nodes]
            self.flipped = np.zeros(self.llr.shape[1], dtype=bool)

            best_llhs = np.max(llhs_top, axis=0)
            self.joint = np.sum(best_llhs) - n_clones * cost_per_edge

            if mut_loc:
                print(np.unique(self.mut_loc))

    def update_tree_llh_marginalized_clones(self, entropy_weight = 0.1, n_clones = 4, mut_loc=False):
        """
        Updates the mutation placement probabilities in the mutation tree
        by marginalizing over all possible locations, and computes the joint likelihood.
        """
        if self.flipped_mutation_direction:

            # Full marginal likelihood over both directions and all placements
            llhs_pos = self.loc_joint_1 + self.llr
            llhs_neg = self.loc_joint_2 - self.llr

            joint_pos = logsumexp(llhs_pos, axis=0)
            joint_neg = logsumexp(llhs_neg, axis=0)

            loc_joint = logsumexp(np.vstack([joint_pos, joint_neg]), axis=0)

            # normalize by the loc_joint likelihood
            p_pos = np.exp(llhs_pos - loc_joint)  # shape: (n_nodes, n_mutations)
            p_neg = np.exp(llhs_neg - loc_joint)  # shape: (n_nodes, n_mutations)

            # Sum across all mutations to get the expected number of attachments per node
            self.expected_mutations_per_edge = p_pos.sum(axis=1) + p_neg.sum(axis=1)
            #
            # # Penalize spread-out mutation assignments (prefer concentration)
            # p = self.expected_mutations_per_edge / (self.n_mut + 1e-12)
            #
            # entropy_penalty = -np.sum(p * np.log(p + 1e-12)) * self.n_cells * self.n_mut
            #
            # self.joint = np.sum(loc_joint) - entropy_weight * entropy_penalty

            # above_threshold = np.where(self.expected_mutations_per_edge >= min_mutations_per_edge)[0]
            # n_clones = max(min_clones, len(above_threshold))
            top_attachment_points = np.argsort(self.expected_mutations_per_edge)[-n_clones:]

            # Step 5: Restrict the log-likelihoods to only those top nodes
            llhs_pos_top = llhs_pos[top_attachment_points]
            llhs_neg_top = llhs_neg[top_attachment_points]

            # Step 6: Marginalize again over direction and top locations only
            joint_pos_top = logsumexp(llhs_pos_top, axis=0)
            joint_neg_top = logsumexp(llhs_neg_top, axis=0)

            loc_joint_top = logsumexp(np.vstack([joint_pos_top, joint_neg_top]), axis=0)
            self.joint = np.sum(loc_joint_top) # - n_clones * cost_per_edge

            if mut_loc:
                llhs_combined_top = logsumexp(np.stack([llhs_pos_top, llhs_neg_top]), axis=0)
                best_node_indices = np.argmax(llhs_combined_top, axis=0)
                self.mut_loc = top_attachment_points[best_node_indices]

                # llhs_combined_top = logsumexp(np.stack([llhs_pos, llhs_neg]), axis=0)
                # self.mut_loc = np.argmax(llhs_combined_top, axis=0)
                print(np.unique(self.mut_loc))

        else:
            # When direction is known, just marginalize over placements
            loc_joint = self.loc_joint_1 + logsumexp(self.llr, axis=0)

            self.mut_loc = self.llr.argmax(axis=0)
            self.joint = np.sum(loc_joint)

    def update_all(self, mut_loc=False):
        """
        Updates the log-likelihood ratios, the optimal mutation locations, and the joint likelihood of the tree.
        """
        self.update_llr()
        self.update_tree_llh_mut_loc()
        # self.update_tree_llh_marginalized(mut_loc=mut_loc)

    def binary_prune(self, subroot):
        """
        Prune a subtree while keeping the main tree strictly binary
        This is achieved by removing the subtree together with its "anchor"
            which is the direct parent of the pruned subtree
        Meanwhile, the original sibling of the pruned subtree is re-assigned to its grandparent
        As a result, the pruned subtree has the "anchor" as its root and is not binary
        """
        self.splice(next(self.siblings(subroot)))

    def greedy_insert_experimental(self):
        """
        Inserts a subtree at its optimal location. Generally a bit faster than greedy_insert
        """

        def search_insertion_loc(target):
            stack = [(target, False, None, None)]
            best_joint = -np.inf
            best_targets = []

            self.current_llr_max_without_anchor = np.delete(self.llr, anchor, axis=0).max(axis=0)

            while stack:
                # visited and original_llr are in order to restore the original llr after all children of this node
                # were visited
                current_target, visited, original_llr, original_max_without_anchor = stack.pop()
                if not visited:
                    self.llr[anchor, :] = self.llr[subroot, :] + self.llr[current_target, :]

                    # self.update_tree_llh_marginalized()
                    self.update_tree_llh_mut_loc()

                    current_joint = self.joint

                    if current_joint == best_joint:
                        best_targets.append(current_target)
                    if current_joint > best_joint:
                        best_joint = current_joint
                        best_targets = [current_target]

                    if not self.isleaf(current_target):
                        current_llr = self.llr[current_target, :].copy()
                        self.llr[current_target, :] += self.llr[subroot, :]

                        stack.append((current_target, True, current_llr,
                                      self.current_llr_max_without_anchor.copy()))  # Mark this node as visited
                        for child in self.children(
                                current_target):  # reverse children list for same results as with recursion
                            stack.append((child, False, current_llr,
                                          None))  # None in case current_llr is not used for joint calculation

                if visited:
                    # Returning to the node after visiting its children, restore the original LLR
                    self.llr[current_target, :] = original_llr
                    self.current_llr_max_without_anchor = original_max_without_anchor

            # print(best_targets, best_joint)
            return np.random.choice(
                best_targets), best_joint

        for anchor in self.pruned_roots():
            subroot = self.children(anchor)[0]
            best_target, best_joint_llh = search_insertion_loc(self.main_root)
            # print(best_target, best_joint_llh)
            self.insert(anchor, best_target)
            # self.update_all()

    def greedy_insert(self):
        """
        Inserts a subtree at its optimal location.
        """

        # Traverse the tree from the root to the leaves.
        def search_insertion_loc(target):
            # calculate LLR at anchor when anchor is inserted above target
            # parent node LLR is the sum of it's children
            self.llr[anchor, :] = self.llr[subroot, :] + self.llr[target, :]
            # highest achievable joint log-likelihood with this insertion
            self.update_mut_loc()

            best_target_node = target
            best_joint = self.joint

            if not self.isleaf(target):
                # for any descendant of target, the LLR at target is the original one plus that of subroot
                # Meaning the subroot is attached underneath the target, so it's likelihood is added to the
                # target likelihood.
                original = self.llr[target, :].copy()
                self.llr[target, :] += self.llr[subroot, :]
                # recursively search all descendants
                for child in self.children(target):
                    child_best_target, child_best_joint = search_insertion_loc(child)
                    if child_best_joint > best_joint:
                        best_target_node = child_best_target
                        best_joint = child_best_joint
                # restore the original LLR at target after searching all descendants
                self.llr[target, :] = original

            return best_target_node, best_joint

        for anchor in self.pruned_roots():
            subroot = self.children(anchor)[0]
            best_target, best_joint_llh = search_insertion_loc(self.main_root)
            # print(best_target, best_joint_llh)
            self.insert(anchor, best_target)
            self.update_all()

    def exhaustive_optimize(self, leaf_only=False):
        """
        Loops through every node and finds the optimal attachment point of the respective subtree.
        """
        self.update_all()

        sr_candidates = list(range(self.n_cells)) if leaf_only else list(range(self.n_vtx))
        np.random.shuffle(sr_candidates)

        for sr in sr_candidates:
            # print(sr)
            if sr == self.main_root:
                continue

            self.binary_prune(sr)
            self.update_llr()
            self.greedy_insert_experimental()

        self.update_all(mut_loc=True)

    def to_graphviz(self, filename=None, engine="dot", leaf_shape="circle", internal_shape="circle", gene_names=None):
        """
        Returns a graphviz Digraph object corresponding to the tree
        """
        dgraph = graphviz.Digraph(filename=filename, engine=engine)

        mutations = [[] for _ in range(self.n_vtx)]
        if gene_names is None:
            for mut, loc in enumerate(self.mut_loc):
                mutations[loc].append(mut)
        else:
            for mut, loc in enumerate(self.mut_loc):
                mutations[loc].append(gene_names[mut])

        for vtx in range(self.n_vtx):
            # node label is the integer that represents it
            node_label = str(vtx)
            # find mutations that should be placed above current node
            edge_label = ','.join([str(j) for j in mutations[vtx]])

            # treat leaf (observed) and internal nodes differently
            if self.isleaf(vtx):
                dgraph.node(node_label, shape=leaf_shape)
            else:
                dgraph.node(node_label, label=node_label, shape=internal_shape, style="filled", color="lightgray")

            if self.isroot(vtx):
                # create a void node with an edge to the root
                dgraph.node(f"void_{vtx}", label="", shape="point")
                dgraph.edge(f"void_{vtx}", node_label, label=edge_label, fontsize="70")
            else:
                dgraph.edge(str(self.parent(vtx)), node_label, label=edge_label, fontsize="70")

        return dgraph

    def postprocess_trees(self):
