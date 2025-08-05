"""
Defines the mutation tree and how it is optimized.
"""
import time

import numpy as np
import graphviz
import warnings

from src_python.tree_base import PruneTree
from src_python.utils import load_config_and_set_random_seed
from scipy.special import logsumexp

config = load_config_and_set_random_seed()


class MutationTree(PruneTree):
    """
    This class provides methods for tree manipulation, visualization and optimization of mutation trees.
    """
    def __init__(self, n_mut=2, n_cells=0):
        if n_mut < 2:
            warnings.warn('Mutation tree too small, nothing to explore.', RuntimeWarning)

        super().__init__(n_mut + 1)
        self.n_mut = n_mut
        self.n_cells = n_cells

        self.flipped = np.zeros(self.n_vtx, dtype=bool)
        self.flipped_mutation_direction = True

        self.reroot(self.wt)
        self.random_mutation_tree()

        self.llr = np.empty((self.n_cells, self.n_vtx))
        self.cumul_llr = np.empty_like(self.llr)
        self.loc_joint_1 = None
        self.loc_joint_2 = None
        self.joint = None
        self.attachment_probs = None  # the expected attachment probability for each cell to the mutations (optional)
        self.expected_llr = None
        self.mut_llh = None

        # self.use_parent_vec([
        #     2, 2, 51, 2, 2, 2, 2, 5, 5, 7, 6, 10, 6, 11, 11, 8, 2, 7, 2, 13, 9, 20, 2, 18,
        #     21, 9, 17, 18, 6, 2, 29, 4, 11, 51, 27, 51, 14, 0, 32, 31, 30, 25, 30, 9, 36, 5,
        #     18, 10, 37, 36, 31, -1
        # ])

    @property
    def wt(self):
        return self.n_mut

    @property
    def n_cells(self):
        return len(self.cell_loc)

    @n_cells.setter
    def n_cells(self, n_cells):
        self.cell_loc = np.ones(n_cells, dtype=int) * -1

    def random_mutation_clone_tree(self, num_clones):
        """
        Generates a mutation tree with roughly equally sized clones.
        """
        random_numbers = np.unique([np.random.randint(1, num_clones) for _ in range(self.n_mut)], return_counts=True)[1]
        muts = list(range(self.n_mut))
        chosen_muts = np.random.choice(muts, random_numbers[0], replace=False)
        muts = [m for m in muts if m not in chosen_muts]
        self.assign_parent(chosen_muts[0], self.wt)
        for mut1, mut2 in zip(chosen_muts, chosen_muts[1:]):
            self.assign_parent(mut2, mut1)
        potential_branch_points = [chosen_muts[-1]]

        for i in range(1, num_clones):
            chosen_muts = np.random.choice(muts, random_numbers[i], replace=False)
            muts = [m for m in muts if m not in chosen_muts]

            branch_point = np.random.choice(potential_branch_points)
            self.assign_parent(chosen_muts[0], branch_point)
            for mut1, mut2 in zip(chosen_muts, chosen_muts[1:]):
                self.assign_parent(mut2, mut1)
            potential_branch_points.append(chosen_muts[-1])

        self.cell_loc = [np.random.choice(potential_branch_points) for _ in range(self.n_cells)]

    def random_mutation_tree(self):
        """
        Generates a random mutation tree
        """
        # Randomly choose one mutation to have self.wt as its parent
        root_assigned = np.random.randint(0, self.n_mut - 1)
        self.assign_parent(root_assigned, self.wt)

        # Assign each mutation a random parent ensuring it's a valid tree
        for vtx in range(self.n_mut):
            if vtx == root_assigned:
                continue  # Skip the already assigned root mutation

            # Assign a random parent from the set of already assigned mutations or self.wt
            potential_parents = [i for i in range(self.n_mut) if i != vtx and self._pvec[i] != -1]
            parent = np.random.choice(potential_parents + [self.wt])
            self.assign_parent(vtx, parent)


    def compute_expected_llr(self):
        """
        Computes the expected log-likelihood ratios marginalized over flip directions,
        independent of tree structure.
        """
        if self.flipped_mutation_direction:
            expected_llr = np.zeros_like(self.llr)

            for j in range(self.n_mut):
                llr_pos = logsumexp(self.llr[:, j])
                llr_neg = logsumexp(-self.llr[:, j])

                logp_pos = self.loc_joint_1[j] + llr_pos
                logp_neg = self.loc_joint_2[j] - llr_neg
                lse = logsumexp([logp_pos, logp_neg])

                p_pos = np.exp(logp_pos - lse)
                p_neg = np.exp(logp_neg - lse)

                expected_llr[:, j] = p_pos * self.llr[:, j] + p_neg * (-self.llr[:, j])
        else:
            expected_llr = np.where(self.flipped, -self.llr, self.llr)

        self.expected_llr = expected_llr

    def compute_mutation_log_likelihood(self):
        """
        Computes the marginalized mutation-level log-likelihood (mut_llh),
        independent of tree structure.
        """

        if self.flipped_mutation_direction:
            mut_llh = np.zeros(self.n_mut)
            for j in range(self.n_mut):
                llh_pos = self.loc_joint_1[j] + logsumexp(self.llr[:, j])
                llh_neg = self.loc_joint_2[j] - logsumexp(-self.llr[:, j])
                mut_llh[j] = logsumexp([llh_pos, llh_neg])
        else:
            mut_llh = np.where(self.flipped[:self.n_mut],
                               self.loc_joint_2 - logsumexp(-self.llr, axis=0),
                               self.loc_joint_1 + logsumexp(self.llr, axis=0))

        self.mut_llh = mut_llh

    def fit_llh(self, llh_1, llh_2):
        """
        Gets ready to run optimization using provided log-likelihoods.
        The two genotypes involved in the mutation, gt1 and gt2, are not required for optimization.

        [Arguments]
            llh1: 2D array in which entry [i,j] is the log-likelihood of cell i having gt1 at locus j
            llh2: 2D array in which entry [i,j] is the log-likelihood of cell i having gt2 at locus j
        """
        assert (llh_1.shape == llh_2.shape)

        if llh_1.shape[1] != self.n_mut:
            self.__init__(llh_1.shape[1], llh_1.shape[0])
        elif llh_1.shape[0] != self.n_cells:
            self.n_cells = llh_1.shape[0]

        self.llr[:, :self.n_mut] = llh_2 - llh_1
        self.llr[:, self.n_mut] = 0  # stands for wildtype

        # joint likelihood of each locus when all cells have genotype 1 or 2
        self.loc_joint_1 = llh_1.sum(axis=0)
        self.loc_joint_2 = llh_2.sum(axis=0)

        self.compute_expected_llr()
        self.compute_mutation_log_likelihood()
        self.update_all()

    def fit_cell_tree(self, ct):
        """
        Fits the mutation tree to the given cell tree.
        """
        assert (self.n_mut == ct.n_mut)
        assert (len(ct.roots) == 1)

        mrm = np.empty(ct.n_vtx + 1, dtype=int)  # mrm for "most recent mutation"
        mrm[-1] = self.wt  # put wildtype at sentinel

        for cvtx in ct.dfs_experimental(ct.main_root):  # cvtx for "cell vertex"
            mut_list = np.where(ct.mut_loc == cvtx)[0]  # mutations attached to the edge above cvtx
            parent_mut = mrm[ct.parent(cvtx)]
            if mut_list.size > 0:
                np.random.shuffle(mut_list)  # randomize the order of mutations at the same edge
                np.sort(mut_list)
                self.assign_parent(mut_list[0], parent_mut)  # assigns the first mutation to the parent_mut
                for mut1, mut2 in zip(mut_list, mut_list[1:]):
                    self.assign_parent(mut2, mut1)

                mrm[cvtx] = mut_list[-1]
            else:
                mrm[cvtx] = mrm[ct.parent(cvtx)]

        self.flipped[:-1] = ct.flipped

    def update_cumul_llr(self):
        """
        Updates the cumulative log-likelihood ratio between the two genotypes
        """
        for rt in self.roots:
            for vtx in self.dfs_experimental(rt):
                llr_summand = -self.llr[:, vtx] if self.flipped[vtx] else self.llr[:, vtx]
                if self.isroot(vtx):
                    self.cumul_llr[:, vtx] = llr_summand
                else:
                    self.cumul_llr[:, vtx] = self.cumul_llr[:, self.parent(vtx)] + llr_summand

    def update_tree_llh_cell_loc(self):
        """
        Updates the optimal cell location in the mutation tree and the joint likelihood of the tree.
        """
        self.cell_loc = self.cumul_llr.argmax(axis=1)
        wt_llh = np.where(self.flipped[:-1], self.loc_joint_2,
                          self.loc_joint_1).sum()  # as filpped isn't updated this could be made a constant
        self.joint = self.cumul_llr.max(axis=1).sum() + wt_llh


    def update_cumul_llr_marginalized(self):
        """
        Updates the cumulative log-likelihood ratio by marginalizing over
        flip directions for each mutation (when enabled).
        """

        for rt in self.roots:
            for vtx in self.dfs_experimental(rt):
                llr_summand = self.expected_llr[:, vtx]
                if self.isroot(vtx):
                    self.cumul_llr[:, vtx] = llr_summand
                else:
                    self.cumul_llr[:, vtx] = self.cumul_llr[:, self.parent(vtx)] + llr_summand


    def update_tree_llh_marginalized(self, store_probs=False):
        """
        Updates the probabilistic cell attachments in the mutation tree by
        marginalizing over all possible attachment points, and optionally over
        mutation directions (per mutation). Computes the joint log-likelihood
        of the tree in a consistent probabilistic framework.
        """

        # Marginalize over all cell attachment points
        marginalized_ll = logsumexp(self.cumul_llr, axis=1)

        if store_probs:
            self.attachment_probs = np.exp(self.cumul_llr - marginalized_ll[:, None])

        # Total joint log-likelihood: sum over cells + mutations
        self.joint = marginalized_ll.sum() + self.mut_llh.sum()
        self.cell_loc = self.cumul_llr.argmax(axis=1)

    def update_all(self, store_probs=False):
        """
        Updates the cumulative log-likelihood ratio and the optimal cell location in the mutation tree.
        """
        self.update_cumul_llr()
        self.update_tree_llh_cell_loc()
        # self.update_cumul_llr_marginalized()
        # self.update_tree_llh_marginalized(store_probs=store_probs)  # True if you want to store the attachment probabilities

    def report_llr_changes(self, old_cumul_llr, new_cumul_llr, atol=0):

        diff = np.abs(new_cumul_llr - old_cumul_llr)
        changed = diff > atol
        delta = new_cumul_llr - old_cumul_llr
        round_digits = 10

        changes_detail = {}
        for j in range(diff.shape[1]):
            row_indices = np.where(changed[:, j])[0]
            if row_indices.size > 0:
                deltas = delta[row_indices, j]
                rounded_deltas = np.round(deltas, round_digits)
                unique_deltas = sorted(set(rounded_deltas))
                changes_detail[j] = {
                    'n_changed': len(row_indices),
                    'deltas': unique_deltas
                }

        if changes_detail:
            for col, info in sorted(changes_detail.items()):
                print(f"  Mutation {col}: {info['n_changed']} cells affected")
        else:
            print("No significant changes in cumul_llr.")

        return changes_detail

    def greedy_attach_marginalized(self):
        """
        Attaches pruned subtrees to the optimal location in the main tree
        using marginalized log-likelihood computations.
        """
        for subroot in self.pruned_roots():
            best_joint_llh = -np.inf
            best_loc = None
            best_locs = []

            main_tree_nodes = [r for r in self.dfs_experimental(self.main_root)]
            subtree_nodes = [r for r in self.dfs_experimental(subroot)]

            marginalized_ll_main_tree = logsumexp(self.cumul_llr[:, main_tree_nodes], axis=1)
            marginalized_ll_subtree = logsumexp(self.cumul_llr[:, subtree_nodes], axis=1)

            for vtx in main_tree_nodes:
                new_ll_subtree = self.cumul_llr[:, vtx] + marginalized_ll_subtree

                marginalized_ll_full = logsumexp(
                    [marginalized_ll_main_tree, new_ll_subtree], axis=0
                )

                self.joint = np.sum(marginalized_ll_full) + self.mut_llh.sum()

                if self.joint == best_joint_llh:
                    best_locs.append(vtx)
                if self.joint > best_joint_llh:
                    best_joint_llh = self.joint
                    best_loc = vtx
                    best_locs = [vtx]

            self.assign_parent(subroot, best_loc)
            self.joint = best_joint_llh

    def greedy_attach(self):
        """
        Attaches the pruned subtrees to the optimal location in the main tree.
        """
        for subroot in self.pruned_roots():
            main_tree_max = self.cumul_llr[:, list(self.dfs_experimental(self.main_root))].max(axis=1)
            subtree_max = self.cumul_llr[:, list(self.dfs_experimental(subroot))].max(axis=1)

            best_llr = -np.inf
            best_loc = None
            best_locs = []
            for vtx in self.dfs_experimental(self.main_root):
                # calculate the llr of the tree with reattached subtree at the vtx
                total_llr = np.maximum(main_tree_max, subtree_max + self.cumul_llr[:, vtx]).sum()
                if total_llr == best_llr:
                    best_locs.append(vtx)
                if total_llr > best_llr:
                    best_llr = total_llr
                    best_loc = vtx
                    best_locs = [vtx]

            self.assign_parent(subroot, best_loc)
            self.update_all()

    def greedy_attach_node(self):
        """
        Inserts the pruned mutation node in the optimal location of the main tree.
        """
        for subroot in self.pruned_roots():
            best_llr_append = -np.inf
            best_llr_insert = -np.inf
            best_loc = None
            best_targets = []

            # Append the pruned mutation to some other mutation
            wt_llh = np.where(self.flipped[:-1], self.loc_joint_2, self.loc_joint_1).sum()
            summand_llr_subroot = -self.llr[:, subroot] if self.flipped[subroot] else self.llr[:, subroot]
            cumul_llr_rest_max = np.delete(self.cumul_llr, subroot, axis=1).max(axis=1)

            for vtx in self.dfs_experimental(self.main_root):
                cumul_llr_subroot = summand_llr_subroot + self.cumul_llr[:, vtx]
                total_llr = np.maximum(cumul_llr_rest_max, cumul_llr_subroot).sum() + wt_llh

                if total_llr == best_llr_append:
                    best_targets.append(vtx)
                if total_llr > best_llr_append:
                    best_llr_append = total_llr
                    best_loc = vtx
                    best_targets = [vtx]

            best_inserts = []
            all_columns = set(range(self.cumul_llr.shape[1]))
            for vtx in self.dfs_experimental(self.main_root):
                children = self.children(vtx).copy()
                for child in children:

                    reduced_dfs = [v for v in self.dfs_experimental(child)]
                    max_subtree = self.cumul_llr[:, reduced_dfs].max(axis=1) + summand_llr_subroot
                    cumul_llr_subroot = self.cumul_llr[:, vtx] + summand_llr_subroot
                    remaining_columns = list(all_columns - set(reduced_dfs + [subroot]))
                    max_maintree = np.max(self.cumul_llr[:, remaining_columns], axis=1)
                    total_llr = np.maximum.reduce([max_subtree, cumul_llr_subroot, max_maintree]).sum() + wt_llh

                    if total_llr == best_llr_insert:
                        best_inserts.append(child)
                    if total_llr > best_llr_insert:
                        best_llr_insert = total_llr
                        best_inserts = [child]

            if best_llr_append > best_llr_insert:
                self.assign_parent(subroot, best_loc)
            else:
                self.insert(subroot, np.random.choice(best_inserts))  # best_child

            self.update_all()

    def exhaustive_optimize(self, prune_single_mutations=True):
        """
        Optimizes the mutation tree exhaustively by iterating though all nodes and pruning and reattaching the
        respective subtrees and individual mutation nodes.
        """
        mut_random_order = list(range(self.n_mut))
        np.random.shuffle(mut_random_order)
        if prune_single_mutations:  # prune single mutations and attach/insert them at their optimal location
            for subroot in mut_random_order:

                if len(self._clist[subroot]) > 1:  # reconstructing the original subtree would be more complex if more
                    # than 1 child is appended to a parent
                    continue

                self.prune_node(subroot)
                self.update_all()
                self.greedy_attach_node()

            print("after Node reattachment ", self.joint)

        np.random.shuffle(mut_random_order)

        for subroot in mut_random_order:  # prune subtrees and insert them at their optimal location
            # self.greedy_attach()

            self.prune(subroot)
            self.update_all()
            self.greedy_attach()
            # self.greedy_attach_marginalized()

        self.update_all()


    def to_graphviz(self, filename=None, engine='dot'):
        """
        Generates a graphviz representation of the mutation tree.
        """
        dgraph = graphviz.Digraph(filename=filename, engine=engine)

        dgraph.node(str(self.wt), label='wt', shape='rectangle', color='gray')
        for vtx in range(self.n_mut):
            dgraph.node(str(vtx), shape='rectangle', style='filled', color='gray')
            if self.isroot(vtx):
                # for root, create a corresponding void node
                dgraph.node(f'void_{vtx}', label='', shape='point')
                dgraph.edge(f'void_{vtx}', str(vtx))
            else:
                dgraph.node(str(vtx), shape='rectangle', style='filled', color='gray')
                dgraph.edge(str(self.parent(vtx)), str(vtx))

        for i in range(self.n_cells):
            name = 'c' + str(i)
            dgraph.node(name, shape='circle')
            dgraph.edge(str(self.cell_loc[i]), name)

        return dgraph
