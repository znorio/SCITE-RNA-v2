"""
To optimize the trees, SCITE-RNA alternates between mutation and cell lineage tree spaces.
"""

import warnings
import numpy as np

from src_python.cell_tree import CellTree
from src_python.mutation_tree import MutationTree
from src_python.utils import load_config_and_set_random_seed

config = load_config_and_set_random_seed()


class SwapOptimizer:
    def __init__(self, sig_digits=9, spaces=None, flipped_mutation_direction=True):
        """
        [Arguments]
            spaces: spaces that will be searched and the order of the search
                    'c' = cell tree space, 'm' = mutation tree space
                    default is ['c','m'], i.e. start with cell tree and search both spaces
            sig_dig: number of significant digits to use when calculating joint probability
        """
        self.mt = None
        self.ct = None
        self.n_decimals = None
        if spaces is None:
            spaces = ["c", "m"]
        self.sig_digits = sig_digits
        self.spaces = spaces
        self.flipped_mutation_direction = flipped_mutation_direction

    @property
    def current_joint(self):
        return round(self.ct.joint, self.n_decimals)

    @property
    def mt_joint(self):
        return round(self.mt.joint, self.n_decimals)

    def postprocess_trees(self):
        """
        Post-process the trees after optimization.
        This includes removing single mutations and updating the joint likelihood.
        """
        self.ct.remove_single_mutations()
        self.mt.remove_single_mutations()
        self.ct.update_all()
        self.mt.update_all()

        # Update the joint likelihoods
        self.ct.update_joint()
        self.mt.update_joint()

    def fit_llh(self, llh_1, llh_2):
        self.ct = CellTree(llh_1.shape[0], llh_1.shape[1], flipped_mutation_direction=self.flipped_mutation_direction)
        self.ct.fit_llh(llh_1, llh_2)

        self.mt = MutationTree(llh_1.shape[1], llh_1.shape[0])
        self.mt.fit_llh(llh_1, llh_2)

        mean_abs = np.sum(np.abs(llh_1 + llh_2)) / 2  # mean abs value when attaching mutations randomly
        self.n_decimals = int(self.sig_digits - np.log10(mean_abs))

    def optimize(self, max_loops=100, reshuffle_nodes=True):
        current_space = 0
        converged = [space not in self.spaces for space in ['c', 'm']]  # choose the spaces to be optimized
        if self.spaces[0] == 'c':  # choose the starting search space
            current_space = 0  # 0 for cell lineage tree, 1 for mutation tree
        elif self.spaces[0] == 'm':
            current_space = 1
        else:
            print("Space has to start with either 'c' or 'm'")

        loop_count = 0
        start_joint = -np.inf

        while not all(converged):
            if loop_count >= max_loops:
                warnings.warn('Maximal loop number exceeded.')
                break
            loop_count += 1

            if current_space == 0:
                print('Optimizing cell lineage tree ...')
                self.ct.exhaustive_optimize()  # loop_count=loop_count)
                self.mt.fit_cell_tree(self.ct)
                self.mt.update_all()

            else:  # i.e. current_space == 1:
                print('Optimizing mutation tree ...')
                self.mt.exhaustive_optimize(prune_single_mutations=reshuffle_nodes)  # loop_count=loop_count)
                self.ct.fit_mutation_tree(self.mt)
                self.ct.update_all()

            print(self.ct.joint, self.mt.joint)

            current_joint = self.current_joint
            if self.spaces == ["m"]:
                current_joint = self.mt_joint
            if start_joint < current_joint:
                converged[current_space] = False
            elif start_joint == current_joint:
                converged[current_space] = True
            else:
                raise RuntimeError('Observed decrease in joint likelihood.')

            start_joint = self.current_joint
            if self.spaces == ["m"]:
                start_joint = self.mt_joint

            if "c" in self.spaces and "m" in self.spaces:  # and converged[current_space] == True
                current_space = 1 - current_space  # switch to the other tree space

        self.ct.postprocess_trees(max_iters=1)

    # def optimize(self, max_loops=100, max_no_improve=3, reshuffle_nodes=True):
    #     """
    #     Joint optimization of cell and mutation trees under a marginalized likelihood model.
    #     Tracks best-scoring parent vectors and restores them at the end.
    #
    #     Parameters:
    #         max_loops (int): Maximum number of iterations.
    #         max_no_improve (int): Stop if no improvement in either tree after these many iterations.
    #         reshuffle_nodes (bool): Whether to allow mutation reshuffling during optimization.
    #     """
    #     current_space = 0 if self.spaces[0] == 'c' else 1
    #     no_improvement = [0, 0]  # [cell tree, mutation tree]
    #     best_joint = [float('-inf'), float('-inf')]  # best likelihood per tree
    #     best_parents = [None, None]  # to store best parent vectors for ct and mt
    #
    #     loop_count = 0
    #     while loop_count < max_loops and (no_improvement[0] < max_no_improve or no_improvement[1] < max_no_improve):
    #         loop_count += 1
    #
    #         if current_space == 0:
    #             print(f"[Loop {loop_count}] Optimizing Cell Lineage Tree ...")
    #             self.ct.exhaustive_optimize()
    #             self.mt.fit_cell_tree(self.ct)
    #             self.mt.update_all()
    #             current_joint = self.mt.joint
    #
    #         else:
    #             print(f"[Loop {loop_count}] Optimizing Mutation Tree ...")
    #             self.mt.exhaustive_optimize(prune_single_mutations=reshuffle_nodes)
    #             self.ct.fit_mutation_tree(self.mt)
    #             self.ct.update_all()
    #             current_joint = self.ct.joint
    #
    #         print(self.ct.joint, self.mt.joint)
    #
    #         if current_joint > best_joint[current_space] + 1e-6:
    #             best_joint[current_space] = current_joint
    #             no_improvement[current_space] = 0
    #             best_parents[current_space] = (
    #                 self.ct.parent_vec if current_space == 0 else self.mt.parent_vec
    #             )
    #         else:
    #             no_improvement[current_space] += 1
    #
    #         # Alternate tree space if both are being optimized
    #         if "c" in self.spaces and "m" in self.spaces:
    #             current_space = 1 - current_space
    #
    #     print("Optimization finished.")
    #     print(f"  Best joint (ct): {best_joint[0]:.6f}")
    #     print(f"  Best joint (mt): {best_joint[1]:.6f}")
    #
    #     # Restore best structures
    #     if best_parents[0] is not None:
    #         self.ct.use_parent_vec(best_parents[0])
    #     if best_parents[1] is not None:
    #         self.mt.use_parent_vec(best_parents[1])
    #
    #     # Final update after restoring structure
    #     self.ct.update_all()
    #     self.mt.update_all()


