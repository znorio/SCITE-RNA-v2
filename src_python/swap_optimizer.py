"""
To optimize the trees, SCITE-RNA alternates between mutation and cell lineage tree spaces.
"""

import warnings
import numpy as np
import yaml

from .cell_tree import CellTree
from .mutation_tree import MutationTree

with open('../config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

seed = config["random_seed"]
np.random.seed(seed)
class SwapOptimizer:
    def __init__(self, sig_digits=10, spaces=["c", "m"], reverse_mutations=True):
        '''
        [Arguments]
            spaces: spaces that will be searched and the order of the search
                    'c' = cell tree space, 'm' = mutation tree space
                    default is ['c','m'], i.e. start with cell tree and search both spaces
            sig_dig: number of significant digits to use when calculating joint probability
        '''
        self.sig_digits = sig_digits
        self.spaces = spaces
        self.reverse_mutations = reverse_mutations

    @property
    def current_joint(self):
        return round(self.ct.joint, self.n_decimals)
    @property
    def mt_joint(self):
        return round(self.mt.joint, self.n_decimals)


    def fit_llh(self, llh_1, llh_2):
        self.ct = CellTree(llh_1.shape[0], llh_1.shape[1], reversible_mut=self.reverse_mutations)
        self.ct.fit_llh(llh_1, llh_2)
        
        self.mt = MutationTree(llh_1.shape[1], llh_1.shape[0])
        self.mt.fit_llh(llh_1, llh_2)

        # determine a rounding precision for joint likelihood calculation
        mean_abs = np.sum(np.abs(llh_1 + llh_2)) / 2 # mean abs value when attaching mutations randomly
        self.n_decimals = int(self.sig_digits - np.log10(mean_abs))
        # Need to round because the sum of floating point numbers can slightly vary depending on the exact process

    def optimize(self, max_loops=100):
        converged = [space not in self.spaces for space in ['c', 'm']] # choose the spaces to be optimized
        if self.spaces[0] == 'c': # choose the starting search space
            current_space = 0 # 0 for cell lineage tree, 1 for mutation tree
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
                self.ct.exhaustive_optimize(loop_count=loop_count)
                self.mt.fit_cell_tree(self.ct)
                self.mt.update_all()

            else: # i.e. current_space == 1:
                print('Optimizing mutation tree ...')
                self.mt.exhaustive_optimize(loop_count=loop_count)
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

            if "c" in self.spaces and "m" in self.spaces: #and converged[current_space] == True # switch to the other tree space
                current_space = 1 - current_space
