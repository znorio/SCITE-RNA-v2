"""
This script is used to generate simulated read counts, ground truth genotypes and ground truth trees.
"""


from numba import njit
import numpy as np
from scipy.stats import poisson, geom

from .cell_tree import CellTree
from .mutation_tree import MutationTree


@njit
def betabinom_rvs(coverage, alpha, beta_param):
    p = np.random.beta(alpha, beta_param)
    return np.random.binomial(coverage, p)


class DataGenerator: 
    mutation_types = ['RH', 'AH', 'HR', 'HA']
    
    
    def __init__(self, n_cells, n_mut,
                 mut_prop=1., genotype_freq=None,
                 coverage_method='geometric', coverage_mean=8, coverage_sample=None,
                 f=0.95, omega=100, omega_h=50):
        self.ct = CellTree(n_cells, n_mut)
        self.mt = MutationTree(n_mut, n_cells)
        self.random_mut_type_params(mut_prop, genotype_freq)
        self.random_coverage_params(coverage_method, coverage_mean, coverage_sample)
        self.betabinom_params(f, omega, omega_h)
        self.genotype = np.empty((self.n_cells, self.n_mut), dtype=str)


    @property
    def n_cells(self):
        return self.ct.n_cells
    

    @property
    def n_mut(self):
        return self.ct.n_mut
    

    def betabinom_params(self, f, omega, omega_h):
        self.alpha = f * omega
        self.beta = omega - self.alpha
        self.omega_h = omega_h
    

    def random_mut_type_params(self, mut_prop, genotype_freq):
        self.mut_prop = mut_prop
        self.genotype_freq = [1/3, 1/3, 1/3] if genotype_freq is None else genotype_freq

    
    def random_coverage_params(self, method, mean=8, sample=None):
        self.coverage_method = method
        self.coverage_mean = mean
        if method == 'sample' and sample is None:
            raise ValueError('Please provide array of coverage values to be sampled from.')
        self.coverage_sample = sample

    
    def random_mut_type(self):
        self.gt1 = np.random.choice(['R', 'H', 'A'], size = self.n_mut, replace = True, p = self.genotype_freq)
        self.gt2 = np.empty_like(self.gt1)
        mutated = np.random.choice(self.n_mut, size = round(self.n_mut * self.mut_prop), replace = False)
        for j in mutated:
            if self.gt1[j] == 'H':
                self.gt2[j] = np.random.choice(['R', 'A']) # mutation HA and HR with equal probability
            else:
                self.gt2[j] = 'H'
    
    
    def random_tree(self, num_clones, stratified=False):
        if stratified and num_clones != "":
            self.mt.random_mutation_clone_tree(num_clones)
            self.ct.fit_mutation_tree(self.mt)
        else:
            self.ct.rand_subtree()
            self.ct.rand_mut_loc(num_clones)

    def random_coverage(self):
        match self.coverage_method:
            case 'constant':
                self.coverage = np.ones((self.n_cells, self.n_mut), dtype=int) * self.coverage_mean
            case 'geometric':
                self.coverage = geom.rvs(p=1/(self.coverage_mean+1), loc=-1, size=(self.n_cells, self.n_mut))
            case 'poisson':
                self.coverage = poisson.rvs(mu=self.coverage_mean, size=(self.n_cells, self.n_mut))
            case 'sample':
                self.coverage = np.random.choice(self.coverage_sample, size=(self.n_cells, self.n_mut), replace=True)
            case _:
                raise ValueError('Invalid coverage sampling method.')
    
    
    def generate_single_read(self, genotype, coverage):
        if genotype == 'R':
            n_ref = betabinom_rvs(coverage, self.alpha, self.beta)
            n_alt = coverage - n_ref
        elif genotype == 'A':
            n_alt = betabinom_rvs(coverage, self.alpha, self.beta)
            n_ref = coverage - n_alt
        elif genotype == 'H':
            n_ref = betabinom_rvs(coverage, self.omega_h / 2, self.omega_h / 2)
            n_alt = coverage - n_ref
        else:
            raise ValueError('[generate_single_read] ERROR: invalid genotype.')
        
        return n_ref, n_alt
    
    
    def generate_reads(self, new_tree=False, new_mut_type=False, new_coverage=True, num_clones=""):
        if new_mut_type:
            self.random_mut_type()
        if new_tree:
            self.random_tree(num_clones)
        if new_coverage:
            self.random_coverage()

        # determine genotypes
        self.genotype = np.empty((self.n_cells, self.n_mut), dtype=str)
        mut_indicator = self.mut_indicator()
        for i in range(self.n_cells): # loop through each cell (leaf)
            for j in range(self.n_mut):
                self.genotype[i,j] = self.gt2[j] if mut_indicator[i,j] else self.gt1[j]
        
        # actual reads
        ref = np.empty((self.n_cells, self.n_mut), dtype=int)
        alt = np.empty((self.n_cells, self.n_mut), dtype=int)
        for i in range(self.n_cells):
            for j in range(self.n_mut):
                ref[i,j], alt[i,j] = self.generate_single_read(self.genotype[i,j], self.coverage[i,j])
        
        return ref, alt
    

    def mut_indicator(self):
        ''' Return a 2D Boolean array in which [i,j] indicates whether cell i is affected by mutation j '''
        res = np.zeros((self.n_cells, self.n_mut), dtype=bool)
        for j in range(self.n_mut): # determine for each mutation the cells below it in the tree
            for vtx in self.ct.leaves(self.ct.mut_loc[j]):
                res[vtx, j] = True

        return res
