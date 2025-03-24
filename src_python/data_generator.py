"""
This script is used to generate simulated read counts, ground truth genotypes and ground truth trees.
"""

# from numba import njit # problem with random seed
import numpy as np
import yaml
from scipy.stats import poisson, geom, nbinom, gamma, beta
import matplotlib.pyplot as plt

from .cell_tree import CellTree
from .mutation_tree import MutationTree

with open('../config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

seed = config["random_seed"]
np.random.seed(seed)


# @njit
def betabinom_rvs(coverage, alpha, beta_param):
    p = np.random.beta(alpha, beta_param)
    return np.random.binomial(coverage, p)


class DataGenerator:
    mutation_types = ['RH', 'AH', 'HR', 'HA']

    def __init__(self, n_cells, n_mut,
                 mut_prop=1., error_rate=0.05, overdispersion=10, genotype_freq=None,
                 alpha_h=2, beta_h=2, dropout_prob=0.2, dropout_direction_prob=0.5,
                 coverage_method='zinb', coverage_mean=60, coverage_sample=None, dropout_alpha=2,
                 dropout_beta=8, dropout_dir_alpha=4, dropout_dir_beta=4, overdispersion_h=6):

        self.ct = CellTree(n_cells, n_mut)
        self.mt = MutationTree(n_mut, n_cells)
        self.random_mut_type_params(mut_prop, genotype_freq)
        self.random_coverage_params(coverage_method, coverage_mean, coverage_sample)
        self.betabinom_params(error_rate, overdispersion, alpha_h, beta_h)
        self.genotype = np.empty((self.n_cells, self.n_mut), dtype=str)
        # self.dropout_prob = dropout_prob
        # self.dropout_direction_prob = dropout_direction_prob
        self.dropout_alpha = dropout_alpha
        self.dropout_beta = dropout_beta
        self.dropout_dir_alpha = dropout_dir_alpha
        self.dropout_dir_beta = dropout_dir_beta
        self.overdispersion_h = overdispersion_h

    @property
    def n_cells(self):
        return self.ct.n_cells

    @property
    def n_mut(self):
        return self.ct.n_mut

    def betabinom_params(self, f, omega, alpha_h, beta_h):
        '''
        [Arguments]
            f: error rate
            omega: uncertainty of f
        '''
        self.alpha_R = f * omega
        self.beta_R = omega - self.alpha_R
        self.alpha_A = (1 - f) * omega
        self.beta_A = omega - self.alpha_A
        self.alpha_H = alpha_h
        self.beta_H = beta_h

    def random_mut_type_params(self, mut_prop, genotype_freq):
        self.mut_prop = mut_prop
        self.genotype_freq = [1 / 3, 1 / 3, 1 / 3] if genotype_freq is None else genotype_freq

    def random_coverage_params(self, method, mean=60, sample=None):
        self.coverage_method = method
        self.coverage_mean = mean
        if method == 'sample' and sample is None:
            raise ValueError('Please provide array of coverage values to be sampled from.')
        self.coverage_sample = sample

    def random_mut_type(self):
        self.gt1 = np.random.choice(['R', 'H', 'A'], size=self.n_mut, replace=True, p=self.genotype_freq)
        self.gt2 = np.empty_like(self.gt1)
        mutated = np.random.choice(self.n_mut, size=round(self.n_mut * self.mut_prop), replace=False)
        for j in mutated:
            if self.gt1[j] == 'H':
                self.gt2[j] = np.random.choice(['R', 'A'])  # mutation HA and HR with equal probability
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
                self.coverage = geom.rvs(p=1 / (self.coverage_mean + 1), loc=-1, size=(self.n_cells, self.n_mut))
            case 'poisson':
                self.coverage = poisson.rvs(mu=self.coverage_mean, size=(self.n_cells, self.n_mut))
            # learned from mm34 scRNA seq dataset
            case "zinb":
                mu, theta, pi = 60, 0.17, 0.38
                nb_samples = nbinom.rvs(theta, theta / (theta + mu), size=(self.n_cells, self.n_mut))
                zero_inflation_mask = np.random.rand(self.n_cells, self.n_mut) < pi
                self.coverage = np.where(zero_inflation_mask, 0, nb_samples)
            case 'sample':
                self.coverage = np.random.choice(self.coverage_sample, size=(self.n_cells, self.n_mut), replace=True)
            case _:
                raise ValueError('Invalid coverage sampling method.')

    def generate_single_read(self, genotype, coverage, dropout_prob, dropout_direction, alpha_H, beta_H):
        if genotype == 'R':
            n_alt = betabinom_rvs(coverage, self.alpha_R, self.beta_R)
        elif genotype == 'A':
            n_alt = betabinom_rvs(coverage, self.alpha_A, self.beta_A)
        elif genotype == 'H':
            # Determine if dropout occurs
            dropout_occurs = np.random.rand() < dropout_prob

            if dropout_occurs:
                # Determine dropout direction based on sampled probabilities
                dropout_to_A = np.random.rand() < dropout_direction
                if dropout_to_A:
                    n_alt = betabinom_rvs(coverage, self.alpha_A, self.beta_A)  # Dropout to A
                else:
                    n_alt = betabinom_rvs(coverage, self.alpha_R, self.beta_R)  # Dropout to R
            else:
                n_alt = betabinom_rvs(coverage, alpha_H, beta_H)  # No dropout
        else:
            raise ValueError('[generate_single_read] ERROR: invalid genotype.')

        n_ref = coverage - n_alt
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
        for i in range(self.n_cells):  # loop through each cell (leaf)
            for j in range(self.n_mut):
                self.genotype[i, j] = self.gt2[j] if mut_indicator[i, j] else self.gt1[j]

        # actual reads
        ref = np.empty((self.n_cells, self.n_mut), dtype=int)
        alt = np.empty((self.n_cells, self.n_mut), dtype=int)

        all_dropout_probs = []
        all_dropout_directions = []
        all_alphas = []
        all_betas = []


        for j in range(self.n_mut):
            # Sample dropout probabilities from beta distributions for each SNV
            dropout_prob = np.random.beta(self.dropout_alpha, self.dropout_beta)  # samples the dropout rate
            dropout_direction = np.random.beta(self.dropout_dir_alpha,
                                               self.dropout_dir_beta)  # samples how imbalanced the dropout is between the alleles

            all_dropout_probs.append(dropout_prob)
            all_dropout_directions.append(dropout_direction)

            # sample shape of the beta distribution for the heterozygous case from gamma distributions
            min_value = 2.5  # Shift
            shape = 2  # Shape parameter (k), adjust if needed

            scale = (self.overdispersion_h - min_value) / shape  # θ = (mean - shift) / k

            overdispersion_H = gamma.rvs(shape, loc=min_value, scale=scale)

            alpha_H = 0.5 * overdispersion_H
            beta_H = overdispersion_H - alpha_H

            # scale_alpha = (self.alpha_H - shift) / (shape - 1) this fixes the max prob to self.alpha_H
            # scale_alpha = (self.alpha_H - shift) / shape # mean prob = self.alpha_H
            # alpha_H = gamma.rvs(shape, loc=shift, scale=scale_alpha)
            # # scale_beta = (self.beta_H - shift) / (shape - 1)
            # scale_beta = (self.beta_H - shift) / shape  # mean prob = self.beta_H
            # beta_H = gamma.rvs(shape, loc=shift, scale=scale_beta)
            all_alphas.append(alpha_H)
            all_betas.append(beta_H)

            for i in range(self.n_cells):
                ref[i, j], alt[i, j] = self.generate_single_read(self.genotype[i, j], self.coverage[i, j],
                                                            dropout_prob, dropout_direction, alpha_H, beta_H)

        # for ttt in range(5):
        #     print(np.unique(self.genotype[:,ttt], return_counts=True))
        #     plt.hist(alt[:,ttt]/(alt[:,ttt] + ref[:,ttt]), bins=100)
        #     plt.show()
        return ref, alt, all_dropout_probs, all_dropout_directions, all_alphas, all_betas

    def mut_indicator(self):
        ''' Return a 2D Boolean array in which [i,j] indicates whether cell i is affected by mutation j '''
        res = np.zeros((self.n_cells, self.n_mut), dtype=bool)
        for j in range(self.n_mut):  # determine for each mutation the cells below it in the tree
            for vtx in self.ct.leaves(self.ct.mut_loc[j]):
                res[vtx, j] = True

        return res
