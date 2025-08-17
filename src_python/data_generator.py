"""
This script is used to generate simulated read counts, ground truth genotypes and ground truth trees.
"""

import numpy as np
from scipy.stats import poisson, geom, nbinom, gamma

from src_python.cell_tree import CellTree
from src_python.mutation_tree import MutationTree
from src_python.utils import load_config_and_set_random_seed

config = load_config_and_set_random_seed()


def betabinom_rvs(coverage, alpha, beta_param):
    """
    Generate random samples from a Beta-Binomial distribution.
    """
    p = np.random.beta(alpha, beta_param)
    return np.random.binomial(coverage, p)


class DataGenerator:
    """
    DataGenerator is a class used to generate simulated read counts, ground truth genotypes, and ground truth trees
    for single-cell sequencing data. Mutation types are ["RH", "AH", "HR", "HA"]

    Attributes:
        n_cells – Number of cells.
        n_mut – Number of mutations.
        mut_prop – Proportion of mutations.
        error_rate – Expected variant allele frequency for homozygous reference genotype
        overdispersion – Overdispersion parameter in homozygous reference and alternative cases
        genotype_freq – Frequencies of genotypes.
        coverage_method – Method to generate coverage values.
        coverage_mean – Mean coverage value.
        coverage_sample – Array of coverage values to sample from.
        dropout_alpha – Alpha parameter of the beta distribution the dropout probability is sampled from.
        dropout_beta – Beta parameter of the beta distribution the dropout probability is sampled from.
        dropout_dir – Dropout direction probability.
        overdispersion_h – Overdispersion parameter for the read counts in the heterozygous case.
    """

    def __init__(self, n_cells, n_mut,
                 mut_prop=1., error_rate=0.05, overdispersion=10, genotype_freq=None,
                 coverage_method="zinb", coverage_mean=60, coverage_sample=None, dropout_alpha=2,
                 dropout_beta=8, dropout_dir=0.5, overdispersion_h=6):

        self.coverage = None
        self.ct = CellTree(n_cells=n_cells, n_mut=n_mut)
        self.mt = MutationTree(n_mut=n_mut, n_cells=n_cells)

        self.genotype = np.empty((self.n_cells, self.n_mut), dtype=str)
        self.mut_prop = mut_prop
        self.genotype_freq = [1 / 3, 1 / 3, 1 / 3] if genotype_freq is None else genotype_freq
        self.gt1 = np.random.choice(["R", "H", "A"], size=self.n_mut, replace=True, p=self.genotype_freq)
        self.gt2 = np.empty_like(self.gt1)

        self.coverage_method = coverage_method
        self.coverage_mean = coverage_mean
        if coverage_method == "sample" and coverage_sample is None:
            raise ValueError("Please provide array of coverage values to be sampled from.")
        self.coverage_sample = coverage_sample

        self.dropout_alpha = dropout_alpha
        self.dropout_beta = dropout_beta
        self.dropout_dir = dropout_dir
        self.overdispersion_h = overdispersion_h

        # Set the beta-binomial parameters
        self.alpha_R = error_rate * overdispersion
        self.beta_R = overdispersion - self.alpha_R
        self.alpha_A = (1 - error_rate) * overdispersion
        self.beta_A = overdispersion - self.alpha_A

    @property
    def n_cells(self):
        return self.ct.n_cells

    @property
    def n_mut(self):
        return self.ct.n_mut

    def random_mut_type(self):
        self.gt1 = np.random.choice(["R", "H", "A"], size=self.n_mut, replace=True, p=self.genotype_freq)
        mutated = np.random.choice(self.n_mut, size=round(self.n_mut * self.mut_prop), replace=False)
        for j in mutated:
            if self.gt1[j] == "H":
                self.gt2[j] = np.random.choice(["R", "A"])  # mutation HA and HR with equal probability
            else:
                self.gt2[j] = "H"

    def random_tree(self, num_clones):
        """
        Generate a random tree

        [Arguments]
            num_clones: Number of clones (cells with the same genotype) in the tree.
        """
        self.ct.rand_subtree()
        self.ct.rand_mut_loc(num_clones)

    def random_coverage(self):
        """
        Generate random coverage values for each cell and mutation based on the specified coverage method.

        The coverage values are generated using one of the following methods:
        - "constant": All coverage values are set to the mean coverage.
        - "geometric": Coverage values are sampled from a geometric distribution.
        - "poisson": Coverage values are sampled from a Poisson distribution.
        - "zinb": Coverage values are sampled from a zero-inflated negative binomial distribution.
        - "sample": Coverage values are sampled from a provided array of coverage values.
        """
        match self.coverage_method:
            case "constant":
                self.coverage = np.ones((self.n_cells, self.n_mut), dtype=int) * self.coverage_mean
            case "geometric":
                self.coverage = geom.rvs(p=1 / (self.coverage_mean + 1), loc=-1, size=(self.n_cells, self.n_mut))
            case "poisson":
                self.coverage = poisson.rvs(mu=self.coverage_mean, size=(self.n_cells, self.n_mut))
            # parameters 60, 0.17, 0.38 learned from mm34 scRNA seq dataset
            case "zinb":
                mu, theta, pi = self.coverage_mean, 0.17, 0.38
                nb_samples = nbinom.rvs(theta, theta / (theta + mu), size=(self.n_cells, self.n_mut))
                zero_inflation_mask = np.random.rand(self.n_cells, self.n_mut) < pi
                self.coverage = np.where(zero_inflation_mask, 0, nb_samples)
            case "sample":
                self.coverage = np.random.choice(self.coverage_sample, size=(self.n_cells, self.n_mut), replace=True)
            case _:
                raise ValueError("Invalid coverage sampling method.")

    def generate_single_read(self, genotype, coverage, dropout_prob, dropout_direction, alpha_h, beta_h):
        """
        Generate read counts for a single cell and mutation.

        [Arguments]
            genotype: the genotype of the cell ("R", "H", or "A")
            coverage: the total read coverage for the cell and mutation
            dropout_prob: the probability of dropout for heterozygous genotypes
            dropout_direction: the probability of dropout direction for heterozygous genotypes
            alpha_h: the alpha parameter for the beta-binomial distribution in the heterozygous case
            beta_h: the beta parameter for the beta-binomial distribution in the heterozygous case

        [Returns]
            n_ref: the number of reference reads
            n_alt: the number of alternative reads
        """
        if genotype == "R":
            n_alt = betabinom_rvs(coverage, self.alpha_R, self.beta_R)
        elif genotype == "A":
            n_alt = betabinom_rvs(coverage, self.alpha_A, self.beta_A)
        elif genotype == "H":
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
                n_alt = betabinom_rvs(coverage, alpha_h, beta_h)  # No dropout
        else:
            raise ValueError("[generate_single_read] ERROR: invalid genotype.")

        n_ref = coverage - n_alt
        return n_ref, n_alt

    def generate_reads(self, new_tree=False, new_mut_type=False, new_coverage=True, num_clones="", min_value=2.5,
                       shape=2):
        """
        Generate read counts for all cells and mutations.
        [Arguments]
            new_tree: if True, generate a new random tree
            new_mut_type: if True, generate new random mutation types
            new_coverage: if True, generate new random coverage values
            num_clones: number of clones in the tree, otherwise random mutation placement is used
            min_value: minimum value for the overdispersion parameter in the heterozygous case
            shape: shape parameter for the gamma distribution, which is used to sample the overdispersion parameter

        [Returns]
            ref: reference read counts
            alt: alternative read counts
            all_dropout_probs: dropout probabilities per SNV
            all_overdispersions_h: overdispersion parameters for heterozygous case per SNV
        """
        if new_mut_type:
            self.random_mut_type()
        if new_tree:
            self.random_tree(num_clones)
        if new_coverage:
            self.random_coverage()

        # determine genotypes
        self.genotype = np.empty((self.n_cells, self.n_mut), dtype=str)
        mut_indicator = self.mut_indicator()
        for i in range(self.n_cells):
            for j in range(self.n_mut):
                self.genotype[i, j] = self.gt2[j] if mut_indicator[i, j] else self.gt1[j]

        ref = np.empty((self.n_cells, self.n_mut), dtype=int)
        alt = np.empty((self.n_cells, self.n_mut), dtype=int)

        all_dropout_probs = []
        all_overdispersions_h = []

        for j in range(self.n_mut):
            # Sample dropout probabilities from beta distributions for each SNV
            dropout_prob = np.random.beta(self.dropout_alpha, self.dropout_beta)
            all_dropout_probs.append(dropout_prob)

            # Sample overdispersion parameter for heterozygous case from gamma distribution for each SNV
            scale = (self.overdispersion_h - min_value) / shape  # θ = (mean - shift) / k

            overdispersion_H = gamma.rvs(shape, loc=min_value, scale=scale)
            all_overdispersions_h.append(overdispersion_H)

            # as the cells are assumed to be independent, allelic imbalances
            # are assumed to be symmetric and only affect the overdispersion
            alpha_H = beta_H = 0.5 * overdispersion_H

            for i in range(self.n_cells):
                ref[i, j], alt[i, j] = self.generate_single_read(self.genotype[i, j], self.coverage[i, j],
                                                                 dropout_prob, config.dropout_direction, alpha_H, beta_H)

        return ref, alt, all_dropout_probs, all_overdispersions_h

    def mut_indicator(self):
        """
        Return a 2D Boolean array in which [i,j] indicates whether cell i is affected by mutation j
        """
        res = np.zeros((self.n_cells, self.n_mut), dtype=bool)
        for j in range(self.n_mut):  # determine for each mutation the cells below it in the tree
            for i in self.ct.leaves(self.ct.mut_loc[j]):
                res[i, j] = True
        return res
