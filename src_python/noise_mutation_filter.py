"""
This code calculates the posterior probability of different mutation types and genotypes given the data
and can filter SNVs based on this posterior.
"""

import numpy as np
from numba import njit
import math
import yaml
from scipy.special import loggamma, logsumexp

with open('../config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

seed = config["random_seed"]
np.random.seed(seed)

@njit
def ln_gamma(z):
    return math.lgamma(z)

@njit
def betaln(x, y):
    return ln_gamma(x) + ln_gamma(y) - ln_gamma(x + y)

@njit
def factorial(n):
    if n == 0 or n == 1:
        return 0.0  # log(1) = 0, log(0) is undefined, return 0.0 as safe placeholder
    result = 0.0
    for i in range(2, n + 1):
        result += np.log(i)
    return result

def logbinom(n, k):
    return loggamma(n+1) - loggamma(k+1) - loggamma(n-k+1)

def lognormalize(array):
    return array - logsumexp(array)

@njit
def log_binomial_coefficient(n, k):
    if 0 <= k <= n:
        log_numerator = factorial(n)
        log_denominator = factorial(k) + factorial(n - k)
        return log_numerator - log_denominator
    else:
        return 0.0  # Return 0.0 for invalid inputs

# custom betabinom_pmf function, as it is called a lot and faster than the scipy version.
@njit
def betabinom_pmf(k, n, a, b):
    if n < 0 or k < 0 or k > n or a <= 0 or b <= 0:
        return 0.0

    log_binom_coef = log_binomial_coefficient(n, k)

    num = betaln(k + a, n - k + b)
    denom = betaln(a, b)

    return np.exp(num - denom + log_binom_coef)


class MutationFilter:
    def __init__(self, error_rate=0.05, overdispersion=10, genotype_freq=None, mut_freq=0.5):
        if genotype_freq is None:
            genotype_freq = {'R': 1 / 4, 'H': 1 / 2, 'A': 1 / 4}
        self.set_betabinom(f, omega, h_factor)
        self.set_mut_type_prior(genotype_freq, mut_freq)

    def set_betabinom(self, f, omega, h_factor):
        '''
        [Arguments]
            f: frequency of correct read (i.e. 1 - error rate)
            omega: uncertainty of f, effective number of prior observations (when determining error rate)
        '''
        self.alpha_R = f * omega
        self.beta_R = omega - self.alpha_R
        self.alpha_A = (1 - f) * omega
        self.beta_A = omega - self.alpha_A
        self.alpha_H = omega/2 * h_factor
        self.beta_H = omega/2 * h_factor

    def set_mut_type_prior(self, genotype_freq, mut_freq):
        '''
        Calculates and stores the log-prior for each possible mutation type of a locus (including non-mutated)

        [Arguments]
            genotype_freq: priors of the root (wildtype) having genotype R, H or A
            mut_freq: a priori proportion of loci that are mutated
        '''
        self.mut_type_prior = {s: None for s in ['R', 'H', 'A', 'RH', 'HR', 'AH', 'HA']}

        # three non-mutated cases
        self.mut_type_prior['R'] = genotype_freq['R'] * (1 - mut_freq)
        self.mut_type_prior['H'] = genotype_freq['H'] * (1 - mut_freq)
        self.mut_type_prior['A'] = genotype_freq['A'] * (1 - mut_freq)
        # four mutated cases
        self.mut_type_prior['RH'] = genotype_freq['R'] * mut_freq 
        self.mut_type_prior['HA'] = genotype_freq['H'] * mut_freq / 2 # can either become alternative or reference
        self.mut_type_prior['HR'] = self.mut_type_prior['HA']
        self.mut_type_prior['AH'] = genotype_freq['A'] * mut_freq
        
        # convert to log scale
        for s in self.mut_type_prior: 
            self.mut_type_prior[s] = np.log(self.mut_type_prior[s])


    def single_read_llh(self, n_ref, n_alt, genotype):
        '''
        [Arguments]
            n_ref: number of ref reads
            n_alt: number of alt reads
            genotype: the genotype of interest

        [Returns]
            the log-likelihood of observing n_ref, n_alt, given genotype
        '''
        if genotype == 'R':
            result = betabinom_pmf(n_ref, n_ref + n_alt, self.alpha_R, self.beta_R)
        elif genotype == 'A':
            result = betabinom_pmf(n_ref, n_ref + n_alt, self.alpha_A, self.beta_A)
        elif genotype == 'H':
            result = betabinom_pmf(n_ref, n_ref + n_alt, self.alpha_H, self.beta_H)
        else:
            raise ValueError('[MutationFilter.single_read_llh] Invalid genotype.')
        
        return np.log(result)


    def k_mut_llh(self, ref, alt, gt1, gt2): 
        '''
        [Arguments]
            ref, alt: 1D array, read counts at a locus for all cells
            gt1, gt2: genotypes before and after the mutation
        
        [Returns]
            If gt1 is the same as gt2 (i.e. there is no mutation), returns a single joint log-likelihood
            Otherwise, returns a 1D array in which entry [k] is the log-likelihood of having k mutated cells
        '''
        
        N = ref.size # number of cells
        
        if gt1 == gt2:
            return np.sum([self.single_read_llh(ref[i], alt[i], gt1) for i in range(N)])
        
        k_in_first_n_llh = np.zeros((N+1, N+1)) # [n,k]: log-likelihood that k among the first n cells are mutated
        k_in_first_n_llh[0,0] = 0 # Trivial case: when there is 0 cell in total, the likelihood of having 0 mutated cell is 1
        
        for n in range(N):
            # log-likelihoods of the n-th cell having gt1 and gt2
            gt1_llh = self.single_read_llh(ref[n], alt[n], gt1)
            gt2_llh = self.single_read_llh(ref[n], alt[n], gt2)

            # k = 0 special case
            k_in_first_n_llh[n+1, 0] = k_in_first_n_llh[n, 0] + gt1_llh

            # k = 1 through n
            k_over_n = np.array([k/(n+1) for k in range(1,n+1)])
            log_summand_1 = np.log(1 - k_over_n) + gt1_llh + k_in_first_n_llh[n, 1:n+1]
            log_summand_2 = np.log(k_over_n) + gt2_llh + k_in_first_n_llh[n, 0:n]
            k_in_first_n_llh[n+1, 1:n+1] = np.logaddexp(log_summand_1, log_summand_2)

            # k = n+1 special case
            k_in_first_n_llh[n+1, n+1] = k_in_first_n_llh[n, n] + gt2_llh
        
        return k_in_first_n_llh[N, :]


    def single_locus_posteriors(self, ref, alt, comp_priors): 
        '''
        Calculates the log-posterior of different mutation types for a single locus

        [Arguments]
            ref, alt: 1D arrays containing ref and alt reads of each cell
            comp_priors: log-prior for each genotype composition
        
        [Returns]
            1D numpy array containing posteriors of each mutation type, in the order ['R', 'H', 'A', 'RH', 'HA', 'HR', 'AH']
        
        NB When a mutation affects a single cell or all cells, it is considered non-mutated and assigned to one
        of 'R', 'H' and 'A', depending on which one is the majority
        '''
        llh_RH = self.k_mut_llh(ref, alt, 'R', 'H')
        llh_HA = self.k_mut_llh(ref, alt, 'H', 'A')
        assert(llh_RH[-1] == llh_HA[0]) # both should be llh of all H

        joint_R = llh_RH[:1] + comp_priors['R'] # llh zero out of n cells with genotype R are mutated given the data + prior of genotype RR
        joint_H = llh_HA[:1] + comp_priors['H']
        joint_A = llh_HA[-1:] + comp_priors['A'] # log likelihood, that all the cells are mutated + prior genotype A
        joint_RH = llh_RH[1:] + comp_priors['RH'] # llh that 1 or more cells with genotype R are mutated given the data + prior of having 1 or more cells with genotype R having a mutation
        joint_HA = llh_HA[1:] + comp_priors['HA']
        joint_HR = np.flip(llh_RH)[1:] + comp_priors['HR'] # llh k cells genotype H -> R  + prior that a mutation affects k cells for genotype H->R
        joint_AH = np.flip(llh_HA)[1:] + comp_priors['AH']

        joint = np.array([
            logsumexp(np.concatenate((joint_R, joint_RH[:1], joint_HR[-1:]))), # RR
            logsumexp(np.concatenate((joint_H, joint_RH[-1:], joint_HR[:1], joint_HA[:1], joint_AH[-1:]))), # HH
            logsumexp(np.concatenate((joint_A, joint_HA[-1:], joint_AH[:1]))), # AA
            logsumexp(joint_RH[1:-1]),
            logsumexp(joint_HA[1:-1]),
            logsumexp(joint_HR[1:-1]),
            logsumexp(joint_AH[1:-1])
        ])

        posteriors = lognormalize(joint) # Bayes' theorem
        return posteriors

    def mut_type_posteriors(self, ref, alt):
        '''
        Calculates the log-prior of different mutation types for all loci
        In case no mutation occurs, all cells have the same genotype (which is either R or H or A)
        In case there is a mutation, each number of mutated cells is considered separately

        [Arguments]
            ref, alt: matrices containing the ref and alt reads
        
        [Returns]
            2D numpy array with n_loci rows and 7 columns, with each column standing for a mutation type
        '''
        n_cells, n_loci = ref.shape
        
        # log-prior for each number of affected cells
        # placing a mutation randomly on one of the edges of a binary tree, how many affected cells would you expect?
        k_mut_priors = np.array([2 * logbinom(n_cells, k) - np.log(2*k-1) - logbinom(2*n_cells, 2*k) for k in range(1, n_cells+1)])

        # composition priors
        comp_priors = {}
        for mut_type in ['R', 'H', 'A']:
            comp_priors[mut_type] = self.mut_type_prior[mut_type]
        for mut_type in ['RH', 'HA', 'HR', 'AH']:
            comp_priors[mut_type] = self.mut_type_prior[mut_type] + k_mut_priors

        # calculate posteriors for all loci with the help of multiprocessing
        result = np.zeros((n_loci, 7))
        for j in range(n_loci):
            result[j] = self.single_locus_posteriors(ref[:,j], alt[:,j], comp_priors)
        return result


    def filter_mutations(self, ref, alt, method='highest_post', t=None, n_exp=None, only_ref_to_alt=False):
        '''
        Filters the loci according to the posteriors of each mutation state

        [Arguments]
            method: criterion that determines which loci are considered mutated
            t: the posterior threshold to be used when using the 'threshold' method
            n_exp: the number of loci to be selected when using the 'first_k' method
        '''
        assert(ref.shape == alt.shape)
        # ['R', 'H', 'A', 'RH', 'HA', 'HR', 'AH']
        posteriors = np.exp(self.mut_type_posteriors(ref, alt))

        if method == 'highest_post': # for each locus, choose the state with highest posterior
            selected = np.where(np.argmax(posteriors, axis=1) >= 3)[0]
        elif method == 'threshold': # choose loci at which mutated posterior > threshold 
            selected = np.where(np.sum(posteriors[:, 3:], axis=1) > t)[0]
        elif method == 'first_k': # choose the k loci with highest mutated posteriors
            mut_posteriors = np.sum(posteriors[:, 3:], axis=1)
            order = np.argsort(mut_posteriors)[::-1]
            selected = order[:n_exp]
        else:
            raise ValueError('[MutationFilter.filter_mutations] Unknown filtering method.')

        if only_ref_to_alt: # only consider mutations from reference to heterozygous or heterozygous to alternative not the reverse
            mut_posteriors = np.sum(posteriors[:, 3:5], axis=1)
            order = np.argsort(mut_posteriors)[::-1]
            selected = order[:n_exp]

            mut_type = np.argmax(posteriors[selected, 3:5], axis=1)
            gt1_inferred = np.choose(mut_type, choices=['R', 'H'])
            gt2_inferred = np.choose(mut_type, choices=['H', 'A'])

        else:
            mut_type = np.argmax(posteriors[selected, 3:], axis=1)
            gt1_inferred = np.choose(mut_type, choices=['R', 'H', 'H', 'A'])
            gt2_inferred = np.choose(mut_type, choices=['H', 'A', 'R', 'H'])

        gt_not_selected = []  # maximum likelihood of genotypes that are not included in the tree learning
        for i in range(posteriors.shape[0]):
            if i in selected:
                continue
            else:
                genotype = np.argmax(posteriors[i, :3])
                gt_not_selected.append(np.choose(genotype, choices=['R', 'H', 'A', 'RH', 'HA', 'HR', 'AH']))

        return selected, gt1_inferred, gt2_inferred, gt_not_selected

    def get_llh_mat(self, ref, alt, gt1, gt2):
        '''
        [Returns]
            llh_mat_1: 2D array in which [i,j] is the log-likelihood of cell i having gt1 at locus j
            llh_mat_2: 2D array in which [i,j] is the log-likelihood of cell i having gt2 at locus j
        '''
        n_cells, n_mut = ref.shape
        llh_mat_1 = np.empty((n_cells, n_mut))
        llh_mat_2 = np.empty((n_cells, n_mut))
        
        for i in range(n_cells):
            for j in range(n_mut):
                llh_mat_1[i,j] = self.single_read_llh(ref[i,j], alt[i,j], gt1[j])
                llh_mat_2[i,j] = self.single_read_llh(ref[i,j], alt[i,j], gt2[j])
        
        return llh_mat_1, llh_mat_2