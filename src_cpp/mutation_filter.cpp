/*
This code calculates the posterior probability of different mutation types and genotypes given the data
and can filter SNVs based on this posterior. Additionally, it is used to optimize model parameters based on
the inferred genotypes and observed read counts.
*/

#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <Eigen/Core>
#include <limits>
#include <LBFGSB.h>
#include <iostream>
#include <random>

#include "utils.h"
#include "mutation_filter.h"

using Eigen::VectorXd;
using Eigen::Map;
using namespace LBFGSpp;

// Constructor for MutationFilter class
MutationFilter::MutationFilter(double error_rate, double overdispersion,
                               const std::map<std::string, double>& genotype_freq, double mut_freq,
                               double dropout_alpha, double dropout_beta,
                               double dropout_direction_prob, double overdispersion_h)
        : error_rate(error_rate), overdispersion_homozygous(overdispersion), genotype_freq_(genotype_freq),
          dropout_direction_prob(dropout_direction_prob), overdispersion_heterozygous(overdispersion_h) {

    mut_type_prior = {{"R", NAN}, {"H", NAN}, {"A", NAN}, {"RH", NAN}, {"HR", NAN}, {"AH", NAN}, {"HA", NAN}};
    set_mut_type_prior(genotype_freq, mut_freq);
    alpha_R = error_rate * overdispersion;
    beta_R = overdispersion - alpha_R;
    alpha_A = (1 - error_rate) * overdispersion;
    beta_A = overdispersion - alpha_A;
    alpha_H = 0.5 * overdispersion_h;
    beta_H = overdispersion_h - alpha_H;
    dropout_prob = dropout_alpha / (dropout_alpha + dropout_beta);
}

// Update alpha and beta parameters based on error rate and overdispersion
void MutationFilter::update_alpha_beta(double error_r, double overdispersion_hom) {
    alpha_R = error_r * overdispersion_hom;
    beta_R = overdispersion_hom - alpha_R;

    alpha_A = (1.0 - error_r) * overdispersion_hom;
    beta_A = overdispersion_hom - alpha_A;
}

// set mutation type prior probabilities
void MutationFilter::set_mut_type_prior(const std::map<std::string, double>& genotype_freq, double mut_freq) {
    mut_type_prior["R"] = std::log(genotype_freq.at("R") * (1 - mut_freq));
    mut_type_prior["H"] = std::log(genotype_freq.at("H") * (1 - mut_freq));
    mut_type_prior["A"] = std::log(genotype_freq.at("A") * (1 - mut_freq));
    mut_type_prior["RH"] = std::log(genotype_freq.at("R") * mut_freq);
    mut_type_prior["HA"] = std::log(genotype_freq.at("H") * mut_freq / 2);
    mut_type_prior["HR"] = mut_type_prior["HA"];
    mut_type_prior["AH"] = std::log(genotype_freq.at("A") * mut_freq);
}


// likelihood of the data at a cell locus position
double MutationFilter::single_read_llh_with_dropout(int n_alt, int n_total, char genotype) const {
    if (genotype == 'R') {
        return log_betabinom_pmf(n_alt, n_total, alpha_R, beta_R);
    } else if (genotype == 'A') {
        return log_betabinom_pmf(n_alt, n_total, alpha_A, beta_A);
    } else if (genotype == 'H') {
        double result_no_dropout = std::log(1 - dropout_prob) +
                                   log_betabinom_pmf(n_alt, n_total, alpha_H, beta_H);
        double dropout_R = std::log(dropout_prob) + std::log(1 - dropout_direction_prob) +
                           log_betabinom_pmf(n_alt, n_total, alpha_R, beta_R);
        double dropout_A = std::log(dropout_prob) + std::log(dropout_direction_prob) +
                           log_betabinom_pmf(n_alt, n_total, alpha_A, beta_A);
        return logsumexp({result_no_dropout, dropout_R, dropout_A});
    } else {
        throw std::invalid_argument("Invalid genotype.");
    }
}

// likelihood of the data at a cell locus position with SNV specific dropout
double MutationFilter::single_read_llh_with_individual_dropout(int n_alt, int n_total, char genotype,
                                                               double dropout_probability, double alpha_h, double beta_h) const {
    if (genotype == 'R') {
        return log_betabinom_pmf(n_alt, n_total, alpha_R, beta_R);
    } else if (genotype == 'A') {
        return log_betabinom_pmf(n_alt, n_total, alpha_A, beta_A);
    } else if (genotype == 'H') {
        double result_no_dropout = std::log(1 - dropout_probability) +
                                   log_betabinom_pmf(n_alt, n_total, alpha_h, beta_h);
        double dropout_R = std::log(dropout_probability) + std::log(1 - dropout_direction_prob) +
                           log_betabinom_pmf(n_alt, n_total, alpha_R, beta_R);
        double dropout_A = std::log(dropout_probability) + std::log(dropout_direction_prob) +
                           log_betabinom_pmf(n_alt, n_total, alpha_A, beta_A);
        return logsumexp({result_no_dropout, dropout_R, dropout_A});
    } else {
        throw std::invalid_argument("Invalid genotype.");
    }
}

// llh that k out of the first n cells are mutated
std::vector<double> MutationFilter::k_mut_llh(const std::vector<int>& ref, const std::vector<int>& alt,
                                              char gt1, char gt2) const {
    size_t N = ref.size();
    std::vector<int> total(N);
    for (size_t i = 0; i < N; ++i) {
        total[i] = ref[i] + alt[i];
    }

    std::vector<std::vector<double>> k_in_first_n_llh(N + 1, std::vector<double>(N + 1, 0.0));
    k_in_first_n_llh[0][0] = 0;

    for (size_t n = 0; n < N; ++n) {
        double gt1_llh = single_read_llh_with_dropout(alt[n], total[n], gt1);
        double gt2_llh = single_read_llh_with_dropout(alt[n], total[n], gt2);

        k_in_first_n_llh[n + 1][0] = k_in_first_n_llh[n][0] + gt1_llh;

        std::vector<double> k_over_n(n);
        for (size_t k = 1; k <= n; ++k) {
            k_over_n[k - 1] = static_cast<double>(k) / static_cast<double>(n + 1);
        }

        std::vector<double> log_summand_1(n);
        std::vector<double> log_summand_2(n);
        for (size_t k = 1; k <= n; ++k) {
            log_summand_1[k - 1] = std::log(1 - k_over_n[k - 1]) + gt1_llh + k_in_first_n_llh[n][k];
            log_summand_2[k - 1] = std::log(k_over_n[k - 1]) + gt2_llh + k_in_first_n_llh[n][k - 1];
        }

        for (size_t k = 1; k <= n; ++k) {
            k_in_first_n_llh[n + 1][k] = logsumexp({log_summand_1[k - 1], log_summand_2[k - 1]});
        }

        k_in_first_n_llh[n + 1][n + 1] = k_in_first_n_llh[n][n] + gt2_llh;
    }

    return k_in_first_n_llh[N];
}


// likelihood of a single locus given the data and priors
std::vector<double> MutationFilter::single_locus_posteriors(const std::vector<int>& ref, const std::vector<int>& alt,
                                                            const std::unordered_map<std::string, std::vector<double>>& comp_priors) const {
    if (ref.size() != alt.size() || ref.size() <= 10) {
        throw std::invalid_argument("ref and alt must have the same size.");
    }
    std::vector<double> llh_RH = k_mut_llh(ref, alt, 'R', 'H');
    std::vector<double> llh_HA = k_mut_llh(ref, alt, 'H', 'A');
    if (llh_RH.back() != llh_HA.front()) {
        throw std::runtime_error("Mismatch in likelihoods");
    }

    std::vector<double> joint_R = add_scalar_to_vector(llh_RH[0], comp_priors.at("R")); // # llh zero out of n cells with genotype R are mutated given the data + prior of genotype RR
    std::vector<double> joint_H = add_scalar_to_vector(llh_HA[0], comp_priors.at("H"));
    std::vector<double> joint_A = add_scalar_to_vector(llh_HA.back(), comp_priors.at("A"));
    std::vector<double> joint_RH =  addVectors({llh_RH.begin() + 1, llh_RH.end()}, comp_priors.at("RH"));
    std::vector<double> joint_HA = addVectors({llh_HA.begin() + 1, llh_HA.end()}, comp_priors.at("HA"));

    std::reverse(llh_RH.begin(), llh_RH.end());
    std::reverse(llh_HA.begin(), llh_HA.end());

    std::vector<double> joint_HR = addVectors({llh_RH.begin() + 1, llh_RH.end()}, comp_priors.at("HR"));
    std::vector<double> joint_AH = addVectors({llh_HA.begin() + 1, llh_HA.end()}, comp_priors.at("AH"));

    std::vector<double> joint(7);
    joint[0] = logsumexp(concat(joint_R, std::vector<double>{joint_RH[0]}, std::vector<double>{joint_HR.back()})); // RR
    joint[1] = logsumexp(concat(joint_H, std::vector<double>{joint_RH.back()}, std::vector<double>{joint_HR[0]}, std::vector<double>{joint_HA[0]}, std::vector<double>{joint_AH.back()})); // HH
    joint[2] = logsumexp(concat(joint_A, std::vector<double>{joint_HA.back()}, std::vector<double>{joint_AH[0]})); // AA
    joint[3] = logsumexp({joint_RH.begin() + 1, joint_RH.end() - 1}); // RH (1:-1)
    joint[4] = logsumexp({joint_HA.begin() + 1, joint_HA.end() - 1}); // HA (1:-1)
    joint[5] = logsumexp({joint_HR.begin() + 1, joint_HR.end() - 1}); // HR (1:-1)
    joint[6] = logsumexp({joint_AH.begin() + 1, joint_AH.end() - 1}); // AH

    std::vector<double> posteriors = lognormalize_exp(joint);  // Bayes' theorem
    return posteriors;
}


// get posterior likelihoods of mutation types
std::vector<std::vector<double>> MutationFilter::mut_type_posteriors(const std::vector<std::vector<int>>& ref,
                                                                     const std::vector<std::vector<int>>& alt) {

    // log-prior for each number of affected cells
    std::vector<double> k_mut_priors(n_cells);
    for (int k = 1; k <= n_cells; ++k) {
        k_mut_priors[k - 1] = 2 * logbinom(n_cells, k) - std::log(2 * k - 1) - logbinom(2 * n_cells, 2 * k);
    }

    // composition priors
    std::unordered_map<std::string, std::vector<double>> comp_priors;
    for (const auto &mut_type: {"R", "H", "A"}) {
        comp_priors[mut_type] = {mut_type_prior[mut_type]};
    }
    for (const auto &mut_type: {"RH", "HA", "HR", "AH"}) {
        comp_priors[mut_type].resize(n_cells);
        for (int i = 0; i < n_cells; ++i) {
            comp_priors[mut_type][i] = mut_type_prior[mut_type] + k_mut_priors[i];
        }
    }
    // calculate posteriors for all loci
    std::vector<std::vector<double>> result(n_loci, std::vector<double>(7));

    for (int j = 0; j < n_loci; ++j) {
        result[j] = single_locus_posteriors(get_column(ref,j), get_column(alt, j), comp_priors);
    }

    return result;
}

// Filter mutations based on the posterior probabilities of mutation types
std::tuple<std::vector<int>, std::vector<char>, std::vector<char>, std::vector<char>> MutationFilter::filter_mutations(
        const std::vector<std::vector<int>>& ref,
        const std::vector<std::vector<int>>& alt,
        const std::string& method,
        double t,
        int n_exp){

    n_loci = static_cast<int>(ref[0].size());
    n_cells = static_cast<int>(ref.size());

    std::vector<std::vector<double>> posteriors = mut_type_posteriors(ref, alt);
    std::vector<int> selected;

    if (method == "highest_post") {
        for (int i = 0; i < static_cast<int>(posteriors.size()); ++i) {
            if (std::max_element(posteriors[i].begin(), posteriors[i].end()) - posteriors[i].begin() >= 3) {
                selected.push_back(i);
            }
        }
    } else if (method == "threshold") {
        for (int i = 0; i < static_cast<int>(posteriors.size()); ++i) {
            if (std::accumulate(posteriors[i].begin() + 3, posteriors[i].end(), 0.0) > t) {
                selected.push_back(i);
            }
        }
    } else if (method == "first_k") {
        std::vector<double> mut_posteriors(posteriors.size());
        for (int i = 0; i < static_cast<int>(posteriors.size()); ++i) {
            mut_posteriors[i] = std::accumulate(posteriors[i].begin() + 3, posteriors[i].end(), 0.0);
        }
        std::vector<int> order(mut_posteriors.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return mut_posteriors[a] > mut_posteriors[b];
        });
        selected = std::vector<int>(order.begin(), order.begin() + n_exp);
    } else {
        throw std::invalid_argument("Unknown filtering method.");
    }

    std::vector<char> gt1_inferred(selected.size());
    std::vector<char> gt2_inferred(selected.size());
    std::vector<char> gt_not_selected;

    for (int i = 0; i < static_cast<int>(selected.size()); ++i) {
        int index = selected[i];
        int mut_type = static_cast<int>(std::max_element(posteriors[index].begin() + 3, posteriors[index].end()) - posteriors[index].begin() - 3);
        gt1_inferred[i] = (mut_type == 0) ? 'R' : (mut_type == 1) ? 'H' : (mut_type == 2) ? 'H' : 'A';
        gt2_inferred[i] = (mut_type == 0) ? 'H' : (mut_type == 1) ? 'A' : (mut_type == 2) ? 'R' : 'H';
    }

    for (int i = 0; i < static_cast<int>(posteriors.size()); ++i) {
        if (std::find(selected.begin(), selected.end(), i) == selected.end()) {
            int genotype = static_cast<int>(std::max_element(posteriors[i].begin(), posteriors[i].begin() + 3) - posteriors[i].begin());
            gt_not_selected.push_back((genotype == 0) ? 'R' : (genotype == 1) ? 'H' : 'A');
        }
    }

    return {selected, gt1_inferred, gt2_inferred, gt_not_selected};
}

// Get the log-likelihood matrix for the given reference and alternative reads, genotypes, and dropout/overdispersion probabilities.
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> MutationFilter::get_llh_mat(
        const std::vector<std::vector<int>>& ref, const std::vector<std::vector<int>>& alt,
        const std::vector<char>& gt1, const std::vector<char>& gt2, bool individual,
        const std::vector<double>& dropout_probs, const std::vector<double>& overdispersions_h) {

    n_loci = static_cast<int>(ref[0].size());
    n_cells = static_cast<int>(ref.size());
    std::vector<std::vector<double>> llh_mat_1(n_cells, std::vector<double>(n_loci));
    std::vector<std::vector<double>> llh_mat_2(n_cells, std::vector<double>(n_loci));
    std::vector<std::vector<int>> total(n_cells, std::vector<int>(n_loci));

    std::vector<double> alphas_h(overdispersions_h.size());
    std::vector<double> betas_h(overdispersions_h.size());
    for (size_t i = 0; i < overdispersions_h.size(); ++i) {
        alphas_h[i] = 0.5 * overdispersions_h[i];
        betas_h[i] = overdispersions_h[i] - alphas_h[i];
    }

    for (size_t i = 0; i < n_cells; ++i) {
        for (size_t j = 0; j < n_loci; ++j) {
            total[i][j] = ref[i][j] + alt[i][j];
        }
    }

    for (size_t j = 0; j < n_loci; ++j) {
        for (size_t i = 0; i < n_cells; ++i) {

            int k = alt[i][j];
            int n = total[i][j];

            if (!individual) {
                llh_mat_1[i][j] = single_read_llh_with_dropout(k, n, gt1[j]);  // + mut_type_prior.at(std::string(1, gt1[j]));
                llh_mat_2[i][j] = single_read_llh_with_dropout(k, n, gt2[j]);  //+ mut_type_prior.at(std::string(1, gt2[j]));
            } else {
                llh_mat_1[i][j] = single_read_llh_with_individual_dropout(k, n, gt1[j], dropout_probs[j], alphas_h[j], betas_h[j]); // + mut_type_prior.at(std::string(1, gt1[j]))
                llh_mat_2[i][j] = single_read_llh_with_individual_dropout(k, n, gt2[j], dropout_probs[j], alphas_h[j], betas_h[j]); // + mut_type_prior.at(std::string(1, gt2[j]))
            }
        }
    }

    return {llh_mat_1, llh_mat_2};
}


// FUNCTIONS TO OPTIMIZE MODEL PARAMETERS

// Calculates the log-likelihood of observing k alternative reads with n coverage for a heterozygous locus.
std::tuple<double, double, double> MutationFilter::calculate_heterozygous_log_likelihoods(int k,
                                                                                          int n,
                                                                                          double dropout_prob,
                                                                                          double dropout_direction,
                                                                                          double alpha_h,
                                                                                          double beta_h,
                                                                                          double error_rate,
                                                                                          double overdispersion) {
    double log_no_dropout = std::log(1 - dropout_prob) + log_betabinom_pmf(k, n, alpha_h, beta_h);

    // Dropout to "R"
    double alpha_R = error_rate * overdispersion;
    double beta_R = overdispersion - alpha_R;
    double log_dropout_R = std::log(dropout_prob) + std::log(1 - dropout_direction) + log_betabinom_pmf(k, n, alpha_R, beta_R);

    // Dropout to "A"
    double alpha_A = (1 - error_rate) * overdispersion;
    double beta_A = overdispersion - alpha_A;
    double log_dropout_A = std::log(dropout_prob) + std::log(dropout_direction) + log_betabinom_pmf(k, n, alpha_A, beta_A);

    return std::make_tuple(log_no_dropout, log_dropout_R, log_dropout_A);
}

// Computes the total log-likelihood of the observations for the given parameters and genotypes.
double MutationFilter::total_log_likelihood(const std::vector<double>& params,
                                            const std::vector<int>& k_obs,
                                            const std::vector<int>& n_obs,
                                            const std::vector<char>& genotypes) const {
    double dropout_probability = params[0];
    double overdispersion = params[1];
    double error = params[2];
    double overdispersion_h = params[3];
    double log_likelihood = 0.0;

    double alpha_h = 0.5 * overdispersion_h;
    double beta_h = overdispersion_h - alpha_h;

    for (size_t i = 0; i < k_obs.size(); ++i) {
        int k = k_obs[i];
        int n = n_obs[i];
        char genotype = genotypes[i];

        if (genotype == 'R') {
            double alphaR = error * overdispersion;
            double betaR = overdispersion - alphaR;
            log_likelihood += log_betabinom_pmf(k, n, alphaR, betaR);
        } else if (genotype == 'H') {
            auto [log_no_dropout, log_dropout_R, log_dropout_A] = calculate_heterozygous_log_likelihoods(
                    k, n, dropout_probability, dropout_direction_prob, alpha_h, beta_h, error, overdispersion);

            log_likelihood += logsumexp({log_no_dropout, log_dropout_R, log_dropout_A});
        } else if (genotype == 'A') {
            double alphaA = (1 - error) * overdispersion;
            double betaA = overdispersion - alphaA;
            log_likelihood += log_betabinom_pmf(k, n, alphaA, betaA);
        } else {
            throw std::invalid_argument("Unexpected genotype: " + std::string(1, genotype));
        }
    }

    return log_likelihood;
}

// Function to compute the log-likelihood of the parameters
double MutationFilter::compute_llh_parameters(
        double dropout, double overdispersion,
        double error_r, double overdispersion_h,
        bool global_opt, double min_value, double shape, double alpha_parameters) const{
    // Compute Beta parameters using prior means
    double alpha_dropout_prob = alpha_parameters;
    double beta_dropout_prob = 1.0 / this->dropout_prob;

    double alpha_error_rate = alpha_parameters;
    double beta_error_rate = 1.0 / this->error_rate;

    // Adjust divisor based on optimization mode
    double divisor = global_opt ? (shape - 1) : shape;

    // Gamma scale values based on mean or mode
    double scale_overdispersion = (this->overdispersion_homozygous - min_value) / divisor;
    double scale_overdispersion_h = (this->overdispersion_heterozygous - min_value) / divisor;

    // Compute log-likelihood of the parameters
    double log_lh = 0.0;
    log_lh += beta_logpdf(dropout, alpha_dropout_prob, beta_dropout_prob);
    log_lh += gamma_logpdf(overdispersion, shape, scale_overdispersion, min_value);
    log_lh += beta_logpdf(error_r, alpha_error_rate, beta_error_rate);
    log_lh += gamma_logpdf(overdispersion_h, shape, scale_overdispersion_h, min_value);

    return log_lh;
}

// Computes the total combined log-likelihood of the parameters and the observations given the genotypes.
double MutationFilter::total_log_posterior(const std::vector<double>& params, const std::vector<int>& k_obs,
                                           const std::vector<int>& n_obs, const std::vector<char>& genotypes) const {
    double dropout = params[0];
    double overdispersion = params[1];
    double error_r = params[2];
    double overdispersion_h = params[3];

    // Compute log-likelihood (same logic as before)
    double log_likelihood = total_log_likelihood(params, k_obs, n_obs, genotypes);

    // Compute log-likelihood of the parameters
    double log_parameters = compute_llh_parameters(dropout, overdispersion, error_r, overdispersion_h, true);

    return -(log_likelihood + log_parameters);  // Negative because we're minimizing
}

// Fit the parameters of the model using the inferred genotypes and observed read counts.
std::vector<double> MutationFilter::fit_parameters(const std::vector<int>& ref,
                                                   const std::vector<int>& alt,
                                                   const std::vector<char>& genotypes,
                                                   std::vector<double> initial_params,
                                                   int max_iterations,
                                                   double tolerance){
    std::vector<int> total(ref.size());
    for (size_t i = 0; i < ref.size(); ++i)
        total[i] = ref[i] + alt[i];

    std::vector<char> genotypes_nonzero;
    std::vector<int> alt_norm;
    std::vector<int> total_norm;

    for (size_t i = 0; i < total.size(); ++i)
    {
        if (total[i] != 0)
        {
            genotypes_nonzero.push_back(genotypes[i]);
            alt_norm.push_back(alt[i]);
            total_norm.push_back(total[i]);
        }
    }

    // Objective function class
    struct Objective
    {
        const MutationFilter& filter;
        const std::vector<int>& alt;
        const std::vector<int>& total;
        const std::vector<char>& genotypes;

        Objective(const MutationFilter& filter_,
                  const std::vector<int>& alt_,
                  const std::vector<int>& total_,
                  const std::vector<char>& genotypes_)
                : filter(filter_), alt(alt_), total(total_), genotypes(genotypes_) {}

        double operator()(const VectorXd& x, VectorXd& grad)
        {
            std::vector<double> params(x.data(), x.data() + x.size());
            double val = filter.total_log_posterior(params, alt, total, genotypes);

            // Use numerical gradient (finite difference)
            const double eps = 1e-7;
            grad.resize(x.size());
            for (int i = 0; i < x.size(); ++i) {
                VectorXd x_eps = x;
                x_eps[i] += eps;
                std::vector<double> params_eps(x_eps.data(), x_eps.data() + x_eps.size());
                double val_eps = filter.total_log_posterior(params_eps, alt, total, genotypes);
                grad[i] = (val_eps - val) / eps;
            }

            return val;
        }
    };

    // Convert initial parameters to Eigen vector
    VectorXd x = Map<VectorXd>(initial_params.data(), static_cast<Eigen::Index>(initial_params.size()));

    // Set bounds
    VectorXd lb(4), ub(4);
    lb << 0.01, 1.0, 0.001, 2.5; // Lower bounds dropout, overdispersion, error rate, overdispersion_h
    ub << 0.99, 100.0, 0.1, 50.0;

    // Optimization config
    LBFGSBParam<double> param;
    param.epsilon = tolerance;
    param.max_iterations = max_iterations;
    param.delta = 1e-4;

    // Run optimizer
    LBFGSBSolver<double> solver(param);
    Objective obj(*this, alt_norm, total_norm, genotypes_nonzero);

    double fx;
    solver.minimize(obj, x, fx, lb, ub);

    return {x.data(), x.data() + x.size()};
}

double MutationFilter::total_log_posterior_individual(const std::vector<double>& params,
                                      const std::vector<int>& k_obs,
                                      const std::vector<int>& n_obs,
                                      double overdispersion,
                                      double errorRate,
                                      double dropoutDirectionProb) const {
    double dropoutProb = params[0];
    double overdispersionH = params[1];

    double log_likelihood = 0.0;

    double alpha_h = overdispersionH * 0.5;
    double beta_h = overdispersionH - alpha_h;

    for (size_t i = 0; i < k_obs.size(); ++i) {
        int k = k_obs[i];
        int n = n_obs[i];

        auto [log_no_dropout, log_dropout_R, log_dropout_A] = calculate_heterozygous_log_likelihoods(
                k, n, dropoutProb, dropoutDirectionProb, alpha_h, beta_h, errorRate, overdispersion);

        log_likelihood += logsumexp({log_no_dropout, log_dropout_R, log_dropout_A});
    }

    // Compute log-likelihood of the parameters
    double log_parameter = compute_llh_parameters(dropoutProb, overdispersion, errorRate, overdispersionH, false);

    return -(log_likelihood + log_parameter);  // Negative because we're minimizing
}

std::vector<double> MutationFilter::fit_parameters_individual(const std::vector<int>& alt_het,
                                                              const std::vector<int>& total_reads,
                                                              double overdispersion,
                                                              double error_r,
                                                              double dropout_direction,
                                                              std::vector<double> initial_params,
                                                              int max_iterations,
                                                              double tolerance) {
    // Nested functor class must not use `this` directly in constructor
    struct Objective {
        const MutationFilter& filter;
        const std::vector<int>& alt_het;
        const std::vector<int>& total_reads;
        double overdispersion_;
        double error_rate_;
        double dropout_direction_;

        Objective(const MutationFilter& f,
                  const std::vector<int>& a,
                  const std::vector<int>& t,
                  double od,
                  double er,
                  double dd)
                : filter(f), alt_het(a), total_reads(t),
                  overdispersion_(od), error_rate_(er), dropout_direction_(dd) {}

        double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
            std::vector<double> params(x.data(), x.data() + x.size());
            double val = filter.total_log_posterior_individual(params, alt_het, total_reads,
                                                               overdispersion_, error_rate_, dropout_direction_);

            // Numerical gradient using finite differences
            const double eps = 1e-7;
            grad.resize(x.size());
            for (int i = 0; i < x.size(); ++i) {
                Eigen::VectorXd x_eps = x;
                x_eps[i] += eps;

                std::vector<double> params_eps(x_eps.data(), x_eps.data() + x_eps.size());
                double val_eps = filter.total_log_posterior_individual(params_eps, alt_het, total_reads,
                                                                       overdispersion_, error_rate_, dropout_direction_);

                grad[i] = (val_eps - val) / eps;
            }

            return val;
        }




    };

    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(initial_params.data(), static_cast<Eigen::Index>(initial_params.size()));

    // Bounds: [dropout_prob, overdispersion_h]
    Eigen::VectorXd lb(2), ub(2);
    lb << 0.01, 2.5;
    ub << 0.99, 50.0;

    LBFGSpp::LBFGSBParam<double> param;
    param.epsilon = tolerance;
    param.max_iterations = max_iterations;
    param.delta = 0.005;

    LBFGSpp::LBFGSBSolver<double> solver(param);
    Objective obj(*this, alt_het, total_reads, overdispersion, error_r, dropout_direction);

    // Number of random initial guesses
    const int num_initial_guesses = 10;
    double best_fx = std::numeric_limits<double>::infinity();
    Eigen::VectorXd best_x = x;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist_dropout(0.05, 0.5);
    std::uniform_real_distribution<double> dist_overdispersion(3, 20);

    // Try optimization multiple times with random initial guesses
    for (int i = 0; i < num_initial_guesses; ++i) {

        Eigen::VectorXd random_x(2);
        random_x(0) = dist_dropout(gen); // Random dropout_prob
        random_x(1) = dist_overdispersion(gen); // Random overdispersion_h

        // Perform optimization with the random initial parameters
        double fx;
        int niter = solver.minimize(obj, random_x, fx, lb, ub);

        if (niter > 0 && fx < best_fx) {
            best_fx = fx;
            best_x = random_x;
        }
    }

    // Return the best result found
    return {best_x.data(), best_x.data() + best_x.size()};
}


std::tuple<double, double, double, double, std::vector<double>, std::vector<double>> MutationFilter::update_parameters(
        const std::vector<std::vector<int>>& ref, const std::vector<std::vector<int>>& alt,
        const std::vector<std::vector<char>>& inferred_genotypes) {

    std::vector<int> all_ref_counts;
    std::vector<int> all_alt_counts;
    std::vector<char> all_genotypes;

    for (size_t i = 0; i < ref.size(); ++i) {
        all_ref_counts.insert(all_ref_counts.end(), ref[i].begin(), ref[i].end());
        all_alt_counts.insert(all_alt_counts.end(), alt[i].begin(), alt[i].end());
        all_genotypes.insert(all_genotypes.end(), inferred_genotypes[i].begin(), inferred_genotypes[i].end());
    }

    // Fit all global parameters at once
    std::vector<double> global_params = fit_parameters(all_ref_counts, all_alt_counts, all_genotypes, {dropout_prob, overdispersion_homozygous, error_rate, overdispersion_heterozygous});
//
//    std::cout << "Global parameters: ";
//    for (double param : global_params) {
//        std::cout << param << " ";
//    }
//    std::cout << std::endl;

    // Fit global parameters in two stages, first overdispersion and error rate, then dropout and overdispersion_h
//    std::vector<double> global_params = fit_parameters_two_stage(
//            all_ref_counts,
//            all_alt_counts,
//            all_genotypes,
//            {dropout_prob, overdispersion_homozygous, error_rate, overdispersion_heterozygous},
//            50,
//            1e-5
//    );

    std::cout << "Global parameters: ";
    for (double param : global_params) {
        std::cout << param << " ";
    }
    std::cout << std::endl;

    double dropout = global_params[0];
    double overdispersion = global_params[1];
    double error_r = global_params[2];
    double overdispersion_h = global_params[3];

    std::vector<double> individual_dropout_probs;
    std::vector<double> individual_overdispersions_h;

    for (size_t snv = 0; snv < inferred_genotypes[0].size(); ++snv) {
        std::vector<size_t> indices;
        for (size_t i = 0; i < inferred_genotypes.size(); ++i) {
            if (inferred_genotypes[i][snv] == 'H') {
                indices.push_back(i);
            }
        }

        std::vector<int> ref_het;
        std::vector<int> alt_het;
        for (size_t idx : indices) {
            ref_het.push_back(ref[idx][snv]);
            alt_het.push_back(alt[idx][snv]);
        }

        std::vector<int> total_reads(ref_het.size());
        for (size_t i = 0; i < ref_het.size(); ++i) {
            total_reads[i] = ref_het[i] + alt_het[i];
        }

        std::vector<int> informative;
        for (int i = 0; i < total_reads.size(); ++i) {
            if (total_reads[i] > 10) {
                informative.push_back(i);
            }
        }

        std::vector<double> individual_params;
        if (informative.size() > 5) { // makes optimization faster to only fit if there is enough data
            // Fit individual parameters
            individual_params = fit_parameters_individual(alt_het, total_reads, overdispersion,
                                                          error_r, dropout_direction_prob, {dropout_prob, overdispersion_heterozygous});
        }
        else {
            individual_params = {dropout, overdispersion_h};
        }

        double posterior_dropout_prob = individual_params[0];
        double posterior_overdispersion_h = individual_params[1];

        individual_dropout_probs.push_back(posterior_dropout_prob);
        individual_overdispersions_h.push_back(posterior_overdispersion_h);
    }
    return {dropout, overdispersion, error_r, overdispersion_h, individual_dropout_probs, individual_overdispersions_h};
}


// Function to compute the natural logarithm of the beta function
double MutationFilter::betaln(double x, double y) {
    return std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y);
}

// Function to compute the natural logarithm of the factorial of n
double MutationFilter::factorial(int n) {
    return std::lgamma(n + 1); // log(n!)
}

// Function to compute the natural logarithm of the binomial coefficient
double MutationFilter::log_binomial_coefficient(int n, int k) {
    if (0 <= k && k <= n) {
        double log_numerator = factorial(n);
        double log_denominator = factorial(k) + factorial(n - k);
        return log_numerator - log_denominator;
    } else {
        return 0.0;
    }
}

// Function to compute the probability mass function of the beta-binomial distribution
double MutationFilter::log_betabinom_pmf(int k, int n, double a, double b) {
    if (n < 0 || k < 0 || k > n || a <= 0 || b <= 0) {
        return -std::numeric_limits<double>::infinity(); // Handle invalid input
    }

    double log_binom_coef = log_binomial_coefficient(n, k);
    double num = betaln(k + a, n - k + b);
    double denom = betaln(a, b);

    return (num - denom + log_binom_coef);
}


// Function to compute the log-PDF of a beta distribution
double MutationFilter::beta_logpdf(double x, double alpha, double beta) {
    if (x <= 0 || x >= 1) {
        return -std::numeric_limits<double>::infinity();
    }
    return (alpha - 1) * std::log(x) + (beta - 1) * std::log(1 - x) - betaln(alpha, beta);
}

// Function to compute the log-PDF of a gamma distribution
double MutationFilter::gamma_logpdf(double x, double shape, double scale, double loc = 0) {
    if (x < loc) {
        return -std::numeric_limits<double>::infinity();
    }
    x -= loc;
    return (shape - 1) * std::log(x) - x / scale - shape * std::log(scale) - std::lgamma(shape);
}

// logbinomial function
double MutationFilter::logbinom(int n, int k) {
    if (k < 0 || k > n) {
        throw std::invalid_argument("k must be between 0 and n inclusive");
    }
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
}

// logsumexp of vector
double MutationFilter::logsumexp(const std::vector<double>& v) {
    double max_val = *std::max_element(v.begin(), v.end());
    double sum = 0.0;
    for (double x : v) {
        sum += std::exp(x - max_val);
    }
    return max_val + std::log(sum);
}

// lognormalize and exp of vector
std::vector<double> MutationFilter::lognormalize_exp(const std::vector<double>& v) {
    double max_val = *std::max_element(v.begin(), v.end());
    std::vector<double> exp_values(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        exp_values[i] = std::exp(v[i] - max_val);
    }
    double sum_exp = std::accumulate(exp_values.begin(), exp_values.end(), 0.0);

    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = exp_values[i] / sum_exp;
    }
    return result;
}

// get column of a 2D matrix
std::vector<int> MutationFilter::get_column(const std::vector<std::vector<int>>& matrix, size_t col_index) {
    std::vector<int> column;
    column.reserve(matrix.size());  // Reserve space to avoid multiple allocations

    for (const auto& row : matrix) {
        if (col_index < row.size()) {
            column.push_back(row[col_index]);
        }
    }
    return column;
}

// concatenate several 1D vectors
template<typename... Vectors>
std::vector<double> MutationFilter::concat(const std::vector<double>& first, const Vectors&... rest) const {
    std::vector<double> result = first;
    (result.insert(result.end(), rest.begin(), rest.end()), ...);
    return result;
}

// add scalar to 1D vector
std::vector<double> MutationFilter::add_scalar_to_vector(double scalar, const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = scalar + vec[i];
    }
    return result;
}
