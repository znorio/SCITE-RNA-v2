/*
This code calculates the posterior probability of different mutation types and genotypes given the data
and can filter SNVs based on this posterior.
*/

#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <numeric>

#include "mutation_filter.h"

// initialize filter mutations based on posterior likelihoods
MutationFilter::MutationFilter(double f, int omega, double h_factor,
                               const std::unordered_map<char, double>& genotype_freq, double mut_freq, int min_grp_size)
        : f_(f), omega_(omega), h_factor_(h_factor), genotype_freq_(genotype_freq), mut_freq_(mut_freq), min_grp_size_(min_grp_size) {
    set_betabinom();
    set_mut_type_prior();
}

// calculate parameters of the beta binomial distribution
void MutationFilter::set_betabinom() {
    alpha_R = f_ * static_cast<float>(omega_);
    beta_R = static_cast<float>(omega_) - alpha_R;
    alpha_A = (1.0f - f_) * static_cast<float>(omega_);
    beta_A = static_cast<float>(omega_) - alpha_A;
    alpha_H = (static_cast<float>(omega_) / 2.0f) * h_factor_;
    beta_H = (static_cast<float>(omega_) / 2.0f) * h_factor_;
}

// prior likelihood of mutation types
void MutationFilter::set_mut_type_prior() {
    mut_type_prior["R"] = genotype_freq_['R'] * (1.0f - mut_freq_);
    mut_type_prior["H"] = genotype_freq_['H'] * (1.0f - mut_freq_);
    mut_type_prior["A"] = genotype_freq_['A'] * (1.0f - mut_freq_);
    mut_type_prior["RH"] = genotype_freq_['R'] * mut_freq_;
    mut_type_prior["HA"] = genotype_freq_['H'] * mut_freq_ / 2.0f;
    mut_type_prior["HR"] = mut_type_prior["HA"];
    mut_type_prior["AH"] = genotype_freq_['A'] * mut_freq_;

    // Convert to log scale
    for (auto& kv : mut_type_prior) {
        kv.second = std::log(kv.second);
    }
}

// likelihood of the data at a cell locus position
double MutationFilter::single_read_llh(int n_ref, int n_alt, char genotype) const {
    double result;
    if (genotype == 'R') {
        result = betabinom_pmf(n_ref, n_ref + n_alt, alpha_R, beta_R);
    } else if (genotype == 'A') {
        result = betabinom_pmf(n_ref, n_ref + n_alt, alpha_A, beta_A);
    } else if (genotype == 'H') {
        result = betabinom_pmf(n_ref, n_ref + n_alt, alpha_H, beta_H);
    } else {
        throw std::invalid_argument("[MutationFilter::single_read_llh] Invalid genotype.");
    }
    return std::log(result);
}

// llh that k out of the first n cells are mutated
std::vector<double> MutationFilter::k_mut_llh(std::vector<int>& ref, std::vector<int>& alt, char gt1, char gt2) const {
    std::vector<double> result(n_cells + 1, 0.0);

    if (gt1 == gt2) {
        for (int i = 0; i < n_cells; ++i) {
            result[0] += single_read_llh(ref[i], alt[i], gt1);
        }
    }
    else {
        std::vector<std::vector<double>> k_in_first_n_llh(n_cells + 1, std::vector<double>(n_cells + 1, 0.0));

        for (int n = 0; n < n_cells; ++n) {
            double gt1_llh = single_read_llh(ref[n], alt[n], gt1);
            double gt2_llh = single_read_llh(ref[n], alt[n], gt2);

            k_in_first_n_llh[n + 1][0] = k_in_first_n_llh[n][0] + gt1_llh;

            for (int k = 1; k <= n; ++k) {
                double k_over_n = static_cast<double>(k) / (n + 1);
                double log_summand_1 = std::log(1 - k_over_n) + gt1_llh + k_in_first_n_llh[n][k];
                double log_summand_2 = std::log(k_over_n) + gt2_llh + k_in_first_n_llh[n][k - 1];
                k_in_first_n_llh[n + 1][k] = logaddexp(log_summand_1, log_summand_2);
            }

            k_in_first_n_llh[n + 1][n + 1] = k_in_first_n_llh[n][n] + gt2_llh;
        }

        result.assign(k_in_first_n_llh[n_cells].begin(), k_in_first_n_llh[n_cells].end());
    }

    return result;
}

// likelihood of a single locus given the data and priors
std::vector<double> MutationFilter::single_locus_posteriors(std::vector<int> ref, std::vector<int> alt,
                                                            const std::unordered_map<std::string, std::vector<double>>& comp_priors) const {
    std::vector<double> llh_RH = k_mut_llh(ref, alt, 'R', 'H');
    std::vector<double> llh_HA = k_mut_llh(ref, alt, 'H', 'A');
    if (llh_RH.back() != llh_HA.front()) {
        throw std::runtime_error("Mismatch in likelihoods");
    }

    std::vector<double> joint_R = add_scalar_to_vector(llh_RH[0], comp_priors.at("R")); // # llh zero out of n cells with genotype R are mutated given the data + prior of genotype RR
    std::vector<double> joint_H = add_scalar_to_vector(llh_HA[0], comp_priors.at("H"));
    std::vector<double> joint_A = add_scalar_to_vector(llh_HA.back(), comp_priors.at("A"));
    std::vector<double> joint_RH =  add_vectors({llh_RH.begin() + 1, llh_RH.end()}, comp_priors.at("RH"));
    std::vector<double> joint_HA = add_vectors({llh_HA.begin() + 1, llh_HA.end()}, comp_priors.at("HA"));

    std::reverse(llh_RH.begin(), llh_RH.end());
    std::reverse(llh_HA.begin(), llh_HA.end());

    std::vector<double> joint_HR = add_vectors({llh_RH.begin() + 1, llh_RH.end()}, comp_priors.at("HR"));
    std::vector<double> joint_AH = add_vectors({llh_HA.begin() + 1, llh_HA.end()}, comp_priors.at("AH"));

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
std::vector<std::vector<double>> MutationFilter::mut_type_posteriors(std::vector<std::vector<int>>& ref,
                                                                     std::vector<std::vector<int>>& alt) {

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
//    for (const auto &mut_type: {"HR", "AH"}) {
//        std::reverse(comp_priors[mut_type].begin(), comp_priors[mut_type].end());
//    }

    // calculate posteriors for all loci
    std::vector<std::vector<double>> result(n_loci, std::vector<double>(7));

    for (int j = 0; j < n_loci; ++j) {
        result[j] = single_locus_posteriors(get_column(ref,j), get_column(alt, j), comp_priors);
    }

    return result;
}

// filter mutations based on the posterior likelihood of mutation types
std::tuple<std::vector<int>, std::vector<char>, std::vector<char>, std::vector<char>> MutationFilter::filter_mutations(
                            std::vector<std::vector<int>>& ref,
                            std::vector<std::vector<int>>& alt,
                            const std::string& method,
                            double t,
                            int n_exp,
                            bool reversible) {

    n_loci = static_cast<int>(ref[0].size());
    n_cells = static_cast<int>(ref.size());

    std::vector<std::vector<double>> posteriors = mut_type_posteriors(ref, alt);

    std::vector<int> selected;

    if (method == "highest_post") {
        for (int j = 0; j < n_loci; ++j) {
            std::vector<double> mut_posterior(posteriors[j].begin() + 3, posteriors[j].end());
            if (*std::max_element(mut_posterior.begin(), mut_posterior.end()) > 0) {
                selected.push_back(j);
            }
        }
    }
    else if (method == "threshold") {
        for (int j = 0; j < n_loci; ++j) {
            double sum_posterior = std::accumulate(posteriors[j].begin() + 3, posteriors[j].end(), 0.0);
            if (sum_posterior > t) {
                selected.push_back(j);
            }
        }
    }
    else if (method == "first_k") {
        std::vector<double> mut_posteriors;
        mut_posteriors.reserve(n_loci);
        for (int j = 0; j < n_loci; ++j) {
            double sum_posterior = std::accumulate(posteriors[j].begin() + 3, posteriors[j].end(), 0.0);
            mut_posteriors.push_back(sum_posterior);
        }

        std::vector<int> order(n_loci);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return mut_posteriors[a] < mut_posteriors[b];
        });

        for (int i = n_loci - 1; i >= std::max(0, n_loci - n_exp); --i) {
            selected.push_back(order[i]);
        }
    }
    else {
        throw std::invalid_argument("[MutationFilter::filter_mutations] Unknown filtering method.");
    }

    if (reversible) {
        std::vector<int> mut_type(selected.size());

        for (size_t i = 0; i < selected.size(); ++i) {
            std::vector<double> mut_posterior(posteriors[selected[i]].begin() + 3, posteriors[selected[i]].end());
            auto it = std::max_element(mut_posterior.begin(), mut_posterior.end());
            auto index = std::distance(mut_posterior.begin(), it);
            mut_type[i] = static_cast<int>(index);
        }


        std::vector<char> gt2_inferred(mut_type.size());
        std::vector<char> gt1_inferred(mut_type.size());
        for (size_t j = 0; j < mut_type.size(); ++j) {
            switch (mut_type[j]) {
                case 0: gt1_inferred[j] = 'R'; break;
                case 1: case 2: gt1_inferred[j] = 'H'; break;
                case 3: gt1_inferred[j] = 'A'; break;
                default: throw std::runtime_error("Invalid mut_type index");
            }
        }
        for (size_t k = 0; k < mut_type.size(); ++k) {
            switch (mut_type[k]) {
                case 0: gt2_inferred[k] = 'H'; break;
                case 1: gt2_inferred[k] = 'A'; break;
                case 2: gt2_inferred[k] = 'R'; break;
                case 3: gt2_inferred[k] = 'H'; break;
                default: throw std::runtime_error("Invalid mut_type index");
            }
        }

        std::vector<char> gt_not_selected;
        for (int i = 0; i < posteriors.size(); ++i) {
            if (std::find(selected.begin(), selected.end(), i) != selected.end()) {
                continue;  // If this locus is selected, skip it.
            }

            // Find the index of the max value in the first 3 entries (0, 1, 2) of the posterior
            auto max_it = std::max_element(posteriors[i].begin(), posteriors[i].begin() + 3);
            int genotype_index = std::distance(posteriors[i].begin(), max_it);

            // Use a switch statement to choose the corresponding genotype (R, H, A, etc.)
            switch (genotype_index) {
                case 0: gt_not_selected.push_back('R'); break;
                case 1: gt_not_selected.push_back('H'); break;
                case 2: gt_not_selected.push_back('A'); break;
                default: throw std::runtime_error("Invalid genotype index");
            }
        }

        return {selected, gt1_inferred, gt2_inferred, gt_not_selected};
    }
    else {
        std::vector<int> mut_type(selected.size());

        for (size_t i = 0; i < selected.size(); ++i) {
            std::vector<double> mut_posterior(posteriors[selected[i]].begin() + 3, posteriors[selected[i]].begin() + 5); // RH. HA
            auto it = std::max_element(mut_posterior.begin(), mut_posterior.end());
            auto index = std::distance(mut_posterior.begin(), it);
            mut_type[i] = static_cast<int>(index);
        }
        std::vector<char> gt1_inferred(mut_type.size());
        std::vector<char> gt2_inferred(mut_type.size());

        for (size_t j = 0; j < mut_type.size(); ++j) {
            switch (mut_type[j]) {
                case 0: gt1_inferred[j] = 'R'; break;
                case 1: gt1_inferred[j] = 'H'; break;
                default: throw std::runtime_error("Invalid mut_type index");
            }
        }
        for (size_t k = 0; k < mut_type.size(); ++k) {
            switch (mut_type[k]) {
                case 0: gt2_inferred[k] = 'H'; break;
                case 1: gt2_inferred[k] = 'A'; break;
                default: throw std::runtime_error("Invalid mut_type index");
            }
        }

        std::vector<char> gt_not_selected;
        for (int i = 0; i < posteriors.size(); ++i) {
            if (std::find(selected.begin(), selected.end(), i) != selected.end()) {
                continue;
            }

            auto max_it = std::max_element(posteriors[i].begin(), posteriors[i].begin() + 3);
            int genotype_index = std::distance(posteriors[i].begin(), max_it);

            // Use a switch statement to choose the corresponding genotype (R, H, A)
            switch (genotype_index) {
                case 0: gt_not_selected.push_back('R'); break;
                case 1: gt_not_selected.push_back('H'); break;
                case 2: gt_not_selected.push_back('A'); break;
                default: throw std::runtime_error("Invalid genotype index");
            }
        }
        return {selected, gt1_inferred, gt2_inferred, gt_not_selected};
    }
}

// get llh matrices given genotypes 1 and 2 and the data
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> MutationFilter::get_llh_mat(
            const std::vector<std::vector<int>>& ref,
            const std::vector<std::vector<int>>& alt,
            const std::vector<char>& gt1,
            const std::vector<char>& gt2) const {

    int n_mut =  static_cast<int>(ref[0].size()); // after filtering n_mut != n_loci
    int n_cell =  static_cast<int>(ref.size()); // necessary for debugging
    // Initialize llh_mat_1 and llh_mat_2
    std::vector<std::vector<double>> llh_mat_1(n_cell, std::vector<double>(n_mut));
    std::vector<std::vector<double>> llh_mat_2(n_cell, std::vector<double>(n_mut));

    // Calculate log-likelihood matrices
    for (size_t i = 0; i < n_cell; ++i) {
        for (size_t j = 0; j < n_mut; ++j) {
            llh_mat_1[i][j] = single_read_llh(ref[i][j], alt[i][j], gt1[j]);
            llh_mat_2[i][j] = single_read_llh(ref[i][j], alt[i][j], gt2[j]);
        }
    }

    return {llh_mat_1, llh_mat_2};
}

// Function to compute the natural logarithm of the beta function
double MutationFilter::betaln(double x, double y) {
    return std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y);
}

// Function to compute the natural logarithm of the factorial of n
double MutationFilter::factorial(int n) {
    if (n == 0 || n == 1) {
        return 0.0; // log(1) = 0
    }
    double result = 0.0;
    for (int i = 2; i <= n; ++i) {
        result += std::log(i);
    }
    return result;
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
double MutationFilter::betabinom_pmf(int k, int n, double a, double b) {
    if (n < 0 || k < 0 || k > n || a <= 0 || b <= 0) {
        return 0.0; // Handle invalid input gracefully
    }

    double log_binom_coef = log_binomial_coefficient(n, k);
    double num = betaln(k + a, n - k + b);
    double denom = betaln(a, b);

    return std::exp(num - denom + log_binom_coef);
}

// logaddexp if two scalars
double MutationFilter::logaddexp(double logx, double logy) {
    if (logx == -std::numeric_limits<double>::infinity()) {
        return logy;
    } else if (logy == -std::numeric_limits<double>::infinity()) {
        return logx;
    } else if (logx > logy) {
        return logx + std::log1p(std::exp(logy - logx));
    } else {
        return logy + std::log1p(std::exp(logx - logy));
    }
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
    double max_val = *std::max(v.begin(), v.end());
    double sum_exp = 0.0;
    for (double x : v) {
        sum_exp += std::exp(x - max_val);
    }
    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = std::exp(v[i] - max_val - std::log(sum_exp));
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

// add two 1D vectors
std::vector<double> MutationFilter::add_vectors(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for element-wise addition.");
    }
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
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