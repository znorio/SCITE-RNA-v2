#include <unordered_map>
#include <iostream>
#include <vector>
#include <map>

#ifndef SCITE_RNA_CPP_MUTATION_FILTER_H
#define SCITE_RNA_CPP_MUTATION_FILTER_H

class MutationFilter {
private:
    int n_cells = 1;
    int n_loci = 1;
    double error_rate;
    double overdispersion_homozygous;
    std::map<std::string, double> genotype_freq_;
    double dropout_direction_prob;
    double overdispersion_heterozygous;
    std::map<std::string, double> mut_type_prior;
    double alpha_R;
    double beta_R;
    double alpha_A;
    double beta_A;
    double alpha_H;
    double beta_H;
    double dropout_prob;

public:
    explicit MutationFilter(double error_rate = 0.05, double overdispersion = 10,
                   const std::map<std::string, double>& genotype_freq = {{"R", 1.0/3}, {"H", 1.0/3}, {"A", 1.0/3}},
                   double mut_freq_ = 0.5, double dropout_alpha = 2, double dropout_beta = 8,
                   double dropout_direction_prob = 0.5, double overdispersion_h = 6);

    void set_mut_type_prior(const std::map<std::string, double>& genotype_freq, double mut_freq);
    [[nodiscard]] double single_read_llh_with_dropout(int n_alt, int n_total, char genotype) const;
    [[nodiscard]] double single_read_llh_with_individual_dropout(int n_alt, int n_total, char genotype,
                                                   double dropout_prob, double alpha_h, double beta_h) const;
    [[nodiscard]] std::vector<double> k_mut_llh(const std::vector<int>& ref, const std::vector<int>& alt,
                                  char gt1, char gt2) const;
    [[nodiscard]] std::vector<double> single_locus_posteriors(const std::vector<int>& ref, const std::vector<int>& alt,
                                                const std::unordered_map<std::string, std::vector<double>>& comp_priors) const;
    std::vector<std::vector<double>> mut_type_posteriors(const std::vector<std::vector<int>>& ref,
                                                         const std::vector<std::vector<int>>& alt);
    std::tuple<std::vector<int>, std::vector<char>, std::vector<char>, std::vector<char>> filter_mutations(
            const std::vector<std::vector<int>>& ref,
            const std::vector<std::vector<int>>& alt,
            const std::string& method = "highest_post", double t = 0.0, int n_exp = 0);
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> get_llh_mat(
            const std::vector<std::vector<int>>& ref, const std::vector<std::vector<int>>& alt,
            const std::vector<char>& gt1, const std::vector<char>& gt2, bool individual,
            const std::vector<double>& dropout_probs = {}, const std::vector<double>& = {});

    [[nodiscard]] double compute_llh_parameters(
            double dropout, double overdispersion,
            double error_r, double overdispersion_h,
            bool global_opt, double min_value=2, double shape=2, double alpha_parameters=2) const;

    [[nodiscard]] double total_log_posterior(const std::vector<double>& params, const std::vector<int>& k_obs,
                               const std::vector<int>& n_obs, const std::vector<char>& genotypes) const;
    [[nodiscard]] double total_log_posterior_individual(const std::vector<double>& params,
                                          const std::vector<int>& k_obs,
                                          const std::vector<int>& n_obs,
                                          double overdispersion,
                                          double errorRate,
                                          double dropoutDirectionProb) const;
    std::vector<double> fit_parameters(const std::vector<int>& ref, const std::vector<int>& alt,
                                       const std::vector<char>& genotypes, std::vector<double> initial_params = {},
                                       int max_iterations = 50, double tolerance = 1e-5);
    std::vector<double> fit_parameters_individual(const std::vector<int>& alt_het, const std::vector<int>& total_reads,
                                                  double overdispersion, double error_rate, double dropout_direction,
                                                  std::vector<double> initial_params = {}, int max_iterations = 50,
                                                  double tolerance = 1e-5);
    std::tuple<double, double, double, double, std::vector<double>, std::vector<double>> update_parameters(
            const std::vector<std::vector<int>>& ref, const std::vector<std::vector<int>>& alt,
            const std::vector<std::vector<char>>& inferred_genotypes);

    void update_alpha_beta(double error_r, double overdispersion_hom);

    // helper functions
    static double betaln(double x, double y);
    static double factorial(int n);
    static double log_binomial_coefficient(int n, int k);
    static double log_betabinom_pmf(int n_ref, int total_reads, double alpha, double beta);
    static double logbinom(int n, int k);
    static double logsumexp(const std::vector<double>& v);
    static std::vector<double> lognormalize_exp(const std::vector<double>& v);
    static std::vector<int> get_column(const std::vector<std::vector<int>>& matrix, size_t col_index);
    template<typename... Vectors>
    std::vector<double> concat(const std::vector<double>& first, const Vectors&... rest) const;
    static std::vector<double> add_scalar_to_vector(double scalar, const std::vector<double>& vec);


    [[nodiscard]] static std::tuple<double, double, double>
    calculate_heterozygous_log_likelihoods(int k, int n, double dropout_prob, double dropout_direction_prob,
                                           double alpha_h, double beta_h, double error_rate, double overdispersion) ;

    [[nodiscard]] double total_log_likelihood(const std::vector<double>& params,
                                                const std::vector<int>& k_obs,
                                                const std::vector<int>& n_obs,
                                                const std::vector<char>& genotypes) const;

    static double beta_logpdf(double x, double alpha, double beta);

    static double gamma_logpdf(double x, double shape, double scale, double loc);
};

#endif //SCITE_RNA_CPP_MUTATION_FILTER_H
