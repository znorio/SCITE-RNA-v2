#include <unordered_map>
#include <iostream>
#include <vector>

#ifndef SCITE_RNA_CPP_MUTATION_FILTER_H
#define SCITE_RNA_CPP_MUTATION_FILTER_H

class MutationFilter {
private:
    double f_;
    int n_cells = 1;
    int n_loci = 1;
    int omega_;
    double h_factor_;
    double mut_freq_;
    int min_grp_size_;
    double alpha_R = 0.5;
    double beta_R = 0.5;
    double alpha_A = 0.5;
    double beta_A = 0.5;
    double alpha_H = 0.5;
    double beta_H = 0.5;
    std::unordered_map<char, double> genotype_freq_;
    std::unordered_map<std::string, double> mut_type_prior;

public:
    // mutation filter functions
    explicit MutationFilter(double f = 0.95, int omega = 100, double h_factor = 0.5,
                   const std::unordered_map<char, double>& genotype_freq = get_genotype_freq(),
                   double mut_freq = 0.5, int min_grp_size = 1);
    void set_betabinom();
    void set_mut_type_prior();
    double single_read_llh(int n_ref, int n_alt, char genotype) const;
    std::vector<double> k_mut_llh(std::vector<int>& ref, std::vector<int>& alt, char gt1, char gt2) const;
    std::vector<double> single_locus_posteriors(std::vector<int> ref, std::vector<int> alt,
                                                const std::unordered_map<std::string, std::vector<double>>& comp_priors) const;
    std::vector<std::vector<double>> mut_type_posteriors(std::vector<std::vector<int>>& ref,
                                                         std::vector<std::vector<int>>& alt);
    std::tuple<std::vector<int>, std::vector<char>, std::vector<char>, std::vector<char>> filter_mutations(
                                    std::vector<std::vector<int>>& ref,
                                    std::vector<std::vector<int>>& alt,
                                    const std::string& method = "highest_post",
                                    double t = 0.0,
                                    int n_exp = 0,
                                    bool reversible = true, int n_test = 0);
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> get_llh_mat(
                const std::vector<std::vector<int>>& ref,
                const std::vector<std::vector<int>>& alt,
                const std::vector<char>& gt1,
                const std::vector<char>& gt2) const;

    // helper functions
    static double betaln(double x, double y);
    static double factorial(int n);
    static double log_binomial_coefficient(int n, int k);
    static double betabinom_pmf(int n_ref, int total_reads, double alpha, double beta);
    static double logaddexp(double logx, double logy);
    static double logbinom(int n, int k);
    static double logsumexp(const std::vector<double>& v);
    static std::vector<double> lognormalize_exp(const std::vector<double>& v);
    static std::vector<int> get_column(const std::vector<std::vector<int>>& matrix, size_t col_index);
    template<typename... Vectors>
    std::vector<double> concat(const std::vector<double>& first, const Vectors&... rest) const;
    static std::vector<double> add_vectors(const std::vector<double>& a, const std::vector<double>& b);
    static std::vector<double> add_scalar_to_vector(double scalar, const std::vector<double>& vec);
    static std::unordered_map<char, double> get_genotype_freq();
};

#endif //SCITE_RNA_CPP_MUTATION_FILTER_H
