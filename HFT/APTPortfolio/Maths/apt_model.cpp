/*
APTPortfolio HFT Problem: Arbitrage Pricing Theory Implementation
================================================================

Problem Statement:
Implement the Arbitrage Pricing Theory (APT) model for multi-factor asset pricing.
Build a factor-based portfolio construction and risk attribution system.

This tests:
1. Multi-factor model implementation
2. Factor exposure calculation
3. Statistical arbitrage strategies
4. Performance attribution

Applications:
- Factor-based investing
- Risk budgeting and attribution
- Statistical arbitrage
- Style analysis
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <map>
#include <string>
#include <memory>
#include <stdexcept>
#include <random>
#include <iomanip>

// Helper functions for matrix operations
class MatrixUtils {
public:
    // Matrix multiplication: C = A * B
    static std::vector<std::vector<double>> multiply(
        const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B) {
        
        size_t rows_A = A.size();
        size_t cols_A = A[0].size();
        size_t cols_B = B[0].size();
        
        std::vector<std::vector<double>> C(rows_A, std::vector<double>(cols_B, 0.0));
        
        for (size_t i = 0; i < rows_A; ++i) {
            for (size_t j = 0; j < cols_B; ++j) {
                for (size_t k = 0; k < cols_A; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }
    
    // Matrix-vector multiplication
    static std::vector<double> multiply(
        const std::vector<std::vector<double>>& A,
        const std::vector<double>& x) {
        
        size_t rows = A.size();
        size_t cols = A[0].size();
        std::vector<double> result(rows, 0.0);
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i] += A[i][j] * x[j];
            }
        }
        return result;
    }
    
    // Transpose matrix
    static std::vector<std::vector<double>> transpose(
        const std::vector<std::vector<double>>& A) {
        
        size_t rows = A.size();
        size_t cols = A[0].size();
        std::vector<std::vector<double>> At(cols, std::vector<double>(rows));
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                At[j][i] = A[i][j];
            }
        }
        return At;
    }
    
    // Calculate column means
    static std::vector<double> colMeans(const std::vector<std::vector<double>>& A) {
        size_t rows = A.size();
        size_t cols = A[0].size();
        std::vector<double> means(cols, 0.0);
        
        for (size_t j = 0; j < cols; ++j) {
            for (size_t i = 0; i < rows; ++i) {
                means[j] += A[i][j];
            }
            means[j] /= rows;
        }
        return means;
    }
    
    // Simple matrix inversion using Gaussian elimination (for small matrices)
    static std::vector<std::vector<double>> inverse(
        const std::vector<std::vector<double>>& A) {
        
        size_t n = A.size();
        std::vector<std::vector<double>> augmented(n, std::vector<double>(2 * n));
        
        // Create augmented matrix [A|I]
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                augmented[i][j] = A[i][j];
                augmented[i][j + n] = (i == j) ? 1.0 : 0.0;
            }
        }
        
        // Gaussian elimination with pivoting
        for (size_t i = 0; i < n; ++i) {
            // Find pivot
            size_t max_row = i;
            for (size_t k = i + 1; k < n; ++k) {
                if ((augmented[k][i] < 0 ? -augmented[k][i] : augmented[k][i]) > 
                    (augmented[max_row][i] < 0 ? -augmented[max_row][i] : augmented[max_row][i])) {
                    max_row = k;
                }
            }
            
            if (max_row != i) {
                std::swap(augmented[i], augmented[max_row]);
            }
            
            // Make diagonal element 1
            double pivot = augmented[i][i];
            if ((pivot < 0 ? -pivot : pivot) < 1e-10) {
                throw std::runtime_error("Matrix is singular");
            }
            for (size_t j = 0; j < 2 * n; ++j) {
                augmented[i][j] /= pivot;
            }
            
            // Eliminate column
            for (size_t k = 0; k < n; ++k) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (size_t j = 0; j < 2 * n; ++j) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }
        
        // Extract inverse
        std::vector<std::vector<double>> result(n, std::vector<double>(n));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                result[i][j] = augmented[i][j + n];
            }
        }
        return result;
    }
};

class APTModel {
private:
    std::vector<std::vector<double>> returns_;     // Asset returns (T x N)
    std::vector<std::vector<double>> factors_;     // Factor returns (T x K)
    std::vector<std::vector<double>> loadings_;    // Factor loadings (N x K)
    std::vector<double> alpha_;                    // Asset alphas (N x 1)
    std::vector<std::vector<double>> residuals_;   // Residual returns (T x N)
    double r_squared_;
    
public:
    APTModel(const std::vector<std::vector<double>>& returns, 
             const std::vector<std::vector<double>>& factors) 
        : returns_(returns), factors_(factors) {
        estimateModel();
    }
    
    void estimateModel() {
        int T = returns_.size();      // Time periods
        int N = returns_[0].size();   // Number of assets
        int K = factors_[0].size();   // Number of factors
        
        loadings_.resize(N, std::vector<double>(K, 0.0));
        alpha_.resize(N, 0.0);
        residuals_.resize(T, std::vector<double>(N, 0.0));
        
        // For each asset, run regression: R_i = alpha_i + beta_i * F + epsilon_i
        for (int i = 0; i < N; ++i) {
            std::vector<double> asset_returns(T);
            for (int t = 0; t < T; ++t) {
                asset_returns[t] = returns_[t][i];
            }
            
            // Create design matrix X = [1, F] (T x (K+1))
            std::vector<std::vector<double>> X(T, std::vector<double>(K + 1));
            for (int t = 0; t < T; ++t) {
                X[t][0] = 1.0;  // Constant term
                for (int k = 0; k < K; ++k) {
                    X[t][k + 1] = factors_[t][k];
                }
            }
            
            // OLS regression: beta = (X'X)^-1 X'y
            auto Xt = MatrixUtils::transpose(X);
            auto XtX = MatrixUtils::multiply(Xt, X);
            auto XtX_inv = MatrixUtils::inverse(XtX);
            auto Xty = MatrixUtils::multiply(Xt, std::vector<std::vector<double>>{asset_returns});
            
            // Convert single column to vector
            std::vector<double> Xty_vec(K + 1);
            for (int j = 0; j < K + 1; ++j) {
                Xty_vec[j] = Xty[j][0];
            }
            
            auto coefficients = MatrixUtils::multiply(XtX_inv, Xty_vec);
            
            alpha_[i] = coefficients[0];
            for (int k = 0; k < K; ++k) {
                loadings_[i][k] = coefficients[k + 1];
            }
            
            // Calculate residuals
            for (int t = 0; t < T; ++t) {
                double fitted = alpha_[i];
                for (int k = 0; k < K; ++k) {
                    fitted += loadings_[i][k] * factors_[t][k];
                }
                residuals_[t][i] = asset_returns[t] - fitted;
            }
        }
        
        // Calculate R-squared
        calculateRSquared();
    }
    
    void calculateRSquared() {
        double total_ss = 0.0;
        double residual_ss = 0.0;
        
        for (int i = 0; i < returns_[0].size(); ++i) {
            // Calculate mean return for asset i
            double mean_return = 0.0;
            for (int t = 0; t < returns_.size(); ++t) {
                mean_return += returns_[t][i];
            }
            mean_return /= returns_.size();
            
            // Calculate total sum of squares and residual sum of squares
            for (int t = 0; t < returns_.size(); ++t) {
                double deviation = returns_[t][i] - mean_return;
                total_ss += deviation * deviation;
                residual_ss += residuals_[t][i] * residuals_[t][i];
            }
        }
        
        r_squared_ = 1.0 - (residual_ss / total_ss);
    }
    
    // Factor exposure analysis
    struct FactorExposure {
        std::vector<double> exposures;
        double tracking_error;
        double information_ratio;
    };
    
    FactorExposure calculatePortfolioExposure(const std::vector<double>& weights) {
        FactorExposure result;
        int K = factors_[0].size();
        result.exposures.resize(K, 0.0);
        
        // Portfolio factor exposures = weighted average of asset exposures
        for (int k = 0; k < K; ++k) {
            for (int i = 0; i < weights.size(); ++i) {
                result.exposures[k] += weights[i] * loadings_[i][k];
            }
        }
        
        // Simplified tracking error calculation
        auto factor_cov = calculateFactorCovariance();
        auto residual_cov = calculateResidualCovariance();
        
        double factor_variance = 0.0;
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                factor_variance += result.exposures[i] * factor_cov[i][j] * result.exposures[j];
            }
        }
        
        double residual_variance = 0.0;
        for (int i = 0; i < weights.size(); ++i) {
            residual_variance += weights[i] * weights[i] * residual_cov[i][i];
        }
        
        result.tracking_error = std::sqrt(factor_variance + residual_variance);
        
        // Information ratio (alpha / tracking error)
        double portfolio_alpha = 0.0;
        for (int i = 0; i < weights.size(); ++i) {
            portfolio_alpha += weights[i] * alpha_[i];
        }
        result.information_ratio = portfolio_alpha / result.tracking_error;
        
        return result;
    }
    
    std::vector<std::vector<double>> calculateFactorCovariance() {
        // Sample covariance of factor returns
        int T = factors_.size();
        int K = factors_[0].size();
        
        auto factor_means = MatrixUtils::colMeans(factors_);
        std::vector<std::vector<double>> cov(K, std::vector<double>(K, 0.0));
        
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                for (int t = 0; t < T; ++t) {
                    double dev_i = factors_[t][i] - factor_means[i];
                    double dev_j = factors_[t][j] - factor_means[j];
                    cov[i][j] += dev_i * dev_j;
                }
                cov[i][j] /= (T - 1);
            }
        }
        return cov;
    }
    
    std::vector<std::vector<double>> calculateResidualCovariance() {
        // Diagonal covariance matrix (assuming uncorrelated residuals)
        int N = residuals_[0].size();
        int T = residuals_.size();
        
        std::vector<std::vector<double>> cov(N, std::vector<double>(N, 0.0));
        for (int i = 0; i < N; ++i) {
            double variance = 0.0;
            for (int t = 0; t < T; ++t) {
                variance += residuals_[t][i] * residuals_[t][i];
            }
            cov[i][i] = variance / (T - 1);
        }
        return cov;
    }
    
    // Statistical arbitrage strategies
    struct PairsTradingSignal {
        double spread;
        double z_score;
        double half_life;
        bool is_stationary;
    };
    
    PairsTradingSignal analyzeCointegration(int asset1, int asset2) {
        int T = returns_.size();
        std::vector<double> returns1(T), returns2(T);
        
        for (int t = 0; t < T; ++t) {
            returns1[t] = returns_[t][asset1];
            returns2[t] = returns_[t][asset2];
        }
        
        // Calculate spread (simplified - should use proper cointegration)
        std::vector<double> spread(T);
        for (int t = 0; t < T; ++t) {
            spread[t] = returns1[t] - returns2[t];  // Simple spread
        }
        
        // Z-score
        double mean_spread = std::accumulate(spread.begin(), spread.end(), 0.0) / T;
        double var_spread = 0.0;
        for (double s : spread) {
            var_spread += (s - mean_spread) * (s - mean_spread);
        }
        var_spread /= (T - 1);
        double std_spread = std::sqrt(var_spread);
        double current_z = (spread.back() - mean_spread) / std_spread;
        
        // Half-life estimation (simplified)
        double beta = 0.9;  // Simplified AR(1) coefficient
        double half_life = -std::log(2.0) / std::log((beta < 0 ? -beta : beta));
        
        bool is_stationary = (beta < 0 ? -beta : beta) < 1.0;
        
        return {spread.back(), current_z, half_life, is_stationary};
    }
    
    // Performance attribution
    struct AttributionResult {
        std::map<std::string, double> factor_returns;
        double selection_return;
        double total_return;
    };
    
    AttributionResult performanceAttribution(const std::vector<double>& weights, 
                                           const std::vector<std::string>& factor_names) {
        AttributionResult result;
        
        // Portfolio return
        double total_return = 0.0;
        for (int t = 0; t < returns_.size(); ++t) {
            double portfolio_return = 0.0;
            for (int i = 0; i < weights.size(); ++i) {
                portfolio_return += weights[i] * returns_[t][i];
            }
            total_return += portfolio_return;
        }
        result.total_return = total_return;
        
        // Factor exposures
        std::vector<double> exposures(factors_[0].size(), 0.0);
        for (int k = 0; k < factors_[0].size(); ++k) {
            for (int i = 0; i < weights.size(); ++i) {
                exposures[k] += weights[i] * loadings_[i][k];
            }
        }
        
        // Factor returns contribution
        std::vector<double> factor_returns(factors_[0].size(), 0.0);
        for (int k = 0; k < factors_[0].size(); ++k) {
            for (int t = 0; t < factors_.size(); ++t) {
                factor_returns[k] += factors_[t][k];
            }
        }
        
        for (int i = 0; i < factor_names.size(); ++i) {
            result.factor_returns[factor_names[i]] = exposures[i] * factor_returns[i];
        }
        
        // Selection return (alpha + residual)
        double alpha_return = 0.0;
        for (int i = 0; i < weights.size(); ++i) {
            alpha_return += weights[i] * alpha_[i];
        }
        
        double residual_return = 0.0;
        for (int t = 0; t < residuals_.size(); ++t) {
            for (int i = 0; i < weights.size(); ++i) {
                residual_return += weights[i] * residuals_[t][i];
            }
        }
        
        result.selection_return = alpha_return + residual_return;
        
        return result;
    }
    
    // Getters
    const std::vector<std::vector<double>>& getLoadings() const { return loadings_; }
    const std::vector<double>& getAlpha() const { return alpha_; }
    const std::vector<std::vector<double>>& getResiduals() const { return residuals_; }
    double getRSquared() const { return r_squared_; }
};

// Example usage
int main() {
    std::cout << "APT Model Implementation\n";
    std::cout << "========================\n\n";
    
    // Generate sample data
    int T = 252;  // One year
    int N = 20;   // 20 assets
    int K = 3;    // 3 factors
    
    // Random factor returns
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> factor_dist(0.0, 0.02);
    std::normal_distribution<double> noise_dist(0.0, 0.01);
    
    std::vector<std::vector<double>> factors(T, std::vector<double>(K));
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            factors[t][k] = factor_dist(gen);
        }
    }
    
    // Generate asset returns based on factor model
    std::vector<std::vector<double>> true_loadings(N, std::vector<double>(K));
    std::vector<double> true_alpha(N);
    
    std::uniform_real_distribution<double> loading_dist(-1.0, 1.0);
    std::normal_distribution<double> alpha_dist(0.0, 0.001);
    
    for (int i = 0; i < N; ++i) {
        true_alpha[i] = alpha_dist(gen);
        for (int k = 0; k < K; ++k) {
            true_loadings[i][k] = loading_dist(gen);
        }
    }
    
    std::vector<std::vector<double>> returns(T, std::vector<double>(N));
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N; ++i) {
            returns[t][i] = true_alpha[i];
            for (int k = 0; k < K; ++k) {
                returns[t][i] += true_loadings[i][k] * factors[t][k];
            }
            returns[t][i] += noise_dist(gen);  // Add noise
        }
    }
    
    // Create APT model
    APTModel apt(returns, factors);
    
    std::cout << "APT Model Results:\n";
    std::cout << "==================\n";
    std::cout << "R-squared: " << std::fixed << std::setprecision(4) << apt.getRSquared() << "\n\n";
    
    // Equal-weighted portfolio analysis
    std::vector<double> equal_weights(N, 1.0/N);
    auto exposure = apt.calculatePortfolioExposure(equal_weights);
    
    std::cout << "Equal-Weighted Portfolio:\n";
    std::cout << "Factor Exposures: [";
    for (int i = 0; i < exposure.exposures.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << exposure.exposures[i];
        if (i < exposure.exposures.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "Tracking Error: " << exposure.tracking_error << "\n";
    std::cout << "Information Ratio: " << exposure.information_ratio << "\n\n";
    
    // Pairs trading analysis
    auto pairs_signal = apt.analyzeCointegration(0, 1);
    std::cout << "Pairs Trading Analysis (Asset 0 vs Asset 1):\n";
    std::cout << "Current Spread: " << std::fixed << std::setprecision(6) << pairs_signal.spread << "\n";
    std::cout << "Z-Score: " << pairs_signal.z_score << "\n";
    std::cout << "Half-Life: " << pairs_signal.half_life << " days\n";
    std::cout << "Is Stationary: " << (pairs_signal.is_stationary ? "Yes" : "No") << "\n";
    
    return 0;
}
