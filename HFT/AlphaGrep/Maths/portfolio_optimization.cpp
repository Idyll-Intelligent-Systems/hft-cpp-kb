/*
AlphaGrep HFT Mathematics Problem: Portfolio Optimization
========================================================

Problem Statement:
Implement Mean-Variance Optimization (Markowitz Model) and Risk Parity algorithms
for portfolio construction. Include real-time rebalancing capabilities.

This tests:
1. Modern Portfolio Theory implementation
2. Optimization algorithms (quadratic programming)
3. Risk management
4. Real-time computation efficiency

Applications:
- Multi-asset portfolio management
- Risk-adjusted return maximization
- Dynamic hedging strategies
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <random>
#include <iomanip>

// Simple Matrix class to replace Eigen dependency
class Matrix {
private:
    std::vector<std::vector<double>> data_;
    size_t rows_, cols_;

public:
    Matrix(size_t rows, size_t cols, double init_val = 0.0) 
        : rows_(rows), cols_(cols), data_(rows, std::vector<double>(cols, init_val)) {}
    
    Matrix(const std::vector<std::vector<double>>& data) 
        : rows_(data.size()), cols_(data.empty() ? 0 : data[0].size()), data_(data) {}
    
    double& operator()(size_t i, size_t j) { return data_[i][j]; }
    const double& operator()(size_t i, size_t j) const { return data_[i][j]; }
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
    std::vector<double> row(size_t i) const { return data_[i]; }
    std::vector<double> col(size_t j) const {
        std::vector<double> result(rows_);
        for (size_t i = 0; i < rows_; ++i) result[i] = data_[i][j];
        return result;
    }
    
    Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(j, i) = data_[i][j];
            }
        }
        return result;
    }
    
    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) throw std::invalid_argument("Matrix dimensions don't match");
        Matrix result(rows_, other.cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.cols_; ++j) {
                for (size_t k = 0; k < cols_; ++k) {
                    result(i, j) += data_[i][k] * other.data_[k][j];
                }
            }
        }
        return result;
    }
    
    std::vector<double> operator*(const std::vector<double>& vec) const {
        if (cols_ != vec.size()) throw std::invalid_argument("Matrix-vector dimensions don't match");
        std::vector<double> result(rows_, 0.0);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result[i] += data_[i][j] * vec[j];
            }
        }
        return result;
    }
    
    Matrix inverse() const {
        if (rows_ != cols_) throw std::invalid_argument("Matrix must be square");
        size_t n = rows_;
        Matrix augmented(n, 2 * n);
        
        // Create augmented matrix [A|I]
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                augmented(i, j) = data_[i][j];
                augmented(i, j + n) = (i == j) ? 1.0 : 0.0;
            }
        }
        
        // Gaussian elimination
        for (size_t i = 0; i < n; ++i) {
            // Find pivot
            size_t max_row = i;
            for (size_t k = i + 1; k < n; ++k) {
                if (std::abs(augmented(k, i)) > std::abs(augmented(max_row, i))) {
                    max_row = k;
                }
            }
            std::swap(augmented.data_[i], augmented.data_[max_row]);
            
            // Make diagonal element 1
            double pivot = augmented(i, i);
            if (std::abs(pivot) < 1e-10) throw std::runtime_error("Matrix is singular");
            for (size_t j = 0; j < 2 * n; ++j) {
                augmented(i, j) /= pivot;
            }
            
            // Eliminate column
            for (size_t k = 0; k < n; ++k) {
                if (k != i) {
                    double factor = augmented(k, i);
                    for (size_t j = 0; j < 2 * n; ++j) {
                        augmented(k, j) -= factor * augmented(i, j);
                    }
                }
            }
        }
        
        // Extract inverse
        Matrix result(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                result(i, j) = augmented(i, j + n);
            }
        }
        return result;
    }
    
    std::vector<double> colwise_mean() const {
        std::vector<double> means(cols_, 0.0);
        for (size_t j = 0; j < cols_; ++j) {
            for (size_t i = 0; i < rows_; ++i) {
                means[j] += data_[i][j];
            }
            means[j] /= rows_;
        }
        return means;
    }
    
    std::vector<double> rowwise_mean() const {
        std::vector<double> means(rows_, 0.0);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                means[i] += data_[i][j];
            }
            means[i] /= cols_;
        }
        return means;
    }
};

using MatrixXd = Matrix;
using VectorXd = std::vector<double>;

class PortfolioOptimizer {
private:
    MatrixXd returns_;
    MatrixXd covariance_matrix_;
    VectorXd expected_returns_;
    double risk_free_rate_;
    
public:
    PortfolioOptimizer(const MatrixXd& returns, double rf_rate = 0.02) 
        : returns_(returns), risk_free_rate_(rf_rate) {
        calculateStatistics();
    }
    
    void calculateStatistics() {
        int n_assets = returns_.cols();
        int n_periods = returns_.rows();
        
        // Calculate expected returns (mean)
        expected_returns_ = returns_.colwise_mean();
        
        // Calculate covariance matrix
        covariance_matrix_ = MatrixXd(n_assets, n_assets);
        for (int i = 0; i < n_assets; ++i) {
            auto col_i = returns_.col(i);
            double mean_i = expected_returns_[i];
            for (int j = 0; j < n_assets; ++j) {
                auto col_j = returns_.col(j);
                double mean_j = expected_returns_[j];
                double covariance = 0.0;
                for (int t = 0; t < n_periods; ++t) {
                    covariance += (col_i[t] - mean_i) * (col_j[t] - mean_j);
                }
                covariance_matrix_(i, j) = covariance / (n_periods - 1);
            }
        }
    }
    
    // Markowitz Mean-Variance Optimization
    struct OptimizationResult {
        VectorXd weights;
        double expected_return;
        double volatility;
        double sharpe_ratio;
    };
    
    OptimizationResult meanVarianceOptimization(double target_return) {
        int n = expected_returns_.size();
        
        // Simplified analytical solution for two-constraint optimization
        // This is a basic implementation - for production use, consider numerical optimization
        
        // Equal weights as starting point
        VectorXd weights(n, 1.0 / n);
        
        // Simple gradient descent for demonstration
        double learning_rate = 0.01;
        int max_iterations = 1000;
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            // Calculate current portfolio return and risk
            double portfolio_return = 0.0;
            for (int i = 0; i < n; ++i) {
                portfolio_return += weights[i] * expected_returns_[i];
            }
            
            // Adjust weights to match target return
            double return_diff = target_return - portfolio_return;
            for (int i = 0; i < n; ++i) {
                weights[i] += learning_rate * return_diff * expected_returns_[i];
            }
            
            // Normalize weights to sum to 1
            double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
            for (int i = 0; i < n; ++i) {
                weights[i] /= sum_weights;
            }
            
            if (std::abs(return_diff) < 1e-6) break;
        }
        
        OptimizationResult result;
        result.weights = weights;
        result.expected_return = 0.0;
        for (int i = 0; i < n; ++i) {
            result.expected_return += weights[i] * expected_returns_[i];
        }
        
        // Calculate portfolio volatility
        auto cov_weights = covariance_matrix_ * weights;
        result.volatility = 0.0;
        for (int i = 0; i < n; ++i) {
            result.volatility += weights[i] * cov_weights[i];
        }
        result.volatility = std::sqrt(result.volatility);
        result.sharpe_ratio = (result.expected_return - risk_free_rate_) / result.volatility;
        
        return result;
    }
    
    // Risk Parity Portfolio (Simplified Implementation)
    VectorXd riskParityOptimization(double tolerance = 1e-6, int max_iterations = 1000) {
        int n = expected_returns_.size();
        VectorXd weights(n, 1.0/n);  // Equal weights initialization
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            VectorXd risk_contrib = calculateRiskContribution(weights);
            double target_contrib = 1.0 / n;
            
            // Simple iterative adjustment
            bool converged = true;
            for (int i = 0; i < n; ++i) {
                double diff = risk_contrib[i] - target_contrib;
                if (std::abs(diff) > tolerance) {
                    converged = false;
                    weights[i] *= (1.0 - 0.1 * diff); // Simple adjustment
                }
            }
            
            // Project onto simplex (weights sum to 1, non-negative)
            projectSimplex(weights);
            
            if (converged) break;
        }
        
        return weights;
    }
    
    VectorXd calculateRiskContribution(const VectorXd& weights) {
        auto marginal_risk = covariance_matrix_ * weights;
        double portfolio_variance = 0.0;
        for (int i = 0; i < weights.size(); ++i) {
            portfolio_variance += weights[i] * marginal_risk[i];
        }
        double portfolio_risk = std::sqrt(portfolio_variance);
        
        VectorXd risk_contrib(weights.size());
        for (int i = 0; i < weights.size(); ++i) {
            risk_contrib[i] = (weights[i] * marginal_risk[i]) / portfolio_risk;
        }
        return risk_contrib;
    }
    
    void projectSimplex(VectorXd& weights) {
        // Project onto unit simplex: sum = 1, all >= 0
        for (double& w : weights) {
            w = std::max(0.0, w);
        }
        double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (sum_weights > 0) {
            for (double& w : weights) {
                w /= sum_weights;
            }
        } else {
            std::fill(weights.begin(), weights.end(), 1.0/weights.size());
        }
    }
    
    // Efficient Frontier calculation
    std::vector<OptimizationResult> calculateEfficientFrontier(int num_points = 50) {
        std::vector<OptimizationResult> frontier;
        
        double min_return = *std::min_element(expected_returns_.begin(), expected_returns_.end());
        double max_return = *std::max_element(expected_returns_.begin(), expected_returns_.end());
        
        for (int i = 0; i < num_points; ++i) {
            double target_return = min_return + i * (max_return - min_return) / (num_points - 1);
            try {
                OptimizationResult result = meanVarianceOptimization(target_return);
                frontier.push_back(result);
            } catch (...) {
                // Skip if optimization fails for this target return
                continue;
            }
        }
        
        return frontier;
    }
    
    // Performance metrics
    double calculateSharpeRatio(const VectorXd& weights) {
        double portfolio_return = 0.0;
        for (int i = 0; i < weights.size(); ++i) {
            portfolio_return += weights[i] * expected_returns_[i];
        }
        auto cov_weights = covariance_matrix_ * weights;
        double portfolio_variance = 0.0;
        for (int i = 0; i < weights.size(); ++i) {
            portfolio_variance += weights[i] * cov_weights[i];
        }
        double portfolio_risk = std::sqrt(portfolio_variance);
        return (portfolio_return - risk_free_rate_) / portfolio_risk;
    }
    
    double calculateMaxDrawdown(const VectorXd& weights) {
        VectorXd portfolio_returns(returns_.rows());
        for (int t = 0; t < returns_.rows(); ++t) {
            portfolio_returns[t] = 0.0;
            for (int i = 0; i < returns_.cols(); ++i) {
                portfolio_returns[t] += weights[i] * returns_(t, i);
            }
        }
        
        VectorXd cumulative_returns(portfolio_returns.size());
        cumulative_returns[0] = portfolio_returns[0];
        for (int i = 1; i < portfolio_returns.size(); ++i) {
            cumulative_returns[i] = cumulative_returns[i-1] + portfolio_returns[i];
        }
        
        double max_drawdown = 0.0;
        double peak = cumulative_returns[0];
        
        for (int i = 1; i < cumulative_returns.size(); ++i) {
            if (cumulative_returns[i] > peak) {
                peak = cumulative_returns[i];
            }
            double drawdown = peak - cumulative_returns[i];
            max_drawdown = std::max(max_drawdown, drawdown);
        }
        
        return max_drawdown;
    }
};

// Example usage and testing
int main() {
    // Generate sample return data
    int n_assets = 5;
    int n_periods = 252;  // One year of daily returns
    
    // Create sample returns matrix
    std::vector<std::vector<double>> return_data(n_periods, std::vector<double>(n_assets));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.02);
    
    for (int t = 0; t < n_periods; ++t) {
        for (int i = 0; i < n_assets; ++i) {
            return_data[t][i] = dist(gen);
        }
    }
    
    MatrixXd returns(return_data);
    PortfolioOptimizer optimizer(returns);
    
    std::cout << "Portfolio Optimization Results:\n";
    std::cout << "================================\n\n";
    
    // Mean-Variance Optimization
    double target_return = 0.10;  // 10% annual return
    auto mv_result = optimizer.meanVarianceOptimization(target_return);
    
    std::cout << "Mean-Variance Optimization (Target Return: " << target_return << "):\n";
    std::cout << "Weights: [";
    for (int i = 0; i < mv_result.weights.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << mv_result.weights[i];
        if (i < mv_result.weights.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "Expected Return: " << mv_result.expected_return << "\n";
    std::cout << "Volatility: " << mv_result.volatility << "\n";
    std::cout << "Sharpe Ratio: " << mv_result.sharpe_ratio << "\n\n";
    
    // Risk Parity Optimization
    VectorXd rp_weights = optimizer.riskParityOptimization();
    double rp_sharpe = optimizer.calculateSharpeRatio(rp_weights);
    
    std::cout << "Risk Parity Optimization:\n";
    std::cout << "Weights: [";
    for (int i = 0; i < rp_weights.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << rp_weights[i];
        if (i < rp_weights.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "Sharpe Ratio: " << rp_sharpe << "\n";
    std::cout << "Max Drawdown: " << optimizer.calculateMaxDrawdown(rp_weights) << "\n\n";
    
    // Efficient Frontier
    auto frontier = optimizer.calculateEfficientFrontier(20);
    std::cout << "Efficient Frontier (first 5 points):\n";
    for (int i = 0; i < std::min(5, (int)frontier.size()); ++i) {
        std::cout << "Return: " << std::fixed << std::setprecision(4) << frontier[i].expected_return 
                  << ", Risk: " << frontier[i].volatility 
                  << ", Sharpe: " << frontier[i].sharpe_ratio << "\n";
    }
    
    return 0;
}
