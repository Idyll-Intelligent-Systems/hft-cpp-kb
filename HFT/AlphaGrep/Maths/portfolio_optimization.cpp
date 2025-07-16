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
#include <Eigen/Dense>

using namespace Eigen;

class PortfolioOptimizer {
private:
    MatrixXd returns_;
    MatrixXd covariance_matrix_;
    VectorXd expected_returns_;
    VectorXd risk_free_rate_;
    
public:
    PortfolioOptimizer(const MatrixXd& returns, double rf_rate = 0.02) 
        : returns_(returns), risk_free_rate_(VectorXd::Constant(1, rf_rate)) {
        calculateStatistics();
    }
    
    void calculateStatistics() {
        int n_assets = returns_.cols();
        int n_periods = returns_.rows();
        
        // Calculate expected returns (mean)
        expected_returns_ = returns_.colwise().mean();
        
        // Calculate covariance matrix
        MatrixXd centered = returns_.rowwise() - expected_returns_.transpose();
        covariance_matrix_ = (centered.transpose() * centered) / (n_periods - 1);
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
        
        // Quadratic programming: minimize w'Σw subject to constraints
        // Using analytical solution for equality constraints
        
        MatrixXd A(2, n);
        A.row(0) = VectorXd::Ones(n);  // weights sum to 1
        A.row(1) = expected_returns_;   // target return constraint
        
        VectorXd b(2);
        b << 1.0, target_return;
        
        // Analytical solution: w = Σ^-1 * A' * (A * Σ^-1 * A')^-1 * b
        MatrixXd inv_cov = covariance_matrix_.inverse();
        MatrixXd temp = A * inv_cov * A.transpose();
        VectorXd weights = inv_cov * A.transpose() * temp.inverse() * b;
        
        OptimizationResult result;
        result.weights = weights;
        result.expected_return = weights.dot(expected_returns_);
        result.volatility = sqrt(weights.transpose() * covariance_matrix_ * weights);
        result.sharpe_ratio = (result.expected_return - risk_free_rate_(0)) / result.volatility;
        
        return result;
    }
    
    // Risk Parity Portfolio
    VectorXd riskParityOptimization(double tolerance = 1e-6, int max_iterations = 1000) {
        int n = expected_returns_.size();
        VectorXd weights = VectorXd::Constant(n, 1.0/n);  // Equal weights initialization
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            VectorXd risk_contrib = calculateRiskContribution(weights);
            double target_contrib = 1.0 / n;
            
            // Newton-Raphson iteration for risk parity
            VectorXd gradient = calculateRiskParityGradient(weights);
            MatrixXd hessian = calculateRiskParityHessian(weights);
            
            VectorXd delta_w = -hessian.inverse() * gradient;
            weights += delta_w;
            
            // Project onto simplex (weights sum to 1, non-negative)
            projectSimplex(weights);
            
            if (gradient.norm() < tolerance) break;
        }
        
        return weights;
    }
    
    VectorXd calculateRiskContribution(const VectorXd& weights) {
        VectorXd marginal_risk = covariance_matrix_ * weights;
        double portfolio_risk = sqrt(weights.transpose() * covariance_matrix_ * weights);
        return (weights.cwiseProduct(marginal_risk)) / portfolio_risk;
    }
    
    VectorXd calculateRiskParityGradient(const VectorXd& weights) {
        int n = weights.size();
        VectorXd risk_contrib = calculateRiskContribution(weights);
        double target_contrib = 1.0 / n;
        
        VectorXd gradient(n);
        for (int i = 0; i < n; ++i) {
            gradient(i) = 2 * (risk_contrib(i) - target_contrib);
        }
        return gradient;
    }
    
    MatrixXd calculateRiskParityHessian(const VectorXd& weights) {
        int n = weights.size();
        MatrixXd hessian = MatrixXd::Zero(n, n);
        
        // Approximate Hessian using finite differences
        double h = 1e-8;
        VectorXd grad_base = calculateRiskParityGradient(weights);
        
        for (int i = 0; i < n; ++i) {
            VectorXd weights_plus = weights;
            weights_plus(i) += h;
            VectorXd grad_plus = calculateRiskParityGradient(weights_plus);
            hessian.col(i) = (grad_plus - grad_base) / h;
        }
        
        return hessian;
    }
    
    void projectSimplex(VectorXd& weights) {
        // Project onto unit simplex: sum = 1, all >= 0
        weights = weights.cwiseMax(0.0);
        double sum_weights = weights.sum();
        if (sum_weights > 0) {
            weights /= sum_weights;
        } else {
            weights = VectorXd::Constant(weights.size(), 1.0/weights.size());
        }
    }
    
    // Efficient Frontier calculation
    std::vector<OptimizationResult> calculateEfficientFrontier(int num_points = 50) {
        std::vector<OptimizationResult> frontier;
        
        double min_return = expected_returns_.minCoeff();
        double max_return = expected_returns_.maxCoeff();
        
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
        double portfolio_return = weights.dot(expected_returns_);
        double portfolio_risk = sqrt(weights.transpose() * covariance_matrix_ * weights);
        return (portfolio_return - risk_free_rate_(0)) / portfolio_risk;
    }
    
    double calculateMaxDrawdown(const VectorXd& weights) {
        VectorXd portfolio_returns = returns_ * weights;
        VectorXd cumulative_returns = VectorXd::Zero(portfolio_returns.size());
        
        cumulative_returns(0) = portfolio_returns(0);
        for (int i = 1; i < portfolio_returns.size(); ++i) {
            cumulative_returns(i) = cumulative_returns(i-1) + portfolio_returns(i);
        }
        
        double max_drawdown = 0.0;
        double peak = cumulative_returns(0);
        
        for (int i = 1; i < cumulative_returns.size(); ++i) {
            if (cumulative_returns(i) > peak) {
                peak = cumulative_returns(i);
            }
            double drawdown = peak - cumulative_returns(i);
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
    
    MatrixXd returns = MatrixXd::Random(n_periods, n_assets) * 0.02;  // Random returns
    
    PortfolioOptimizer optimizer(returns);
    
    std::cout << "Portfolio Optimization Results:\n";
    std::cout << "================================\n\n";
    
    // Mean-Variance Optimization
    double target_return = 0.10;  // 10% annual return
    auto mv_result = optimizer.meanVarianceOptimization(target_return);
    
    std::cout << "Mean-Variance Optimization (Target Return: " << target_return << "):\n";
    std::cout << "Weights: " << mv_result.weights.transpose() << "\n";
    std::cout << "Expected Return: " << mv_result.expected_return << "\n";
    std::cout << "Volatility: " << mv_result.volatility << "\n";
    std::cout << "Sharpe Ratio: " << mv_result.sharpe_ratio << "\n\n";
    
    // Risk Parity Optimization
    VectorXd rp_weights = optimizer.riskParityOptimization();
    double rp_sharpe = optimizer.calculateSharpeRatio(rp_weights);
    
    std::cout << "Risk Parity Optimization:\n";
    std::cout << "Weights: " << rp_weights.transpose() << "\n";
    std::cout << "Sharpe Ratio: " << rp_sharpe << "\n";
    std::cout << "Max Drawdown: " << optimizer.calculateMaxDrawdown(rp_weights) << "\n\n";
    
    // Efficient Frontier
    auto frontier = optimizer.calculateEfficientFrontier(20);
    std::cout << "Efficient Frontier (first 5 points):\n";
    for (int i = 0; i < std::min(5, (int)frontier.size()); ++i) {
        std::cout << "Return: " << frontier[i].expected_return 
                  << ", Risk: " << frontier[i].volatility 
                  << ", Sharpe: " << frontier[i].sharpe_ratio << "\n";
    }
    
    return 0;
}
