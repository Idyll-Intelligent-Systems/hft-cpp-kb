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
#include <Eigen/Dense>
#include <map>
#include <string>

using namespace Eigen;
using namespace std;

class APTModel {
private:
    MatrixXd returns_;           // Asset returns (T x N)
    MatrixXd factors_;           // Factor returns (T x K)
    MatrixXd loadings_;          // Factor loadings (N x K)
    VectorXd alpha_;             // Asset alphas (N x 1)
    MatrixXd residuals_;         // Residual returns (T x N)
    double r_squared_;
    
public:
    APTModel(const MatrixXd& returns, const MatrixXd& factors) 
        : returns_(returns), factors_(factors) {
        estimateModel();
    }
    
    void estimateModel() {
        int T = returns_.rows();    // Time periods
        int N = returns_.cols();    // Number of assets
        int K = factors_.cols();    // Number of factors
        
        loadings_ = MatrixXd::Zero(N, K);
        alpha_ = VectorXd::Zero(N);
        residuals_ = MatrixXd::Zero(T, N);
        
        // For each asset, run regression: R_i = alpha_i + beta_i * F + epsilon_i
        for (int i = 0; i < N; ++i) {
            VectorXd asset_returns = returns_.col(i);
            
            // Add constant term for alpha
            MatrixXd X(T, K + 1);
            X.col(0) = VectorXd::Ones(T);
            X.rightCols(K) = factors_;
            
            // OLS regression: beta = (X'X)^-1 X'y
            VectorXd coefficients = (X.transpose() * X).inverse() * X.transpose() * asset_returns;
            
            alpha_(i) = coefficients(0);
            loadings_.row(i) = coefficients.tail(K);
            
            // Calculate residuals
            VectorXd fitted = X * coefficients;
            residuals_.col(i) = asset_returns - fitted;
        }
        
        // Calculate R-squared
        calculateRSquared();
    }
    
    void calculateRSquared() {
        double total_ss = 0.0;
        double residual_ss = 0.0;
        
        for (int i = 0; i < returns_.cols(); ++i) {
            VectorXd asset_returns = returns_.col(i);
            double mean_return = asset_returns.mean();
            
            total_ss += (asset_returns.array() - mean_return).square().sum();
            residual_ss += residuals_.col(i).array().square().sum();
        }
        
        r_squared_ = 1.0 - (residual_ss / total_ss);
    }
    
    // Factor exposure analysis
    struct FactorExposure {
        VectorXd exposures;
        double tracking_error;
        double information_ratio;
    };
    
    FactorExposure calculatePortfolioExposure(const VectorXd& weights) {
        FactorExposure result;
        
        // Portfolio factor exposures = weighted average of asset exposures
        result.exposures = loadings_.transpose() * weights;
        
        // Portfolio tracking error
        MatrixXd factor_cov = calculateFactorCovariance();
        MatrixXd residual_cov = calculateResidualCovariance();
        
        double factor_variance = result.exposures.transpose() * factor_cov * result.exposures;
        double residual_variance = weights.transpose() * residual_cov * weights;
        
        result.tracking_error = sqrt(factor_variance + residual_variance);
        
        // Information ratio (alpha / tracking error)
        double portfolio_alpha = alpha_.dot(weights);
        result.information_ratio = portfolio_alpha / result.tracking_error;
        
        return result;
    }
    
    MatrixXd calculateFactorCovariance() {
        // Sample covariance of factor returns
        int T = factors_.rows();
        MatrixXd centered = factors_.rowwise() - factors_.colwise().mean();
        return (centered.transpose() * centered) / (T - 1);
    }
    
    MatrixXd calculateResidualCovariance() {
        // Diagonal covariance matrix (assuming uncorrelated residuals)
        int N = residuals_.cols();
        int T = residuals_.rows();
        
        MatrixXd cov = MatrixXd::Zero(N, N);
        for (int i = 0; i < N; ++i) {
            double variance = residuals_.col(i).array().square().sum() / (T - 1);
            cov(i, i) = variance;
        }
        return cov;
    }
    
    // Factor-based portfolio optimization
    VectorXd factorNeutralPortfolio(const VectorXd& target_exposures) {
        int N = returns_.cols();
        int K = factors_.cols();
        
        // Minimize tracking error subject to factor constraints
        // min w'Σw subject to B'w = target_exposures and 1'w = 1
        
        MatrixXd residual_cov = calculateResidualCovariance();
        MatrixXd constraints(K + 1, N);
        constraints.topRows(K) = loadings_.transpose();
        constraints.bottomRows(1) = VectorXd::Ones(N).transpose();
        
        VectorXd targets(K + 1);
        targets.head(K) = target_exposures;
        targets(K) = 1.0;  // weights sum to 1
        
        // Analytical solution using Lagrange multipliers
        MatrixXd inv_cov = residual_cov.inverse();
        MatrixXd temp = constraints * inv_cov * constraints.transpose();
        VectorXd weights = inv_cov * constraints.transpose() * temp.inverse() * targets;
        
        return weights;
    }
    
    // Statistical arbitrage strategies
    struct PairsTradingSignal {
        double spread;
        double z_score;
        double half_life;
        bool is_stationary;
    };
    
    PairsTradingSignal analyzeCointegration(int asset1, int asset2) {
        VectorXd returns1 = returns_.col(asset1);
        VectorXd returns2 = returns_.col(asset2);
        
        // Calculate spread
        VectorXd spread = returns1 - returns2;
        
        // Z-score
        double mean_spread = spread.mean();
        double std_spread = sqrt((spread.array() - mean_spread).square().mean());
        double current_z = (spread(spread.size()-1) - mean_spread) / std_spread;
        
        // Half-life estimation using AR(1) model
        VectorXd lagged_spread = spread.head(spread.size()-1);
        VectorXd current_spread = spread.tail(spread.size()-1);
        
        double beta = (lagged_spread.array() * current_spread.array()).sum() / 
                     lagged_spread.array().square().sum();
        
        double half_life = -log(2.0) / log(beta);
        
        // Augmented Dickey-Fuller test (simplified)
        bool is_stationary = abs(beta) < 1.0;
        
        return {spread(spread.size()-1), current_z, half_life, is_stationary};
    }
    
    // Performance attribution
    struct AttributionResult {
        map<string, double> factor_returns;
        double selection_return;
        double total_return;
    };
    
    AttributionResult performanceAttribution(const VectorXd& weights, 
                                           const vector<string>& factor_names) {
        AttributionResult result;
        
        // Portfolio return
        VectorXd portfolio_returns = returns_ * weights;
        result.total_return = portfolio_returns.sum();
        
        // Factor exposures
        VectorXd exposures = loadings_.transpose() * weights;
        
        // Factor returns contribution
        VectorXd factor_returns = factors_.colwise().sum();
        for (int i = 0; i < factor_names.size(); ++i) {
            result.factor_returns[factor_names[i]] = exposures(i) * factor_returns(i);
        }
        
        // Selection return (alpha + residual)
        double alpha_return = alpha_.dot(weights);
        VectorXd residual_returns = residuals_ * weights;
        result.selection_return = alpha_return + residual_returns.sum();
        
        return result;
    }
    
    // Risk decomposition
    struct RiskDecomposition {
        map<string, double> factor_risk;
        double specific_risk;
        double total_risk;
    };
    
    RiskDecomposition decomposeRisk(const VectorXd& weights, 
                                   const vector<string>& factor_names) {
        RiskDecomposition result;
        
        VectorXd exposures = loadings_.transpose() * weights;
        MatrixXd factor_cov = calculateFactorCovariance();
        MatrixXd residual_cov = calculateResidualCovariance();
        
        // Factor risk contributions
        for (int i = 0; i < factor_names.size(); ++i) {
            double factor_var = exposures(i) * exposures(i) * factor_cov(i, i);
            result.factor_risk[factor_names[i]] = sqrt(factor_var);
        }
        
        // Specific risk
        double specific_var = weights.transpose() * residual_cov * weights;
        result.specific_risk = sqrt(specific_var);
        
        // Total risk
        double total_var = exposures.transpose() * factor_cov * exposures + specific_var;
        result.total_risk = sqrt(total_var);
        
        return result;
    }
    
    // Getters
    const MatrixXd& getLoadings() const { return loadings_; }
    const VectorXd& getAlpha() const { return alpha_; }
    const MatrixXd& getResiduals() const { return residuals_; }
    double getRSquared() const { return r_squared_; }
};

// Factor construction utilities
class FactorConstructor {
public:
    // Momentum factor
    static VectorXd constructMomentumFactor(const MatrixXd& prices, int lookback = 12) {
        int T = prices.rows();
        VectorXd momentum = VectorXd::Zero(T);
        
        for (int t = lookback; t < T; ++t) {
            VectorXd current_prices = prices.row(t);
            VectorXd past_prices = prices.row(t - lookback);
            VectorXd returns = (current_prices.array() / past_prices.array()).log();
            momentum(t) = returns.mean();
        }
        
        return momentum;
    }
    
    // Value factor (P/B ratio based)
    static VectorXd constructValueFactor(const MatrixXd& prices, 
                                        const VectorXd& book_values) {
        int T = prices.rows();
        VectorXd value_factor = VectorXd::Zero(T);
        
        for (int t = 0; t < T; ++t) {
            VectorXd pb_ratios = prices.row(t).array() / book_values.array();
            value_factor(t) = -pb_ratios.mean();  // Negative because low P/B is value
        }
        
        return value_factor;
    }
    
    // Size factor
    static VectorXd constructSizeFactor(const MatrixXd& market_caps) {
        int T = market_caps.rows();
        VectorXd size_factor = VectorXd::Zero(T);
        
        for (int t = 0; t < T; ++t) {
            VectorXd log_caps = market_caps.row(t).array().log();
            size_factor(t) = -log_caps.mean();  // Negative because small cap outperforms
        }
        
        return size_factor;
    }
};

// Example usage
int main() {
    // Generate sample data
    int T = 252;  // One year
    int N = 20;   // 20 assets
    int K = 3;    // 3 factors
    
    // Random factor returns
    MatrixXd factors = MatrixXd::Random(T, K) * 0.02;
    
    // Generate asset returns based on factor model
    MatrixXd true_loadings = MatrixXd::Random(N, K);
    VectorXd true_alpha = VectorXd::Random(N) * 0.001;
    MatrixXd noise = MatrixXd::Random(T, N) * 0.01;
    
    MatrixXd returns = factors * true_loadings.transpose() + 
                      VectorXd::Ones(T) * true_alpha.transpose() + noise;
    
    // Create APT model
    APTModel apt(returns, factors);
    
    cout << "APT Model Results:\n";
    cout << "==================\n";
    cout << "R-squared: " << apt.getRSquared() << "\n\n";
    
    // Equal-weighted portfolio analysis
    VectorXd equal_weights = VectorXd::Constant(N, 1.0/N);
    auto exposure = apt.calculatePortfolioExposure(equal_weights);
    
    cout << "Equal-Weighted Portfolio:\n";
    cout << "Factor Exposures: " << exposure.exposures.transpose() << "\n";
    cout << "Tracking Error: " << exposure.tracking_error << "\n";
    cout << "Information Ratio: " << exposure.information_ratio << "\n\n";
    
    // Factor-neutral portfolio
    VectorXd zero_exposures = VectorXd::Zero(K);
    VectorXd neutral_weights = apt.factorNeutralPortfolio(zero_exposures);
    
    cout << "Factor-Neutral Portfolio:\n";
    cout << "Weights: " << neutral_weights.transpose() << "\n";
    
    auto neutral_exposure = apt.calculatePortfolioExposure(neutral_weights);
    cout << "Factor Exposures: " << neutral_exposure.exposures.transpose() << "\n";
    
    // Pairs trading analysis
    auto pairs_signal = apt.analyzeCointegration(0, 1);
    cout << "\nPairs Trading Analysis (Asset 0 vs Asset 1):\n";
    cout << "Current Spread: " << pairs_signal.spread << "\n";
    cout << "Z-Score: " << pairs_signal.z_score << "\n";
    cout << "Half-Life: " << pairs_signal.half_life << " days\n";
    cout << "Is Stationary: " << (pairs_signal.is_stationary ? "Yes" : "No") << "\n";
    
    return 0;
}
