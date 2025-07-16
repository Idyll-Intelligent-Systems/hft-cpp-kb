/*
Advanced Dynamic Programming: Matrix Chain Multiplication with Trading Applications
==================================================================================

Problem: Given a sequence of matrices A1, A2, ..., An, find the most efficient way to multiply 
these matrices together. The problem is not actually to perform the multiplications, but merely 
to decide in which order to perform the multiplications.

This problem is highly relevant in quantitative finance for:
- Portfolio optimization calculations
- Risk model computations
- Monte Carlo simulations
- Covariance matrix operations

Time Complexity: O(n³)
Space Complexity: O(n²)
*/

#include <iostream>
#include <vector>
#include <climits>
#include <string>
#include <sstream>
#include <iomanip>

class MatrixChainMultiplication {
private:
    std::vector<std::vector<int>> dp;
    std::vector<std::vector<int>> split;
    std::vector<int> dimensions;
    
public:
    // Standard matrix chain multiplication
    int matrixChainOrder(std::vector<int>& dims) {
        int n = dims.size() - 1; // Number of matrices
        dimensions = dims;
        
        // dp[i][j] = minimum cost to multiply matrices from i to j
        dp = std::vector<std::vector<int>>(n, std::vector<int>(n, 0));
        split = std::vector<std::vector<int>>(n, std::vector<int>(n, 0));
        
        // l is chain length
        for (int l = 2; l <= n; l++) {
            for (int i = 0; i <= n - l; i++) {
                int j = i + l - 1;
                dp[i][j] = INT_MAX;
                
                for (int k = i; k < j; k++) {
                    int cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1];
                    if (cost < dp[i][j]) {
                        dp[i][j] = cost;
                        split[i][j] = k;
                    }
                }
            }
        }
        
        return dp[0][n-1];
    }
    
    // Get optimal parenthesization
    std::string getOptimalParenthesization(int i, int j) {
        if (i == j) {
            return "M" + std::to_string(i);
        }
        
        int k = split[i][j];
        return "(" + getOptimalParenthesization(i, k) + " x " + 
               getOptimalParenthesization(k+1, j) + ")";
    }
    
    // Print the DP table for visualization
    void printDPTable() {
        int n = dp.size();
        std::cout << "\nDP Table (minimum multiplication costs):\n";
        std::cout << std::setw(6) << "";
        for (int j = 0; j < n; j++) {
            std::cout << std::setw(8) << j;
        }
        std::cout << "\n";
        
        for (int i = 0; i < n; i++) {
            std::cout << std::setw(4) << i << ": ";
            for (int j = 0; j < n; j++) {
                if (j >= i) {
                    std::cout << std::setw(8) << dp[i][j];
                } else {
                    std::cout << std::setw(8) << "-";
                }
            }
            std::cout << "\n";
        }
    }
};

// Trading-specific applications
class TradingMatrixOperations {
public:
    // Portfolio optimization: (w^T * Σ * w) where w is weights, Σ is covariance matrix
    static int optimizePortfolioCalculation(int num_assets, int num_scenarios) {
        // Matrices: weights(1 x n), covariance(n x n), weights_transposed(n x 1)
        // Plus scenario matrices for Monte Carlo
        std::vector<int> dims;
        
        // Weight vector: 1 x num_assets
        dims.push_back(1);
        dims.push_back(num_assets);
        
        // Covariance matrix: num_assets x num_assets
        dims.push_back(num_assets);
        
        // Weight vector transposed: num_assets x 1
        dims.push_back(1);
        
        // Add scenario matrices for Monte Carlo simulation
        for (int i = 0; i < num_scenarios; i++) {
            dims.push_back(num_assets);  // Each scenario is num_assets x num_assets
        }
        dims.push_back(1); // Final result vector
        
        MatrixChainMultiplication mcm;
        return mcm.matrixChainOrder(dims);
    }
    
    // Risk calculation: multiple factor models
    static int optimizeRiskFactorCalculation(int num_assets, int num_factors, int num_periods) {
        std::vector<int> dims;
        
        // Asset returns: num_periods x num_assets
        dims.push_back(num_periods);
        dims.push_back(num_assets);
        
        // Factor loadings: num_assets x num_factors  
        dims.push_back(num_factors);
        
        // Factor covariance: num_factors x num_factors
        dims.push_back(num_factors);
        
        // Factor loadings transposed: num_factors x num_assets
        dims.push_back(num_assets);
        
        // Portfolio weights: num_assets x 1
        dims.push_back(1);
        
        MatrixChainMultiplication mcm;
        return mcm.matrixChainOrder(dims);
    }
    
    // Options pricing: Monte Carlo with multiple underlyings
    static int optimizeOptionsPricingMC(int num_paths, int num_steps, int num_underlyings) {
        std::vector<int> dims;
        
        // Random numbers: num_paths x num_steps
        dims.push_back(num_paths);
        dims.push_back(num_steps);
        
        // Correlation matrix: num_steps x num_underlyings
        dims.push_back(num_underlyings);
        
        // Drift vector: num_underlyings x 1
        dims.push_back(1);
        
        // Volatility matrix: 1 x num_underlyings (for broadcasting)
        dims.push_back(num_underlyings);
        
        // Payoff calculation matrix: num_underlyings x num_paths
        dims.push_back(num_paths);
        
        // Discount factors: num_paths x 1
        dims.push_back(1);
        
        MatrixChainMultiplication mcm;
        return mcm.matrixChainOrder(dims);
    }
};

// Advanced DP variants for specific trading scenarios
class AdvancedMatrixChainDP {
public:
    // Matrix chain with different operation costs (useful for different data types)
    static int matrixChainWithCosts(std::vector<int>& dims, 
                                   std::vector<std::vector<double>>& operation_costs) {
        int n = dims.size() - 1;
        std::vector<std::vector<double>> dp(n, std::vector<double>(n, 0.0));
        
        for (int l = 2; l <= n; l++) {
            for (int i = 0; i <= n - l; i++) {
                int j = i + l - 1;
                dp[i][j] = DBL_MAX;
                
                for (int k = i; k < j; k++) {
                    double cost = dp[i][k] + dp[k+1][j] + 
                                 operation_costs[i][j] * dims[i] * dims[k+1] * dims[j+1];
                    dp[i][j] = std::min(dp[i][j], cost);
                }
            }
        }
        
        return static_cast<int>(dp[0][n-1]);
    }
    
    // Matrix chain with memory constraints (cache-aware optimization)
    static int matrixChainWithMemory(std::vector<int>& dims, int memory_limit) {
        int n = dims.size() - 1;
        std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));
        std::vector<std::vector<int>> memory_usage(n, std::vector<int>(n, 0));
        
        // Calculate memory usage for each subproblem
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                memory_usage[i][j] = dims[i] * dims[j+1]; // Memory for result matrix
            }
        }
        
        for (int l = 2; l <= n; l++) {
            for (int i = 0; i <= n - l; i++) {
                int j = i + l - 1;
                dp[i][j] = INT_MAX;
                
                for (int k = i; k < j; k++) {
                    int total_memory = memory_usage[i][k] + memory_usage[k+1][j] + 
                                      memory_usage[i][j];
                    
                    if (total_memory <= memory_limit) {
                        int cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1];
                        dp[i][j] = std::min(dp[i][j], cost);
                    }
                }
            }
        }
        
        return dp[0][n-1];
    }
};

// Performance analyzer for trading computations
class PerformanceAnalyzer {
public:
    static void analyzePortfolioOptimization() {
        std::cout << "\nPortfolio Optimization Analysis:\n";
        std::cout << "================================\n";
        
        std::vector<int> asset_counts = {10, 50, 100, 500, 1000};
        std::vector<int> scenario_counts = {1000, 5000, 10000};
        
        for (int assets : asset_counts) {
            for (int scenarios : scenario_counts) {
                int operations = TradingMatrixOperations::optimizePortfolioCalculation(assets, scenarios);
                std::cout << "Assets: " << std::setw(4) << assets 
                         << ", Scenarios: " << std::setw(5) << scenarios
                         << ", Operations: " << std::setw(12) << operations << "\n";
            }
        }
    }
    
    static void analyzeRiskCalculations() {
        std::cout << "\nRisk Factor Model Analysis:\n";
        std::cout << "===========================\n";
        
        std::vector<int> asset_counts = {100, 500, 1000};
        std::vector<int> factor_counts = {5, 10, 20, 50};
        std::vector<int> period_counts = {252, 1000, 2520}; // 1 year, ~4 years, 10 years
        
        for (int assets : asset_counts) {
            for (int factors : factor_counts) {
                for (int periods : period_counts) {
                    int operations = TradingMatrixOperations::optimizeRiskFactorCalculation(
                        assets, factors, periods);
                    std::cout << "Assets: " << std::setw(4) << assets 
                             << ", Factors: " << std::setw(3) << factors
                             << ", Periods: " << std::setw(4) << periods
                             << ", Operations: " << std::setw(12) << operations << "\n";
                }
            }
        }
    }
    
    static void compareOptimizationStrategies() {
        std::cout << "\nOptimization Strategy Comparison:\n";
        std::cout << "=================================\n";
        
        // Test case: 5 matrices with dimensions [1, 5, 4, 6, 2, 7]
        std::vector<int> dims = {1, 5, 4, 6, 2, 7};
        
        MatrixChainMultiplication mcm;
        int standard_cost = mcm.matrixChainOrder(dims);
        
        std::cout << "Standard DP cost: " << standard_cost << "\n";
        std::cout << "Optimal parenthesization: " << mcm.getOptimalParenthesization(0, 4) << "\n";
        
        mcm.printDPTable();
        
        // Test with operation costs (simulating different computational complexities)
        int n = dims.size() - 1;
        std::vector<std::vector<double>> costs(n, std::vector<double>(n, 1.0));
        
        // Higher cost for larger matrices (simulating cache misses)
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (dims[i] * dims[j+1] > 20) {
                    costs[i][j] = 1.5; // 50% penalty for large operations
                }
            }
        }
        
        int cost_aware = AdvancedMatrixChainDP::matrixChainWithCosts(dims, costs);
        std::cout << "\nCost-aware optimization: " << cost_aware << "\n";
        
        // Test with memory constraints
        int memory_constrained = AdvancedMatrixChainDP::matrixChainWithMemory(dims, 50);
        std::cout << "Memory-constrained optimization: " << memory_constrained << "\n";
    }
};

// Real-world example generator
class RealWorldExamples {
public:
    static void generatePortfolioExample() {
        std::cout << "\nReal-World Example: Portfolio Risk Calculation\n";
        std::cout << "==============================================\n";
        
        // Scenario: Calculate portfolio risk w^T * Σ * w for 100 assets
        std::vector<int> dims = {1, 100, 100, 1}; // w^T (1x100), Σ (100x100), w (100x1)
        
        MatrixChainMultiplication mcm;
        int cost = mcm.matrixChainOrder(dims);
        
        std::cout << "Portfolio risk calculation (100 assets):\n";
        std::cout << "Matrices: w^T(1x100) * Σ(100x100) * w(100x1)\n";
        std::cout << "Minimum operations: " << cost << "\n";
        std::cout << "Optimal order: " << mcm.getOptimalParenthesization(0, 2) << "\n";
        
        // Compare with suboptimal ordering
        int suboptimal_cost = 1*100*100 + 1*100*1; // (w^T * Σ) * w
        int optimal_cost = 100*100*1 + 1*100*1;     // w^T * (Σ * w)
        
        std::cout << "\nComparison:\n";
        std::cout << "Left-associative (w^T * Σ) * w: " << suboptimal_cost << " operations\n";
        std::cout << "Right-associative w^T * (Σ * w): " << optimal_cost << " operations\n";
        std::cout << "Savings: " << suboptimal_cost - optimal_cost << " operations ("
                 << std::fixed << std::setprecision(1) 
                 << 100.0 * (suboptimal_cost - optimal_cost) / suboptimal_cost 
                 << "% reduction)\n";
    }
};

int main() {
    std::cout << "Matrix Chain Multiplication for Trading Applications\n";
    std::cout << "===================================================\n";
    
    // Basic example
    std::vector<int> dimensions = {1, 2, 3, 4, 5};
    MatrixChainMultiplication mcm;
    
    int min_cost = mcm.matrixChainOrder(dimensions);
    std::cout << "Example: Matrices with dimensions [1x2], [2x3], [3x4], [4x5]\n";
    std::cout << "Minimum multiplication cost: " << min_cost << "\n";
    std::cout << "Optimal parenthesization: " << mcm.getOptimalParenthesization(0, 3) << "\n";
    
    mcm.printDPTable();
    
    // Trading applications
    PerformanceAnalyzer::analyzePortfolioOptimization();
    PerformanceAnalyzer::analyzeRiskCalculations();
    PerformanceAnalyzer::compareOptimizationStrategies();
    
    // Real-world example
    RealWorldExamples::generatePortfolioExample();
    
    return 0;
}
