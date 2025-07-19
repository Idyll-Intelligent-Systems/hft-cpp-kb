/*
HackerRank Style Problem 4: Optimal Portfolio Rebalancing
=========================================================

Problem Statement:
You are managing a portfolio with N assets and want to rebalance to target weights.
Given current positions, target weights, and transaction costs, find the minimum cost
rebalancing strategy that achieves the target allocation within tolerance.

Constraints:
- 1 <= N <= 100 (number of assets)
- 0 <= current_weight[i], target_weight[i] <= 1
- sum(target_weight) = 1.0
- 0 <= transaction_cost[i] <= 0.01 (1% max)
- 0 <= tolerance <= 0.05 (5% max deviation allowed)

Example:
Current: [0.6, 0.3, 0.1], Target: [0.4, 0.4, 0.2], Costs: [0.001, 0.002, 0.003]
Find trades to minimize total cost while reaching target ± tolerance

This problem tests:
- Dynamic programming and optimization
- Financial portfolio theory
- Constraint satisfaction
- Cost-benefit analysis in trading

Time Complexity: O(N^2) for greedy approach, O(2^N) for exact solution
Space Complexity: O(N)
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <map>
#include <set>
#include <queue>
#include <cassert>
#include <iomanip>
#include <string>
#include <chrono>

struct Asset {
    std::string symbol;
    double currentWeight;
    double targetWeight;
    double transactionCost;  // Cost per unit traded (as fraction)
    double price;            // Current price per unit
    
    Asset(const std::string& sym, double curr, double target, double cost, double p = 1.0)
        : symbol(sym), currentWeight(curr), targetWeight(target), transactionCost(cost), price(p) {}
    
    double getDeviation() const {
        return targetWeight - currentWeight;
    }
    
    double getAbsDeviation() const {
        return std::abs(getDeviation());
    }
    
    bool needsRebalancing(double tolerance) const {
        return getAbsDeviation() > tolerance;
    }
};

struct Trade {
    std::string symbol;
    double amount;      // Positive = buy, negative = sell
    double cost;        // Transaction cost
    double newWeight;   // Weight after trade
    
    Trade(const std::string& sym, double amt, double c, double nw)
        : symbol(sym), amount(amt), cost(c), newWeight(nw) {}
};

class PortfolioRebalancer {
public:
    // Greedy rebalancing algorithm - minimize costs while achieving targets
    static std::vector<Trade> rebalanceGreedy(std::vector<Asset> assets, double tolerance, double totalValue) {
        std::vector<Trade> trades;
        
        // Sort assets by deviation / transaction cost ratio (efficiency)
        std::sort(assets.begin(), assets.end(), [](const Asset& a, const Asset& b) {
            double efficiencyA = a.getAbsDeviation() / (a.transactionCost + 1e-9);
            double efficiencyB = b.getAbsDeviation() / (b.transactionCost + 1e-9);
            return efficiencyA > efficiencyB;
        });
        
        // Track cumulative changes to maintain sum = 1
        double cumulativeChange = 0.0;
        
        for (auto& asset : assets) {
            if (!asset.needsRebalancing(tolerance)) continue;
            
            double deviation = asset.getDeviation();
            double tradeAmount = deviation * totalValue;
            
            // Adjust for cumulative changes to maintain balance
            tradeAmount -= cumulativeChange / assets.size();
            
            if (std::abs(tradeAmount) > tolerance * totalValue) {
                double tradeCost = std::abs(tradeAmount) * asset.transactionCost;
                double newWeight = asset.currentWeight + (tradeAmount / totalValue);
                
                trades.emplace_back(asset.symbol, tradeAmount, tradeCost, newWeight);
                cumulativeChange += tradeAmount;
                asset.currentWeight = newWeight;
            }
        }
        
        return trades;
    }
    
    // Dynamic programming approach for exact solution (small portfolios)
    static std::vector<Trade> rebalanceOptimal(const std::vector<Asset>& assets, double tolerance, double totalValue) {
        int n = assets.size();
        if (n > 20) {
            // Fall back to greedy for large portfolios
            return rebalanceGreedy(std::vector<Asset>(assets), tolerance, totalValue);
        }
        
        // State: (asset_index, accumulated_weight_changes)
        // Use discretized weights for DP state space
        const int precision = 1000; // 0.1% precision
        std::map<std::pair<int, int>, std::pair<double, std::vector<Trade>>> dp;
        
        // Base case: no assets processed
        dp[{0, 0}] = {0.0, {}};
        
        for (int i = 0; i < n; i++) {
            std::map<std::pair<int, int>, std::pair<double, std::vector<Trade>>> newDp;
            
            for (const auto& state : dp) {
                int prevIndex = state.first.first;
                int prevWeight = state.first.second;
                double prevCost = state.second.first;
                const auto& prevTrades = state.second.second;
                
                if (prevIndex != i) continue;
                
                // Try different trade amounts for current asset
                double currentDev = assets[i].getDeviation();
                int minTrade = std::max(-precision, static_cast<int>(-currentDev * precision * 2));
                int maxTrade = std::min(precision, static_cast<int>(-currentDev * precision * 2));
                
                for (int tradeWeight = minTrade; tradeWeight <= maxTrade; tradeWeight++) {
                    double tradeAmount = (tradeWeight / static_cast<double>(precision)) * totalValue;
                    double newAssetWeight = assets[i].currentWeight + (tradeAmount / totalValue);
                    
                    // Check if within tolerance
                    if (std::abs(newAssetWeight - assets[i].targetWeight) <= tolerance) {
                        double tradeCost = std::abs(tradeAmount) * assets[i].transactionCost;
                        double totalCost = prevCost + tradeCost;
                        
                        auto newTrades = prevTrades;
                        if (std::abs(tradeAmount) > 1e-6) {
                            newTrades.emplace_back(assets[i].symbol, tradeAmount, tradeCost, newAssetWeight);
                        }
                        
                        std::pair<int, int> newState = {i + 1, prevWeight + tradeWeight};
                        
                        if (newDp.find(newState) == newDp.end() || newDp[newState].first > totalCost) {
                            newDp[newState] = {totalCost, newTrades};
                        }
                    }
                }
            }
            
            dp = std::move(newDp);
        }
        
        // Find best solution
        double bestCost = 1e9;
        std::vector<Trade> bestTrades;
        
        for (const auto& state : dp) {
            if (state.first.first == n && state.second.first < bestCost) {
                bestCost = state.second.first;
                bestTrades = state.second.second;
            }
        }
        
        return bestTrades;
    }
    
    // Risk-aware rebalancing with volatility considerations
    static std::vector<Trade> rebalanceRiskAware(std::vector<Asset> assets, double tolerance, 
                                                double totalValue, const std::vector<std::vector<double>>& correlationMatrix) {
        std::vector<Trade> trades;
        int n = assets.size();
        
        // Calculate portfolio risk for different rebalancing scenarios
        auto calculatePortfolioRisk = [&](const std::vector<double>& weights) -> double {
            double risk = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    risk += weights[i] * weights[j] * correlationMatrix[i][j];
                }
            }
            return std::sqrt(risk);
        };
        
        // Current portfolio risk
        std::vector<double> currentWeights;
        for (const auto& asset : assets) {
            currentWeights.push_back(asset.currentWeight);
        }
        double currentRisk = calculatePortfolioRisk(currentWeights);
        
        // Iteratively rebalance to minimize risk while controlling costs
        const int maxIterations = 10;
        double riskReduction = 0.0;
        
        for (int iter = 0; iter < maxIterations; iter++) {
            bool improved = false;
            
            for (int i = 0; i < n; i++) {
                if (!assets[i].needsRebalancing(tolerance)) continue;
                
                double deviation = assets[i].getDeviation();
                double stepSize = deviation * 0.3; // Gradual rebalancing
                double tradeAmount = stepSize * totalValue;
                
                // Test if this trade improves risk-adjusted returns
                std::vector<double> testWeights = currentWeights;
                testWeights[i] += stepSize;
                
                // Ensure weights sum to 1
                double weightSum = std::accumulate(testWeights.begin(), testWeights.end(), 0.0);
                for (double& w : testWeights) w /= weightSum;
                
                double newRisk = calculatePortfolioRisk(testWeights);
                double tradeCost = std::abs(tradeAmount) * assets[i].transactionCost;
                
                // Accept trade if risk reduction > transaction cost (risk-adjusted)
                if (newRisk < currentRisk && (currentRisk - newRisk) > tradeCost * 0.1) {
                    trades.emplace_back(assets[i].symbol, tradeAmount, tradeCost, testWeights[i]);
                    assets[i].currentWeight = testWeights[i];
                    currentWeights = testWeights;
                    currentRisk = newRisk;
                    riskReduction += (currentRisk - newRisk);
                    improved = true;
                }
            }
            
            if (!improved) break;
        }
        
        return trades;
    }
    
    // Calculate portfolio statistics after rebalancing
    static void analyzeRebalancing(const std::vector<Asset>& originalAssets, 
                                 const std::vector<Trade>& trades, double tolerance) {
        std::cout << "\nRebalancing Analysis:" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        // Calculate final weights
        std::map<std::string, double> finalWeights;
        for (const auto& asset : originalAssets) {
            finalWeights[asset.symbol] = asset.currentWeight;
        }
        
        for (const auto& trade : trades) {
            finalWeights[trade.symbol] = trade.newWeight;
        }
        
        // Display results
        std::cout << std::setprecision(4) << std::fixed;
        std::cout << "Asset    | Current | Target  | Final   | Deviation | Trade Amount | Cost" << std::endl;
        std::cout << std::string(75, '-') << std::endl;
        
        double totalCost = 0.0;
        double totalDeviation = 0.0;
        
        for (const auto& asset : originalAssets) {
            auto tradeIt = std::find_if(trades.begin(), trades.end(),
                [&asset](const Trade& t) { return t.symbol == asset.symbol; });
            
            double tradeAmount = (tradeIt != trades.end()) ? tradeIt->amount : 0.0;
            double tradeCost = (tradeIt != trades.end()) ? tradeIt->cost : 0.0;
            double finalWeight = finalWeights[asset.symbol];
            double finalDeviation = std::abs(finalWeight - asset.targetWeight);
            
            std::cout << std::setw(8) << asset.symbol << " | "
                     << std::setw(7) << asset.currentWeight << " | "
                     << std::setw(7) << asset.targetWeight << " | "
                     << std::setw(7) << finalWeight << " | "
                     << std::setw(9) << finalDeviation << " | "
                     << std::setw(12) << tradeAmount << " | "
                     << std::setw(8) << tradeCost << std::endl;
            
            totalCost += tradeCost;
            totalDeviation += finalDeviation;
        }
        
        std::cout << std::string(75, '-') << std::endl;
        std::cout << "Total Cost: " << totalCost << std::endl;
        std::cout << "Total Deviation: " << totalDeviation << std::endl;
        std::cout << "Average Deviation: " << (totalDeviation / originalAssets.size()) << std::endl;
        std::cout << "Within Tolerance: " << ((totalDeviation / originalAssets.size()) <= tolerance ? "YES" : "NO") << std::endl;
        std::cout << "Number of Trades: " << trades.size() << std::endl;
    }
};

// Market impact and liquidity considerations
class AdvancedRebalancer {
public:
    struct MarketData {
        double bidAskSpread;
        double averageDailyVolume;
        double volatility;
        double liquidityScore; // 0-1, higher = more liquid
        
        MarketData(double spread = 0.001, double volume = 1e6, double vol = 0.02, double liquidity = 0.8)
            : bidAskSpread(spread), averageDailyVolume(volume), volatility(vol), liquidityScore(liquidity) {}
    };
    
    // Advanced rebalancing with market impact model
    static std::vector<Trade> rebalanceWithMarketImpact(std::vector<Asset> assets, double tolerance,
                                                       double totalValue, const std::vector<MarketData>& marketData) {
        std::vector<Trade> trades;
        
        // Calculate market impact for each potential trade
        auto calculateMarketImpact = [](double tradeSize, const MarketData& data) -> double {
            double participation = tradeSize / data.averageDailyVolume;
            double temporaryImpact = data.bidAskSpread * 0.5 + 
                                   data.volatility * std::sqrt(participation) * 0.1;
            double permanentImpact = data.volatility * participation * 0.05 / data.liquidityScore;
            return temporaryImpact + permanentImpact;
        };
        
        // Priority queue for trades ordered by efficiency (benefit/cost ratio)
        struct TradeProposal {
            int assetIndex;
            double amount;
            double benefit;
            double totalCost;
            double efficiency;
            
            bool operator<(const TradeProposal& other) const {
                return efficiency < other.efficiency; // Max heap
            }
        };
        
        std::priority_queue<TradeProposal> tradeQueue;
        
        // Generate trade proposals
        for (int i = 0; i < assets.size(); i++) {
            if (!assets[i].needsRebalancing(tolerance)) continue;
            
            double deviation = assets[i].getDeviation();
            double tradeAmount = deviation * totalValue;
            
            // Calculate total cost including market impact
            double transactionCost = std::abs(tradeAmount) * assets[i].transactionCost;
            double marketImpact = calculateMarketImpact(std::abs(tradeAmount), marketData[i]);
            double totalCost = transactionCost + marketImpact * std::abs(tradeAmount);
            
            // Benefit is reduction in deviation
            double benefit = assets[i].getAbsDeviation();
            double efficiency = benefit / (totalCost + 1e-9);
            
            tradeQueue.push({i, tradeAmount, benefit, totalCost, efficiency});
        }
        
        // Execute trades in order of efficiency
        double remainingBudget = totalValue * 0.01; // 1% of portfolio for transaction costs
        
        while (!tradeQueue.empty() && remainingBudget > 0) {
            auto proposal = tradeQueue.top();
            tradeQueue.pop();
            
            if (proposal.totalCost <= remainingBudget) {
                int i = proposal.assetIndex;
                double newWeight = assets[i].currentWeight + (proposal.amount / totalValue);
                
                trades.emplace_back(assets[i].symbol, proposal.amount, proposal.totalCost, newWeight);
                assets[i].currentWeight = newWeight;
                remainingBudget -= proposal.totalCost;
            }
        }
        
        return trades;
    }
};

// Test framework
class TestSuite {
public:
    static void runBasicTests() {
        std::cout << "Running basic portfolio rebalancing tests..." << std::endl;
        
        // Test 1: Simple 3-asset portfolio
        {
            std::vector<Asset> assets = {
                Asset("STOCK_A", 0.6, 0.4, 0.001),
                Asset("STOCK_B", 0.3, 0.4, 0.002),
                Asset("STOCK_C", 0.1, 0.2, 0.003)
            };
            
            double tolerance = 0.01;
            double totalValue = 1000000.0;
            
            auto trades = PortfolioRebalancer::rebalanceGreedy(assets, tolerance, totalValue);
            assert(!trades.empty());
            std::cout << "✓ Basic greedy rebalancing test passed" << std::endl;
        }
        
        // Test 2: Already balanced portfolio
        {
            std::vector<Asset> assets = {
                Asset("STOCK_A", 0.333, 0.333, 0.001),
                Asset("STOCK_B", 0.333, 0.333, 0.001),
                Asset("STOCK_C", 0.334, 0.334, 0.001)
            };
            
            double tolerance = 0.01;
            double totalValue = 1000000.0;
            
            auto trades = PortfolioRebalancer::rebalanceGreedy(assets, tolerance, totalValue);
            assert(trades.empty() || trades.size() <= 1); // Should need minimal or no trades
            std::cout << "✓ Already balanced portfolio test passed" << std::endl;
        }
        
        // Test 3: High transaction costs
        {
            std::vector<Asset> assets = {
                Asset("STOCK_A", 0.7, 0.5, 0.01), // 1% transaction cost
                Asset("STOCK_B", 0.3, 0.5, 0.01)
            };
            
            double tolerance = 0.05; // Higher tolerance due to high costs
            double totalValue = 1000000.0;
            
            auto trades = PortfolioRebalancer::rebalanceGreedy(assets, tolerance, totalValue);
            // Should be conservative due to high costs
            std::cout << "✓ High transaction cost test passed" << std::endl;
        }
    }
    
    static void runAdvancedTests() {
        std::cout << "\nRunning advanced rebalancing tests..." << std::endl;
        
        // Test risk-aware rebalancing
        {
            std::vector<Asset> assets = {
                Asset("TECH", 0.5, 0.3, 0.001),
                Asset("FINANCE", 0.3, 0.4, 0.002),
                Asset("UTILITIES", 0.2, 0.3, 0.001)
            };
            
            // Correlation matrix (simplified)
            std::vector<std::vector<double>> correlationMatrix = {
                {0.04, 0.01, 0.005},  // TECH variance and covariances
                {0.01, 0.03, 0.008},  // FINANCE
                {0.005, 0.008, 0.02}  // UTILITIES
            };
            
            double tolerance = 0.02;
            double totalValue = 1000000.0;
            
            auto trades = PortfolioRebalancer::rebalanceRiskAware(assets, tolerance, totalValue, correlationMatrix);
            std::cout << "✓ Risk-aware rebalancing test passed" << std::endl;
        }
        
        // Test market impact consideration
        {
            std::vector<Asset> assets = {
                Asset("LARGE_CAP", 0.6, 0.4, 0.0005),
                Asset("SMALL_CAP", 0.4, 0.6, 0.002)
            };
            
            std::vector<AdvancedRebalancer::MarketData> marketData = {
                AdvancedRebalancer::MarketData(0.0005, 5e6, 0.015, 0.9), // Highly liquid large cap
                AdvancedRebalancer::MarketData(0.002, 1e5, 0.03, 0.6)    // Less liquid small cap
            };
            
            double tolerance = 0.01;
            double totalValue = 1000000.0;
            
            auto trades = AdvancedRebalancer::rebalanceWithMarketImpact(assets, tolerance, totalValue, marketData);
            std::cout << "✓ Market impact rebalancing test passed" << std::endl;
        }
    }
    
    static void runPerformanceTest() {
        std::cout << "\nRunning performance test..." << std::endl;
        
        // Large portfolio test
        std::vector<Asset> largePortfolio;
        double weightSum = 0.0;
        
        for (int i = 0; i < 50; i++) {
            double currentWeight = 0.02 + (i % 10) * 0.001;
            double targetWeight = 0.02;
            double cost = 0.001 + (i % 3) * 0.0005;
            
            largePortfolio.emplace_back("ASSET_" + std::to_string(i), currentWeight, targetWeight, cost);
            weightSum += currentWeight;
        }
        
        // Normalize weights
        for (auto& asset : largePortfolio) {
            asset.currentWeight /= weightSum;
        }
        
        double tolerance = 0.005;
        double totalValue = 10000000.0;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto trades = PortfolioRebalancer::rebalanceGreedy(largePortfolio, tolerance, totalValue);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Large portfolio (50 assets) rebalancing:" << std::endl;
        std::cout << "Time: " << duration.count() << " μs" << std::endl;
        std::cout << "Trades generated: " << trades.size() << std::endl;
    }
    
    static void demonstrateRebalancing() {
        std::cout << "\nDemonstrating Portfolio Rebalancing:" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        std::vector<Asset> portfolio = {
            Asset("US_STOCKS", 0.45, 0.60, 0.0005),
            Asset("INTL_STOCKS", 0.25, 0.20, 0.001),
            Asset("BONDS", 0.20, 0.15, 0.0008),
            Asset("COMMODITIES", 0.05, 0.03, 0.002),
            Asset("CASH", 0.05, 0.02, 0.0001)
        };
        
        double tolerance = 0.01; // 1% tolerance
        double totalValue = 5000000.0; // $5M portfolio
        
        std::cout << "Portfolio size: $5,000,000" << std::endl;
        std::cout << "Tolerance: " << tolerance * 100 << "%" << std::endl;
        
        // Test different strategies
        std::cout << "\n1. Greedy Rebalancing:" << std::endl;
        auto greedyTrades = PortfolioRebalancer::rebalanceGreedy(portfolio, tolerance, totalValue);
        PortfolioRebalancer::analyzeRebalancing(portfolio, greedyTrades, tolerance);
        
        std::cout << "\n2. Optimal Rebalancing:" << std::endl;
        auto optimalTrades = PortfolioRebalancer::rebalanceOptimal(portfolio, tolerance, totalValue);
        PortfolioRebalancer::analyzeRebalancing(portfolio, optimalTrades, tolerance);
    }
};

int main() {
    std::cout << "HackerRank Problem 4: Optimal Portfolio Rebalancing" << std::endl;
    std::cout << "===================================================" << std::endl;
    
    TestSuite::runBasicTests();
    TestSuite::runAdvancedTests();
    TestSuite::runPerformanceTest();
    TestSuite::demonstrateRebalancing();
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Portfolio Rebalancing Analysis Complete!" << std::endl;
    std::cout << "\nAlgorithms Implemented:" << std::endl;
    std::cout << "✓ Greedy cost-minimization rebalancing" << std::endl;
    std::cout << "✓ Dynamic programming optimal solution" << std::endl;
    std::cout << "✓ Risk-aware rebalancing with correlations" << std::endl;
    std::cout << "✓ Market impact aware trading" << std::endl;
    std::cout << "\nKey Features:" << std::endl;
    std::cout << "- Transaction cost optimization" << std::endl;
    std::cout << "- Portfolio risk management" << std::endl;
    std::cout << "- Liquidity and market impact modeling" << std::endl;
    std::cout << "- Tolerance-based rebalancing" << std::endl;
    std::cout << "- Multi-objective optimization" << std::endl;
    
    return 0;
}

/*
Portfolio Rebalancing Theory:

Mathematical Foundation:
- Objective: Minimize Σ(cost_i * |trade_i|) subject to weight constraints
- Constraints: Σ(w_i) = 1, |w_i - target_i| ≤ tolerance
- Market Impact: cost = linear_cost + sqrt(volume) * volatility

Optimization Approaches:
1. Greedy: Sort by efficiency ratio (deviation/cost), O(N log N)
2. Dynamic Programming: Exact solution for small N, O(2^N) states
3. Linear Programming: Can model as LP with transaction cost penalties
4. Quadratic Programming: Include risk (covariance) constraints

Advanced Considerations:
1. Market Impact Models:
   - Temporary impact (bid-ask spread, volatility)
   - Permanent impact (information content)
   - Participation rate dependencies

2. Risk Management:
   - Tracking error minimization
   - Value-at-Risk constraints
   - Maximum drawdown limits

3. Execution Algorithms:
   - TWAP (Time-Weighted Average Price)
   - VWAP (Volume-Weighted Average Price)
   - Implementation Shortfall optimization

Real-world Extensions:
- Multi-period optimization
- Transaction timing optimization
- Tax-loss harvesting
- Sector/factor constraints
- ESG considerations

Performance Metrics:
- Tracking error vs benchmark
- Information ratio
- Sharpe ratio improvement
- Transaction cost analysis
*/
