/*
D.E. Shaw HFT Problem: Statistical Arbitrage Engine
=================================================

Problem Statement:
Implement a comprehensive statistical arbitrage system that can:
1. Identify mean-reverting relationships between assets
2. Build and maintain trading signals
3. Execute trades with optimal timing
4. Manage risk in real-time

This tests:
1. Time series analysis and cointegration
2. Signal processing and filtering
3. Execution algorithms
4. Risk management systems

Applications:
- Pairs trading and statistical arbitrage
- Market neutral strategies
- High-frequency mean reversion
- Portfolio hedging
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <deque>
#include <unordered_map>
#include <memory>
#include <chrono>

class StatisticalArbitrageEngine {
private:
    struct Asset {
        std::string symbol;
        std::deque<double> prices;
        std::deque<double> returns;
        double current_position;
        double max_position;
        
        Asset(const std::string& sym, double max_pos = 1000.0) 
            : symbol(sym), current_position(0.0), max_position(max_pos) {}
    };
    
    struct TradingPair {
        std::string asset1, asset2;
        double hedge_ratio;
        std::deque<double> spread;
        double spread_mean;
        double spread_std;
        double half_life;
        double adf_statistic;
        bool is_cointegrated;
        
        // Trading parameters
        double entry_threshold;    // Z-score threshold for entry
        double exit_threshold;     // Z-score threshold for exit
        double stop_loss;          // Maximum loss tolerance
        
        TradingPair(const std::string& a1, const std::string& a2) 
            : asset1(a1), asset2(a2), hedge_ratio(1.0), 
              entry_threshold(2.0), exit_threshold(0.5), stop_loss(0.05) {}
    };
    
    struct Signal {
        std::string pair_id;
        double z_score;
        double confidence;
        std::string direction;  // "LONG_SPREAD" or "SHORT_SPREAD"
        std::chrono::system_clock::time_point timestamp;
        bool is_active;
    };
    
    std::unordered_map<std::string, std::unique_ptr<Asset>> assets_;
    std::unordered_map<std::string, std::unique_ptr<TradingPair>> pairs_;
    std::vector<Signal> signals_;
    
    // Risk management parameters
    double max_portfolio_var_;
    double max_correlation_exposure_;
    double max_sector_exposure_;
    
public:
    StatisticalArbitrageEngine(double max_var = 0.02, double max_corr = 0.8, double max_sector = 0.3)
        : max_portfolio_var_(max_var), max_correlation_exposure_(max_corr), max_sector_exposure_(max_sector) {}
    
    // Asset management
    void addAsset(const std::string& symbol, double max_position = 1000.0) {
        assets_[symbol] = std::make_unique<Asset>(symbol, max_position);
    }
    
    void updatePrice(const std::string& symbol, double price) {
        auto& asset = assets_[symbol];
        if (!asset) return;
        
        if (!asset->prices.empty()) {
            double return_val = std::log(price / asset->prices.back());
            asset->returns.push_back(return_val);
            
            // Keep only recent data (e.g., 500 observations)
            if (asset->returns.size() > 500) {
                asset->returns.pop_front();
            }
        }
        
        asset->prices.push_back(price);
        if (asset->prices.size() > 500) {
            asset->prices.pop_front();
        }
    }
    
    // Pair management and cointegration analysis
    void addTradingPair(const std::string& asset1, const std::string& asset2) {
        std::string pair_id = asset1 + "_" + asset2;
        pairs_[pair_id] = std::make_unique<TradingPair>(asset1, asset2);
    }
    
    // Engle-Granger cointegration test
    bool testCointegration(const std::string& pair_id) {
        auto& pair = pairs_[pair_id];
        if (!pair) return false;
        
        auto& asset1 = assets_[pair->asset1];
        auto& asset2 = assets_[pair->asset2];
        
        if (!asset1 || !asset2 || asset1->prices.size() < 50 || asset2->prices.size() < 50) {
            return false;
        }
        
        // Step 1: Estimate hedge ratio using OLS regression
        std::vector<double> x, y;
        size_t min_size = std::min(asset1->prices.size(), asset2->prices.size());
        
        for (size_t i = 0; i < min_size; ++i) {
            x.push_back(asset2->prices[i]);
            y.push_back(asset1->prices[i]);
        }
        
        pair->hedge_ratio = calculateOLSSlope(x, y);
        
        // Step 2: Calculate spread and test for stationarity
        pair->spread.clear();
        for (size_t i = 0; i < min_size; ++i) {
            double spread = asset1->prices[i] - pair->hedge_ratio * asset2->prices[i];
            pair->spread.push_back(spread);
        }
        
        // Calculate spread statistics
        calculateSpreadStatistics(pair_id);
        
        // Step 3: Augmented Dickey-Fuller test on spread
        pair->adf_statistic = augmentedDickeyFullerTest(pair->spread);
        
        // Critical value at 5% significance level (approximate)
        pair->is_cointegrated = pair->adf_statistic < -2.86;
        
        return pair->is_cointegrated;
    }
    
    double calculateOLSSlope(const std::vector<double>& x, const std::vector<double>& y) {
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        size_t n = x.size();
        
        for (size_t i = 0; i < n; ++i) {
            sum_x += x[i];
            sum_y += y[i];
            sum_xy += x[i] * y[i];
            sum_xx += x[i] * x[i];
        }
        
        double mean_x = sum_x / n;
        double mean_y = sum_y / n;
        
        return (sum_xy - n * mean_x * mean_y) / (sum_xx - n * mean_x * mean_x);
    }
    
    void calculateSpreadStatistics(const std::string& pair_id) {
        auto& pair = pairs_[pair_id];
        if (!pair || pair->spread.empty()) return;
        
        // Calculate mean
        double sum = std::accumulate(pair->spread.begin(), pair->spread.end(), 0.0);
        pair->spread_mean = sum / pair->spread.size();
        
        // Calculate standard deviation
        double sq_sum = 0.0;
        for (double spread : pair->spread) {
            sq_sum += (spread - pair->spread_mean) * (spread - pair->spread_mean);
        }
        pair->spread_std = std::sqrt(sq_sum / (pair->spread.size() - 1));
        
        // Estimate half-life using AR(1) model
        pair->half_life = estimateHalfLife(pair->spread);
    }
    
    double estimateHalfLife(const std::deque<double>& series) {
        if (series.size() < 2) return 0.0;
        
        std::vector<double> lagged, current;
        for (size_t i = 1; i < series.size(); ++i) {
            lagged.push_back(series[i-1]);
            current.push_back(series[i]);
        }
        
        double beta = calculateOLSSlope(lagged, current);
        return -std::log(2.0) / std::log(std::abs(beta));
    }
    
    double augmentedDickeyFullerTest(const std::deque<double>& series) {
        if (series.size() < 3) return 0.0;
        
        // Simple ADF test: Δy_t = α + βy_{t-1} + ε_t
        std::vector<double> y_lagged, delta_y;
        
        for (size_t i = 1; i < series.size(); ++i) {
            y_lagged.push_back(series[i-1]);
            delta_y.push_back(series[i] - series[i-1]);
        }
        
        // Calculate beta coefficient
        double beta = calculateOLSSlope(y_lagged, delta_y);
        
        // Calculate standard error (simplified)
        double mse = 0.0;
        double mean_y_lagged = std::accumulate(y_lagged.begin(), y_lagged.end(), 0.0) / y_lagged.size();
        
        for (size_t i = 0; i < y_lagged.size(); ++i) {
            double fitted = beta * y_lagged[i];
            mse += (delta_y[i] - fitted) * (delta_y[i] - fitted);
        }
        mse /= (y_lagged.size() - 1);
        
        double sse = 0.0;
        for (double y : y_lagged) {
            sse += (y - mean_y_lagged) * (y - mean_y_lagged);
        }
        
        double se_beta = std::sqrt(mse / sse);
        
        // T-statistic
        return beta / se_beta;
    }
    
    // Signal generation
    void generateSignals() {
        signals_.clear();
        
        for (auto& [pair_id, pair] : pairs_) {
            if (!pair->is_cointegrated || pair->spread.empty()) continue;
            
            // Current spread z-score
            double current_spread = pair->spread.back();
            double z_score = (current_spread - pair->spread_mean) / pair->spread_std;
            
            // Generate signal based on z-score
            Signal signal;
            signal.pair_id = pair_id;
            signal.z_score = z_score;
            signal.timestamp = std::chrono::system_clock::now();
            signal.is_active = false;
            
            if (std::abs(z_score) > pair->entry_threshold) {
                signal.direction = (z_score > 0) ? "SHORT_SPREAD" : "LONG_SPREAD";
                signal.confidence = std::min(std::abs(z_score) / 3.0, 1.0);  // Max confidence at 3 std
                signal.is_active = true;
                signals_.push_back(signal);
            }
        }
        
        // Sort signals by confidence
        std::sort(signals_.begin(), signals_.end(), 
                 [](const Signal& a, const Signal& b) {
                     return a.confidence > b.confidence;
                 });
    }
    
    // Execution and position management
    struct TradeRecommendation {
        std::string asset;
        double quantity;
        std::string side;  // "BUY" or "SELL"
        double confidence;
        std::string reason;
    };
    
    std::vector<TradeRecommendation> generateTradeRecommendations() {
        std::vector<TradeRecommendation> recommendations;
        
        for (const auto& signal : signals_) {
            if (!signal.is_active) continue;
            
            auto& pair = pairs_[signal.pair_id];
            auto& asset1 = assets_[pair->asset1];
            auto& asset2 = assets_[pair->asset2];
            
            if (!asset1 || !asset2) continue;
            
            // Calculate position sizes
            double base_size = 100.0;  // Base position size
            double size_multiplier = signal.confidence;
            
            if (signal.direction == "LONG_SPREAD") {
                // Buy asset1, sell asset2
                TradeRecommendation rec1, rec2;
                
                rec1.asset = pair->asset1;
                rec1.quantity = base_size * size_multiplier;
                rec1.side = "BUY";
                rec1.confidence = signal.confidence;
                rec1.reason = "Long spread signal";
                
                rec2.asset = pair->asset2;
                rec2.quantity = base_size * size_multiplier * pair->hedge_ratio;
                rec2.side = "SELL";
                rec2.confidence = signal.confidence;
                rec2.reason = "Long spread signal";
                
                recommendations.push_back(rec1);
                recommendations.push_back(rec2);
            } else {
                // Sell asset1, buy asset2
                TradeRecommendation rec1, rec2;
                
                rec1.asset = pair->asset1;
                rec1.quantity = base_size * size_multiplier;
                rec1.side = "SELL";
                rec1.confidence = signal.confidence;
                rec1.reason = "Short spread signal";
                
                rec2.asset = pair->asset2;
                rec2.quantity = base_size * size_multiplier * pair->hedge_ratio;
                rec2.side = "BUY";
                rec2.confidence = signal.confidence;
                rec2.reason = "Short spread signal";
                
                recommendations.push_back(rec1);
                recommendations.push_back(rec2);
            }
        }
        
        return recommendations;
    }
    
    // Risk management
    bool checkRiskLimits(const std::vector<TradeRecommendation>& trades) {
        // Check individual position limits
        for (const auto& trade : trades) {
            auto& asset = assets_[trade.asset];
            if (!asset) continue;
            
            double new_position = asset->current_position;
            new_position += (trade.side == "BUY") ? trade.quantity : -trade.quantity;
            
            if (std::abs(new_position) > asset->max_position) {
                return false;
            }
        }
        
        // Check portfolio-level risk
        double portfolio_var = calculatePortfolioVaR(trades);
        if (portfolio_var > max_portfolio_var_) {
            return false;
        }
        
        return true;
    }
    
    double calculatePortfolioVaR(const std::vector<TradeRecommendation>& trades) {
        // Simplified VaR calculation
        double portfolio_volatility = 0.0;
        
        for (const auto& trade : trades) {
            auto& asset = assets_[trade.asset];
            if (!asset || asset->returns.empty()) continue;
            
            // Calculate asset volatility
            double mean_return = std::accumulate(asset->returns.begin(), asset->returns.end(), 0.0) / asset->returns.size();
            double variance = 0.0;
            for (double ret : asset->returns) {
                variance += (ret - mean_return) * (ret - mean_return);
            }
            variance /= (asset->returns.size() - 1);
            
            double asset_volatility = std::sqrt(variance);
            double position_value = trade.quantity * asset->prices.back();
            
            portfolio_volatility += position_value * position_value * asset_volatility * asset_volatility;
        }
        
        portfolio_volatility = std::sqrt(portfolio_volatility);
        
        // 95% VaR (1.645 standard deviations)
        return 1.645 * portfolio_volatility;
    }
    
    // Performance monitoring
    struct PerformanceMetrics {
        double total_pnl;
        double sharpe_ratio;
        double max_drawdown;
        double win_rate;
        int total_trades;
    };
    
    PerformanceMetrics calculatePerformance() {
        PerformanceMetrics metrics = {};
        
        // Calculate total P&L
        for (auto& [symbol, asset] : assets_) {
            if (asset->prices.size() >= 2) {
                double price_change = asset->prices.back() - asset->prices.front();
                metrics.total_pnl += asset->current_position * price_change;
            }
        }
        
        // Additional metrics would be calculated based on trade history
        // This is a simplified implementation
        
        return metrics;
    }
    
    // Getters for monitoring
    const std::vector<Signal>& getActiveSignals() const { return signals_; }
    const std::unordered_map<std::string, std::unique_ptr<TradingPair>>& getPairs() const { return pairs_; }
    
    void printStatus() {
        std::cout << "Statistical Arbitrage Engine Status:\n";
        std::cout << "====================================\n";
        std::cout << "Total Assets: " << assets_.size() << "\n";
        std::cout << "Total Pairs: " << pairs_.size() << "\n";
        
        int cointegrated_pairs = 0;
        for (auto& [pair_id, pair] : pairs_) {
            if (pair->is_cointegrated) cointegrated_pairs++;
        }
        std::cout << "Cointegrated Pairs: " << cointegrated_pairs << "\n";
        std::cout << "Active Signals: " << signals_.size() << "\n";
        
        if (!signals_.empty()) {
            std::cout << "\nTop Signals:\n";
            for (size_t i = 0; i < std::min(size_t(3), signals_.size()); ++i) {
                const auto& signal = signals_[i];
                std::cout << "  " << signal.pair_id << ": " << signal.direction 
                         << " (Z-score: " << signal.z_score 
                         << ", Confidence: " << signal.confidence << ")\n";
            }
        }
    }
};

// Example usage
int main() {
    StatisticalArbitrageEngine engine;
    
    // Add assets
    engine.addAsset("AAPL", 1000);
    engine.addAsset("MSFT", 1000);
    engine.addAsset("GOOGL", 1000);
    engine.addAsset("AMZN", 1000);
    
    // Simulate price updates
    std::vector<double> aapl_prices = {150, 152, 151, 153, 152, 154, 153, 155, 154, 156};
    std::vector<double> msft_prices = {300, 305, 302, 308, 305, 310, 307, 312, 309, 315};
    
    for (size_t i = 0; i < aapl_prices.size(); ++i) {
        engine.updatePrice("AAPL", aapl_prices[i]);
        engine.updatePrice("MSFT", msft_prices[i]);
    }
    
    // Add trading pair
    engine.addTradingPair("AAPL", "MSFT");
    
    // Test cointegration
    bool is_cointegrated = engine.testCointegration("AAPL_MSFT");
    std::cout << "AAPL-MSFT Cointegration: " << (is_cointegrated ? "Yes" : "No") << "\n\n";
    
    // Generate signals
    engine.generateSignals();
    
    // Get trade recommendations
    auto recommendations = engine.generateTradeRecommendations();
    
    std::cout << "Trade Recommendations:\n";
    for (const auto& rec : recommendations) {
        std::cout << rec.side << " " << rec.quantity << " " << rec.asset 
                 << " (Confidence: " << rec.confidence << ")\n";
    }
    
    // Print engine status
    engine.printStatus();
    
    return 0;
}
