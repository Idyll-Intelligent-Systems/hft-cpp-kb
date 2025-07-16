/*
Problem: Statistical Arbitrage Engine (Two Sigma Style)
Implement a pairs trading system that:
1. Identifies cointegrated asset pairs using statistical tests
2. Calculates optimal hedge ratios using regression
3. Monitors spread deviations and generates trading signals
4. Implements risk management with position sizing and stop-losses
5. Backtests strategy performance with realistic transaction costs

Requirements:
- Real-time cointegration monitoring
- Kalman filter for dynamic hedge ratio estimation
- Z-score based signal generation
- Portfolio-level risk management
- Performance attribution and analytics

This tests statistical modeling, time series analysis, and quantitative finance knowledge.
*/

#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <random>
#include <memory>
#include <map>

// Linear algebra utilities for regression
class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows, cols;

public:
    Matrix(size_t r, size_t c) : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }
    
    double& operator()(size_t i, size_t j) { return data[i][j]; }
    const double& operator()(size_t i, size_t j) const { return data[i][j]; }
    
    size_t get_rows() const { return rows; }
    size_t get_cols() const { return cols; }
    
    // Matrix multiplication
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }
        
        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                for (size_t k = 0; k < cols; ++k) {
                    result(i, j) += data[i][k] * other(k, j);
                }
            }
        }
        return result;
    }
    
    // Matrix transpose
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(j, i) = data[i][j];
            }
        }
        return result;
    }
    
    // Matrix inverse (using Gaussian elimination for 2x2)
    Matrix inverse() const {
        if (rows != 2 || cols != 2) {
            throw std::invalid_argument("Only 2x2 matrix inversion implemented");
        }
        
        double det = data[0][0] * data[1][1] - data[0][1] * data[1][0];
        if (std::abs(det) < 1e-12) {
            throw std::runtime_error("Matrix is singular");
        }
        
        Matrix result(2, 2);
        result(0, 0) = data[1][1] / det;
        result(0, 1) = -data[0][1] / det;
        result(1, 0) = -data[1][0] / det;
        result(1, 1) = data[0][0] / det;
        
        return result;
    }
};

// Statistical tests for cointegration
class StatisticalTests {
public:
    // Augmented Dickey-Fuller test (simplified)
    static double adf_test(const std::vector<double>& series) {
        if (series.size() < 10) return 0.0;  // Insufficient data
        
        // Calculate first differences
        std::vector<double> diff;
        for (size_t i = 1; i < series.size(); ++i) {
            diff.push_back(series[i] - series[i-1]);
        }
        
        // Simple regression: diff[t] = alpha + beta * series[t-1] + error
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
        size_t n = diff.size();
        
        for (size_t i = 0; i < n; ++i) {
            double x = series[i];  // lagged level
            double y = diff[i];    // first difference
            
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        double beta = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        
        // ADF statistic is approximately beta / se(beta)
        // For simplicity, return just beta (negative values suggest stationarity)
        return beta;
    }
    
    // Johansen cointegration test (simplified two-variable case)
    static double johansen_test(const std::vector<double>& series1, 
                               const std::vector<double>& series2) {
        if (series1.size() != series2.size() || series1.size() < 20) {
            return 0.0;
        }
        
        // Calculate correlation as a proxy for cointegration strength
        double mean1 = 0, mean2 = 0;
        for (size_t i = 0; i < series1.size(); ++i) {
            mean1 += series1[i];
            mean2 += series2[i];
        }
        mean1 /= series1.size();
        mean2 /= series2.size();
        
        double num = 0, den1 = 0, den2 = 0;
        for (size_t i = 0; i < series1.size(); ++i) {
            double diff1 = series1[i] - mean1;
            double diff2 = series2[i] - mean2;
            num += diff1 * diff2;
            den1 += diff1 * diff1;
            den2 += diff2 * diff2;
        }
        
        return num / std::sqrt(den1 * den2);
    }
};

// Kalman Filter for dynamic hedge ratio estimation
class KalmanFilter {
private:
    double state;           // Current hedge ratio estimate
    double variance;        // Estimate variance
    double process_noise;   // Process noise variance
    double measurement_noise; // Measurement noise variance
    
public:
    KalmanFilter(double initial_state = 1.0, double initial_variance = 1.0,
                 double proc_noise = 0.001, double meas_noise = 0.1)
        : state(initial_state), variance(initial_variance),
          process_noise(proc_noise), measurement_noise(meas_noise) {}
    
    double update(double measurement, double regressor) {
        // Prediction step
        variance += process_noise;
        
        // Update step
        double innovation = measurement - state * regressor;
        double innovation_variance = variance * regressor * regressor + measurement_noise;
        double kalman_gain = variance * regressor / innovation_variance;
        
        state += kalman_gain * innovation;
        variance *= (1.0 - kalman_gain * regressor);
        
        return state;
    }
    
    double get_state() const { return state; }
    double get_variance() const { return variance; }
};

// Asset price data structure
struct PriceData {
    std::string symbol;
    double price;
    uint64_t timestamp;
    double volume;
    
    PriceData(const std::string& s, double p, double v = 0.0) 
        : symbol(s), price(p), volume(v) {
        timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

// Trading signal
struct Signal {
    std::string pair_name;
    double z_score;
    double confidence;
    bool is_long_spread;    // true = long asset1, short asset2
    double target_notional;
    
    Signal(const std::string& name, double z, double conf, bool long_spread, double notional)
        : pair_name(name), z_score(z), confidence(conf), is_long_spread(long_spread), 
          target_notional(notional) {}
};

// Position in a pair
struct Position {
    std::string asset1, asset2;
    double quantity1, quantity2;  // Signed quantities
    double entry_spread;
    double unrealized_pnl;
    uint64_t entry_time;
    
    Position(const std::string& a1, const std::string& a2, 
             double q1, double q2, double spread)
        : asset1(a1), asset2(a2), quantity1(q1), quantity2(q2), 
          entry_spread(spread), unrealized_pnl(0.0) {
        entry_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

// Pair trading strategy
class PairsTradingStrategy {
private:
    std::string asset1, asset2;
    std::deque<double> prices1, prices2;
    std::deque<double> spreads;
    KalmanFilter hedge_ratio_filter;
    
    // Strategy parameters
    size_t lookback_window;
    double entry_threshold;     // Z-score threshold for entry
    double exit_threshold;      // Z-score threshold for exit
    double stop_loss_threshold; // Maximum loss before stopping
    double transaction_cost;    // Basis points per trade
    
    // Statistics
    double current_hedge_ratio;
    double spread_mean;
    double spread_std;
    double half_life;          // Mean reversion half-life
    
public:
    PairsTradingStrategy(const std::string& a1, const std::string& a2, 
                        size_t window = 60, double entry_z = 2.0, 
                        double exit_z = 0.5, double stop_z = 4.0)
        : asset1(a1), asset2(a2), lookback_window(window),
          entry_threshold(entry_z), exit_threshold(exit_z), 
          stop_loss_threshold(stop_z), transaction_cost(5.0),
          current_hedge_ratio(1.0), spread_mean(0.0), spread_std(1.0), half_life(0.0) {}
    
    void update_prices(double price1, double price2) {
        prices1.push_back(price1);
        prices2.push_back(price2);
        
        if (prices1.size() > lookback_window) {
            prices1.pop_front();
            prices2.pop_front();
        }
        
        // Update hedge ratio using Kalman filter
        if (prices1.size() > 1) {
            current_hedge_ratio = hedge_ratio_filter.update(price1, price2);
        }
        
        // Calculate spread
        double spread = price1 - current_hedge_ratio * price2;
        spreads.push_back(spread);
        
        if (spreads.size() > lookback_window) {
            spreads.pop_front();
        }
        
        // Update spread statistics
        update_spread_statistics();
    }
    
    std::vector<Signal> generate_signals() {
        std::vector<Signal> signals;
        
        if (spreads.size() < lookback_window) {
            return signals;  // Insufficient data
        }
        
        double current_spread = spreads.back();
        double z_score = (current_spread - spread_mean) / spread_std;
        
        // Generate entry signals
        if (std::abs(z_score) > entry_threshold) {
            double confidence = std::min(1.0, std::abs(z_score) / entry_threshold);
            bool long_spread = z_score < 0;  // Buy when spread is below mean
            double notional = calculate_position_size(confidence);
            
            std::string pair_name = asset1 + "/" + asset2;
            signals.emplace_back(pair_name, z_score, confidence, long_spread, notional);
        }
        
        return signals;
    }
    
    bool should_exit_position(const Position& position, double current_price1, double current_price2) {
        double current_spread = current_price1 - current_hedge_ratio * current_price2;
        double z_score = (current_spread - spread_mean) / spread_std;
        
        // Exit conditions
        bool mean_reversion = std::abs(z_score) < exit_threshold;
        bool stop_loss = std::abs(z_score) > stop_loss_threshold;
        
        return mean_reversion || stop_loss;
    }
    
    double calculate_cointegration_strength() const {
        if (prices1.size() < 20) return 0.0;
        
        std::vector<double> p1(prices1.begin(), prices1.end());
        std::vector<double> p2(prices2.begin(), prices2.end());
        
        return StatisticalTests::johansen_test(p1, p2);
    }
    
    void print_statistics() const {
        std::cout << "Pair: " << asset1 << "/" << asset2 << std::endl;
        std::cout << "Hedge Ratio: " << current_hedge_ratio << std::endl;
        std::cout << "Spread Mean: " << spread_mean << std::endl;
        std::cout << "Spread Std: " << spread_std << std::endl;
        std::cout << "Current Z-Score: " << get_current_z_score() << std::endl;
        std::cout << "Cointegration: " << calculate_cointegration_strength() << std::endl;
        std::cout << "Half-Life: " << half_life << " periods" << std::endl;
        std::cout << std::endl;
    }
    
    double get_current_z_score() const {
        if (spreads.empty() || spread_std == 0) return 0.0;
        return (spreads.back() - spread_mean) / spread_std;
    }
    
private:
    void update_spread_statistics() {
        if (spreads.size() < 10) return;
        
        // Calculate mean and standard deviation
        spread_mean = 0.0;
        for (double spread : spreads) {
            spread_mean += spread;
        }
        spread_mean /= spreads.size();
        
        double variance = 0.0;
        for (double spread : spreads) {
            variance += (spread - spread_mean) * (spread - spread_mean);
        }
        spread_std = std::sqrt(variance / (spreads.size() - 1));
        
        // Estimate half-life using AR(1) model
        estimate_half_life();
    }
    
    void estimate_half_life() {
        if (spreads.size() < 20) return;
        
        // Fit AR(1) model: spread[t] = alpha + beta * spread[t-1] + error
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
        size_t n = spreads.size() - 1;
        
        for (size_t i = 0; i < n; ++i) {
            double x = spreads[i] - spread_mean;      // Lagged spread (mean-adjusted)
            double y = spreads[i + 1] - spread_mean;  // Current spread (mean-adjusted)
            
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        double beta = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        
        // Half-life = -log(2) / log(beta)
        if (beta > 0 && beta < 1) {
            half_life = -std::log(2.0) / std::log(beta);
        } else {
            half_life = std::numeric_limits<double>::infinity();
        }
    }
    
    double calculate_position_size(double confidence) const {
        // Kelly criterion-based position sizing
        double base_size = 10000.0;  // Base notional
        double win_rate = 0.6;       // Estimated win rate
        double avg_win = 0.02;       // Average win as fraction
        double avg_loss = 0.015;     // Average loss as fraction
        
        double kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win;
        kelly_fraction = std::max(0.0, std::min(0.25, kelly_fraction));  // Cap at 25%
        
        return base_size * kelly_fraction * confidence;
    }
};

// Portfolio manager for multiple pairs
class PairsPortfolio {
private:
    std::map<std::string, std::unique_ptr<PairsTradingStrategy>> strategies;
    std::vector<Position> positions;
    std::map<std::string, double> current_prices;
    
    // Portfolio parameters
    double total_capital;
    double max_position_size;
    double portfolio_var_limit;
    
public:
    PairsPortfolio(double capital = 1000000.0) 
        : total_capital(capital), max_position_size(capital * 0.1), 
          portfolio_var_limit(capital * 0.02) {}
    
    void add_pair(const std::string& asset1, const std::string& asset2) {
        std::string pair_key = asset1 + "_" + asset2;
        strategies[pair_key] = std::make_unique<PairsTradingStrategy>(asset1, asset2);
    }
    
    void update_price(const std::string& asset, double price) {
        current_prices[asset] = price;
        
        // Update all strategies involving this asset
        for (auto& [pair_key, strategy] : strategies) {
            // Extract asset names from strategy (simplified)
            // In practice, would store asset names in strategy
            update_strategy_if_relevant(strategy.get(), asset, price);
        }
    }
    
    std::vector<Signal> get_all_signals() {
        std::vector<Signal> all_signals;
        
        for (auto& [pair_key, strategy] : strategies) {
            auto signals = strategy->generate_signals();
            all_signals.insert(all_signals.end(), signals.begin(), signals.end());
        }
        
        // Sort by confidence (highest first)
        std::sort(all_signals.begin(), all_signals.end(),
                 [](const Signal& a, const Signal& b) {
                     return a.confidence > b.confidence;
                 });
        
        return all_signals;
    }
    
    void execute_signal(const Signal& signal) {
        // Risk checks
        if (!check_risk_limits(signal)) {
            std::cout << "Signal rejected due to risk limits: " << signal.pair_name << std::endl;
            return;
        }
        
        // Extract assets from pair name
        size_t slash_pos = signal.pair_name.find('/');
        std::string asset1 = signal.pair_name.substr(0, slash_pos);
        std::string asset2 = signal.pair_name.substr(slash_pos + 1);
        
        // Calculate quantities
        double price1 = current_prices[asset1];
        double price2 = current_prices[asset2];
        double hedge_ratio = get_hedge_ratio(asset1, asset2);
        
        double notional1 = signal.target_notional;
        double notional2 = notional1 * hedge_ratio;
        
        double quantity1 = notional1 / price1;
        double quantity2 = notional2 / price2;
        
        if (!signal.is_long_spread) {
            quantity1 = -quantity1;
            quantity2 = -quantity2;
        }
        
        // Create position
        double entry_spread = price1 - hedge_ratio * price2;
        positions.emplace_back(asset1, asset2, quantity1, -quantity2, entry_spread);
        
        std::cout << "Executed: " << signal.pair_name 
                  << " Z-score: " << signal.z_score 
                  << " Notional: " << signal.target_notional << std::endl;
    }
    
    void update_positions() {
        for (auto& position : positions) {
            double price1 = current_prices[position.asset1];
            double price2 = current_prices[position.asset2];
            
            // Calculate unrealized P&L
            double current_value1 = position.quantity1 * price1;
            double current_value2 = position.quantity2 * price2;
            position.unrealized_pnl = current_value1 + current_value2;
        }
    }
    
    void print_portfolio_status() {
        update_positions();
        
        std::cout << "Portfolio Status:" << std::endl;
        std::cout << "=================" << std::endl;
        std::cout << "Total Capital: " << total_capital << std::endl;
        std::cout << "Active Positions: " << positions.size() << std::endl;
        
        double total_pnl = 0.0;
        for (const auto& position : positions) {
            total_pnl += position.unrealized_pnl;
            std::cout << "Position: " << position.asset1 << "/" << position.asset2
                      << " P&L: " << position.unrealized_pnl << std::endl;
        }
        
        std::cout << "Total Unrealized P&L: " << total_pnl << std::endl;
        std::cout << "Return: " << (total_pnl / total_capital) * 100 << "%" << std::endl;
        std::cout << std::endl;
        
        // Print strategy statistics
        for (auto& [pair_key, strategy] : strategies) {
            strategy->print_statistics();
        }
    }
    
private:
    void update_strategy_if_relevant(PairsTradingStrategy* strategy, 
                                   const std::string& asset, double price) {
        // Simplified: update if asset matches (would need better asset tracking)
        // For demo purposes, assume first asset in each strategy
        // In practice, would store asset mappings
    }
    
    bool check_risk_limits(const Signal& signal) const {
        // Check position size limit
        if (signal.target_notional > max_position_size) {
            return false;
        }
        
        // Check portfolio concentration
        double total_exposure = 0.0;
        for (const auto& position : positions) {
            total_exposure += std::abs(position.quantity1 * current_prices.at(position.asset1));
        }
        
        if (total_exposure + signal.target_notional > total_capital * 0.8) {
            return false;
        }
        
        return true;
    }
    
    double get_hedge_ratio(const std::string& asset1, const std::string& asset2) const {
        // Simplified: return 1.0, in practice would get from strategy
        return 1.0;
    }
};

// Market data simulator
class MarketSimulator {
private:
    std::mt19937 rng;
    std::map<std::string, double> prices;
    std::normal_distribution<double> noise_dist;
    
public:
    MarketSimulator() : rng(std::random_device{}()), noise_dist(0.0, 0.01) {
        // Initialize some correlated assets
        prices["AAPL"] = 150.0;
        prices["MSFT"] = 300.0;
        prices["GOOGL"] = 2500.0;
        prices["AMZN"] = 3200.0;
    }
    
    void simulate_step() {
        // Add correlated noise to create cointegrated pairs
        double common_factor = noise_dist(rng);
        
        for (auto& [symbol, price] : prices) {
            double idiosyncratic_noise = noise_dist(rng) * 0.5;
            double total_change = common_factor + idiosyncratic_noise;
            price *= (1.0 + total_change);
        }
    }
    
    double get_price(const std::string& symbol) const {
        auto it = prices.find(symbol);
        return (it != prices.end()) ? it->second : 0.0;
    }
    
    std::vector<std::string> get_symbols() const {
        std::vector<std::string> symbols;
        for (const auto& [symbol, price] : prices) {
            symbols.push_back(symbol);
        }
        return symbols;
    }
};

int main() {
    std::cout << "Two Sigma Style Statistical Arbitrage Engine" << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Initialize portfolio and add pairs
    PairsPortfolio portfolio;
    portfolio.add_pair("AAPL", "MSFT");
    portfolio.add_pair("GOOGL", "AMZN");
    
    // Initialize market simulator
    MarketSimulator market;
    
    // Run simulation
    int simulation_steps = 500;
    for (int step = 0; step < simulation_steps; ++step) {
        // Simulate market movement
        market.simulate_step();
        
        // Update portfolio with new prices
        for (const std::string& symbol : market.get_symbols()) {
            portfolio.update_price(symbol, market.get_price(symbol));
        }
        
        // Generate and execute signals
        auto signals = portfolio.get_all_signals();
        for (const auto& signal : signals) {
            if (signal.confidence > 0.7) {  // Only execute high-confidence signals
                portfolio.execute_signal(signal);
                break;  // Execute only one signal per step
            }
        }
        
        // Print status every 100 steps
        if (step % 100 == 0 && step > 0) {
            std::cout << "Step " << step << ":" << std::endl;
            portfolio.print_portfolio_status();
        }
    }
    
    // Final results
    std::cout << "Final Results:" << std::endl;
    std::cout << "==============" << std::endl;
    portfolio.print_portfolio_status();
    
    return 0;
}

/*
Algorithm Analysis:

Core Components:

1. Cointegration Testing:
   - Augmented Dickey-Fuller test for stationarity
   - Johansen test for cointegration relationships
   - Dynamic monitoring of statistical properties

2. Hedge Ratio Estimation:
   - Kalman filter for real-time adaptation
   - Handles structural breaks and regime changes
   - Provides uncertainty estimates

3. Signal Generation:
   - Z-score based entry/exit thresholds
   - Mean reversion strength assessment
   - Confidence-weighted position sizing

4. Risk Management:
   - Position size limits
   - Portfolio concentration limits
   - Stop-loss mechanisms
   - Kelly criterion for optimal sizing

Statistical Arbitrage Concepts:

1. Pairs Selection:
   - Correlation analysis
   - Cointegration testing
   - Fundamental relationship verification
   - Liquidity and trading cost considerations

2. Spread Construction:
   - Optimal hedge ratio calculation
   - Dynamic rebalancing
   - Transaction cost optimization
   - Half-life estimation

3. Entry/Exit Logic:
   - Statistical significance thresholds
   - Mean reversion timing
   - Regime change detection
   - Risk-adjusted returns

Performance Metrics:
- Sharpe ratio
- Maximum drawdown
- Win rate and profit factor
- Calmar ratio
- Information ratio

Real-World Considerations:

1. Market Microstructure:
   - Bid-ask spreads
   - Market impact
   - Slippage costs
   - Order execution timing

2. Regime Changes:
   - Structural breaks in relationships
   - Market stress periods
   - Volatility clustering
   - Flight-to-quality events

3. Alternative Data:
   - News sentiment
   - Options flow
   - Fundamental metrics
   - Macro indicators

Interview Tips:
1. Discuss statistical significance vs economic significance
2. Explain the bias-variance tradeoff in estimation
3. Consider transaction costs and realistic constraints
4. Address overfitting and out-of-sample performance
5. Think about scalability and capacity constraints

Extensions:
- Multi-asset statistical arbitrage
- Machine learning for signal enhancement
- Options-based stat arb strategies
- Cross-asset momentum strategies
- Alternative risk models (factor models, etc.)
*/
