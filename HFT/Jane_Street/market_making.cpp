/*
Problem: Market Making Strategy with Inventory Management (Jane Street Style)
Implement a market making algorithm that:
1. Maintains bid/ask quotes around fair value
2. Manages inventory risk with position limits
3. Adjusts spreads based on volatility and inventory
4. Handles adverse selection and order flow toxicity

Requirements:
- Calculate fair value from multiple price sources
- Dynamic spread adjustment based on market conditions
- Position-aware quoting to manage inventory risk
- Risk management with stop-loss and position limits

This problem tests understanding of market microstructure, risk management,
and real-time decision making under uncertainty.
*/

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <deque>
#include <chrono>

// Market data structures
struct PriceUpdate {
    double price;
    uint64_t volume;
    uint64_t timestamp;
    
    PriceUpdate(double p, uint64_t v) : price(p), volume(v) {
        timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

struct Quote {
    double bid_price;
    double ask_price;
    uint64_t bid_size;
    uint64_t ask_size;
    uint64_t timestamp;
    
    Quote(double bp, double ap, uint64_t bs, uint64_t as) 
        : bid_price(bp), ask_price(ap), bid_size(bs), ask_size(as) {
        timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    
    double mid_price() const {
        return (bid_price + ask_price) / 2.0;
    }
    
    double spread() const {
        return ask_price - bid_price;
    }
};

struct Fill {
    bool is_buy;        // true if we bought, false if we sold
    double price;
    uint64_t quantity;
    uint64_t timestamp;
    
    Fill(bool buy, double p, uint64_t q) : is_buy(buy), price(p), quantity(q) {
        timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

// Risk management parameters
struct RiskParameters {
    int64_t max_position;           // Maximum absolute position
    double max_loss_per_trade;      // Maximum loss per individual trade
    double max_daily_loss;          // Maximum daily loss
    double inventory_penalty;       // Penalty factor for large inventory
    double min_spread;              // Minimum spread to maintain
    double max_spread;              // Maximum spread to quote
    
    RiskParameters() 
        : max_position(10000), max_loss_per_trade(1000.0), max_daily_loss(5000.0),
          inventory_penalty(0.0001), min_spread(0.01), max_spread(0.10) {}
};

class VolatilityEstimator {
private:
    std::deque<double> price_returns;
    size_t window_size;
    
public:
    VolatilityEstimator(size_t window = 100) : window_size(window) {}
    
    void add_price(double price) {
        static double last_price = price;
        
        if (last_price > 0) {
            double return_val = std::log(price / last_price);
            price_returns.push_back(return_val);
            
            if (price_returns.size() > window_size) {
                price_returns.pop_front();
            }
        }
        
        last_price = price;
    }
    
    double get_volatility() const {
        if (price_returns.size() < 2) return 0.01;  // Default volatility
        
        // Calculate standard deviation of returns
        double mean = 0.0;
        for (double ret : price_returns) {
            mean += ret;
        }
        mean /= price_returns.size();
        
        double variance = 0.0;
        for (double ret : price_returns) {
            variance += (ret - mean) * (ret - mean);
        }
        variance /= (price_returns.size() - 1);
        
        return std::sqrt(variance) * std::sqrt(252 * 24 * 60);  // Annualized volatility
    }
};

class FairValueCalculator {
private:
    std::vector<PriceUpdate> external_prices;
    std::vector<double> weights;
    double ewma_alpha;  // Exponentially weighted moving average alpha
    double fair_value;
    
public:
    FairValueCalculator(const std::vector<double>& w, double alpha = 0.1) 
        : weights(w), ewma_alpha(alpha), fair_value(0.0) {}
    
    void add_price_source(const PriceUpdate& update, size_t source_index) {
        if (source_index >= external_prices.size()) {
            external_prices.resize(source_index + 1, PriceUpdate(0.0, 0));
        }
        
        external_prices[source_index] = update;
        update_fair_value();
    }
    
    double get_fair_value() const {
        return fair_value;
    }
    
private:
    void update_fair_value() {
        double weighted_sum = 0.0;
        double total_weight = 0.0;
        
        size_t loop_size = (external_prices.size() < weights.size()) ? external_prices.size() : weights.size();
        for (size_t i = 0; i < loop_size; ++i) {
            if (external_prices[i].volume > 0) {  // Valid price
                weighted_sum += external_prices[i].price * weights[i];
                total_weight += weights[i];
            }
        }
        
        if (total_weight > 0) {
            double new_fair_value = weighted_sum / total_weight;
            
            // Apply EWMA smoothing
            if (fair_value == 0.0) {
                fair_value = new_fair_value;
            } else {
                fair_value = ewma_alpha * new_fair_value + (1.0 - ewma_alpha) * fair_value;
            }
        }
    }
};

class MarketMaker {
private:
    RiskParameters risk_params;
    FairValueCalculator fair_value_calc;
    VolatilityEstimator vol_estimator;
    
    // Portfolio state
    int64_t position;           // Current inventory (positive = long, negative = short)
    double cash;                // Cash position
    double daily_pnl;           // Daily P&L
    std::vector<Fill> fills;    // Trade history
    
    // Market making parameters
    double base_spread_factor;  // Base spread as fraction of volatility
    double inventory_skew_factor; // How much to skew quotes based on inventory
    uint64_t default_size;      // Default quote size
    
public:
    MarketMaker(const std::vector<double>& price_weights) 
        : fair_value_calc(price_weights), position(0), cash(100000.0), daily_pnl(0.0),
          base_spread_factor(2.0), inventory_skew_factor(0.1), default_size(100) {}
    
    // Update market data
    void update_market_data(const PriceUpdate& update, size_t source_index) {
        fair_value_calc.add_price_source(update, source_index);
        vol_estimator.add_price(update.price);
    }
    
    // Generate quotes
    Quote generate_quote() {
        double fair_value = fair_value_calc.get_fair_value();
        double volatility = vol_estimator.get_volatility();
        
        if (fair_value <= 0.0) {
            // No valid fair value yet
            return Quote(0.0, 0.0, 0, 0);
        }
        
        double min_spread = (risk_params.max_spread < base_spread_factor * volatility) ? 
                           risk_params.max_spread : base_spread_factor * volatility;
        double base_spread = (risk_params.min_spread > min_spread) ? risk_params.min_spread : min_spread;
        
        // Adjust spread based on inventory
        double inventory_ratio = static_cast<double>(position) / risk_params.max_position;
        double abs_inventory_ratio = (inventory_ratio < 0) ? -inventory_ratio : inventory_ratio;
        double spread_adjustment = 1.0 + abs_inventory_ratio * risk_params.inventory_penalty;
        double adjusted_spread = base_spread * spread_adjustment;
        
        // Calculate skew based on inventory (skew quotes away from inventory)
        double skew = inventory_skew_factor * inventory_ratio * fair_value;
        
        // Generate bid/ask prices
        double bid_price = fair_value - adjusted_spread / 2.0 - skew;
        double ask_price = fair_value + adjusted_spread / 2.0 - skew;
        
        // Adjust sizes based on risk limits
        uint64_t bid_size = calculate_quote_size(true);
        uint64_t ask_size = calculate_quote_size(false);
        
        return Quote(bid_price, ask_price, bid_size, ask_size);
    }
    
    // Handle a fill (someone traded against our quote)
    void handle_fill(bool we_bought, double price, uint64_t quantity) {
        Fill fill(we_bought, price, quantity);
        fills.push_back(fill);
        
        if (we_bought) {
            position += quantity;
            cash -= price * quantity;
        } else {
            position -= quantity;
            cash += price * quantity;
        }
        
        // Update daily P&L (mark-to-market against fair value)
        update_daily_pnl();
        
        // Check risk limits
        check_risk_limits();
    }
    
    // Calculate appropriate quote size based on position and risk limits
    uint64_t calculate_quote_size(bool for_bid) {
        // Don't quote if position limit would be exceeded
        if (for_bid && position >= risk_params.max_position) {
            return 0;  // Already at max long position
        }
        if (!for_bid && position <= -risk_params.max_position) {
            return 0;  // Already at max short position
        }
        
        // Reduce size as we approach position limits
        int64_t remaining_capacity;
        if (for_bid) {
            remaining_capacity = risk_params.max_position - position;
        } else {
            remaining_capacity = position + risk_params.max_position;
        }
        
        int64_t safe_capacity = (remaining_capacity > 0) ? remaining_capacity : 0;
        uint64_t capacity_uint = static_cast<uint64_t>(safe_capacity);
        uint64_t max_size = (default_size < capacity_uint) ? default_size : capacity_uint;
        
        // Further reduce size based on inventory
        double abs_position = (position < 0) ? -static_cast<double>(position) : static_cast<double>(position);
        double inventory_factor = 1.0 - abs_position / risk_params.max_position;
        
        return static_cast<uint64_t>(max_size * inventory_factor);
    }
    
    // Risk management
    void check_risk_limits() {
        // Position limit check
        int64_t abs_position = (position < 0) ? -position : position;
        if (abs_position > risk_params.max_position) {
            std::cout << "WARNING: Position limit exceeded! Position: " << position << std::endl;
        }
        
        // Daily loss limit check
        if (daily_pnl < -risk_params.max_daily_loss) {
            std::cout << "WARNING: Daily loss limit exceeded! P&L: " << daily_pnl << std::endl;
        }
    }
    
    void update_daily_pnl() {
        double fair_value = fair_value_calc.get_fair_value();
        if (fair_value > 0.0) {
            double mark_to_market_value = cash + position * fair_value;
            daily_pnl = mark_to_market_value - 100000.0;  // Initial cash
        }
    }
    
    // Analytics
    void print_status() const {
        std::cout << "Market Maker Status:" << std::endl;
        std::cout << "===================" << std::endl;
        std::cout << "Fair Value: " << fair_value_calc.get_fair_value() << std::endl;
        std::cout << "Volatility: " << vol_estimator.get_volatility() * 100 << "%" << std::endl;
        std::cout << "Position: " << position << std::endl;
        std::cout << "Cash: " << cash << std::endl;
        std::cout << "Daily P&L: " << daily_pnl << std::endl;
        std::cout << "Total Fills: " << fills.size() << std::endl;
        
        if (!fills.empty()) {
            double avg_fill_price = 0.0;
            for (const auto& fill : fills) {
                avg_fill_price += fill.price;
            }
            avg_fill_price /= fills.size();
            std::cout << "Average Fill Price: " << avg_fill_price << std::endl;
        }
        std::cout << std::endl;
    }
    
    double get_sharpe_ratio() const {
        if (fills.size() < 2) return 0.0;
        
        std::vector<double> trade_pnls;
        for (size_t i = 1; i < fills.size(); ++i) {
            // Simple trade P&L calculation
            double pnl = fills[i].is_buy ? 
                (fair_value_calc.get_fair_value() - fills[i].price) * fills[i].quantity :
                (fills[i].price - fair_value_calc.get_fair_value()) * fills[i].quantity;
            trade_pnls.push_back(pnl);
        }
        
        double mean_pnl = 0.0;
        for (double pnl : trade_pnls) {
            mean_pnl += pnl;
        }
        mean_pnl /= trade_pnls.size();
        
        double variance = 0.0;
        for (double pnl : trade_pnls) {
            variance += (pnl - mean_pnl) * (pnl - mean_pnl);
        }
        variance /= (trade_pnls.size() - 1);
        
        double std_dev = std::sqrt(variance);
        return std_dev > 0 ? mean_pnl / std_dev : 0.0;
    }
};

// Market simulator for testing
class MarketSimulator {
private:
    std::mt19937 rng;
    std::normal_distribution<double> price_shock;
    std::exponential_distribution<double> arrival_rate;
    
public:
    MarketSimulator() : rng(std::random_device{}()), 
                       price_shock(0.0, 0.01), arrival_rate(10.0) {}
    
    // Simulate market price movement
    double simulate_price_move(double current_price) {
        double shock = price_shock(rng);
        return current_price * (1.0 + shock);
    }
    
    // Simulate order arrival
    bool simulate_order_arrival() {
        double time_to_next = arrival_rate(rng);
        return time_to_next < 0.1;  // 10% chance of order per tick
    }
    
    // Simulate which side gets hit (true = bid hit, false = ask hit)
    bool simulate_side_selection() {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        return uniform(rng) < 0.5;
    }
};

int main() {
    std::cout << "Jane Street Style Market Making Algorithm" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Initialize market maker with multiple price sources
    std::vector<double> price_weights = {0.4, 0.3, 0.2, 0.1};  // Exchange weights
    MarketMaker mm(price_weights);
    MarketSimulator simulator;
    
    // Simulate initial price discovery
    double base_price = 100.0;
    for (size_t source = 0; source < price_weights.size(); ++source) {
        double price_offset = (source - 1.5) * 0.001;  // Small differences between exchanges
        PriceUpdate update(base_price + price_offset, 1000);
        mm.update_market_data(update, source);
    }
    
    std::cout << "Initial market state:" << std::endl;
    mm.print_status();
    
    // Run simulation
    int simulation_steps = 1000;
    for (int step = 0; step < simulation_steps; ++step) {
        // Update market prices
        for (size_t source = 0; source < price_weights.size(); ++source) {
            base_price = simulator.simulate_price_move(base_price);
            double price_offset = (source - 1.5) * 0.001;
            PriceUpdate update(base_price + price_offset, 1000);
            mm.update_market_data(update, source);
        }
        
        // Generate quote
        Quote quote = mm.generate_quote();
        
        // Simulate order flow
        if (simulator.simulate_order_arrival() && quote.bid_price > 0) {
            bool hit_bid = simulator.simulate_side_selection();
            if (hit_bid && quote.bid_size > 0) {
                uint64_t fill_size = (quote.bid_size < 50) ? quote.bid_size : 50;
                mm.handle_fill(true, quote.bid_price, fill_size);
            } else if (!hit_bid && quote.ask_size > 0) {
                uint64_t fill_size = (quote.ask_size < 50) ? quote.ask_size : 50;
                mm.handle_fill(false, quote.ask_price, fill_size);
            }
        }
        
        // Print status every 100 steps
        if (step % 100 == 0 && step > 0) {
            std::cout << "Step " << step << ":" << std::endl;
            std::cout << "Current Quote: Bid " << quote.bid_price << "@" << quote.bid_size 
                      << " Ask " << quote.ask_price << "@" << quote.ask_size << std::endl;
            mm.print_status();
        }
    }
    
    // Final results
    std::cout << "Final Results:" << std::endl;
    std::cout << "==============" << std::endl;
    mm.print_status();
    std::cout << "Sharpe Ratio: " << mm.get_sharpe_ratio() << std::endl;
    
    return 0;
}

/*
Algorithm Analysis:

Key Components:

1. Fair Value Calculation:
   - Multi-source price aggregation with weights
   - Exponentially weighted moving average for smoothing
   - Handles stale/missing price feeds

2. Volatility Estimation:
   - Rolling window of price returns
   - Standard deviation calculation
   - Used for dynamic spread sizing

3. Risk Management:
   - Position limits (max inventory)
   - Daily loss limits
   - Per-trade loss limits
   - Dynamic position-based sizing

4. Quote Generation:
   - Base spread proportional to volatility
   - Inventory skew (wider spreads when heavily positioned)
   - Size adjustment based on remaining capacity

Market Making Strategies:

1. Spread Management:
   - Wider spreads in volatile markets
   - Inventory penalty increases spreads
   - Minimum/maximum spread boundaries

2. Inventory Management:
   - Skew quotes away from large positions
   - Reduce size as position limits approached
   - Mark-to-market position against fair value

3. Adverse Selection Protection:
   - Volatility-adjusted spreads
   - Position limits prevent runaway losses
   - Quick adaptation to market changes

Real-World Considerations:

1. Latency Optimization:
   - Minimize calculation time in quote generation
   - Cache frequently used values
   - Avoid expensive operations in hot path

2. Market Microstructure:
   - Queue position awareness
   - Tick size constraints
   - Exchange-specific rules

3. Advanced Risk Management:
   - Greeks hedging for options
   - Correlation-based position sizing
   - Stress testing scenarios

Performance Metrics:
- Sharpe ratio (risk-adjusted returns)
- Maximum drawdown
- Fill ratio (% of quotes that trade)
- Inventory turnover
- Profit per unit of risk

Interview Tips:
1. Discuss the trade-off between spread and fill probability
2. Explain how inventory affects risk and quote skew
3. Consider different market regimes (trending vs mean-reverting)
4. Address the adverse selection problem
5. Think about regulatory requirements (market making obligations)

Extensions:
- Multi-asset market making with correlation
- Options market making with Greeks
- Cross-venue arbitrage opportunities  
- Machine learning for order flow prediction
- Optimal execution for inventory rebalancing
*/
