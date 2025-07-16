/*
Optiver HFT Problem: Market Making Algorithm with Inventory Management
=====================================================================

Problem Statement:
Design and implement a market making algorithm that:
1. Provides bid/ask quotes continuously
2. Manages inventory risk
3. Adjusts spreads based on market conditions
4. Handles adverse selection
5. Maximizes profit while controlling risk

This tests:
1. Market microstructure understanding
2. Risk management algorithms
3. Real-time decision making
4. Statistical modeling
5. Optimization techniques

Applications:
- Options market making
- ETF arbitrage
- Electronic trading systems
- Liquidity provision
*/

#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>

struct Order {
    int id;
    char side;     // 'B' for bid, 'A' for ask
    double price;
    int quantity;
    std::chrono::system_clock::time_point timestamp;
    
    Order(int _id, char _side, double _price, int _qty) 
        : id(_id), side(_side), price(_price), quantity(_qty),
          timestamp(std::chrono::system_clock::now()) {}
};

struct MarketData {
    double bid_price;
    double ask_price;
    int bid_size;
    int ask_size;
    double last_trade_price;
    int volume;
    double volatility;
    std::chrono::system_clock::time_point timestamp;
    
    MarketData() : bid_price(0), ask_price(0), bid_size(0), ask_size(0),
                   last_trade_price(0), volume(0), volatility(0.02),
                   timestamp(std::chrono::system_clock::now()) {}
};

class MarketMaker {
private:
    // Risk parameters
    double max_position_;
    double max_exposure_;
    double target_spread_;
    double min_spread_;
    double inventory_penalty_;
    
    // Current state
    double current_position_;
    double current_pnl_;
    double fair_value_;
    MarketData latest_market_data_;
    
    // Order management
    std::unordered_map<int, Order> active_orders_;
    int next_order_id_;
    
    // Historical data for modeling
    std::vector<double> price_history_;
    std::vector<double> volatility_history_;
    std::vector<int> volume_history_;
    
    // Performance metrics
    double total_volume_traded_;
    int number_of_trades_;
    double max_drawdown_;
    double sharpe_ratio_;
    
public:
    MarketMaker(double max_pos = 1000, double max_exp = 50000, double target_spr = 0.002)
        : max_position_(max_pos), max_exposure_(max_exp), target_spread_(target_spr),
          min_spread_(0.001), inventory_penalty_(0.0001), current_position_(0),
          current_pnl_(0), fair_value_(100.0), next_order_id_(1),
          total_volume_traded_(0), number_of_trades_(0), max_drawdown_(0), sharpe_ratio_(0) {}
    
    // Core market making logic
    void onMarketData(const MarketData& data) {
        latest_market_data_ = data;
        updateFairValue(data);
        updateVolatility(data);
        
        // Cancel existing orders if market has moved significantly
        if (shouldCancelOrders(data)) {
            cancelAllOrders();
        }
        
        // Generate new quotes
        auto quotes = generateQuotes(data);
        sendQuotes(quotes.first, quotes.second);
    }
    
    void updateFairValue(const MarketData& data) {
        // Weighted average of bid/ask with recent trade information
        double mid_price = (data.bid_price + data.ask_price) / 2.0;
        double trade_weight = 0.3;
        double market_weight = 0.7;
        
        if (data.last_trade_price > 0) {
            fair_value_ = trade_weight * data.last_trade_price + market_weight * mid_price;
        } else {
            fair_value_ = mid_price;
        }
        
        // Adjust for inventory bias
        double inventory_adjustment = current_position_ * inventory_penalty_;
        fair_value_ -= inventory_adjustment;
        
        price_history_.push_back(fair_value_);
        if (price_history_.size() > 1000) {
            price_history_.erase(price_history_.begin());
        }
    }
    
    void updateVolatility(const MarketData& data) {
        if (price_history_.size() >= 2) {
            // Calculate realized volatility
            double sum_squared_returns = 0.0;
            for (size_t i = 1; i < std::min(price_history_.size(), size_t(50)); ++i) {
                double return_val = std::log(price_history_[i] / price_history_[i-1]);
                sum_squared_returns += return_val * return_val;
            }
            
            double realized_vol = std::sqrt(sum_squared_returns / 49.0) * std::sqrt(252); // Annualized
            
            // Blend with market implied volatility
            latest_market_data_.volatility = 0.7 * realized_vol + 0.3 * data.volatility;
        }
        
        volatility_history_.push_back(latest_market_data_.volatility);
        if (volatility_history_.size() > 100) {
            volatility_history_.erase(volatility_history_.begin());
        }
    }
    
    bool shouldCancelOrders(const MarketData& data) {
        // Cancel if market has moved more than 2 ticks
        double market_move_threshold = 0.02;
        
        for (const auto& [id, order] : active_orders_) {
            if (order.side == 'B' && data.bid_price > order.price + market_move_threshold) {
                return true;
            }
            if (order.side == 'A' && data.ask_price < order.price - market_move_threshold) {
                return true;
            }
        }
        
        return false;
    }
    
    void cancelAllOrders() {
        active_orders_.clear();
        std::cout << "Cancelled all orders due to market move\n";
    }
    
    std::pair<Order, Order> generateQuotes(const MarketData& data) {
        // Calculate optimal spread based on volatility and inventory
        double base_spread = calculateOptimalSpread(data);
        
        // Adjust for inventory position
        double inventory_skew = calculateInventorySkew();
        
        // Calculate bid and ask prices
        double bid_price = fair_value_ - base_spread/2.0 + inventory_skew;
        double ask_price = fair_value_ + base_spread/2.0 + inventory_skew;
        
        // Ensure minimum spread
        if (ask_price - bid_price < min_spread_) {
            double mid = (bid_price + ask_price) / 2.0;
            bid_price = mid - min_spread_/2.0;
            ask_price = mid + min_spread_/2.0;
        }
        
        // Calculate optimal sizes
        int bid_size = calculateOptimalSize('B', data);
        int ask_size = calculateOptimalSize('A', data);
        
        Order bid_order(next_order_id_++, 'B', bid_price, bid_size);
        Order ask_order(next_order_id_++, 'A', ask_price, ask_size);
        
        return {bid_order, ask_order};
    }
    
    double calculateOptimalSpread(const MarketData& data) {
        // Base spread on volatility
        double vol_component = data.volatility * 2.0; // 2x daily vol
        
        // Adjust for market conditions
        double liquidity_component = target_spread_ * std::max(1.0, 1000.0 / (data.bid_size + data.ask_size));
        
        // Adverse selection protection
        double adverse_selection = 0.001 * std::sqrt(data.volume / 1000.0);
        
        return std::max(min_spread_, vol_component + liquidity_component + adverse_selection);
    }
    
    double calculateInventorySkew() {
        // Skew quotes away from current position
        double position_ratio = current_position_ / max_position_;
        return position_ratio * target_spread_; // Skew by up to one spread width
    }
    
    int calculateOptimalSize(char side, const MarketData& data) {
        // Base size
        int base_size = 100;
        
        // Adjust for inventory limits
        if (side == 'B' && current_position_ > max_position_ * 0.8) {
            base_size /= 2; // Reduce bid size when long
        }
        if (side == 'A' && current_position_ < -max_position_ * 0.8) {
            base_size /= 2; // Reduce ask size when short
        }
        
        // Adjust for market conditions
        if (data.volatility > 0.03) {
            base_size = static_cast<int>(base_size * 0.7); // Reduce size in high vol
        }
        
        return std::max(1, base_size);
    }
    
    void sendQuotes(const Order& bid, const Order& ask) {
        active_orders_[bid.id] = bid;
        active_orders_[ask.id] = ask;
        
        std::cout << "Quotes: Bid " << bid.quantity << "@" << std::fixed << std::setprecision(4) 
                  << bid.price << " | Ask " << ask.quantity << "@" << ask.price 
                  << " | Fair: " << fair_value_ << " | Pos: " << current_position_ << "\n";
    }
    
    // Order execution handling
    void onOrderFilled(int order_id, int filled_quantity, double fill_price) {
        auto it = active_orders_.find(order_id);
        if (it == active_orders_.end()) return;
        
        Order& order = it->second;
        
        // Update position and P&L
        if (order.side == 'B') {
            current_position_ += filled_quantity;
            current_pnl_ -= filled_quantity * fill_price; // Paid cash
        } else {
            current_position_ -= filled_quantity;
            current_pnl_ += filled_quantity * fill_price; // Received cash
        }
        
        // Update trade statistics
        total_volume_traded_ += filled_quantity;
        number_of_trades_++;
        
        std::cout << "Fill: " << order.side << " " << filled_quantity << "@" << fill_price 
                  << " | New position: " << current_position_ << " | P&L: " << current_pnl_ << "\n";
        
        // Remove or update order
        order.quantity -= filled_quantity;
        if (order.quantity <= 0) {
            active_orders_.erase(it);
        }
        
        // Check risk limits
        checkRiskLimits();
    }
    
    void checkRiskLimits() {
        // Position limit
        if (std::abs(current_position_) > max_position_) {
            std::cout << "WARNING: Position limit exceeded!\n";
            if (current_position_ > max_position_) {
                // Only quote asks to reduce position
                cancelBidOrders();
            } else {
                // Only quote bids to reduce position
                cancelAskOrders();
            }
        }
        
        // Exposure limit (mark-to-market)
        double mark_to_market = current_position_ * fair_value_ + current_pnl_;
        if (std::abs(mark_to_market) > max_exposure_) {
            std::cout << "WARNING: Exposure limit exceeded!\n";
            // Could implement emergency hedging here
        }
    }
    
    void cancelBidOrders() {
        for (auto it = active_orders_.begin(); it != active_orders_.end();) {
            if (it->second.side == 'B') {
                it = active_orders_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    void cancelAskOrders() {
        for (auto it = active_orders_.begin(); it != active_orders_.end();) {
            if (it->second.side == 'A') {
                it = active_orders_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    // Performance analysis
    void calculatePerformanceMetrics() {
        // Update mark-to-market P&L
        double mtm_pnl = current_position_ * fair_value_ + current_pnl_;
        
        // Calculate Sharpe ratio (simplified)
        if (!price_history_.empty()) {
            std::vector<double> pnl_series;
            for (size_t i = 1; i < price_history_.size(); ++i) {
                double position_pnl = current_position_ * (price_history_[i] - price_history_[i-1]);
                pnl_series.push_back(position_pnl);
            }
            
            double mean_pnl = 0.0;
            for (double pnl : pnl_series) mean_pnl += pnl;
            mean_pnl /= pnl_series.size();
            
            double variance = 0.0;
            for (double pnl : pnl_series) variance += (pnl - mean_pnl) * (pnl - mean_pnl);
            variance /= (pnl_series.size() - 1);
            
            sharpe_ratio_ = mean_pnl / std::sqrt(variance) * std::sqrt(252); // Annualized
        }
    }
    
    void printPerformanceReport() {
        calculatePerformanceMetrics();
        
        std::cout << "\n=== Performance Report ===\n";
        std::cout << "Total P&L: $" << current_pnl_ << "\n";
        std::cout << "Current Position: " << current_position_ << "\n";
        std::cout << "Mark-to-Market P&L: $" << (current_position_ * fair_value_ + current_pnl_) << "\n";
        std::cout << "Total Volume Traded: " << total_volume_traded_ << "\n";
        std::cout << "Number of Trades: " << number_of_trades_ << "\n";
        std::cout << "Average Trade Size: " << (number_of_trades_ > 0 ? total_volume_traded_ / number_of_trades_ : 0) << "\n";
        std::cout << "Sharpe Ratio: " << sharpe_ratio_ << "\n";
        std::cout << "Current Fair Value: $" << fair_value_ << "\n";
        std::cout << "Active Orders: " << active_orders_.size() << "\n";
    }
    
    // Getters
    double getCurrentPosition() const { return current_position_; }
    double getCurrentPnL() const { return current_pnl_; }
    double getFairValue() const { return fair_value_; }
    size_t getActiveOrderCount() const { return active_orders_.size(); }
};

// Market simulator for testing
class MarketSimulator {
private:
    std::mt19937 gen_;
    std::normal_distribution<double> price_dist_;
    std::poisson_distribution<int> volume_dist_;
    
public:
    MarketSimulator() : gen_(std::chrono::steady_clock::now().time_since_epoch().count()),
                       price_dist_(0.0, 0.01), volume_dist_(500) {}
    
    MarketData generateMarketData(double base_price = 100.0) {
        MarketData data;
        
        // Simulate bid/ask spread
        double spread = 0.02 + std::abs(price_dist_(gen_)) * 0.02;
        double mid_price = base_price + price_dist_(gen_);
        
        data.bid_price = mid_price - spread/2.0;
        data.ask_price = mid_price + spread/2.0;
        data.bid_size = 100 + volume_dist_(gen_) % 500;
        data.ask_size = 100 + volume_dist_(gen_) % 500;
        data.last_trade_price = mid_price + price_dist_(gen_) * 0.5;
        data.volume = volume_dist_(gen_);
        data.volatility = 0.02 + std::abs(price_dist_(gen_)) * 0.01;
        
        return data;
    }
    
    bool shouldFillOrder(const Order& order, const MarketData& market) {
        // Simple fill logic: fill if price crosses our quote
        if (order.side == 'B' && market.ask_price <= order.price) {
            return true;
        }
        if (order.side == 'A' && market.bid_price >= order.price) {
            return true;
        }
        return false;
    }
};

// Example usage and testing
int main() {
    std::cout << "Optiver Market Making Algorithm\n";
    std::cout << "==============================\n\n";
    
    MarketMaker mm(1000, 50000, 0.002); // Max position 1000, max exposure $50k, target spread 0.2%
    MarketSimulator simulator;
    
    double base_price = 100.0;
    
    // Simulate trading session
    for (int i = 0; i < 100; ++i) {
        // Generate market data
        MarketData market = simulator.generateMarketData(base_price);
        base_price = (market.bid_price + market.ask_price) / 2.0; // Update base price
        
        // Send market data to market maker
        mm.onMarketData(market);
        
        // Simulate order fills (simplified)
        std::mt19937 gen(std::chrono::steady_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<double> fill_prob(0.0, 1.0);
        
        if (fill_prob(gen) < 0.1) { // 10% chance of fill per tick
            // Simulate a fill
            std::uniform_int_distribution<int> side_choice(0, 1);
            char fill_side = (side_choice(gen) == 0) ? 'B' : 'A';
            
            int fill_quantity = 50 + gen() % 100;
            double fill_price = (fill_side == 'B') ? market.ask_price : market.bid_price;
            
            // Create a dummy order ID for the fill
            static int fill_id = 1000;
            mm.onOrderFilled(fill_id++, fill_quantity, fill_price);
        }
        
        // Print status every 20 ticks
        if (i % 20 == 19) {
            std::cout << "\n--- Tick " << i+1 << " ---\n";
            mm.printPerformanceReport();
            std::cout << "\n";
        }
    }
    
    // Final performance report
    std::cout << "\n=== Final Performance Report ===\n";
    mm.printPerformanceReport();
    
    return 0;
}
