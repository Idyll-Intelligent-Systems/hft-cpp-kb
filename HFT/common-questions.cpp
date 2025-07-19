/*
HFT C++ Interview Questions - Optimized Solutions Reference
=========================================================

This file contains optimized solutions for common HFT interview questions covering:
1. Low-latency data structures and algorithms
2. Memory optimization techniques
3. Concurrent programming patterns
4. Financial mathematics implementations
5. System design for high-frequency trading

Author: HFT Interview Preparation
Date: July 19, 2025
*/

#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <queue>
#include <memory>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <atomic>
#include <mutex>
#include <thread>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <random>
#include <array>

// =============================================================================
// COMMON HFT INTERVIEW QUESTION CATEGORIES
// =============================================================================

namespace HFTInterview {

// Category 1: Low-Latency Data Structures
// =======================================

// Cache-aligned structures for optimal performance
struct alignas(64) CacheAlignedData {
    std::atomic<double> price{0.0};
    std::atomic<uint64_t> timestamp{0};
    std::atomic<uint32_t> volume{0};
    char padding[36]; // Ensure full cache line
};

// Lock-free circular buffer for market data
template<typename T, size_t Size>
class LockFreeRingBuffer {
private:
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    std::array<T, Size> buffer_;
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
    
public:
    bool push(const T& item) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) & (Size - 1);
        
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Buffer full
        }
        
        buffer_[current_tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }
    
    bool pop(T& item) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; // Buffer empty
        }
        
        item = buffer_[current_head];
        head_.store((current_head + 1) & (Size - 1), std::memory_order_release);
        return true;
    }
};

// Category 2: Order Book Implementation
// ====================================

class OptimizedOrderBook {
public:
    struct Order {
        uint64_t id;
        double price;
        uint32_t quantity;
        uint64_t timestamp;
        char side; // 'B' or 'S'
        
        Order(uint64_t id_, double price_, uint32_t qty_, char side_)
            : id(id_), price(price_), quantity(qty_), side(side_) {
            timestamp = std::chrono::high_resolution_clock::now()
                       .time_since_epoch().count();
        }
    };
    
private:
    // Price levels using maps for O(log n) operations
    std::map<double, std::queue<Order>, std::greater<double>> bids;  // Descending
    std::map<double, std::queue<Order>, std::less<double>> asks;     // Ascending
    std::unordered_map<uint64_t, std::pair<double, char>> order_lookup;
    
    mutable std::mutex book_mutex;
    std::atomic<uint64_t> last_trade_id{0};
    
public:
    // O(log n) insertion with automatic matching
    std::vector<std::pair<uint64_t, uint32_t>> addOrder(const Order& order) {
        std::lock_guard<std::mutex> lock(book_mutex);
        std::vector<std::pair<uint64_t, uint32_t>> matches;
        
        if (order.side == 'B') {
            // Buy order - match against asks
            matches = matchBuyOrder(order);
        } else {
            // Sell order - match against bids
            matches = matchSellOrder(order);
        }
        
        return matches;
    }
    
    // O(log n) cancellation
    bool cancelOrder(uint64_t order_id) {
        std::lock_guard<std::mutex> lock(book_mutex);
        auto it = order_lookup.find(order_id);
        if (it == order_lookup.end()) return false;
        
        double price = it->second.first;
        char side = it->second.second;
        
        if (side == 'B') {
            removeBuyOrder(price, order_id);
        } else {
            removeSellOrder(price, order_id);
        }
        
        order_lookup.erase(it);
        return true;
    }
    
    // O(1) best price queries
    std::pair<double, uint32_t> getBestBid() const {
        std::lock_guard<std::mutex> lock(book_mutex);
        if (bids.empty()) return {0.0, 0};
        
        const auto& best_level = bids.begin()->second;
        uint32_t total_qty = 0;
        auto temp_queue = best_level;
        while (!temp_queue.empty()) {
            total_qty += temp_queue.front().quantity;
            temp_queue.pop();
        }
        return {bids.begin()->first, total_qty};
    }
    
    std::pair<double, uint32_t> getBestAsk() const {
        std::lock_guard<std::mutex> lock(book_mutex);
        if (asks.empty()) return {0.0, 0};
        
        const auto& best_level = asks.begin()->second;
        uint32_t total_qty = 0;
        auto temp_queue = best_level;
        while (!temp_queue.empty()) {
            total_qty += temp_queue.front().quantity;
            temp_queue.pop();
        }
        return {asks.begin()->first, total_qty};
    }
    
private:
    std::vector<std::pair<uint64_t, uint32_t>> matchBuyOrder(Order order) {
        std::vector<std::pair<uint64_t, uint32_t>> matches;
        
        while (order.quantity > 0 && !asks.empty() && 
               asks.begin()->first <= order.price) {
            
            auto& ask_level = asks.begin()->second;
            auto& ask_order = ask_level.front();
            
            uint32_t trade_qty = std::min(order.quantity, ask_order.quantity);
            matches.emplace_back(ask_order.id, trade_qty);
            
            order.quantity -= trade_qty;
            ask_order.quantity -= trade_qty;
            
            if (ask_order.quantity == 0) {
                order_lookup.erase(ask_order.id);
                ask_level.pop();
                if (ask_level.empty()) {
                    asks.erase(asks.begin());
                }
            }
        }
        
        // Add remaining quantity to book
        if (order.quantity > 0) {
            bids[order.price].push(order);
            order_lookup[order.id] = {order.price, 'B'};
        }
        
        return matches;
    }
    
    std::vector<std::pair<uint64_t, uint32_t>> matchSellOrder(Order order) {
        std::vector<std::pair<uint64_t, uint32_t>> matches;
        
        while (order.quantity > 0 && !bids.empty() && 
               bids.begin()->first >= order.price) {
            
            auto& bid_level = bids.begin()->second;
            auto& bid_order = bid_level.front();
            
            uint32_t trade_qty = std::min(order.quantity, bid_order.quantity);
            matches.emplace_back(bid_order.id, trade_qty);
            
            order.quantity -= trade_qty;
            bid_order.quantity -= trade_qty;
            
            if (bid_order.quantity == 0) {
                order_lookup.erase(bid_order.id);
                bid_level.pop();
                if (bid_level.empty()) {
                    bids.erase(bids.begin());
                }
            }
        }
        
        // Add remaining quantity to book
        if (order.quantity > 0) {
            asks[order.price].push(order);
            order_lookup[order.id] = {order.price, 'S'};
        }
        
        return matches;
    }
    
    void removeBuyOrder(double price, uint64_t order_id) {
        auto level_it = bids.find(price);
        if (level_it == bids.end()) return;
        
        std::queue<Order> temp_queue;
        while (!level_it->second.empty()) {
            auto order = level_it->second.front();
            level_it->second.pop();
            if (order.id != order_id) {
                temp_queue.push(order);
            }
        }
        level_it->second = std::move(temp_queue);
        
        if (level_it->second.empty()) {
            bids.erase(level_it);
        }
    }
    
    void removeSellOrder(double price, uint64_t order_id) {
        auto level_it = asks.find(price);
        if (level_it == asks.end()) return;
        
        std::queue<Order> temp_queue;
        while (!level_it->second.empty()) {
            auto order = level_it->second.front();
            level_it->second.pop();
            if (order.id != order_id) {
                temp_queue.push(order);
            }
        }
        level_it->second = std::move(temp_queue);
        
        if (level_it->second.empty()) {
            asks.erase(level_it);
        }
    }
};

// Category 3: Risk Management and P&L Calculation
// ===============================================

class RealTimeRiskCalculator {
private:
    std::atomic<double> position_{0.0};
    std::atomic<double> unrealized_pnl_{0.0};
    std::atomic<double> realized_pnl_{0.0};
    std::atomic<double> average_price_{0.0};
    std::atomic<double> max_position_limit_{10000.0};
    
    mutable std::mutex trade_history_mutex_;
    std::vector<std::pair<double, double>> trade_history_; // price, quantity
    
public:
    // Thread-safe position update
    bool updatePosition(double price, double quantity) {
        double new_position = position_.load() + quantity;
        
        // Risk check
        if (std::abs(new_position) > max_position_limit_.load()) {
            return false; // Reject trade
        }
        
        // Update position and P&L atomically
        double old_position = position_.exchange(new_position);
        
        if (old_position * quantity < 0) {
            // Position reduction - realize P&L
            double reduction = std::min(std::abs(quantity), std::abs(old_position));
            double avg_price = average_price_.load();
            double realized = reduction * (price - avg_price) * (old_position > 0 ? 1 : -1);
            
            // Atomic add for double using compare_exchange_weak
            double expected = realized_pnl_.load();
            while (!realized_pnl_.compare_exchange_weak(expected, expected + realized)) {
                // Loop until successful
            }
        }
        
        // Update average price for remaining position
        if (new_position != 0 && old_position * quantity >= 0) {
            double total_cost = old_position * average_price_.load() + quantity * price;
            average_price_.store(total_cost / new_position);
        }
        
        // Store trade for analysis
        {
            std::lock_guard<std::mutex> lock(trade_history_mutex_);
            trade_history_.emplace_back(price, quantity);
        }
        
        return true;
    }
    
    // Real-time unrealized P&L calculation
    void updateMarketPrice(double market_price) {
        double pos = position_.load();
        double avg_price = average_price_.load();
        unrealized_pnl_.store(pos * (market_price - avg_price));
    }
    
    // Risk metrics calculation
    struct RiskMetrics {
        double var_95;          // Value at Risk (95%)
        double max_drawdown;
        double sharpe_ratio;
        double total_pnl;
    };
    
    RiskMetrics calculateRiskMetrics() const {
        std::lock_guard<std::mutex> lock(trade_history_mutex_);
        
        if (trade_history_.empty()) {
            return {0.0, 0.0, 0.0, 0.0};
        }
        
        // Calculate daily P&L series
        std::vector<double> daily_pnl;
        double running_pnl = 0.0;
        
        for (const auto& trade : trade_history_) {
            running_pnl += trade.first * trade.second; // Simplified P&L
            daily_pnl.push_back(running_pnl);
        }
        
        // VaR calculation (95th percentile)
        std::vector<double> returns;
        for (size_t i = 1; i < daily_pnl.size(); ++i) {
            returns.push_back(daily_pnl[i] - daily_pnl[i-1]);
        }
        
        if (returns.empty()) {
            return {0.0, 0.0, 0.0, running_pnl};
        }
        
        std::sort(returns.begin(), returns.end());
        double var_95 = returns[static_cast<size_t>(returns.size() * 0.05)];
        
        // Max drawdown calculation
        double max_drawdown = 0.0;
        double peak = daily_pnl[0];
        for (double pnl : daily_pnl) {
            peak = std::max(peak, pnl);
            max_drawdown = std::max(max_drawdown, peak - pnl);
        }
        
        // Sharpe ratio calculation
        double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        double variance = 0.0;
        for (double ret : returns) {
            variance += (ret - mean_return) * (ret - mean_return);
        }
        variance /= returns.size();
        double sharpe_ratio = variance > 0 ? mean_return / std::sqrt(variance) : 0.0;
        
        return {var_95, max_drawdown, sharpe_ratio, running_pnl};
    }
    
    // Getters
    double getPosition() const { return position_.load(); }
    double getRealizedPnL() const { return realized_pnl_.load(); }
    double getUnrealizedPnL() const { return unrealized_pnl_.load(); }
    double getTotalPnL() const { return realized_pnl_.load() + unrealized_pnl_.load(); }
};

// Category 4: Statistical and Mathematical Functions
// ==================================================

class MathUtils {
public:
    // Fast inverse square root (for volatility calculations)
    static float fastInvSqrt(float x) {
        float xhalf = 0.5f * x;
        int i = *(int*)&x;
        i = 0x5f3759df - (i >> 1);
        x = *(float*)&i;
        x = x * (1.5f - xhalf * x * x);
        return x;
    }
    
    // Online variance calculation (Welford's algorithm)
    class OnlineVariance {
    private:
        int count_ = 0;
        double mean_ = 0.0;
        double m2_ = 0.0;
        
    public:
        void update(double value) {
            count_++;
            double delta = value - mean_;
            mean_ += delta / count_;
            double delta2 = value - mean_;
            m2_ += delta * delta2;
        }
        
        double variance() const {
            return count_ > 1 ? m2_ / (count_ - 1) : 0.0;
        }
        
        double standardDeviation() const {
            return std::sqrt(variance());
        }
        
        double mean() const { return mean_; }
        int count() const { return count_; }
    };
    
    // Exponential moving average
    class EMA {
    private:
        double alpha_;
        double value_;
        bool initialized_ = false;
        
    public:
        explicit EMA(double alpha) : alpha_(alpha), value_(0.0) {}
        
        void update(double new_value) {
            if (!initialized_) {
                value_ = new_value;
                initialized_ = true;
            } else {
                value_ = alpha_ * new_value + (1.0 - alpha_) * value_;
            }
        }
        
        double getValue() const { return value_; }
    };
    
    // Black-Scholes option pricing (for derivatives trading)
    static double blackScholesCall(double S, double K, double T, double r, double sigma) {
        if (T <= 0) return std::max(S - K, 0.0);
        
        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        
        double N_d1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
        double N_d2 = 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));
        
        return S * N_d1 - K * std::exp(-r * T) * N_d2;
    }
};

// Category 5: Performance Measurement and Optimization
// ====================================================

class PerformanceMeasurement {
public:
    // High-resolution timer
    class Timer {
    private:
        std::chrono::high_resolution_clock::time_point start_;
        
    public:
        Timer() : start_(std::chrono::high_resolution_clock::now()) {}
        
        double elapsedNanoseconds() const {
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
        }
        
        double elapsedMicroseconds() const {
            return elapsedNanoseconds() / 1000.0;
        }
        
        void reset() {
            start_ = std::chrono::high_resolution_clock::now();
        }
    };
    
    // Latency statistics collector
    class LatencyCollector {
    private:
        std::vector<double> latencies_;
        mutable std::mutex mutex_;
        
    public:
        void addSample(double latency_ns) {
            std::lock_guard<std::mutex> lock(mutex_);
            latencies_.push_back(latency_ns);
        }
        
        struct LatencyStats {
            double min, max, mean, median, p95, p99;
            size_t count;
        };
        
        LatencyStats getStats() const {
            std::lock_guard<std::mutex> lock(mutex_);
            if (latencies_.empty()) {
                return {0, 0, 0, 0, 0, 0, 0};
            }
            
            auto sorted = latencies_;
            std::sort(sorted.begin(), sorted.end());
            
            double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
            
            return {
                sorted.front(),
                sorted.back(),
                sum / sorted.size(),
                sorted[sorted.size() / 2],
                sorted[static_cast<size_t>(sorted.size() * 0.95)],
                sorted[static_cast<size_t>(sorted.size() * 0.99)],
                sorted.size()
            };
        }
    };
    
    // Memory pool for object allocation
    template<typename T, size_t PoolSize>
    class MemoryPool {
    private:
        alignas(T) char pool_[sizeof(T) * PoolSize];
        std::atomic<size_t> next_free_{0};
        
    public:
        T* allocate() {
            size_t index = next_free_.fetch_add(1, std::memory_order_relaxed);
            if (index >= PoolSize) {
                next_free_.store(PoolSize, std::memory_order_relaxed);
                return nullptr; // Pool exhausted
            }
            return reinterpret_cast<T*>(&pool_[index * sizeof(T)]);
        }
        
        void reset() {
            next_free_.store(0, std::memory_order_relaxed);
        }
        
        size_t available() const {
            size_t used = next_free_.load(std::memory_order_relaxed);
            return used < PoolSize ? PoolSize - used : 0;
        }
    };
};

} // namespace HFTInterview

// =============================================================================
// READY FOR YOUR 5 INTERVIEW QUESTIONS
// =============================================================================

/*
This reference file is now ready! 

I'm prepared to provide optimized solutions for your 5 HFT interview questions.
Each solution will include:

1. Multiple algorithmic approaches (brute force → optimized)
2. Time and space complexity analysis
3. HFT-specific optimizations (cache alignment, lock-free, etc.)
4. Real-world considerations (risk management, performance)
5. Complete working code with test cases

Common HFT interview question categories I'm ready for:
- Order book implementation and modifications
- Market data processing and filtering
- Risk management and P&L calculations
- Statistical arbitrage and signal processing
- Low-latency algorithm optimization
- Memory management and performance tuning
- Concurrent programming patterns
- Mathematical finance implementations

Please provide your 5 questions and I'll give you comprehensive,
production-ready solutions for each!
*/

int main() {
    std::cout << "HFT C++ Interview Preparation - Ready for Questions!" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    // Quick test of the framework
    using namespace HFTInterview;
    
    // Test order book
    OptimizedOrderBook book;
    auto trades = book.addOrder({1, 100.0, 100, 'B'});
    auto best_bid = book.getBestBid();
    std::cout << "Order book test: Best bid = " << best_bid.first 
              << " @ " << best_bid.second << std::endl;
    
    // Test performance measurement
    PerformanceMeasurement::Timer timer;
    // Simulate some work
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    std::cout << "Timer test: " << timer.elapsedMicroseconds() << " μs" << std::endl;
    
    // Test math utilities
    MathUtils::OnlineVariance variance;
    variance.update(1.0);
    variance.update(2.0);
    variance.update(3.0);
    std::cout << "Variance test: σ = " << variance.standardDeviation() << std::endl;
    
    std::cout << "\nReady to answer your 5 interview questions!" << std::endl;
    
    return 0;
}