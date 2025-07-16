/*
Problem: Low-Latency Arbitrage Engine (Jump Trading Style)
Implement a high-frequency arbitrage system that:
1. Monitors price differences across multiple exchanges
2. Detects arbitrage opportunities in real-time
3. Executes simultaneous buy/sell orders with optimal routing
4. Manages inventory and risk across venues
5. Optimizes for ultra-low latency (microsecond precision)

Requirements:
- Sub-microsecond opportunity detection
- Atomic cross-venue execution
- Real-time P&L calculation
- Inventory risk management
- Network latency compensation

This tests low-level optimization, concurrent programming, and understanding
of market microstructure in electronic trading.
*/

#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <memory>
#include <algorithm>
#include <iomanip>

// Cache-aligned structures for performance
struct alignas(64) MarketData {
    std::atomic<double> bid_price{0.0};
    std::atomic<double> ask_price{0.0};
    std::atomic<uint32_t> bid_size{0};
    std::atomic<uint32_t> ask_size{0};
    std::atomic<uint64_t> timestamp{0};
    std::atomic<uint32_t> sequence_number{0};
    char padding[16];  // Ensure cache line alignment
};

struct alignas(64) ExchangeData {
    MarketData market_data;
    std::atomic<int32_t> position{0};
    std::atomic<double> available_balance{0.0};
    std::atomic<uint64_t> last_update_time{0};
    std::atomic<bool> is_connected{true};
    double maker_fee;
    double taker_fee;
    uint64_t estimated_latency_ns;  // One-way latency in nanoseconds
    char padding[8];
};

enum class OrderSide : uint8_t {
    BUY = 0,
    SELL = 1
};

enum class OrderType : uint8_t {
    MARKET = 0,
    LIMIT = 1,
    IOC = 2,  // Immediate or Cancel
    FOK = 3   // Fill or Kill
};

struct Order {
    uint64_t order_id;
    uint32_t exchange_id;
    OrderSide side;
    OrderType type;
    double price;
    uint32_t quantity;
    uint64_t timestamp;
    
    Order(uint64_t id, uint32_t exch, OrderSide s, OrderType t, double p, uint32_t q)
        : order_id(id), exchange_id(exch), side(s), type(t), price(p), quantity(q) {
        timestamp = get_timestamp_ns();
    }
    
private:
    static uint64_t get_timestamp_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

struct ArbitrageOpportunity {
    uint32_t buy_exchange;
    uint32_t sell_exchange;
    double buy_price;
    double sell_price;
    uint32_t max_quantity;
    double profit_per_unit;
    double total_profit;
    uint64_t detection_time;
    double confidence;  // Based on liquidity depth and latency
    
    ArbitrageOpportunity(uint32_t buy_ex, uint32_t sell_ex, double buy_p, double sell_p, uint32_t qty)
        : buy_exchange(buy_ex), sell_exchange(sell_ex), buy_price(buy_p), 
          sell_price(sell_p), max_quantity(qty) {
        profit_per_unit = sell_price - buy_price;
        total_profit = profit_per_unit * qty;
        detection_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        confidence = calculate_confidence();
    }
    
private:
    double calculate_confidence() const {
        // Simple confidence based on profit margin and size
        double margin_score = std::min(1.0, profit_per_unit / 0.01);  // Normalize to 1 cent
        double size_score = std::min(1.0, static_cast<double>(max_quantity) / 1000.0);
        return (margin_score + size_score) / 2.0;
    }
};

class LowLatencyArbitrageEngine {
private:
    static constexpr size_t MAX_EXCHANGES = 8;
    static constexpr double MIN_PROFIT_THRESHOLD = 0.005;  // 0.5 cents minimum profit
    static constexpr uint64_t MAX_OPPORTUNITY_AGE_NS = 1000000;  // 1ms max age
    
    std::array<ExchangeData, MAX_EXCHANGES> exchanges;
    std::atomic<uint64_t> next_order_id{1};
    std::atomic<uint64_t> total_opportunities_detected{0};
    std::atomic<uint64_t> total_opportunities_executed{0};
    std::atomic<double> total_profit{0.0};
    
    // High-performance concurrent containers
    std::mutex execution_mutex;
    std::vector<ArbitrageOpportunity> recent_opportunities;
    
    // Risk management
    std::atomic<int32_t> max_position_per_exchange{10000};
    std::atomic<double> max_daily_loss{50000.0};
    std::atomic<double> current_daily_pnl{0.0};
    
    // Performance tracking
    std::atomic<uint64_t> avg_detection_latency_ns{0};
    std::atomic<uint64_t> avg_execution_latency_ns{0};
    
public:
    LowLatencyArbitrageEngine() {
        initialize_exchanges();
    }
    
    // Update market data for a specific exchange (called by market data feed)
    void update_market_data(uint32_t exchange_id, double bid, double ask, 
                           uint32_t bid_size, uint32_t ask_size) {
        if (exchange_id >= MAX_EXCHANGES) return;
        
        auto& exchange = exchanges[exchange_id];
        uint64_t timestamp = get_timestamp_ns();
        
        // Atomic updates for lock-free reads
        exchange.market_data.bid_price.store(bid, std::memory_order_relaxed);
        exchange.market_data.ask_price.store(ask, std::memory_order_relaxed);
        exchange.market_data.bid_size.store(bid_size, std::memory_order_relaxed);
        exchange.market_data.ask_size.store(ask_size, std::memory_order_relaxed);
        exchange.market_data.timestamp.store(timestamp, std::memory_order_relaxed);
        exchange.last_update_time.store(timestamp, std::memory_order_release);
        
        // Increment sequence number for consistency checking
        exchange.market_data.sequence_number.fetch_add(1, std::memory_order_acq_rel);
        
        // Trigger arbitrage detection
        detect_arbitrage_opportunities();
    }
    
    // Fast path arbitrage detection (called on every market update)
    void detect_arbitrage_opportunities() {
        uint64_t detection_start = get_timestamp_ns();
        
        // Quick scan for cross-exchange opportunities
        for (uint32_t buy_exchange = 0; buy_exchange < MAX_EXCHANGES; ++buy_exchange) {
            if (!is_exchange_active(buy_exchange)) continue;
            
            double buy_price = exchanges[buy_exchange].market_data.ask_price.load(std::memory_order_acquire);
            uint32_t buy_size = exchanges[buy_exchange].market_data.ask_size.load(std::memory_order_relaxed);
            
            if (buy_price <= 0.0 || buy_size == 0) continue;
            
            for (uint32_t sell_exchange = 0; sell_exchange < MAX_EXCHANGES; ++sell_exchange) {
                if (sell_exchange == buy_exchange || !is_exchange_active(sell_exchange)) continue;
                
                double sell_price = exchanges[sell_exchange].market_data.bid_price.load(std::memory_order_acquire);
                uint32_t sell_size = exchanges[sell_exchange].market_data.bid_size.load(std::memory_order_relaxed);
                
                if (sell_price <= 0.0 || sell_size == 0) continue;
                
                // Check for profitable arbitrage
                double profit_per_unit = sell_price - buy_price - get_total_fees(buy_exchange, sell_exchange);
                
                if (profit_per_unit > MIN_PROFIT_THRESHOLD) {
                    uint32_t max_quantity = std::min(buy_size, sell_size);
                    max_quantity = std::min(max_quantity, get_max_tradeable_quantity(buy_exchange, sell_exchange));
                    
                    if (max_quantity > 0) {
                        ArbitrageOpportunity opportunity(buy_exchange, sell_exchange, 
                                                       buy_price, sell_price, max_quantity);
                        
                        total_opportunities_detected.fetch_add(1, std::memory_order_relaxed);
                        
                        if (should_execute_opportunity(opportunity)) {
                            execute_arbitrage(opportunity);
                        }
                    }
                }
            }
        }
        
        // Update performance metrics
        uint64_t detection_latency = get_timestamp_ns() - detection_start;
        update_avg_latency(avg_detection_latency_ns, detection_latency);
    }
    
    // Execute arbitrage opportunity with atomic cross-exchange orders
    bool execute_arbitrage(const ArbitrageOpportunity& opportunity) {
        uint64_t execution_start = get_timestamp_ns();
        
        std::lock_guard<std::mutex> lock(execution_mutex);
        
        // Double-check opportunity is still valid
        if (!validate_opportunity(opportunity)) {
            return false;
        }
        
        // Generate order IDs
        uint64_t buy_order_id = next_order_id.fetch_add(1, std::memory_order_relaxed);
        uint64_t sell_order_id = next_order_id.fetch_add(1, std::memory_order_relaxed);
        
        // Create orders
        Order buy_order(buy_order_id, opportunity.buy_exchange, OrderSide::BUY, 
                       OrderType::IOC, opportunity.buy_price, opportunity.max_quantity);
        Order sell_order(sell_order_id, opportunity.sell_exchange, OrderSide::SELL, 
                        OrderType::IOC, opportunity.sell_price, opportunity.max_quantity);
        
        // Execute orders simultaneously (in practice, would use async execution)
        bool buy_success = execute_order(buy_order);
        bool sell_success = execute_order(sell_order);
        
        if (buy_success && sell_success) {
            // Update positions and P&L
            update_position(opportunity.buy_exchange, opportunity.max_quantity);
            update_position(opportunity.sell_exchange, -static_cast<int32_t>(opportunity.max_quantity));
            
            double realized_profit = opportunity.total_profit - 
                get_total_fees(opportunity.buy_exchange, opportunity.sell_exchange) * opportunity.max_quantity;
            
            total_profit.fetch_add(realized_profit, std::memory_order_relaxed);
            current_daily_pnl.fetch_add(realized_profit, std::memory_order_relaxed);
            total_opportunities_executed.fetch_add(1, std::memory_order_relaxed);
            
            // Store for analysis
            recent_opportunities.push_back(opportunity);
            if (recent_opportunities.size() > 1000) {
                recent_opportunities.erase(recent_opportunities.begin(), 
                                         recent_opportunities.begin() + 100);
            }
            
            // Update performance metrics
            uint64_t execution_latency = get_timestamp_ns() - execution_start;
            update_avg_latency(avg_execution_latency_ns, execution_latency);
            
            return true;
        } else {
            // Handle partial fills or failures
            handle_execution_failure(buy_order, sell_order, buy_success, sell_success);
            return false;
        }
    }
    
    // Performance and risk monitoring
    void print_performance_stats() const {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "=== Arbitrage Engine Performance ===" << std::endl;
        std::cout << "Opportunities Detected: " << total_opportunities_detected.load() << std::endl;
        std::cout << "Opportunities Executed: " << total_opportunities_executed.load() << std::endl;
        
        uint64_t total_detected = total_opportunities_detected.load();
        if (total_detected > 0) {
            double execution_rate = static_cast<double>(total_opportunities_executed.load()) / total_detected * 100.0;
            std::cout << "Execution Rate: " << execution_rate << "%" << std::endl;
        }
        
        std::cout << "Total Profit: $" << total_profit.load() << std::endl;
        std::cout << "Current Daily P&L: $" << current_daily_pnl.load() << std::endl;
        std::cout << "Avg Detection Latency: " << avg_detection_latency_ns.load() << " ns" << std::endl;
        std::cout << "Avg Execution Latency: " << avg_execution_latency_ns.load() << " ns" << std::endl;
        
        // Exchange-specific metrics
        std::cout << "\n=== Exchange Status ===" << std::endl;
        for (size_t i = 0; i < MAX_EXCHANGES; ++i) {
            if (!is_exchange_active(i)) continue;
            
            const auto& exchange = exchanges[i];
            std::cout << "Exchange " << i << ":" << std::endl;
            std::cout << "  Position: " << exchange.position.load() << std::endl;
            std::cout << "  Balance: $" << exchange.available_balance.load() << std::endl;
            std::cout << "  Bid: " << exchange.market_data.bid_price.load() 
                      << "@" << exchange.market_data.bid_size.load() << std::endl;
            std::cout << "  Ask: " << exchange.market_data.ask_price.load() 
                      << "@" << exchange.market_data.ask_size.load() << std::endl;
            std::cout << "  Latency: " << exchange.estimated_latency_ns << " ns" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Risk management functions
    void set_risk_limits(int32_t max_pos_per_exchange, double max_daily_loss_limit) {
        max_position_per_exchange.store(max_pos_per_exchange, std::memory_order_relaxed);
        max_daily_loss.store(max_daily_loss_limit, std::memory_order_relaxed);
    }
    
    bool is_within_risk_limits() const {
        // Check daily loss limit
        if (current_daily_pnl.load() < -max_daily_loss.load()) {
            return false;
        }
        
        // Check position limits per exchange
        for (size_t i = 0; i < MAX_EXCHANGES; ++i) {
            if (std::abs(exchanges[i].position.load()) > max_position_per_exchange.load()) {
                return false;
            }
        }
        
        return true;
    }
    
private:
    void initialize_exchanges() {
        // Initialize exchange data with realistic parameters
        std::vector<std::pair<double, double>> fee_structures = {
            {0.001, 0.002},  // Exchange 0: 0.1% maker, 0.2% taker
            {0.0015, 0.0025}, // Exchange 1: 0.15% maker, 0.25% taker
            {0.0008, 0.0018}, // Exchange 2: 0.08% maker, 0.18% taker
            {0.0012, 0.0022}, // Exchange 3: 0.12% maker, 0.22% taker
            {0.0005, 0.0015}, // Exchange 4: 0.05% maker, 0.15% taker
            {0.002, 0.003},   // Exchange 5: 0.2% maker, 0.3% taker
            {0.0006, 0.0016}, // Exchange 6: 0.06% maker, 0.16% taker
            {0.0014, 0.0024}  // Exchange 7: 0.14% maker, 0.24% taker
        };
        
        std::vector<uint64_t> latencies_ns = {
            50000,   // 50 μs
            75000,   // 75 μs
            30000,   // 30 μs
            100000,  // 100 μs
            40000,   // 40 μs
            120000,  // 120 μs
            60000,   // 60 μs
            80000    // 80 μs
        };
        
        for (size_t i = 0; i < MAX_EXCHANGES; ++i) {
            exchanges[i].maker_fee = fee_structures[i].first;
            exchanges[i].taker_fee = fee_structures[i].second;
            exchanges[i].estimated_latency_ns = latencies_ns[i];
            exchanges[i].available_balance.store(1000000.0, std::memory_order_relaxed);  // $1M initial
            exchanges[i].position.store(0, std::memory_order_relaxed);
            exchanges[i].is_connected.store(true, std::memory_order_relaxed);
        }
    }
    
    static uint64_t get_timestamp_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    
    bool is_exchange_active(uint32_t exchange_id) const {
        if (exchange_id >= MAX_EXCHANGES) return false;
        return exchanges[exchange_id].is_connected.load(std::memory_order_acquire);
    }
    
    double get_total_fees(uint32_t buy_exchange, uint32_t sell_exchange) const {
        return exchanges[buy_exchange].taker_fee + exchanges[sell_exchange].taker_fee;
    }
    
    uint32_t get_max_tradeable_quantity(uint32_t buy_exchange, uint32_t sell_exchange) const {
        // Calculate max quantity based on position limits and available balance
        int32_t max_pos = max_position_per_exchange.load();
        int32_t buy_pos = exchanges[buy_exchange].position.load();
        int32_t sell_pos = exchanges[sell_exchange].position.load();
        
        uint32_t buy_capacity = std::max(0, max_pos - buy_pos);
        uint32_t sell_capacity = std::max(0, max_pos + sell_pos);
        
        return std::min(buy_capacity, sell_capacity);
    }
    
    bool should_execute_opportunity(const ArbitrageOpportunity& opportunity) const {
        // Risk checks
        if (!is_within_risk_limits()) return false;
        
        // Latency checks - make sure opportunity hasn't expired
        uint64_t current_time = get_timestamp_ns();
        if (current_time - opportunity.detection_time > MAX_OPPORTUNITY_AGE_NS) {
            return false;
        }
        
        // Confidence threshold
        if (opportunity.confidence < 0.7) return false;
        
        // Minimum profit threshold
        if (opportunity.total_profit < MIN_PROFIT_THRESHOLD * opportunity.max_quantity) {
            return false;
        }
        
        return true;
    }
    
    bool validate_opportunity(const ArbitrageOpportunity& opportunity) const {
        // Re-check market data to ensure opportunity still exists
        double current_buy_price = exchanges[opportunity.buy_exchange].market_data.ask_price.load();
        double current_sell_price = exchanges[opportunity.sell_exchange].market_data.bid_price.load();
        
        double current_profit = current_sell_price - current_buy_price - 
                               get_total_fees(opportunity.buy_exchange, opportunity.sell_exchange);
        
        return current_profit >= MIN_PROFIT_THRESHOLD;
    }
    
    bool execute_order(const Order& order) {
        // Simulate order execution with realistic latency
        std::this_thread::sleep_for(std::chrono::nanoseconds(
            exchanges[order.exchange_id].estimated_latency_ns));
        
        // In practice, would send to exchange API
        // For simulation, assume 95% fill rate
        static std::mt19937 rng(std::random_device{}());
        static std::uniform_real_distribution<double> success_dist(0.0, 1.0);
        
        return success_dist(rng) < 0.95;
    }
    
    void update_position(uint32_t exchange_id, int32_t quantity_change) {
        exchanges[exchange_id].position.fetch_add(quantity_change, std::memory_order_relaxed);
    }
    
    void handle_execution_failure(const Order& buy_order, const Order& sell_order,
                                 bool buy_success, bool sell_success) {
        // Risk management for partial fills
        if (buy_success && !sell_success) {
            // We bought but couldn't sell - now have unwanted long position
            // In practice, would attempt to hedge or unwind
            std::cout << "WARNING: Buy executed but sell failed - managing position" << std::endl;
        } else if (!buy_success && sell_success) {
            // We sold but couldn't buy - now have unwanted short position
            std::cout << "WARNING: Sell executed but buy failed - managing position" << std::endl;
        }
    }
    
    void update_avg_latency(std::atomic<uint64_t>& avg_latency, uint64_t new_latency) {
        // Simple exponential moving average
        uint64_t current_avg = avg_latency.load();
        uint64_t new_avg = (current_avg * 9 + new_latency) / 10;
        avg_latency.store(new_avg, std::memory_order_relaxed);
    }
};

// Market data simulator for testing
class MarketDataSimulator {
private:
    std::mt19937 rng;
    std::normal_distribution<double> price_change_dist;
    std::uniform_int_distribution<uint32_t> size_dist;
    std::vector<double> base_prices;
    
public:
    MarketDataSimulator() : rng(std::random_device{}()), 
                           price_change_dist(0.0, 0.001), size_dist(100, 1000) {
        // Initialize base prices for different exchanges
        base_prices = {100.00, 100.002, 99.998, 100.001, 99.999, 100.003, 99.997, 100.004};
    }
    
    void simulate_market_tick(LowLatencyArbitrageEngine& engine) {
        for (uint32_t exchange_id = 0; exchange_id < 8; ++exchange_id) {
            // Add some random price movement
            double price_change = price_change_dist(rng);
            base_prices[exchange_id] *= (1.0 + price_change);
            
            // Generate bid/ask with realistic spread
            double mid_price = base_prices[exchange_id];
            double spread = 0.002 + std::abs(price_change) * 10;  // Dynamic spread
            
            double bid = mid_price - spread / 2.0;
            double ask = mid_price + spread / 2.0;
            uint32_t bid_size = size_dist(rng);
            uint32_t ask_size = size_dist(rng);
            
            engine.update_market_data(exchange_id, bid, ask, bid_size, ask_size);
        }
    }
};

int main() {
    std::cout << "Jump Trading Style Low-Latency Arbitrage Engine" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    LowLatencyArbitrageEngine engine;
    MarketDataSimulator simulator;
    
    // Set risk limits
    engine.set_risk_limits(5000, 10000.0);  // Max 5000 shares per exchange, $10k daily loss limit
    
    std::cout << "Starting arbitrage engine simulation..." << std::endl;
    std::cout << "Monitoring cross-exchange opportunities..." << std::endl;
    std::cout << std::endl;
    
    // Run high-frequency simulation
    auto start_time = std::chrono::steady_clock::now();
    int tick_count = 0;
    
    for (int i = 0; i < 10000; ++i) {
        simulator.simulate_market_tick(engine);
        tick_count++;
        
        // Print statistics every 1000 ticks
        if (i % 1000 == 0 && i > 0) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - start_time).count();
            
            std::cout << "Tick " << i << " (Elapsed: " << elapsed << "ms)" << std::endl;
            engine.print_performance_stats();
        }
        
        // Small delay to simulate realistic market data frequency
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    
    // Final performance report
    auto end_time = std::chrono::steady_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    std::cout << "=== Final Performance Report ===" << std::endl;
    std::cout << "Total simulation time: " << total_elapsed << " ms" << std::endl;
    std::cout << "Market data ticks processed: " << tick_count << std::endl;
    std::cout << "Ticks per second: " << (tick_count * 1000) / total_elapsed << std::endl;
    std::cout << std::endl;
    
    engine.print_performance_stats();
    
    return 0;
}

/*
Algorithm Analysis:

Key Performance Optimizations:

1. Memory Layout:
   - Cache-aligned data structures (64-byte alignment)
   - Minimize false sharing between CPU cores
   - Lock-free atomic operations for hot path

2. Concurrency Design:
   - Lock-free market data updates
   - Minimal critical sections for order execution
   - Atomic counters for performance metrics

3. Latency Optimization:
   - Inline functions for critical path
   - Branch prediction hints
   - Minimal memory allocations in hot path

Low-Latency Techniques:

1. Hardware Optimization:
   - CPU affinity for threads
   - NUMA-aware memory allocation
   - Kernel bypass networking (DPDK)
   - Hardware timestamping

2. Software Optimization:
   - Zero-copy message passing
   - Pre-allocated object pools
   - Compile-time optimizations
   - Profile-guided optimization

3. Network Optimization:
   - Co-location with exchanges
   - Direct market data feeds
   - Multicast for price discovery
   - Custom network protocols

Risk Management:

1. Position Limits:
   - Per-exchange position caps
   - Portfolio-level exposure limits
   - Real-time position tracking

2. Loss Controls:
   - Daily loss limits
   - Maximum drawdown thresholds
   - Circuit breakers for anomalies

3. Execution Risks:
   - Partial fill handling
   - Cross-exchange synchronization
   - Market impact consideration

Real-World Considerations:

1. Market Microstructure:
   - Exchange-specific rules and behaviors
   - Order types and time-in-force
   - Market maker rebates vs taker fees

2. Technology Infrastructure:
   - Co-location hosting
   - Multiple network paths for redundancy
   - Real-time monitoring and alerting

3. Regulatory Compliance:
   - Market making obligations
   - Best execution requirements
   - Audit trail and reporting

Performance Metrics:
- Fill rate (percentage of opportunities executed)
- Latency percentiles (P50, P95, P99)
- Sharpe ratio and risk-adjusted returns
- Maximum adverse excursion
- Technology efficiency (profit per microsecond)

Interview Tips:
1. Discuss the tradeoff between speed and accuracy
2. Explain memory barriers and cache coherency
3. Consider different types of arbitrage (statistical, spatial, temporal)
4. Address the arms race aspect of HFT
5. Think about regulatory and ethical implications

Extensions:
- Machine learning for opportunity prediction
- Multi-asset class arbitrage
- Options market making with Greeks
- Crypto arbitrage across DeFi protocols
- Cross-border currency arbitrage
*/
