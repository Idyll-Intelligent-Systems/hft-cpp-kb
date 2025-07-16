/*
Problem: Order Book Implementation with Fast Lookups (Citadel Style)
Implement an order book that supports:
1. Add order (BUY/SELL, price, quantity, order_id)
2. Cancel order (order_id)
3. Get best bid/ask prices
4. Get volume at price level
5. Match orders when possible

Requirements:
- O(log n) for add/cancel operations
- O(1) for best bid/ask queries
- Handle order matching automatically
- Memory efficient implementation

This is a typical HFT interview question focusing on low-latency data structures.
*/

#include <iostream>
#include <map>
#include <unordered_map>
#include <queue>
#include <memory>
#include <chrono>

enum class OrderSide {
    BUY,
    SELL
};

struct Order {
    uint64_t order_id;
    OrderSide side;
    double price;
    uint64_t quantity;
    uint64_t timestamp;
    
    Order(uint64_t id, OrderSide s, double p, uint64_t q) 
        : order_id(id), side(s), price(p), quantity(q) {
        timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
};

struct PriceLevel {
    double price;
    uint64_t total_quantity;
    std::queue<std::shared_ptr<Order>> orders;  // FIFO for same price
    
    PriceLevel(double p) : price(p), total_quantity(0) {}
    
    void add_order(std::shared_ptr<Order> order) {
        orders.push(order);
        total_quantity += order->quantity;
    }
    
    bool remove_order(uint64_t order_id) {
        // In production, would use more efficient data structure
        std::queue<std::shared_ptr<Order>> temp_queue;
        bool found = false;
        
        while (!orders.empty()) {
            auto order = orders.front();
            orders.pop();
            
            if (order->order_id == order_id) {
                total_quantity -= order->quantity;
                found = true;
            } else {
                temp_queue.push(order);
            }
        }
        
        orders = std::move(temp_queue);
        return found;
    }
    
    std::shared_ptr<Order> get_next_order() {
        if (orders.empty()) return nullptr;
        return orders.front();
    }
    
    void remove_front_order(uint64_t executed_quantity) {
        if (orders.empty()) return;
        
        auto order = orders.front();
        if (order->quantity <= executed_quantity) {
            total_quantity -= order->quantity;
            orders.pop();
        } else {
            order->quantity -= executed_quantity;
            total_quantity -= executed_quantity;
        }
    }
    
    bool is_empty() const {
        return orders.empty();
    }
};

struct Trade {
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    double price;
    uint64_t quantity;
    uint64_t timestamp;
    
    Trade(uint64_t bid, uint64_t sid, double p, uint64_t q) 
        : buy_order_id(bid), sell_order_id(sid), price(p), quantity(q) {
        timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
};

class OrderBook {
private:
    // Price levels - using map for sorted prices
    std::map<double, std::shared_ptr<PriceLevel>, std::greater<double>> bids;  // Descending order
    std::map<double, std::shared_ptr<PriceLevel>> asks;  // Ascending order
    
    // Fast order lookup
    std::unordered_map<uint64_t, std::shared_ptr<Order>> orders;
    
    // Trade history
    std::vector<Trade> trades;
    
    // Statistics
    uint64_t total_trades;
    uint64_t total_volume;
    
public:
    OrderBook() : total_trades(0), total_volume(0) {}
    
    // Add order and attempt to match
    std::vector<Trade> add_order(uint64_t order_id, OrderSide side, double price, uint64_t quantity) {
        auto order = std::make_shared<Order>(order_id, side, price, quantity);
        orders[order_id] = order;
        
        std::vector<Trade> new_trades;
        
        if (side == OrderSide::BUY) {
            new_trades = match_buy_order(order);
            if (order->quantity > 0) {
                add_to_book(order);
            }
        } else {
            new_trades = match_sell_order(order);
            if (order->quantity > 0) {
                add_to_book(order);
            }
        }
        
        // Update statistics
        for (const auto& trade : new_trades) {
            total_trades++;
            total_volume += trade.quantity;
            trades.push_back(trade);
        }
        
        return new_trades;
    }
    
    // Cancel order
    bool cancel_order(uint64_t order_id) {
        auto it = orders.find(order_id);
        if (it == orders.end()) {
            return false;
        }
        
        auto order = it->second;
        bool removed = false;
        
        if (order->side == OrderSide::BUY) {
            auto price_it = bids.find(order->price);
            if (price_it != bids.end()) {
                removed = price_it->second->remove_order(order_id);
                if (price_it->second->is_empty()) {
                    bids.erase(price_it);
                }
            }
        } else {
            auto price_it = asks.find(order->price);
            if (price_it != asks.end()) {
                removed = price_it->second->remove_order(order_id);
                if (price_it->second->is_empty()) {
                    asks.erase(price_it);
                }
            }
        }
        
        if (removed) {
            orders.erase(it);
        }
        
        return removed;
    }
    
    // Get best bid price
    double get_best_bid() const {
        return bids.empty() ? 0.0 : bids.begin()->first;
    }
    
    // Get best ask price
    double get_best_ask() const {
        return asks.empty() ? 0.0 : asks.begin()->first;
    }
    
    // Get bid-ask spread
    double get_spread() const {
        if (bids.empty() || asks.empty()) return 0.0;
        return get_best_ask() - get_best_bid();
    }
    
    // Get volume at price level
    uint64_t get_volume_at_price(OrderSide side, double price) const {
        if (side == OrderSide::BUY) {
            auto it = bids.find(price);
            return (it != bids.end()) ? it->second->total_quantity : 0;
        } else {
            auto it = asks.find(price);
            return (it != asks.end()) ? it->second->total_quantity : 0;
        }
    }
    
    // Get market depth (top N levels)
    void print_market_depth(int levels = 5) const {
        std::cout << "Market Depth (Top " << levels << " levels):" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        // Print asks (ascending order, but display from highest to lowest)
        auto ask_it = asks.rbegin();
        for (int i = 0; i < levels && ask_it != asks.rend(); ++i, ++ask_it) {
            std::cout << "ASK: " << ask_it->first << " @ " << ask_it->second->total_quantity << std::endl;
        }
        
        std::cout << "------- SPREAD: " << get_spread() << " -------" << std::endl;
        
        // Print bids (descending order)
        auto bid_it = bids.begin();
        for (int i = 0; i < levels && bid_it != bids.end(); ++i, ++bid_it) {
            std::cout << "BID: " << bid_it->first << " @ " << bid_it->second->total_quantity << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Get statistics
    void print_statistics() const {
        std::cout << "Order Book Statistics:" << std::endl;
        std::cout << "=====================" << std::endl;
        std::cout << "Total Trades: " << total_trades << std::endl;
        std::cout << "Total Volume: " << total_volume << std::endl;
        std::cout << "Active Orders: " << orders.size() << std::endl;
        std::cout << "Bid Levels: " << bids.size() << std::endl;
        std::cout << "Ask Levels: " << asks.size() << std::endl;
        std::cout << "Best Bid: " << get_best_bid() << std::endl;
        std::cout << "Best Ask: " << get_best_ask() << std::endl;
        std::cout << "Spread: " << get_spread() << std::endl;
        std::cout << std::endl;
    }
    
private:
    std::vector<Trade> match_buy_order(std::shared_ptr<Order> buy_order) {
        std::vector<Trade> trades;
        
        while (buy_order->quantity > 0 && !asks.empty()) {
            auto best_ask_it = asks.begin();
            double ask_price = best_ask_it->first;
            
            if (buy_order->price < ask_price) {
                break;  // No more matches possible
            }
            
            auto ask_level = best_ask_it->second;
            auto sell_order = ask_level->get_next_order();
            
            if (!sell_order) {
                asks.erase(best_ask_it);
                continue;
            }
            
            uint64_t trade_quantity = std::min(buy_order->quantity, sell_order->quantity);
            
            // Create trade
            trades.emplace_back(buy_order->order_id, sell_order->order_id, 
                              ask_price, trade_quantity);
            
            // Update quantities
            buy_order->quantity -= trade_quantity;
            ask_level->remove_front_order(trade_quantity);
            
            // Remove empty price level
            if (ask_level->is_empty()) {
                asks.erase(best_ask_it);
            }
            
            // Remove fully executed sell order
            if (sell_order->quantity == 0) {
                orders.erase(sell_order->order_id);
            }
        }
        
        return trades;
    }
    
    std::vector<Trade> match_sell_order(std::shared_ptr<Order> sell_order) {
        std::vector<Trade> trades;
        
        while (sell_order->quantity > 0 && !bids.empty()) {
            auto best_bid_it = bids.begin();
            double bid_price = best_bid_it->first;
            
            if (sell_order->price > bid_price) {
                break;  // No more matches possible
            }
            
            auto bid_level = best_bid_it->second;
            auto buy_order = bid_level->get_next_order();
            
            if (!buy_order) {
                bids.erase(best_bid_it);
                continue;
            }
            
            uint64_t trade_quantity = std::min(sell_order->quantity, buy_order->quantity);
            
            // Create trade
            trades.emplace_back(buy_order->order_id, sell_order->order_id, 
                              bid_price, trade_quantity);
            
            // Update quantities
            sell_order->quantity -= trade_quantity;
            bid_level->remove_front_order(trade_quantity);
            
            // Remove empty price level
            if (bid_level->is_empty()) {
                bids.erase(best_bid_it);
            }
            
            // Remove fully executed buy order
            if (buy_order->quantity == 0) {
                orders.erase(buy_order->order_id);
            }
        }
        
        return trades;
    }
    
    void add_to_book(std::shared_ptr<Order> order) {
        if (order->side == OrderSide::BUY) {
            auto it = bids.find(order->price);
            if (it == bids.end()) {
                bids[order->price] = std::make_shared<PriceLevel>(order->price);
            }
            bids[order->price]->add_order(order);
        } else {
            auto it = asks.find(order->price);
            if (it == asks.end()) {
                asks[order->price] = std::make_shared<PriceLevel>(order->price);
            }
            asks[order->price]->add_order(order);
        }
    }
};

// Performance testing
class PerformanceTest {
public:
    static void run_latency_test(OrderBook& book, int num_operations = 10000) {
        std::cout << "Running Latency Test (" << num_operations << " operations):" << std::endl;
        std::cout << "=======================================================" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Add orders
        for (int i = 0; i < num_operations / 2; i++) {
            book.add_order(i, OrderSide::BUY, 100.0 - (i % 10), 100);
            book.add_order(i + num_operations / 2, OrderSide::SELL, 100.0 + (i % 10), 100);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        std::cout << "Average latency per operation: " 
                  << duration.count() / num_operations << " ns" << std::endl;
        std::cout << "Operations per second: " 
                  << (num_operations * 1000000000LL) / duration.count() << std::endl;
        std::cout << std::endl;
    }
};

int main() {
    OrderBook book;
    
    std::cout << "Citadel-Style Order Book Implementation" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // Test basic functionality
    std::cout << "1. Adding initial orders:" << std::endl;
    book.add_order(1, OrderSide::BUY, 99.50, 1000);
    book.add_order(2, OrderSide::BUY, 99.45, 500);
    book.add_order(3, OrderSide::BUY, 99.40, 2000);
    
    book.add_order(4, OrderSide::SELL, 100.50, 800);
    book.add_order(5, OrderSide::SELL, 100.55, 1200);
    book.add_order(6, OrderSide::SELL, 100.60, 300);
    
    book.print_market_depth();
    book.print_statistics();
    
    // Test order matching
    std::cout << "2. Adding aggressive orders (should match):" << std::endl;
    auto trades1 = book.add_order(7, OrderSide::BUY, 100.55, 1500);
    std::cout << "Trades executed: " << trades1.size() << std::endl;
    for (const auto& trade : trades1) {
        std::cout << "Trade: Buy#" << trade.buy_order_id << " Sell#" << trade.sell_order_id
                  << " @ " << trade.price << " qty:" << trade.quantity << std::endl;
    }
    
    book.print_market_depth();
    
    // Test order cancellation
    std::cout << "3. Testing order cancellation:" << std::endl;
    bool cancelled = book.cancel_order(2);
    std::cout << "Order 2 cancelled: " << (cancelled ? "Yes" : "No") << std::endl;
    
    book.print_market_depth();
    
    // Test volume queries
    std::cout << "4. Volume at price levels:" << std::endl;
    std::cout << "Volume at bid 99.40: " << book.get_volume_at_price(OrderSide::BUY, 99.40) << std::endl;
    std::cout << "Volume at ask 100.60: " << book.get_volume_at_price(OrderSide::SELL, 100.60) << std::endl;
    
    book.print_statistics();
    
    // Performance test
    OrderBook perf_book;
    PerformanceTest::run_latency_test(perf_book, 100000);
    
    return 0;
}

/*
Algorithm Analysis:

Time Complexities:
- Add order: O(log n) for insertion + O(k) for matching where k = matched orders
- Cancel order: O(log n) for lookup + O(m) for removal where m = orders at price level
- Best bid/ask: O(1) with map::begin()
- Volume at price: O(log n) for map lookup

Space Complexity: O(n) where n = number of active orders

Key Design Decisions:

1. Data Structures:
   - std::map for price levels (sorted order)
   - std::queue for FIFO order execution at same price
   - std::unordered_map for O(1) order lookup by ID

2. Price Level Management:
   - Automatic creation/deletion of price levels
   - Aggregated quantity tracking per level
   - FIFO order execution within level

3. Order Matching:
   - Price-time priority (price first, then FIFO)
   - Immediate execution when possible
   - Partial fills supported

Optimizations for HFT:

1. Memory Layout:
   - Use object pools for frequent allocation/deallocation
   - Consider cache-friendly data structures
   - Minimize pointer indirection

2. Lock-Free Alternatives:
   - Atomic operations for statistics
   - Lock-free queues for order processing
   - Memory barriers for consistency

3. SIMD Operations:
   - Vectorized price comparisons
   - Parallel order matching
   - Bulk operations on price levels

Real-World Considerations:

1. Market Data:
   - Level 2 data feeds
   - Market by order vs market by price
   - Timestamp precision (nanoseconds)

2. Risk Management:
   - Position limits
   - Credit checks
   - Pre-trade risk controls

3. Compliance:
   - Order audit trails
   - Market making obligations
   - Regulatory reporting

Interview Tips:
1. Discuss trade-offs between different data structures
2. Consider memory vs CPU optimization
3. Think about scalability and concurrent access
4. Address error handling and edge cases
5. Mention relevant HFT concepts (latency, throughput, fairness)

Extensions:
- Hidden orders (iceberg orders)
- Stop orders and conditional logic
- Multi-symbol order books
- Cross-trading and internalization
- Market making algorithms
*/
