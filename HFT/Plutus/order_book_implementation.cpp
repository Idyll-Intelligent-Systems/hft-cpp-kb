/*
Plutus HFT Interview Problem: Order Book Implementation and Market Making
========================================================================

Problem Statement:
Implement a high-performance order book that supports:
1. Adding/canceling limit orders
2. Processing market orders
3. Tracking best bid/ask
4. Maintaining price-time priority
5. Market making strategy with inventory management

This tests:
- Data structure design and optimization
- Real-time system performance
- Trading system knowledge
- Risk management concepts

Time Complexity Requirements:
- Add order: O(log n)
- Cancel order: O(1)
- Market order: O(k) where k is orders filled
- Best bid/ask: O(1)
*/

#include <iostream>
#include <unordered_map>
#include <map>
#include <queue>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <memory>
#include <atomic>

enum class Side { BUY, SELL };
enum class OrderType { LIMIT, MARKET, IOC, FOK };

struct Order {
    uint64_t orderId;
    Side side;
    OrderType type;
    double price;
    uint64_t quantity;
    uint64_t originalQuantity;
    uint64_t timestamp;
    
    Order(uint64_t id, Side s, OrderType t, double p, uint64_t q, uint64_t ts)
        : orderId(id), side(s), type(t), price(p), quantity(q), 
          originalQuantity(q), timestamp(ts) {}
};

struct Trade {
    uint64_t buyOrderId;
    uint64_t sellOrderId;
    double price;
    uint64_t quantity;
    uint64_t timestamp;
    
    Trade(uint64_t bid, uint64_t aid, double p, uint64_t q, uint64_t ts)
        : buyOrderId(bid), sellOrderId(aid), price(p), quantity(q), timestamp(ts) {}
};

class OrderBook {
private:
    // Price level structure for efficient order management
    struct PriceLevel {
        double price;
        uint64_t totalQuantity;
        std::queue<std::shared_ptr<Order>> orders;
        
        PriceLevel(double p) : price(p), totalQuantity(0) {}
    };
    
    // Buy side: higher prices first (max heap behavior)
    std::map<double, std::shared_ptr<PriceLevel>, std::greater<double>> buySide;
    // Sell side: lower prices first (min heap behavior)  
    std::map<double, std::shared_ptr<PriceLevel>, std::less<double>> sellSide;
    
    // Order tracking for O(1) cancellation
    std::unordered_map<uint64_t, std::shared_ptr<Order>> activeOrders;
    
    // Market data
    std::atomic<double> bestBid{0.0};
    std::atomic<double> bestAsk{std::numeric_limits<double>::max()};
    std::atomic<uint64_t> bidQuantity{0};
    std::atomic<uint64_t> askQuantity{0};
    
    // Trade history
    std::vector<Trade> trades;
    
    // Sequence number for timestamp generation
    std::atomic<uint64_t> sequenceNumber{0};
    
    uint64_t getTimestamp() {
        return ++sequenceNumber;
    }
    
    void updateBestBidAsk() {
        // Update best bid
        if (!buySide.empty()) {
            auto& topBuy = buySide.begin()->second;
            bestBid = topBuy->price;
            bidQuantity = topBuy->totalQuantity;
        } else {
            bestBid = 0.0;
            bidQuantity = 0;
        }
        
        // Update best ask
        if (!sellSide.empty()) {
            auto& topSell = sellSide.begin()->second;
            bestAsk = topSell->price;
            askQuantity = topSell->totalQuantity;
        } else {
            bestAsk = std::numeric_limits<double>::max();
            askQuantity = 0;
        }
    }
    
    void addOrderToLevel(std::shared_ptr<Order> order, 
                        std::map<double, std::shared_ptr<PriceLevel>, std::greater<double>>& side) {
        auto it = side.find(order->price);
        if (it == side.end()) {
            auto level = std::make_shared<PriceLevel>(order->price);
            side[order->price] = level;
            it = side.find(order->price);
        }
        
        it->second->orders.push(order);
        it->second->totalQuantity += order->quantity;
        activeOrders[order->orderId] = order;
    }
    
    void addOrderToLevel(std::shared_ptr<Order> order,
                        std::map<double, std::shared_ptr<PriceLevel>, std::less<double>>& side) {
        auto it = side.find(order->price);
        if (it == side.end()) {
            auto level = std::make_shared<PriceLevel>(order->price);
            side[order->price] = level;
            it = side.find(order->price);
        }
        
        it->second->orders.push(order);
        it->second->totalQuantity += order->quantity;
        activeOrders[order->orderId] = order;
    }
    
    template<typename SideMap>
    void cleanupEmptyLevel(double price, SideMap& side) {
        auto it = side.find(price);
        if (it != side.end() && it->second->orders.empty()) {
            side.erase(it);
        }
    }

public:
    OrderBook() = default;
    
    // Add limit order to the book
    bool addOrder(uint64_t orderId, Side side, double price, uint64_t quantity) {
        if (quantity == 0 || price <= 0) return false;
        
        auto order = std::make_shared<Order>(orderId, side, OrderType::LIMIT, 
                                           price, quantity, getTimestamp());
        
        if (side == Side::BUY) {
            addOrderToLevel(order, buySide);
        } else {
            addOrderToLevel(order, sellSide);
        }
        
        updateBestBidAsk();
        return true;
    }
    
    // Cancel order
    bool cancelOrder(uint64_t orderId) {
        auto it = activeOrders.find(orderId);
        if (it == activeOrders.end()) return false;
        
        auto order = it->second;
        order->quantity = 0; // Mark as cancelled
        activeOrders.erase(it);
        
        updateBestBidAsk();
        return true;
    }
    
    // Process market order and return fills
    std::vector<Trade> processMarketOrder(Side side, uint64_t quantity) {
        std::vector<Trade> fills;
        uint64_t remainingQuantity = quantity;
        uint64_t marketOrderId = getTimestamp(); // Use timestamp as order ID
        
        if (side == Side::BUY) {
            // Buy market order matches against sell side
            while (remainingQuantity > 0 && !sellSide.empty()) {
                auto& level = sellSide.begin()->second;
                
                while (remainingQuantity > 0 && !level->orders.empty()) {
                    auto& frontOrder = level->orders.front();
                    
                    if (frontOrder->quantity == 0) {
                        level->orders.pop();
                        continue;
                    }
                    
                    uint64_t fillQuantity = std::min(remainingQuantity, frontOrder->quantity);
                    
                    // Create trade
                    fills.emplace_back(marketOrderId, frontOrder->orderId, 
                                     level->price, fillQuantity, getTimestamp());
                    
                    // Update quantities
                    frontOrder->quantity -= fillQuantity;
                    level->totalQuantity -= fillQuantity;
                    remainingQuantity -= fillQuantity;
                    
                    if (frontOrder->quantity == 0) {
                        activeOrders.erase(frontOrder->orderId);
                        level->orders.pop();
                    }
                }
                
                if (level->orders.empty()) {
                    cleanupEmptyLevel(level->price, sellSide);
                }
            }
        } else {
            // Sell market order matches against buy side
            while (remainingQuantity > 0 && !buySide.empty()) {
                auto& level = buySide.begin()->second;
                
                while (remainingQuantity > 0 && !level->orders.empty()) {
                    auto& frontOrder = level->orders.front();
                    
                    if (frontOrder->quantity == 0) {
                        level->orders.pop();
                        continue;
                    }
                    
                    uint64_t fillQuantity = std::min(remainingQuantity, frontOrder->quantity);
                    
                    // Create trade
                    fills.emplace_back(frontOrder->orderId, marketOrderId,
                                     level->price, fillQuantity, getTimestamp());
                    
                    // Update quantities
                    frontOrder->quantity -= fillQuantity;
                    level->totalQuantity -= fillQuantity;
                    remainingQuantity -= fillQuantity;
                    
                    if (frontOrder->quantity == 0) {
                        activeOrders.erase(frontOrder->orderId);
                        level->orders.pop();
                    }
                }
                
                if (level->orders.empty()) {
                    cleanupEmptyLevel(level->price, buySide);
                }
            }
        }
        
        // Store trades
        trades.insert(trades.end(), fills.begin(), fills.end());
        updateBestBidAsk();
        
        return fills;
    }
    
    // Get market data
    double getBestBid() const { return bestBid.load(); }
    double getBestAsk() const { return bestAsk.load(); }
    uint64_t getBidQuantity() const { return bidQuantity.load(); }
    uint64_t getAskQuantity() const { return askQuantity.load(); }
    double getSpread() const { 
        double bid = bestBid.load();
        double ask = bestAsk.load();
        return (ask == std::numeric_limits<double>::max()) ? 0.0 : ask - bid;
    }
    
    // Get order book depth
    std::vector<std::pair<double, uint64_t>> getBidDepth(int levels = 5) const {
        std::vector<std::pair<double, uint64_t>> depth;
        int count = 0;
        
        for (const auto& level : buySide) {
            if (count >= levels) break;
            depth.emplace_back(level.second->price, level.second->totalQuantity);
            count++;
        }
        
        return depth;
    }
    
    std::vector<std::pair<double, uint64_t>> getAskDepth(int levels = 5) const {
        std::vector<std::pair<double, uint64_t>> depth;
        int count = 0;
        
        for (const auto& level : sellSide) {
            if (count >= levels) break;
            depth.emplace_back(level.second->price, level.second->totalQuantity);
            count++;
        }
        
        return depth;
    }
    
    // Statistics
    size_t getActiveOrderCount() const { return activeOrders.size(); }
    size_t getTradeCount() const { return trades.size(); }
    
    void printOrderBook(int levels = 5) const {
        auto askDepth = getAskDepth(levels);
        auto bidDepth = getBidDepth(levels);
        
        std::cout << "\nOrder Book:" << std::endl;
        std::cout << "Ask Side:" << std::endl;
        for (auto it = askDepth.rbegin(); it != askDepth.rend(); ++it) {
            std::printf("  %8.2f | %6lu\n", it->first, it->second);
        }
        std::cout << "         Spread: " << getSpread() << std::endl;
        std::cout << "Bid Side:" << std::endl;
        for (const auto& level : bidDepth) {
            std::printf("  %8.2f | %6lu\n", level.first, level.second);
        }
    }
};

// Market Making Strategy
class MarketMaker {
private:
    OrderBook* orderBook;
    double targetSpread;
    double maxInventory;
    double currentInventory;
    uint64_t orderIdCounter;
    std::unordered_map<uint64_t, bool> activeQuotes; // orderId -> isBid
    
public:
    MarketMaker(OrderBook* ob, double spread, double maxInv) 
        : orderBook(ob), targetSpread(spread), maxInventory(maxInv), 
          currentInventory(0), orderIdCounter(1000000) {}
    
    void updateQuotes() {
        // Cancel existing quotes
        cancelAllQuotes();
        
        double bestBid = orderBook->getBestBid();
        double bestAsk = orderBook->getBestAsk();
        
        if (bestBid == 0.0 || bestAsk == std::numeric_limits<double>::max()) {
            return; // No market to make
        }
        
        double midPrice = (bestBid + bestAsk) / 2.0;
        double halfSpread = targetSpread / 2.0;
        
        // Adjust for inventory
        double inventorySkew = currentInventory / maxInventory * 0.01; // 1 cent per unit inventory
        
        double bidPrice = midPrice - halfSpread - inventorySkew;
        double askPrice = midPrice + halfSpread - inventorySkew;
        
        // Place quotes if inventory allows
        if (std::abs(currentInventory) < maxInventory) {
            if (currentInventory > -maxInventory * 0.8) {
                // Place bid
                uint64_t bidOrderId = ++orderIdCounter;
                orderBook->addOrder(bidOrderId, Side::BUY, bidPrice, 100);
                activeQuotes[bidOrderId] = true;
            }
            
            if (currentInventory < maxInventory * 0.8) {
                // Place ask
                uint64_t askOrderId = ++orderIdCounter;
                orderBook->addOrder(askOrderId, Side::SELL, askPrice, 100);
                activeQuotes[askOrderId] = false;
            }
        }
    }
    
    void cancelAllQuotes() {
        for (const auto& quote : activeQuotes) {
            orderBook->cancelOrder(quote.first);
        }
        activeQuotes.clear();
    }
    
    void handleFill(uint64_t orderId, uint64_t quantity, Side side) {
        if (side == Side::BUY) {
            currentInventory += quantity;
        } else {
            currentInventory -= quantity;
        }
        
        // Remove from active quotes if fully filled
        activeQuotes.erase(orderId);
    }
    
    double getCurrentInventory() const { return currentInventory; }
    void setInventory(double inventory) { currentInventory = inventory; }
};

// Performance testing framework
class OrderBookTest {
public:
    static void runPerformanceTest() {
        std::cout << "Plutus HFT Order Book Performance Test" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        OrderBook orderBook;
        const int numOrders = 100000;
        
        // Test order insertion performance
        auto start = std::chrono::high_resolution_clock::now();
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> priceDist(99.0, 101.0);
        std::uniform_int_distribution<> quantityDist(100, 1000);
        std::uniform_int_distribution<> sideDist(0, 1);
        
        std::vector<uint64_t> orderIds;
        
        for (int i = 0; i < numOrders; i++) {
            uint64_t orderId = i + 1;
            Side side = (sideDist(gen) == 0) ? Side::BUY : Side::SELL;
            double price = priceDist(gen);
            uint64_t quantity = quantityDist(gen);
            
            orderBook.addOrder(orderId, side, price, quantity);
            orderIds.push_back(orderId);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto insertionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Order Insertion Performance:" << std::endl;
        std::cout << "Total orders: " << numOrders << std::endl;
        std::cout << "Total time: " << insertionTime.count() << " μs" << std::endl;
        std::cout << "Average per order: " << (insertionTime.count() / (double)numOrders) << " μs" << std::endl;
        
        // Test market order performance
        start = std::chrono::high_resolution_clock::now();
        
        auto trades = orderBook.processMarketOrder(Side::BUY, 50000);
        
        end = std::chrono::high_resolution_clock::now();
        auto marketOrderTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "\nMarket Order Performance:" << std::endl;
        std::cout << "Quantity: 50000" << std::endl;
        std::cout << "Trades generated: " << trades.size() << std::endl;
        std::cout << "Execution time: " << marketOrderTime.count() << " μs" << std::endl;
        
        // Test cancellation performance
        start = std::chrono::high_resolution_clock::now();
        
        int cancellations = 0;
        for (int i = 0; i < std::min(10000, (int)orderIds.size()); i++) {
            if (orderBook.cancelOrder(orderIds[i])) {
                cancellations++;
            }
        }
        
        end = std::chrono::high_resolution_clock::now();
        auto cancellationTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "\nCancellation Performance:" << std::endl;
        std::cout << "Cancellations: " << cancellations << std::endl;
        std::cout << "Total time: " << cancellationTime.count() << " μs" << std::endl;
        std::cout << "Average per cancellation: " << (cancellationTime.count() / (double)cancellations) << " μs" << std::endl;
        
        // Market data access performance
        start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < 1000000; i++) {
            volatile double bid = orderBook.getBestBid();
            volatile double ask = orderBook.getBestAsk();
            volatile double spread = orderBook.getSpread();
        }
        
        end = std::chrono::high_resolution_clock::now();
        auto marketDataTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "\nMarket Data Access Performance:" << std::endl;
        std::cout << "Accesses: 1000000" << std::endl;
        std::cout << "Total time: " << marketDataTime.count() << " μs" << std::endl;
        std::cout << "Average per access: " << (marketDataTime.count() / 1000000.0) << " μs" << std::endl;
        
        orderBook.printOrderBook();
    }
    
    static void runMarketMakingSimulation() {
        std::cout << "\n\nMarket Making Simulation" << std::endl;
        std::cout << "========================" << std::endl;
        
        OrderBook orderBook;
        MarketMaker marketMaker(&orderBook, 0.02, 1000); // 2 cent spread, max 1000 inventory
        
        // Seed the order book
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> priceDist(99.8, 100.2);
        std::uniform_int_distribution<> quantityDist(100, 500);
        
        // Add some initial orders
        for (int i = 0; i < 20; i++) {
            orderBook.addOrder(i, Side::BUY, priceDist(gen), quantityDist(gen));
            orderBook.addOrder(i + 1000, Side::SELL, priceDist(gen), quantityDist(gen));
        }
        
        std::cout << "Initial order book:" << std::endl;
        orderBook.printOrderBook();
        
        // Run market making simulation
        for (int round = 0; round < 10; round++) {
            std::cout << "\nRound " << (round + 1) << ":" << std::endl;
            
            // Update market maker quotes
            marketMaker.updateQuotes();
            
            std::cout << "After market maker quotes:" << std::endl;
            orderBook.printOrderBook();
            std::cout << "Market maker inventory: " << marketMaker.getCurrentInventory() << std::endl;
            
            // Simulate some market activity
            if (gen() % 2) {
                // Market buy order
                auto trades = orderBook.processMarketOrder(Side::BUY, 50 + gen() % 100);
                std::cout << "Market buy executed, " << trades.size() << " fills" << std::endl;
                
                // Update market maker inventory for any fills
                for (const auto& trade : trades) {
                    // Simplified: assume market maker was the seller
                    marketMaker.setInventory(marketMaker.getCurrentInventory() - trade.quantity);
                }
            } else {
                // Market sell order
                auto trades = orderBook.processMarketOrder(Side::SELL, 50 + gen() % 100);
                std::cout << "Market sell executed, " << trades.size() << " fills" << std::endl;
                
                // Update market maker inventory for any fills
                for (const auto& trade : trades) {
                    // Simplified: assume market maker was the buyer
                    marketMaker.setInventory(marketMaker.getCurrentInventory() + trade.quantity);
                }
            }
        }
        
        std::cout << "\nFinal state:" << std::endl;
        orderBook.printOrderBook();
        std::cout << "Final market maker inventory: " << marketMaker.getCurrentInventory() << std::endl;
        std::cout << "Total trades: " << orderBook.getTradeCount() << std::endl;
    }
    
    static void runLatencyTest() {
        std::cout << "\n\nLatency Test (Critical Path)" << std::endl;
        std::cout << "============================" << std::endl;
        
        OrderBook orderBook;
        
        // Warm up
        for (int i = 0; i < 1000; i++) {
            orderBook.addOrder(i, Side::BUY, 100.0 - i * 0.01, 100);
            orderBook.addOrder(i + 10000, Side::SELL, 100.0 + i * 0.01, 100);
        }
        
        // Test critical path: market order processing
        const int iterations = 10000;
        std::vector<long> latencies(iterations);
        
        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Critical trading path: process small market order
            auto trades = orderBook.processMarketOrder(Side::BUY, 100);
            
            auto end = std::chrono::high_resolution_clock::now();
            latencies[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            
            // Add order back to maintain book state
            orderBook.addOrder(i + 20000, Side::SELL, 100.0 + (i % 100) * 0.01, 100);
        }
        
        // Calculate latency statistics
        std::sort(latencies.begin(), latencies.end());
        
        double avgLatency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / iterations;
        long p50 = latencies[iterations * 0.5];
        long p95 = latencies[iterations * 0.95];
        long p99 = latencies[iterations * 0.99];
        long p999 = latencies[iterations * 0.999];
        
        std::cout << "Market Order Latency Statistics (nanoseconds):" << std::endl;
        std::cout << "Average: " << avgLatency << " ns" << std::endl;
        std::cout << "50th percentile: " << p50 << " ns" << std::endl;
        std::cout << "95th percentile: " << p95 << " ns" << std::endl;
        std::cout << "99th percentile: " << p99 << " ns" << std::endl;
        std::cout << "99.9th percentile: " << p999 << " ns" << std::endl;
        
        std::cout << "\nLatency in microseconds:" << std::endl;
        std::cout << "Average: " << (avgLatency / 1000.0) << " μs" << std::endl;
        std::cout << "99th percentile: " << (p99 / 1000.0) << " μs" << std::endl;
    }
};

int main() {
    OrderBookTest::runPerformanceTest();
    OrderBookTest::runMarketMakingSimulation();
    OrderBookTest::runLatencyTest();
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "Plutus Order Book Implementation Complete!" << std::endl;
    std::cout << "\nKey Features Demonstrated:" << std::endl;
    std::cout << "1. High-performance order book with O(log n) insertion" << std::endl;
    std::cout << "2. O(1) order cancellation using hash map tracking" << std::endl;
    std::cout << "3. Efficient market order processing" << std::endl;
    std::cout << "4. Real-time market data with atomic operations" << std::endl;
    std::cout << "5. Market making strategy with inventory management" << std::endl;
    std::cout << "6. Comprehensive performance testing framework" << std::endl;
    std::cout << "\nOptimizations for Production:" << std::endl;
    std::cout << "- Memory pools for order allocation" << std::endl;
    std::cout << "- Lock-free data structures" << std::endl;
    std::cout << "- SIMD operations for bulk calculations" << std::endl;
    std::cout << "- CPU affinity and NUMA optimization" << std::endl;
    std::cout << "- Direct memory access for market data" << std::endl;
    
    return 0;
}

/*
Interview Discussion Points:

1. Data Structure Design:
   - Why use std::map vs std::unordered_map for price levels?
   - Trade-offs between memory usage and performance
   - Cache locality considerations

2. Concurrency:
   - How to make this thread-safe for multiple trading threads?
   - Lock-free vs lock-based approaches
   - Memory ordering and atomic operations

3. Performance Optimization:
   - Memory allocation strategies (pools vs dynamic)
   - Branch prediction optimization
   - SIMD for bulk operations
   - Network and I/O considerations

4. Risk Management:
   - Position limits and risk checks
   - Circuit breakers and kill switches
   - Real-time P&L calculation

5. Market Microstructure:
   - Order types (IOC, FOK, Hidden, Iceberg)
   - Priority schemes (price-time, pro-rata)
   - Market data dissemination

6. System Design:
   - How to scale to multiple symbols?
   - Cross-connect and co-location strategies
   - Disaster recovery and failover

This implementation demonstrates the core concepts needed for
high-frequency trading systems at firms like Plutus.
*/
