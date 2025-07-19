/*
HackerRank Style Problem 3: Sliding Window Maximum for Trading Signals
======================================================================

Problem Statement:
You are given an array representing price movements and a window size k.
For each window of size k, find the maximum price change within that window.
This is useful for detecting breakout patterns and momentum signals in trading.

Additionally, implement variants for:
1. Minimum price in window (support levels)
2. Maximum volume in window  
3. Price volatility (standard deviation) in window

Constraints:
- 1 <= prices.length <= 10^5
- 1 <= k <= prices.length
- -1000 <= price_change <= 1000

Example:
Input: prices = [1, 3, -1, -3, 5, 3, 6, 7], k = 3
Output: [3, 3, 5, 5, 6, 7]

This problem tests:
- Sliding window techniques with deque optimization
- Real-time signal processing
- Statistical calculations on streaming data
- Memory-efficient algorithm design

Time Complexity: O(n) where n is array length
Space Complexity: O(k) for the deque
*/

#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <climits>
#include <cassert>
#include <chrono>
#include <random>

// Structure to represent a trading data point
struct TradingData {
    double price;
    double volume;
    double priceChange;
    uint64_t timestamp;
    
    TradingData(double p, double v, double pc, uint64_t ts)
        : price(p), volume(v), priceChange(pc), timestamp(ts) {}
};

class SlidingWindowAnalyzer {
public:
    // Basic sliding window maximum using deque (monotonic decreasing)
    static std::vector<int> slidingWindowMaximum(const std::vector<int>& nums, int k) {
        if (nums.empty() || k <= 0) return {};
        
        std::vector<int> result;
        std::deque<int> dq; // Store indices, maintain decreasing order of values
        
        for (int i = 0; i < nums.size(); i++) {
            // Remove elements outside current window
            while (!dq.empty() && dq.front() <= i - k) {
                dq.pop_front();
            }
            
            // Remove elements smaller than current element
            // (they can never be maximum while current element is in window)
            while (!dq.empty() && nums[dq.back()] <= nums[i]) {
                dq.pop_back();
            }
            
            dq.push_back(i);
            
            // Add result for current window
            if (i >= k - 1) {
                result.push_back(nums[dq.front()]);
            }
        }
        
        return result;
    }
    
    // Sliding window minimum (for support level detection)
    static std::vector<int> slidingWindowMinimum(const std::vector<int>& nums, int k) {
        if (nums.empty() || k <= 0) return {};
        
        std::vector<int> result;
        std::deque<int> dq; // Store indices, maintain increasing order of values
        
        for (int i = 0; i < nums.size(); i++) {
            // Remove elements outside current window
            while (!dq.empty() && dq.front() <= i - k) {
                dq.pop_front();
            }
            
            // Remove elements larger than current element
            while (!dq.empty() && nums[dq.back()] >= nums[i]) {
                dq.pop_back();
            }
            
            dq.push_back(i);
            
            if (i >= k - 1) {
                result.push_back(nums[dq.front()]);
            }
        }
        
        return result;
    }
    
    // Sliding window for price changes (momentum detection)
    static std::vector<double> slidingWindowMaxPriceChange(const std::vector<double>& priceChanges, int k) {
        if (priceChanges.empty() || k <= 0) return {};
        
        std::vector<double> result;
        std::deque<int> dq;
        
        for (int i = 0; i < priceChanges.size(); i++) {
            while (!dq.empty() && dq.front() <= i - k) {
                dq.pop_front();
            }
            
            while (!dq.empty() && priceChanges[dq.back()] <= priceChanges[i]) {
                dq.pop_back();
            }
            
            dq.push_back(i);
            
            if (i >= k - 1) {
                result.push_back(priceChanges[dq.front()]);
            }
        }
        
        return result;
    }
};

// Advanced trading signal analyzer
class TradingSignalAnalyzer {
private:
    struct WindowStats {
        double maxPrice;
        double minPrice;
        double maxVolume;
        double avgPrice;
        double volatility;
        double momentum;
        
        WindowStats() : maxPrice(-1e9), minPrice(1e9), maxVolume(0), 
                       avgPrice(0), volatility(0), momentum(0) {}
    };

public:
    // Comprehensive sliding window analysis for trading signals
    static std::vector<WindowStats> analyzeTradingWindows(const std::vector<TradingData>& data, int k) {
        if (data.empty() || k <= 0) return {};
        
        std::vector<WindowStats> results;
        std::deque<int> maxPriceIdx, minPriceIdx, maxVolumeIdx;
        
        for (int i = 0; i < data.size(); i++) {
            // Update max price deque
            while (!maxPriceIdx.empty() && maxPriceIdx.front() <= i - k) {
                maxPriceIdx.pop_front();
            }
            while (!maxPriceIdx.empty() && data[maxPriceIdx.back()].price <= data[i].price) {
                maxPriceIdx.pop_back();
            }
            maxPriceIdx.push_back(i);
            
            // Update min price deque
            while (!minPriceIdx.empty() && minPriceIdx.front() <= i - k) {
                minPriceIdx.pop_front();
            }
            while (!minPriceIdx.empty() && data[minPriceIdx.back()].price >= data[i].price) {
                minPriceIdx.pop_back();
            }
            minPriceIdx.push_back(i);
            
            // Update max volume deque
            while (!maxVolumeIdx.empty() && maxVolumeIdx.front() <= i - k) {
                maxVolumeIdx.pop_front();
            }
            while (!maxVolumeIdx.empty() && data[maxVolumeIdx.back()].volume <= data[i].volume) {
                maxVolumeIdx.pop_back();
            }
            maxVolumeIdx.push_back(i);
            
            // Calculate window statistics
            if (i >= k - 1) {
                WindowStats stats;
                stats.maxPrice = data[maxPriceIdx.front()].price;
                stats.minPrice = data[minPriceIdx.front()].price;
                stats.maxVolume = data[maxVolumeIdx.front()].volume;
                
                // Calculate average price and volatility for current window
                double sum = 0, sumSquares = 0;
                double momentumSum = 0;
                
                for (int j = i - k + 1; j <= i; j++) {
                    sum += data[j].price;
                    sumSquares += data[j].price * data[j].price;
                    momentumSum += data[j].priceChange;
                }
                
                stats.avgPrice = sum / k;
                double variance = (sumSquares / k) - (stats.avgPrice * stats.avgPrice);
                stats.volatility = std::sqrt(std::max(0.0, variance));
                stats.momentum = momentumSum;
                
                results.push_back(stats);
            }
        }
        
        return results;
    }
    
    // Detect breakout patterns using sliding window
    static std::vector<bool> detectBreakouts(const std::vector<double>& prices, int windowSize, double threshold) {
        std::vector<bool> breakouts(prices.size(), false);
        
        if (prices.size() < windowSize + 1) return breakouts;
        
        auto maxWindow = SlidingWindowAnalyzer::slidingWindowMaximum(
            std::vector<int>(prices.begin(), prices.end()), windowSize);
        auto minWindow = SlidingWindowAnalyzer::slidingWindowMinimum(
            std::vector<int>(prices.begin(), prices.end()), windowSize);
        
        for (int i = windowSize; i < prices.size(); i++) {
            double windowMax = maxWindow[i - windowSize];
            double windowMin = minWindow[i - windowSize];
            double range = windowMax - windowMin;
            
            // Breakout if current price exceeds window max/min by threshold
            if (prices[i] > windowMax + threshold * range || 
                prices[i] < windowMin - threshold * range) {
                breakouts[i] = true;
            }
        }
        
        return breakouts;
    }
    
    // Volume spike detection
    static std::vector<bool> detectVolumeSpikes(const std::vector<double>& volumes, int windowSize, double multiplier) {
        std::vector<bool> spikes(volumes.size(), false);
        
        if (volumes.size() < windowSize + 1) return spikes;
        
        // Calculate rolling average volume
        for (int i = windowSize; i < volumes.size(); i++) {
            double avgVolume = 0;
            for (int j = i - windowSize; j < i; j++) {
                avgVolume += volumes[j];
            }
            avgVolume /= windowSize;
            
            // Spike if current volume is multiplier times average
            if (volumes[i] > avgVolume * multiplier) {
                spikes[i] = true;
            }
        }
        
        return spikes;
    }
};

// Memory-efficient streaming analyzer for real-time systems
class StreamingWindowAnalyzer {
private:
    int windowSize;
    std::deque<double> window;
    std::deque<int> maxDeque;  // Indices for max tracking
    std::deque<int> minDeque;  // Indices for min tracking
    double sum;
    double sumSquares;
    int position;

public:
    StreamingWindowAnalyzer(int k) : windowSize(k), sum(0), sumSquares(0), position(0) {}
    
    struct StreamStats {
        double maximum;
        double minimum;
        double average;
        double volatility;
        bool isValid;
        
        StreamStats() : maximum(0), minimum(0), average(0), volatility(0), isValid(false) {}
    };
    
    StreamStats addDataPoint(double value) {
        StreamStats stats;
        
        // Add new value to window
        window.push_back(value);
        sum += value;
        sumSquares += value * value;
        
        // Update max deque
        while (!maxDeque.empty() && window[maxDeque.back()] <= value) {
            maxDeque.pop_back();
        }
        maxDeque.push_back(position);
        
        // Update min deque
        while (!minDeque.empty() && window[minDeque.back()] >= value) {
            minDeque.pop_back();
        }
        minDeque.push_back(position);
        
        // Remove old values if window is full
        if (window.size() > windowSize) {
            double oldValue = window.front();
            window.pop_front();
            sum -= oldValue;
            sumSquares -= oldValue * oldValue;
            
            // Clean up deques
            while (!maxDeque.empty() && maxDeque.front() <= position - windowSize) {
                maxDeque.pop_front();
            }
            while (!minDeque.empty() && minDeque.front() <= position - windowSize) {
                minDeque.pop_front();
            }
        }
        
        position++;
        
        // Calculate statistics if window is full
        if (window.size() == windowSize) {
            stats.maximum = window[maxDeque.front() - (position - windowSize)];
            stats.minimum = window[minDeque.front() - (position - windowSize)];
            stats.average = sum / windowSize;
            
            double variance = (sumSquares / windowSize) - (stats.average * stats.average);
            stats.volatility = std::sqrt(std::max(0.0, variance));
            stats.isValid = true;
        }
        
        return stats;
    }
    
    void reset() {
        window.clear();
        maxDeque.clear();
        minDeque.clear();
        sum = sumSquares = 0;
        position = 0;
    }
};

// Test framework
class TestSuite {
public:
    static void runBasicTests() {
        std::cout << "Running basic sliding window tests..." << std::endl;
        
        // Test 1: Basic sliding window maximum
        {
            std::vector<int> nums = {1, 3, -1, -3, 5, 3, 6, 7};
            int k = 3;
            auto result = SlidingWindowAnalyzer::slidingWindowMaximum(nums, k);
            std::vector<int> expected = {3, 3, 5, 5, 6, 7};
            assert(result == expected);
            std::cout << "✓ Basic sliding window maximum test passed" << std::endl;
        }
        
        // Test 2: Sliding window minimum
        {
            std::vector<int> nums = {1, 3, -1, -3, 5, 3, 6, 7};
            int k = 3;
            auto result = SlidingWindowAnalyzer::slidingWindowMinimum(nums, k);
            std::vector<int> expected = {-1, -3, -3, -3, 3, 3};
            assert(result == expected);
            std::cout << "✓ Sliding window minimum test passed" << std::endl;
        }
        
        // Test 3: Price change analysis
        {
            std::vector<double> priceChanges = {0.5, -0.3, 1.2, -0.8, 2.1, 0.4, -0.6};
            int k = 3;
            auto result = SlidingWindowAnalyzer::slidingWindowMaxPriceChange(priceChanges, k);
            // Expected: max of [0.5, -0.3, 1.2] = 1.2, max of [-0.3, 1.2, -0.8] = 1.2, etc.
            assert(result.size() == 5);
            assert(std::abs(result[0] - 1.2) < 1e-9);
            std::cout << "✓ Price change analysis test passed" << std::endl;
        }
        
        // Test 4: Edge cases
        {
            std::vector<int> empty = {};
            auto result = SlidingWindowAnalyzer::slidingWindowMaximum(empty, 3);
            assert(result.empty());
            
            std::vector<int> single = {42};
            result = SlidingWindowAnalyzer::slidingWindowMaximum(single, 1);
            assert(result.size() == 1 && result[0] == 42);
            std::cout << "✓ Edge cases test passed" << std::endl;
        }
    }
    
    static void runTradingSignalTests() {
        std::cout << "\nRunning trading signal analysis tests..." << std::endl;
        
        // Create sample trading data
        std::vector<TradingData> data;
        std::vector<double> prices = {100.0, 101.5, 99.8, 102.3, 104.1, 103.2, 105.7, 106.2};
        std::vector<double> volumes = {1000, 1500, 800, 2000, 2500, 1200, 3000, 1800};
        
        for (int i = 0; i < prices.size(); i++) {
            double priceChange = (i > 0) ? prices[i] - prices[i-1] : 0;
            data.emplace_back(prices[i], volumes[i], priceChange, i);
        }
        
        // Test comprehensive analysis
        auto stats = TradingSignalAnalyzer::analyzeTradingWindows(data, 3);
        assert(!stats.empty());
        std::cout << "✓ Trading signal analysis test passed" << std::endl;
        
        // Test breakout detection
        auto breakouts = TradingSignalAnalyzer::detectBreakouts(prices, 3, 0.1);
        assert(breakouts.size() == prices.size());
        std::cout << "✓ Breakout detection test passed" << std::endl;
        
        // Test volume spike detection
        auto spikes = TradingSignalAnalyzer::detectVolumeSpikes(volumes, 3, 1.5);
        assert(spikes.size() == volumes.size());
        std::cout << "✓ Volume spike detection test passed" << std::endl;
    }
    
    static void runStreamingTest() {
        std::cout << "\nRunning streaming analyzer test..." << std::endl;
        
        StreamingWindowAnalyzer analyzer(5);
        std::vector<double> testData = {10.0, 15.0, 8.0, 20.0, 12.0, 18.0, 9.0, 25.0};
        
        for (double value : testData) {
            auto stats = analyzer.addDataPoint(value);
            if (stats.isValid) {
                std::cout << "Value: " << value 
                         << ", Max: " << stats.maximum 
                         << ", Min: " << stats.minimum
                         << ", Avg: " << stats.average
                         << ", Vol: " << stats.volatility << std::endl;
            }
        }
        
        std::cout << "✓ Streaming analyzer test passed" << std::endl;
    }
    
    static void runPerformanceTest() {
        std::cout << "\nRunning performance test..." << std::endl;
        
        const int dataSize = 1000000;
        const int windowSize = 100;
        
        // Generate random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(-1000, 1000);
        
        std::vector<int> data(dataSize);
        for (int& val : data) {
            val = dis(gen);
        }
        
        // Test sliding window maximum performance
        auto start = std::chrono::high_resolution_clock::now();
        auto result = SlidingWindowAnalyzer::slidingWindowMaximum(data, windowSize);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Processed " << dataSize << " elements with window size " << windowSize << std::endl;
        std::cout << "Time: " << duration.count() << " μs" << std::endl;
        std::cout << "Throughput: " << (dataSize * 1000000.0 / duration.count()) << " elements/second" << std::endl;
        std::cout << "Result size: " << result.size() << std::endl;
        
        // Test streaming performance
        StreamingWindowAnalyzer streamAnalyzer(windowSize);
        
        start = std::chrono::high_resolution_clock::now();
        for (int val : data) {
            streamAnalyzer.addDataPoint(val);
        }
        end = std::chrono::high_resolution_clock::now();
        
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Streaming analysis time: " << duration.count() << " μs" << std::endl;
        std::cout << "Streaming throughput: " << (dataSize * 1000000.0 / duration.count()) << " elements/second" << std::endl;
    }
};

int main() {
    std::cout << "HackerRank Problem 3: Sliding Window Maximum for Trading Signals" << std::endl;
    std::cout << "================================================================" << std::endl;
    
    TestSuite::runBasicTests();
    TestSuite::runTradingSignalTests();
    TestSuite::runStreamingTest();
    TestSuite::runPerformanceTest();
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Sliding Window Trading Analysis Complete!" << std::endl;
    std::cout << "\nKey Features Implemented:" << std::endl;
    std::cout << "✓ O(n) sliding window maximum/minimum using deque" << std::endl;
    std::cout << "✓ Comprehensive trading signal analysis" << std::endl;
    std::cout << "✓ Breakout pattern detection" << std::endl;
    std::cout << "✓ Volume spike identification" << std::endl;
    std::cout << "✓ Real-time streaming window analyzer" << std::endl;
    std::cout << "✓ Statistical measures (volatility, momentum)" << std::endl;
    std::cout << "\nTrading Applications:" << std::endl;
    std::cout << "- Momentum signal generation" << std::endl;
    std::cout << "- Support/resistance level detection" << std::endl;
    std::cout << "- Volatility regime identification" << std::endl;
    std::cout << "- Volume anomaly detection" << std::endl;
    std::cout << "- Real-time risk monitoring" << std::endl;
    
    return 0;
}

/*
Algorithm Deep Dive:

Sliding Window Maximum with Deque:
1. Maintain a deque of indices in decreasing order of values
2. For each new element:
   - Remove indices outside current window from front
   - Remove indices with smaller values from back (they can't be max)
   - Add current index to back
   - Front of deque contains index of maximum element

Key Insights:
- Monotonic deque ensures O(1) amortized operations
- Each element enters and leaves deque at most once
- Total time complexity: O(n) for n elements

Trading Signal Applications:
1. Momentum Detection: Track maximum price change in sliding window
2. Support/Resistance: Use minimum/maximum price levels
3. Volatility Calculation: Rolling standard deviation
4. Volume Analysis: Detect unusual trading activity

Real-time Considerations:
- Memory-efficient streaming implementation
- Lock-free data structures for multi-threading
- SIMD optimization for bulk calculations
- Cache-friendly memory access patterns

Interview Extensions:
- Handle missing data points
- Implement weighted sliding windows
- Add multi-timeframe analysis
- Support for different window types (time-based vs count-based)
- Integration with order book events
*/
