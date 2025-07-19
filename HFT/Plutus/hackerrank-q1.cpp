/*
HackerRank Style Problem 1: Maximum Profit from Stock Trading
============================================================

Problem Statement:
You are given an array of stock prices where prices[i] is the price of a stock on day i.
You can complete at most k transactions (buy-sell pairs). Design an algorithm to find the maximum profit.

Constraints:
- 1 <= k <= 100
- 1 <= prices.length <= 1000
- 0 <= prices[i] <= 1000

Example:
Input: k = 2, prices = [2,4,1]
Output: 2
Explanation: Buy on day 1 (price = 2) and sell on day 2 (price = 4), profit = 2.

This problem tests:
- Dynamic Programming optimization
- State space reduction
- Time/Space complexity analysis
- Real trading constraints understanding

Time Complexity: O(min(k, n) * n)
Space Complexity: O(min(k, n))
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <cassert>

class StockTrader {
public:
    // Optimized solution for at most k transactions
    static int maxProfit(int k, std::vector<int>& prices) {
        int n = prices.size();
        if (n <= 1 || k == 0) return 0;
        
        // If k >= n/2, we can do as many transactions as we want
        if (k >= n / 2) {
            return maxProfitUnlimited(prices);
        }
        
        // DP approach: dp[i][j] = max profit using at most i transactions up to day j
        // Optimize space: only need current and previous transaction states
        std::vector<int> buy(k + 1, -prices[0]);  // Max profit after buying (negative cost)
        std::vector<int> sell(k + 1, 0);          // Max profit after selling
        
        for (int i = 1; i < n; i++) {
            // Process from k down to 1 to avoid using updated values
            for (int j = k; j >= 1; j--) {
                // Sell: either keep previous sell state or sell today
                sell[j] = std::max(sell[j], buy[j] + prices[i]);
                // Buy: either keep previous buy state or buy today
                buy[j] = std::max(buy[j], sell[j-1] - prices[i]);
            }
        }
        
        return sell[k];
    }
    
private:
    // Helper for unlimited transactions
    static int maxProfitUnlimited(const std::vector<int>& prices) {
        int profit = 0;
        for (int i = 1; i < prices.size(); i++) {
            if (prices[i] > prices[i-1]) {
                profit += prices[i] - prices[i-1];
            }
        }
        return profit;
    }
};

// Alternative solution using state machine approach
class StockTraderStateMachine {
public:
    static int maxProfit(int k, std::vector<int>& prices) {
        int n = prices.size();
        if (n <= 1 || k == 0) return 0;
        
        if (k >= n / 2) {
            return quickSolve(prices);
        }
        
        // States: hold[i] = max profit after buying with i transactions
        //         sold[i] = max profit after selling with i transactions
        std::vector<int> hold(k + 1, INT_MIN);
        std::vector<int> sold(k + 1, 0);
        
        for (int price : prices) {
            for (int i = k; i >= 1; i--) {
                sold[i] = std::max(sold[i], hold[i] + price);
                hold[i] = std::max(hold[i], sold[i-1] - price);
            }
        }
        
        return sold[k];
    }
    
private:
    static int quickSolve(const std::vector<int>& prices) {
        int profit = 0;
        for (int i = 1; i < prices.size(); i++) {
            profit += std::max(0, prices[i] - prices[i-1]);
        }
        return profit;
    }
};

// Test framework
class TestFramework {
public:
    static void runTests() {
        std::cout << "Running Stock Trading Tests..." << std::endl;
        
        // Test Case 1: Basic example
        {
            std::vector<int> prices = {2, 4, 1};
            int k = 2;
            int expected = 2;
            int result = StockTrader::maxProfit(k, prices);
            assert(result == expected);
            std::cout << "✓ Test 1 passed: k=" << k << ", prices=[2,4,1], result=" << result << std::endl;
        }
        
        // Test Case 2: No profit possible
        {
            std::vector<int> prices = {3, 2, 1};
            int k = 2;
            int expected = 0;
            int result = StockTrader::maxProfit(k, prices);
            assert(result == expected);
            std::cout << "✓ Test 2 passed: k=" << k << ", prices=[3,2,1], result=" << result << std::endl;
        }
        
        // Test Case 3: Multiple transactions
        {
            std::vector<int> prices = {3, 2, 6, 5, 0, 3};
            int k = 2;
            int expected = 7; // Buy at 2, sell at 6 (profit=4), buy at 0, sell at 3 (profit=3)
            int result = StockTrader::maxProfit(k, prices);
            assert(result == expected);
            std::cout << "✓ Test 3 passed: k=" << k << ", prices=[3,2,6,5,0,3], result=" << result << std::endl;
        }
        
        // Test Case 4: Large k (unlimited transactions)
        {
            std::vector<int> prices = {1, 2, 3, 4, 5};
            int k = 10;
            int expected = 4; // Buy at 1, sell at 5
            int result = StockTrader::maxProfit(k, prices);
            assert(result == expected);
            std::cout << "✓ Test 4 passed: k=" << k << ", prices=[1,2,3,4,5], result=" << result << std::endl;
        }
        
        // Test Case 5: Edge case - single day
        {
            std::vector<int> prices = {5};
            int k = 1;
            int expected = 0;
            int result = StockTrader::maxProfit(k, prices);
            assert(result == expected);
            std::cout << "✓ Test 5 passed: k=" << k << ", prices=[5], result=" << result << std::endl;
        }
        
        // Performance test
        performanceTest();
        
        std::cout << "All tests passed! ✓" << std::endl;
    }
    
private:
    static void performanceTest() {
        std::cout << "\nRunning performance test..." << std::endl;
        
        // Generate large test case
        std::vector<int> prices;
        for (int i = 0; i < 1000; i++) {
            prices.push_back(i % 100 + (i / 100) * 10);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        int result = StockTrader::maxProfit(50, prices);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Performance test: 1000 prices, k=50" << std::endl;
        std::cout << "Result: " << result << std::endl;
        std::cout << "Time: " << duration.count() << " μs" << std::endl;
    }
};

int main() {
    std::cout << "HackerRank Problem 1: Maximum Profit from Stock Trading" << std::endl;
    std::cout << "=======================================================" << std::endl;
    
    TestFramework::runTests();
    
    // Interactive example
    std::cout << "\nInteractive Example:" << std::endl;
    std::vector<int> prices = {7, 1, 5, 3, 6, 4};
    int k = 2;
    
    std::cout << "Stock prices: [";
    for (int i = 0; i < prices.size(); i++) {
        std::cout << prices[i];
        if (i < prices.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Maximum transactions: " << k << std::endl;
    
    int maxProfitResult = StockTrader::maxProfit(k, prices);
    std::cout << "Maximum profit: " << maxProfitResult << std::endl;
    
    // Compare with state machine approach
    int stateMachineResult = StockTraderStateMachine::maxProfit(k, prices);
    std::cout << "State machine result: " << stateMachineResult << std::endl;
    assert(maxProfitResult == stateMachineResult);
    
    std::cout << "\nAlgorithm Analysis:" << std::endl;
    std::cout << "- Time Complexity: O(min(k, n) * n)" << std::endl;
    std::cout << "- Space Complexity: O(min(k, n))" << std::endl;
    std::cout << "- Key insight: When k >= n/2, problem reduces to unlimited transactions" << std::endl;
    std::cout << "- DP state: buy[i] and sell[i] for i transactions" << std::endl;
    
    return 0;
}

/*
Solution Explanation:

1. State Definition:
   - buy[i]: Maximum profit after buying stock with at most i transactions
   - sell[i]: Maximum profit after selling stock with at most i transactions

2. Recurrence Relation:
   - sell[i] = max(sell[i], buy[i] + price) // Sell today or keep previous state
   - buy[i] = max(buy[i], sell[i-1] - price) // Buy today or keep previous state

3. Optimization:
   - When k >= n/2, we can perform unlimited transactions
   - This reduces the problem to buying before every price increase

4. Applications in HFT:
   - Portfolio optimization with transaction limits
   - Risk management with position constraints
   - Market making with inventory limits

Interview Follow-ups:
- How to handle transaction costs?
- What if we have cooldown periods?
- How to extend to multiple assets?
- Real-time streaming version?
*/
