/*
LeetCode Problem: Sliding Window Maximum
======================================

Problem: Given an array nums and a sliding window of size k which is moving from the very left 
of the array to the very right, find the maximum number in the window at each position.

Example:
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]

Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

This problem is commonly asked in trading firms as it relates to:
- Real-time data processing
- Sliding window algorithms
- Efficient data structure usage
*/

#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <chrono>

class SlidingWindowMaximum {
public:
    // Solution 1: Using Deque (Monotonic Queue) - O(n) time, O(k) space
    std::vector<int> maxSlidingWindowDeque(std::vector<int>& nums, int k) {
        std::vector<int> result;
        std::deque<int> dq;  // Store indices
        
        for (int i = 0; i < nums.size(); i++) {
            // Remove indices that are out of current window
            while (!dq.empty() && dq.front() <= i - k) {
                dq.pop_front();
            }
            
            // Remove indices whose corresponding values are smaller than current
            while (!dq.empty() && nums[dq.back()] < nums[i]) {
                dq.pop_back();
            }
            
            dq.push_back(i);
            
            // The front of deque contains the index of maximum element
            if (i >= k - 1) {
                result.push_back(nums[dq.front()]);
            }
        }
        
        return result;
    }
    
    // Solution 2: Using Two Stacks - O(n) time, O(k) space
    std::vector<int> maxSlidingWindowTwoStacks(std::vector<int>& nums, int k) {
        std::vector<int> result;
        std::vector<int> left_max(nums.size());
        std::vector<int> right_max(nums.size());
        
        // Precompute maximum for each block of size k
        for (int i = 0; i < nums.size(); i++) {
            if (i % k == 0) {
                left_max[i] = nums[i];
            } else {
                left_max[i] = std::max(left_max[i-1], nums[i]);
            }
        }
        
        for (int i = nums.size() - 1; i >= 0; i--) {
            if (i == nums.size() - 1 || (i + 1) % k == 0) {
                right_max[i] = nums[i];
            } else {
                right_max[i] = std::max(right_max[i+1], nums[i]);
            }
        }
        
        for (int i = 0; i <= nums.size() - k; i++) {
            result.push_back(std::max(right_max[i], left_max[i + k - 1]));
        }
        
        return result;
    }
    
    // Solution 3: Segment Tree approach - O(n log k) time, O(k) space
    class SegmentTree {
    private:
        std::vector<int> tree;
        int n;
        
        void build(std::vector<int>& arr, int node, int start, int end) {
            if (start == end) {
                tree[node] = arr[start];
            } else {
                int mid = (start + end) / 2;
                build(arr, 2*node, start, mid);
                build(arr, 2*node+1, mid+1, end);
                tree[node] = std::max(tree[2*node], tree[2*node+1]);
            }
        }
        
        int query(int node, int start, int end, int l, int r) {
            if (r < start || end < l) {
                return INT_MIN;
            }
            if (l <= start && end <= r) {
                return tree[node];
            }
            int mid = (start + end) / 2;
            return std::max(query(2*node, start, mid, l, r),
                           query(2*node+1, mid+1, end, l, r));
        }
        
    public:
        SegmentTree(std::vector<int>& arr) {
            n = arr.size();
            tree.resize(4 * n);
            build(arr, 1, 0, n-1);
        }
        
        int getMax(int l, int r) {
            return query(1, 0, n-1, l, r);
        }
    };
    
    std::vector<int> maxSlidingWindowSegmentTree(std::vector<int>& nums, int k) {
        std::vector<int> result;
        SegmentTree st(nums);
        
        for (int i = 0; i <= nums.size() - k; i++) {
            result.push_back(st.getMax(i, i + k - 1));
        }
        
        return result;
    }
    
    // Solution 4: Brute Force - O(nk) time, O(1) space
    std::vector<int> maxSlidingWindowBruteForce(std::vector<int>& nums, int k) {
        std::vector<int> result;
        
        for (int i = 0; i <= nums.size() - k; i++) {
            int max_val = nums[i];
            for (int j = i + 1; j < i + k; j++) {
                max_val = std::max(max_val, nums[j]);
            }
            result.push_back(max_val);
        }
        
        return result;
    }
};

// Performance testing and comparison
class PerformanceTester {
public:
    static void testPerformance() {
        std::vector<int> sizes = {1000, 5000, 10000, 50000};
        std::vector<int> window_sizes = {10, 50, 100};
        
        SlidingWindowMaximum solver;
        
        std::cout << "Performance Comparison:\n";
        std::cout << "=====================\n\n";
        
        for (int size : sizes) {
            std::vector<int> nums(size);
            for (int i = 0; i < size; i++) {
                nums[i] = rand() % 10000;
            }
            
            for (int k : window_sizes) {
                if (k >= size) continue;
                
                std::cout << "Array size: " << size << ", Window size: " << k << "\n";
                
                // Test Deque solution
                auto start = std::chrono::high_resolution_clock::now();
                auto result1 = solver.maxSlidingWindowDeque(nums, k);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "  Deque solution: " << duration1.count() << " microseconds\n";
                
                // Test Two Stacks solution
                start = std::chrono::high_resolution_clock::now();
                auto result2 = solver.maxSlidingWindowTwoStacks(nums, k);
                end = std::chrono::high_resolution_clock::now();
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "  Two Stacks solution: " << duration2.count() << " microseconds\n";
                
                // Test Segment Tree solution (only for smaller inputs due to complexity)
                if (size <= 10000) {
                    start = std::chrono::high_resolution_clock::now();
                    auto result3 = solver.maxSlidingWindowSegmentTree(nums, k);
                    end = std::chrono::high_resolution_clock::now();
                    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    std::cout << "  Segment Tree solution: " << duration3.count() << " microseconds\n";
                }
                
                // Verify results are the same
                bool results_match = (result1 == result2);
                std::cout << "  Results match: " << (results_match ? "Yes" : "No") << "\n\n";
            }
        }
    }
};

// Real-world applications in trading
class TradingApplications {
public:
    // Calculate maximum price in sliding window for stop-loss
    static std::vector<double> calculateTrailingStopLoss(const std::vector<double>& prices, 
                                                          int window_size, 
                                                          double stop_percentage) {
        std::vector<double> stop_levels;
        std::deque<int> dq;
        
        for (int i = 0; i < prices.size(); i++) {
            // Remove indices outside window
            while (!dq.empty() && dq.front() <= i - window_size) {
                dq.pop_front();
            }
            
            // Maintain decreasing order
            while (!dq.empty() && prices[dq.back()] < prices[i]) {
                dq.pop_back();
            }
            
            dq.push_back(i);
            
            if (i >= window_size - 1) {
                double max_price = prices[dq.front()];
                double stop_level = max_price * (1.0 - stop_percentage);
                stop_levels.push_back(stop_level);
            }
        }
        
        return stop_levels;
    }
    
    // Calculate maximum volume in sliding window for liquidity analysis
    static std::vector<long long> calculateMaxVolumeWindow(const std::vector<long long>& volumes, 
                                                            int window_size) {
        std::vector<long long> max_volumes;
        std::deque<int> dq;
        
        for (int i = 0; i < volumes.size(); i++) {
            while (!dq.empty() && dq.front() <= i - window_size) {
                dq.pop_front();
            }
            
            while (!dq.empty() && volumes[dq.back()] < volumes[i]) {
                dq.pop_back();
            }
            
            dq.push_back(i);
            
            if (i >= window_size - 1) {
                max_volumes.push_back(volumes[dq.front()]);
            }
        }
        
        return max_volumes;
    }
};

// Test cases and examples
int main() {
    SlidingWindowMaximum solver;
    
    // Test case 1: Basic example
    std::vector<int> nums1 = {1, 3, -1, -3, 5, 3, 6, 7};
    int k1 = 3;
    
    std::cout << "Test Case 1:\n";
    std::cout << "Input: [1,3,-1,-3,5,3,6,7], k = 3\n";
    
    auto result1 = solver.maxSlidingWindowDeque(nums1, k1);
    std::cout << "Output: [";
    for (int i = 0; i < result1.size(); i++) {
        std::cout << result1[i];
        if (i < result1.size() - 1) std::cout << ",";
    }
    std::cout << "]\n\n";
    
    // Test case 2: All same elements
    std::vector<int> nums2 = {5, 5, 5, 5, 5};
    int k2 = 3;
    
    std::cout << "Test Case 2:\n";
    std::cout << "Input: [5,5,5,5,5], k = 3\n";
    
    auto result2 = solver.maxSlidingWindowDeque(nums2, k2);
    std::cout << "Output: [";
    for (int i = 0; i < result2.size(); i++) {
        std::cout << result2[i];
        if (i < result2.size() - 1) std::cout << ",";
    }
    std::cout << "]\n\n";
    
    // Test case 3: Decreasing array
    std::vector<int> nums3 = {7, 6, 5, 4, 3, 2, 1};
    int k3 = 3;
    
    std::cout << "Test Case 3:\n";
    std::cout << "Input: [7,6,5,4,3,2,1], k = 3\n";
    
    auto result3 = solver.maxSlidingWindowDeque(nums3, k3);
    std::cout << "Output: [";
    for (int i = 0; i < result3.size(); i++) {
        std::cout << result3[i];
        if (i < result3.size() - 1) std::cout << ",";
    }
    std::cout << "]\n\n";
    
    // Trading application example
    std::cout << "Trading Application Example:\n";
    std::cout << "===========================\n";
    
    std::vector<double> stock_prices = {100.0, 102.5, 101.8, 105.2, 103.7, 107.1, 106.3, 109.8};
    auto stop_levels = TradingApplications::calculateTrailingStopLoss(stock_prices, 3, 0.05);
    
    std::cout << "Stock prices: ";
    for (double price : stock_prices) {
        std::cout << price << " ";
    }
    std::cout << "\n";
    
    std::cout << "Trailing stop levels (5% below 3-period max): ";
    for (double stop : stop_levels) {
        std::cout << stop << " ";
    }
    std::cout << "\n\n";
    
    // Performance testing
    std::cout << "Running performance tests...\n";
    PerformanceTester::testPerformance();
    
    return 0;
}
