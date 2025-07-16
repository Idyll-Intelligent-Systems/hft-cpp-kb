/*
Google SDE Interview Problem 3: Smallest Range Covering Elements from K Lists (Hard)
You have k lists of sorted integers in non-decreasing order. Find the smallest range that includes 
at least one number from each of the k lists.

We define the range [a, b] is smaller than range [c, d] if b - a < d - c or a < c if b - a == d - c.

Example:
Input: nums = [[4,10,15,24,26],[0,9,12,20],[5,18,22,30]]
Output: [20,24]
Explanation: 
List 1: [4, 10, 15, 24, 26], 24 is in range [20,24].
List 2: [0, 9, 12, 20], 20 is in range [20,24].
List 3: [5, 18, 22, 30], 22 is in range [20,24].

Time Complexity: O(n * log k) where n is total elements, k is number of lists
Space Complexity: O(k) for priority queue
*/

#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <algorithm>

class Solution {
public:
    // Approach 1: Min Heap with Sliding Window
    std::vector<int> smallestRange(std::vector<std::vector<int>>& nums) {
        // Priority queue to store {value, list_index, element_index}
        auto cmp = [](const std::vector<int>& a, const std::vector<int>& b) {
            return a[0] > b[0]; // Min heap based on value
        };
        std::priority_queue<std::vector<int>, std::vector<std::vector<int>>, decltype(cmp)> pq(cmp);
        
        int maxVal = INT_MIN;
        int rangeStart = 0, rangeEnd = INT_MAX;
        
        // Initialize: add first element from each list
        for (int i = 0; i < nums.size(); i++) {
            pq.push({nums[i][0], i, 0});
            maxVal = std::max(maxVal, nums[i][0]);
        }
        
        while (pq.size() == nums.size()) {
            auto current = pq.top();
            pq.pop();
            
            int minVal = current[0];
            int listIdx = current[1];
            int elemIdx = current[2];
            
            // Update range if current is better
            if (maxVal - minVal < rangeEnd - rangeStart) {
                rangeStart = minVal;
                rangeEnd = maxVal;
            }
            
            // Move to next element in the same list
            if (elemIdx + 1 < nums[listIdx].size()) {
                int nextVal = nums[listIdx][elemIdx + 1];
                pq.push({nextVal, listIdx, elemIdx + 1});
                maxVal = std::max(maxVal, nextVal);
            }
        }
        
        return {rangeStart, rangeEnd};
    }
    
    // Approach 2: Two Pointers with Sorted Merge
    std::vector<int> smallestRangeAdvanced(std::vector<std::vector<int>>& nums) {
        // Create merged array with {value, list_index}
        std::vector<std::pair<int, int>> merged;
        
        for (int i = 0; i < nums.size(); i++) {
            for (int val : nums[i]) {
                merged.push_back({val, i});
            }
        }
        
        std::sort(merged.begin(), merged.end());
        
        // Sliding window to find minimum range
        std::vector<int> count(nums.size(), 0);
        int left = 0, validLists = 0;
        int rangeStart = 0, rangeEnd = INT_MAX;
        
        for (int right = 0; right < merged.size(); right++) {
            int listIdx = merged[right].second;
            
            if (count[listIdx] == 0) {
                validLists++;
            }
            count[listIdx]++;
            
            // Shrink window while maintaining all lists
            while (validLists == nums.size()) {
                int currentRange = merged[right].first - merged[left].first;
                
                if (currentRange < rangeEnd - rangeStart) {
                    rangeStart = merged[left].first;
                    rangeEnd = merged[right].first;
                }
                
                int leftListIdx = merged[left].second;
                count[leftListIdx]--;
                if (count[leftListIdx] == 0) {
                    validLists--;
                }
                left++;
            }
        }
        
        return {rangeStart, rangeEnd};
    }
    
    // Approach 3: Optimized with Early Termination
    std::vector<int> smallestRangeOptimized(std::vector<std::vector<int>>& nums) {
        // Find global min and max for early termination
        int globalMin = INT_MAX, globalMax = INT_MIN;
        for (const auto& list : nums) {
            if (!list.empty()) {
                globalMin = std::min(globalMin, list.front());
                globalMax = std::max(globalMax, list.back());
            }
        }
        
        // If range is already minimal (one element per list at same value)
        bool allSame = true;
        for (const auto& list : nums) {
            if (!list.empty() && (list.front() != globalMin || list.back() != globalMin)) {
                allSame = false;
                break;
            }
        }
        
        if (allSame) {
            return {globalMin, globalMin};
        }
        
        // Use standard min heap approach
        return smallestRange(nums);
    }
    
    // Approach 4: Memory Optimized
    std::vector<int> smallestRangeMemoryOptimized(std::vector<std::vector<int>>& nums) {
        struct Element {
            int value;
            int listIdx;
            int elemIdx;
            
            bool operator>(const Element& other) const {
                return value > other.value;
            }
        };
        
        std::priority_queue<Element, std::vector<Element>, std::greater<Element>> pq;
        int maxVal = INT_MIN;
        
        // Initialize
        for (int i = 0; i < nums.size(); i++) {
            if (!nums[i].empty()) {
                pq.push({nums[i][0], i, 0});
                maxVal = std::max(maxVal, nums[i][0]);
            }
        }
        
        int rangeStart = 0, rangeEnd = INT_MAX;
        
        while (pq.size() == nums.size()) {
            Element current = pq.top();
            pq.pop();
            
            // Update range if better
            if (maxVal - current.value < rangeEnd - rangeStart) {
                rangeStart = current.value;
                rangeEnd = maxVal;
            }
            
            // Add next element from same list
            if (current.elemIdx + 1 < nums[current.listIdx].size()) {
                int nextVal = nums[current.listIdx][current.elemIdx + 1];
                pq.push({nextVal, current.listIdx, current.elemIdx + 1});
                maxVal = std::max(maxVal, nextVal);
            }
        }
        
        return {rangeStart, rangeEnd};
    }
};

// Advanced test cases for Google interviews
class SmallestRangeTest {
public:
    static void runComprehensiveTests() {
        Solution solution;
        
        std::cout << "Google Smallest Range Tests:" << std::endl;
        std::cout << "===========================" << std::endl;
        
        // Test case 1: Basic example
        std::vector<std::vector<int>> nums1 = {{4,10,15,24,26},{0,9,12,20},{5,18,22,30}};
        auto result1 = solution.smallestRange(nums1);
        std::cout << "Test 1: [" << result1[0] << "," << result1[1] << "] (Expected: [20,24])" << std::endl;
        
        // Test case 2: Single elements
        std::vector<std::vector<int>> nums2 = {{1},{2},{3}};
        auto result2 = solution.smallestRange(nums2);
        std::cout << "Test 2: [" << result2[0] << "," << result2[1] << "] (Expected: [1,3])" << std::endl;
        
        // Test case 3: Overlapping ranges
        std::vector<std::vector<int>> nums3 = {{1,2,3},{1,2,3},{1,2,3}};
        auto result3 = solution.smallestRange(nums3);
        std::cout << "Test 3: [" << result3[0] << "," << result3[1] << "] (Expected: [1,1])" << std::endl;
        
        // Test case 4: Large differences
        std::vector<std::vector<int>> nums4 = {{1,3,5,7,9},{2,4,6,8,10}};
        auto result4 = solution.smallestRange(nums4);
        std::cout << "Test 4: [" << result4[0] << "," << result4[1] << "] (Expected: [1,2])" << std::endl;
        
        // Test case 5: Different lengths
        std::vector<std::vector<int>> nums5 = {{1},{2,3,4,5,6,7},{8,9}};
        auto result5 = solution.smallestRange(nums5);
        std::cout << "Test 5: [" << result5[0] << "," << result5[1] << "] (Expected: [1,8])" << std::endl;
        
        // Performance comparison
        std::cout << "\nPerformance Comparison:" << std::endl;
        std::vector<std::vector<int>> largeNums;
        for (int i = 0; i < 10; i++) {
            std::vector<int> list;
            for (int j = 0; j < 1000; j++) {
                list.push_back(i * 1000 + j);
            }
            largeNums.push_back(list);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result_heap = solution.smallestRange(largeNums);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_heap = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        auto result_merge = solution.smallestRangeAdvanced(largeNums);
        end = std::chrono::high_resolution_clock::now();
        auto duration_merge = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Min Heap: [" << result_heap[0] << "," << result_heap[1] << "] in " 
                  << duration_heap.count() << " μs" << std::endl;
        std::cout << "Merge+Sliding: [" << result_merge[0] << "," << result_merge[1] << "] in " 
                  << duration_merge.count() << " μs" << std::endl;
    }
    
    // Edge case testing
    static void testEdgeCases() {
        Solution solution;
        
        std::cout << "\nEdge Case Testing:" << std::endl;
        std::cout << "==================" << std::endl;
        
        // Empty lists (though problem guarantees non-empty)
        std::vector<std::vector<int>> nums_empty = {{}};
        // auto result_empty = solution.smallestRange(nums_empty);
        
        // Single list
        std::vector<std::vector<int>> nums_single = {{1,2,3,4,5}};
        // auto result_single = solution.smallestRange(nums_single);
        
        // Negative numbers
        std::vector<std::vector<int>> nums_negative = {{-3,-1,1},{-2,2,4},{-1,0,3}};
        auto result_negative = solution.smallestRange(nums_negative);
        std::cout << "Negative numbers: [" << result_negative[0] << "," << result_negative[1] << "]" << std::endl;
        
        // Large numbers
        std::vector<std::vector<int>> nums_large = {{100000,200000},{150000,300000},{175000,250000}};
        auto result_large = solution.smallestRange(nums_large);
        std::cout << "Large numbers: [" << result_large[0] << "," << result_large[1] << "]" << std::endl;
    }
};

int main() {
    SmallestRangeTest::runComprehensiveTests();
    SmallestRangeTest::testEdgeCases();
    return 0;
}

/*
Algorithm Analysis:

Approach 1: Min Heap
- Maintain one element from each list in priority queue
- Track maximum value among current elements
- Move minimum element forward, update range
- Time: O(n log k), Space: O(k)

Approach 2: Merge + Sliding Window
- Merge all lists with list indices
- Use sliding window to find minimum range covering all lists
- Time: O(n log n), Space: O(n)

Key Insights:
1. Always need one element from each list
2. Range defined by min and max of current selection
3. Move minimum element to potentially improve range
4. Greedy approach works due to sorted nature

Google Interview Focus:
- Multiple optimization approaches
- Memory vs time complexity tradeoffs
- Edge case handling (empty lists, single elements)
- Code clarity and modularity
- Scalability discussion

Optimizations:
1. Early termination for trivial cases
2. Memory-efficient data structures
3. Cache-friendly access patterns
4. Bounds checking and validation

Edge Cases:
- Lists of different lengths
- Duplicate values across lists
- Single element lists
- Negative numbers
- Maximum value constraints

Interview Tips:
1. Start with brute force explanation
2. Optimize step by step
3. Discuss space-time tradeoffs
4. Handle edge cases gracefully
5. Write clean, testable code
*/
