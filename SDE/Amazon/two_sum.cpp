/*
Amazon SDE Interview Problem 1: Two Sum (Classic Amazon Problem)
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:
Input: nums = [3,3], target = 6
Output: [0,1]

This is THE most asked Amazon interview question. Multiple approaches showcase problem-solving evolution.

Time Complexity: O(n) with hash map, O(n²) with brute force
Space Complexity: O(n) with hash map, O(1) with brute force
*/

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

class Solution {
public:
    // Approach 1: Hash Map (Optimal and most common in interviews)
    std::vector<int> twoSum(std::vector<int>& nums, int target) {
        std::unordered_map<int, int> numToIndex;
        
        for (int i = 0; i < nums.size(); i++) {
            int complement = target - nums[i];
            
            if (numToIndex.find(complement) != numToIndex.end()) {
                return {numToIndex[complement], i};
            }
            
            numToIndex[nums[i]] = i;
        }
        
        return {}; // No solution found (shouldn't happen per problem statement)
    }
    
    // Approach 2: Brute Force (starting point for optimization discussion)
    std::vector<int> twoSumBruteForce(std::vector<int>& nums, int target) {
        for (int i = 0; i < nums.size(); i++) {
            for (int j = i + 1; j < nums.size(); j++) {
                if (nums[i] + nums[j] == target) {
                    return {i, j};
                }
            }
        }
        return {};
    }
    
    // Approach 3: Two Pointers (requires sorting, loses original indices)
    std::vector<int> twoSumTwoPointers(std::vector<int>& nums, int target) {
        std::vector<std::pair<int, int>> indexedNums;
        for (int i = 0; i < nums.size(); i++) {
            indexedNums.push_back({nums[i], i});
        }
        
        std::sort(indexedNums.begin(), indexedNums.end());
        
        int left = 0, right = indexedNums.size() - 1;
        
        while (left < right) {
            int sum = indexedNums[left].first + indexedNums[right].first;
            
            if (sum == target) {
                return {indexedNums[left].second, indexedNums[right].second};
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
        
        return {};
    }
    
    // Approach 4: Modified for finding all pairs (common follow-up)
    std::vector<std::vector<int>> twoSumAllPairs(std::vector<int>& nums, int target) {
        std::vector<std::vector<int>> result;
        std::unordered_map<int, std::vector<int>> numToIndices;
        
        // Build map of number to all its indices
        for (int i = 0; i < nums.size(); i++) {
            numToIndices[nums[i]].push_back(i);
        }
        
        std::unordered_set<int> processed;
        
        for (int i = 0; i < nums.size(); i++) {
            int complement = target - nums[i];
            
            if (processed.count(nums[i])) continue;
            
            if (complement == nums[i]) {
                // Same number case: need at least 2 occurrences
                if (numToIndices[nums[i]].size() >= 2) {
                    for (int j = 0; j < numToIndices[nums[i]].size(); j++) {
                        for (int k = j + 1; k < numToIndices[nums[i]].size(); k++) {
                            result.push_back({numToIndices[nums[i]][j], numToIndices[nums[i]][k]});
                        }
                    }
                }
            } else if (numToIndices.count(complement) && !processed.count(complement)) {
                // Different numbers case
                for (int idx1 : numToIndices[nums[i]]) {
                    for (int idx2 : numToIndices[complement]) {
                        if (idx1 < idx2) {
                            result.push_back({idx1, idx2});
                        }
                    }
                }
            }
            
            processed.insert(nums[i]);
        }
        
        return result;
    }
    
    // Approach 5: Optimized for sorted input (if allowed to modify)
    std::vector<int> twoSumSorted(std::vector<int>& nums, int target) {
        // Store original indices
        std::vector<std::pair<int, int>> indexedNums;
        for (int i = 0; i < nums.size(); i++) {
            indexedNums.push_back({nums[i], i});
        }
        
        // Sort by value
        std::sort(indexedNums.begin(), indexedNums.end());
        
        int left = 0, right = indexedNums.size() - 1;
        
        while (left < right) {
            int sum = indexedNums[left].first + indexedNums[right].first;
            
            if (sum == target) {
                int idx1 = indexedNums[left].second;
                int idx2 = indexedNums[right].second;
                return {std::min(idx1, idx2), std::max(idx1, idx2)};
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
        
        return {};
    }
    
    // Approach 6: Space-optimized for specific constraints
    std::vector<int> twoSumSpaceOptimized(std::vector<int>& nums, int target) {
        // If we know the range of numbers is small, we can use array instead of hash map
        // Assuming numbers are in range [-1000, 1000] for this example
        const int OFFSET = 1000;
        const int SIZE = 2001;
        std::vector<int> indices(SIZE, -1);
        
        for (int i = 0; i < nums.size(); i++) {
            int complement = target - nums[i];
            
            // Check if complement is in valid range
            if (complement >= -1000 && complement <= 1000) {
                int complementIdx = complement + OFFSET;
                if (indices[complementIdx] != -1) {
                    return {indices[complementIdx], i};
                }
            }
            
            // Store current number's index
            if (nums[i] >= -1000 && nums[i] <= 1000) {
                indices[nums[i] + OFFSET] = i;
            }
        }
        
        return {};
    }
};

// Comprehensive test framework
class TwoSumTest {
public:
    static void runTests() {
        Solution solution;
        
        std::cout << "Amazon Two Sum Tests:" << std::endl;
        std::cout << "====================" << std::endl;
        
        testBasicCases(solution);
        testEdgeCases(solution);
        compareApproaches(solution);
        amazonSpecificScenarios(solution);
        performanceAnalysis(solution);
    }
    
    static void testBasicCases(Solution& solution) {
        std::cout << "\nBasic Test Cases:" << std::endl;
        std::cout << "=================" << std::endl;
        
        std::vector<std::tuple<std::vector<int>, int, std::vector<int>>> testCases = {
            {{2, 7, 11, 15}, 9, {0, 1}},
            {{3, 2, 4}, 6, {1, 2}},
            {{3, 3}, 6, {0, 1}},
            {{1, 2, 3, 4, 5}, 8, {2, 4}},
            {{-1, -2, -3, -4, -5}, -8, {2, 4}}
        };
        
        for (int i = 0; i < testCases.size(); i++) {
            auto [nums, target, expected] = testCases[i];
            auto result = solution.twoSum(nums, target);
            
            // Sort both results for comparison since order doesn't matter
            std::sort(result.begin(), result.end());
            std::sort(expected.begin(), expected.end());
            
            std::cout << "Test " << (i + 1) << ": nums=[";
            for (int j = 0; j < nums.size(); j++) {
                std::cout << nums[j];
                if (j < nums.size() - 1) std::cout << ",";
            }
            std::cout << "], target=" << target << std::endl;
            std::cout << "Result: [" << result[0] << "," << result[1] << "]" << std::endl;
            std::cout << "Expected: [" << expected[0] << "," << expected[1] << "]" << std::endl;
            std::cout << "Status: " << (result == expected ? "✅ PASS" : "❌ FAIL") << std::endl;
            std::cout << "---" << std::endl;
        }
    }
    
    static void testEdgeCases(Solution& solution) {
        std::cout << "\nEdge Case Tests:" << std::endl;
        std::cout << "================" << std::endl;
        
        // Minimum size array
        std::vector<int> minArray = {1, 2};
        auto result1 = solution.twoSum(minArray, 3);
        std::cout << "Minimum array [1,2], target=3: [" << result1[0] << "," << result1[1] << "] " 
                  << (result1 == std::vector<int>{0, 1} ? "✅" : "❌") << std::endl;
        
        // Negative numbers
        std::vector<int> negArray = {-3, 4, 3, 90};
        auto result2 = solution.twoSum(negArray, 0);
        std::sort(result2.begin(), result2.end());
        std::cout << "Negative numbers [-3,4,3,90], target=0: [" << result2[0] << "," << result2[1] << "] " 
                  << (result2 == std::vector<int>{0, 2} ? "✅" : "❌") << std::endl;
        
        // Large numbers
        std::vector<int> largeArray = {1000000, 2000000, 3000000};
        auto result3 = solution.twoSum(largeArray, 3000000);
        std::sort(result3.begin(), result3.end());
        std::cout << "Large numbers, target=3000000: [" << result3[0] << "," << result3[1] << "] " 
                  << (result3 == std::vector<int>{0, 1} ? "✅" : "❌") << std::endl;
        
        // Zero target
        std::vector<int> zeroArray = {-1, 0, 1, 2};
        auto result4 = solution.twoSum(zeroArray, 0);
        std::sort(result4.begin(), result4.end());
        std::cout << "Zero target [-1,0,1,2]: [" << result4[0] << "," << result4[1] << "] " 
                  << (result4 == std::vector<int>{0, 2} ? "✅" : "❌") << std::endl;
        
        // Duplicate elements
        std::vector<int> dupArray = {3, 3, 4};
        auto result5 = solution.twoSum(dupArray, 6);
        std::sort(result5.begin(), result5.end());
        std::cout << "Duplicates [3,3,4], target=6: [" << result5[0] << "," << result5[1] << "] " 
                  << (result5 == std::vector<int>{0, 1} ? "✅" : "❌") << std::endl;
    }
    
    static void compareApproaches(Solution& solution) {
        std::cout << "\nApproach Comparison:" << std::endl;
        std::cout << "===================" << std::endl;
        
        std::vector<int> testArray = {2, 7, 11, 15, 3, 6, 8, 1, 9, 4};
        int target = 10;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result1 = solution.twoSum(testArray, target);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        auto result2 = solution.twoSumBruteForce(testArray, target);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        auto result3 = solution.twoSumTwoPointers(testArray, target);
        end = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        auto result4 = solution.twoSumSpaceOptimized(testArray, target);
        end = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        std::cout << "Hash Map: [" << result1[0] << "," << result1[1] << "] (" << duration1.count() << " ns)" << std::endl;
        std::cout << "Brute Force: [" << result2[0] << "," << result2[1] << "] (" << duration2.count() << " ns)" << std::endl;
        std::cout << "Two Pointers: [" << result3[0] << "," << result3[1] << "] (" << duration3.count() << " ns)" << std::endl;
        std::cout << "Space Optimized: [" << result4[0] << "," << result4[1] << "] (" << duration4.count() << " ns)" << std::endl;
    }
    
    static void amazonSpecificScenarios(Solution& solution) {
        std::cout << "\nAmazon-Specific Scenarios:" << std::endl;
        std::cout << "=========================" << std::endl;
        
        // Scenario 1: Product prices (common Amazon context)
        std::vector<int> prices = {1299, 899, 1599, 699, 2099}; // Prices in cents
        int budget = 2198; // Budget in cents
        auto result1 = solution.twoSum(prices, budget);
        std::cout << "Product prices scenario: [" << result1[0] << "," << result1[1] << "] "
                  << "(Products at $" << prices[result1[0]]/100.0 << " and $" << prices[result1[1]]/100.0 << ")" << std::endl;
        
        // Scenario 2: Warehouse capacities
        std::vector<int> capacities = {1000, 1500, 2000, 500, 3000};
        int requiredCapacity = 3500;
        auto result2 = solution.twoSum(capacities, requiredCapacity);
        std::cout << "Warehouse capacities: [" << result2[0] << "," << result2[1] << "] "
                  << "(Capacities: " << capacities[result2[0]] << " + " << capacities[result2[1]] << ")" << std::endl;
        
        // Scenario 3: Delivery times
        std::vector<int> deliveryTimes = {30, 45, 60, 90, 120}; // Minutes
        int totalTime = 105;
        auto result3 = solution.twoSum(deliveryTimes, totalTime);
        std::cout << "Delivery times: [" << result3[0] << "," << result3[1] << "] "
                  << "(Times: " << deliveryTimes[result3[0]] << " + " << deliveryTimes[result3[1]] << " minutes)" << std::endl;
        
        // Scenario 4: Package weights
        std::vector<int> weights = {5, 10, 15, 20, 25}; // Kg
        int maxWeight = 30;
        auto result4 = solution.twoSum(weights, maxWeight);
        std::cout << "Package weights: [" << result4[0] << "," << result4[1] << "] "
                  << "(Weights: " << weights[result4[0]] << " + " << weights[result4[1]] << " kg)" << std::endl;
        
        // Scenario 5: Customer ratings (scaled to integers)
        std::vector<int> ratings = {42, 38, 45, 35, 48}; // Ratings * 10
        int targetRating = 80; // 4.0 * 2 * 10
        auto result5 = solution.twoSum(ratings, targetRating);
        std::cout << "Customer ratings: [" << result5[0] << "," << result5[1] << "] "
                  << "(Ratings: " << ratings[result5[0]]/10.0 << " + " << ratings[result5[1]]/10.0 << ")" << std::endl;
    }
    
    static void performanceAnalysis(Solution& solution) {
        std::cout << "\nPerformance Analysis:" << std::endl;
        std::cout << "====================" << std::endl;
        
        std::vector<int> sizes = {100, 1000, 10000, 100000};
        
        for (int size : sizes) {
            std::vector<int> largeArray = generateRandomArray(size);
            int target = largeArray[0] + largeArray[size-1]; // Ensure solution exists
            
            // Test hash map approach
            auto start = std::chrono::high_resolution_clock::now();
            auto result = solution.twoSum(largeArray, target);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            std::cout << "Array size " << size << ": " << duration.count() << " μs" << std::endl;
        }
        
        // Memory usage analysis
        std::cout << "\nMemory Usage Analysis:" << std::endl;
        std::cout << "======================" << std::endl;
        
        int testSize = 10000;
        std::vector<int> testArray = generateRandomArray(testSize);
        
        // Estimate memory usage for different approaches
        size_t hashMapMemory = testSize * (sizeof(int) + sizeof(int)) + sizeof(std::unordered_map<int,int>);
        size_t twoPointerMemory = testSize * sizeof(std::pair<int,int>) + sizeof(std::vector<std::pair<int,int>>);
        size_t bruteForceMemory = sizeof(int) * 2; // Just for loop variables
        
        std::cout << "Hash Map approach: ~" << hashMapMemory / 1024 << " KB" << std::endl;
        std::cout << "Two Pointer approach: ~" << twoPointerMemory / 1024 << " KB" << std::endl;
        std::cout << "Brute Force approach: ~" << bruteForceMemory << " bytes" << std::endl;
    }
    
    static std::vector<int> generateRandomArray(int size) {
        std::vector<int> arr;
        arr.reserve(size);
        
        srand(42); // Fixed seed for reproducible results
        for (int i = 0; i < size; i++) {
            arr.push_back(rand() % 10000);
        }
        
        return arr;
    }
    
    static void followUpQuestions(Solution& solution) {
        std::cout << "\nFollow-up Questions:" << std::endl;
        std::cout << "===================" << std::endl;
        
        // 1. Find all pairs that sum to target
        std::vector<int> nums1 = {1, 2, 3, 4, 2, 3, 1};
        int target1 = 5;
        auto allPairs = solution.twoSumAllPairs(nums1, target1);
        std::cout << "All pairs summing to " << target1 << ":" << std::endl;
        for (const auto& pair : allPairs) {
            std::cout << "[" << pair[0] << "," << pair[1] << "] ";
        }
        std::cout << std::endl;
        
        // 2. What if array is sorted?
        std::vector<int> sortedNums = {1, 2, 3, 4, 5, 6, 7};
        int target2 = 9;
        auto sortedResult = solution.twoSumSorted(sortedNums, target2);
        std::cout << "Sorted array result: [" << sortedResult[0] << "," << sortedResult[1] << "]" << std::endl;
        
        // 3. Memory constraints
        std::cout << "Space-optimized approach suitable for: limited memory environments" << std::endl;
        
        // 4. Multiple solutions
        std::cout << "Multiple solutions handling: Use set to avoid duplicates or return first found" << std::endl;
    }
};

int main() {
    TwoSumTest::runTests();
    TwoSumTest::followUpQuestions(Solution{});
    return 0;
}

/*
Algorithm Analysis:

Core Problem: Find two array elements that sum to a target value

Approach Comparison:

1. Hash Map (Optimal for interviews):
   - Time: O(n), Space: O(n)
   - Single pass through array
   - Immediate lookup for complement
   - Most commonly expected solution

2. Brute Force:
   - Time: O(n²), Space: O(1)
   - Good starting point for optimization discussion
   - Shows problem-solving evolution
   - Clear but inefficient

3. Two Pointers:
   - Time: O(n log n), Space: O(n)
   - Requires sorting first
   - Loses original indices without extra work
   - Good for sorted input follow-up

4. Space Optimized:
   - Time: O(n), Space: O(k) where k is value range
   - Uses array instead of hash map
   - Only works for limited value ranges
   - Shows constraint-based optimization

Amazon Interview Focus:
- Problem-solving approach evolution
- Time/space complexity trade-offs
- Real-world application scenarios
- Follow-up question handling
- Multiple solution approaches

Key Optimizations:
1. Single-pass hash map approach
2. Early termination when solution found
3. Space optimization for constrained ranges
4. Handling duplicates and edge cases

Real-world Amazon Applications:
- Product recommendation pairing
- Warehouse capacity planning
- Delivery route optimization
- Budget allocation problems
- Resource matching systems

Edge Cases:
- Minimum array size (exactly 2 elements)
- Negative numbers
- Zero as target or element
- Duplicate elements
- Large number ranges

Interview Tips:
1. Start with brute force explanation
2. Optimize to hash map approach
3. Discuss space/time trade-offs
4. Handle follow-up questions
5. Consider real-world constraints

Common Mistakes:
1. Using same element twice
2. Not handling duplicates correctly
3. Assuming sorted input
4. Poor hash map implementation
5. Forgetting edge cases

Follow-up Questions (Amazon loves these):
1. What if array is sorted?
2. Find all pairs instead of first pair?
3. What about three sum or k-sum?
4. Memory constraints?
5. Stream of numbers instead of array?

Performance Considerations:
- Hash map vs array for value storage
- Memory allocation patterns
- Cache locality in large arrays
- Parallel processing opportunities
- Incremental updates

Testing Strategy:
- Basic functionality verification
- Edge case coverage
- Performance benchmarking
- Amazon-specific scenarios
- Follow-up question validation

Production Considerations:
- Input validation and sanitization
- Error handling for no solution
- Memory usage monitoring
- Concurrent access patterns
- Logging and debugging support

This problem is fundamental to Amazon interviews because:
1. Tests basic algorithmic thinking
2. Shows optimization skills
3. Has many real-world applications
4. Leads to interesting follow-ups
5. Demonstrates coding proficiency
*/
