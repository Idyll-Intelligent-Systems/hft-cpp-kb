/*
Problem: Median of Two Sorted Arrays (Hard)
Given two sorted arrays nums1 and nums2 of size m and n respectively, 
return the median of the two arrays.

The overall run time complexity should be O(log (m+n)).

Example:
Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000 (merged array = [1,2,3], median = 2)

Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000 (merged array = [1,2,3,4], median = (2+3)/2 = 2.5)

Time Complexity: O(log(min(m,n)))
Space Complexity: O(1)
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>

class Solution {
public:
    double findMedianSortedArrays(std::vector<int>& nums1, std::vector<int>& nums2) {
        // Ensure nums1 is the smaller array for optimization
        if (nums1.size() > nums2.size()) {
            return findMedianSortedArrays(nums2, nums1);
        }
        
        int m = nums1.size();
        int n = nums2.size();
        int total = m + n;
        int half = (total + 1) / 2;  // Elements in left partition
        
        int left = 0, right = m;
        
        while (left <= right) {
            int partitionX = (left + right) / 2;
            int partitionY = half - partitionX;
            
            // Handle edge cases for partition boundaries
            int maxLeftX = (partitionX == 0) ? INT_MIN : nums1[partitionX - 1];
            int minRightX = (partitionX == m) ? INT_MAX : nums1[partitionX];
            
            int maxLeftY = (partitionY == 0) ? INT_MIN : nums2[partitionY - 1];
            int minRightY = (partitionY == n) ? INT_MAX : nums2[partitionY];
            
            // Check if we found the correct partition
            if (maxLeftX <= minRightY && maxLeftY <= minRightX) {
                // Perfect partition found
                if (total % 2 == 1) {
                    // Odd total length - median is max of left partition
                    return std::max(maxLeftX, maxLeftY);
                } else {
                    // Even total length - median is average of middle two elements
                    return (std::max(maxLeftX, maxLeftY) + std::min(minRightX, minRightY)) / 2.0;
                }
            }
            else if (maxLeftX > minRightY) {
                // partitionX is too far right, move left
                right = partitionX - 1;
            }
            else {
                // partitionX is too far left, move right
                left = partitionX + 1;
            }
        }
        
        // Should never reach here for valid input
        throw std::invalid_argument("Input arrays are not sorted");
    }
    
    // Alternative approach: Merge and find median (O(m+n) time, O(m+n) space)
    double findMedianBruteForce(std::vector<int>& nums1, std::vector<int>& nums2) {
        std::vector<int> merged;
        merged.reserve(nums1.size() + nums2.size());
        
        int i = 0, j = 0;
        
        // Merge two sorted arrays
        while (i < nums1.size() && j < nums2.size()) {
            if (nums1[i] <= nums2[j]) {
                merged.push_back(nums1[i++]);
            } else {
                merged.push_back(nums2[j++]);
            }
        }
        
        // Add remaining elements
        while (i < nums1.size()) {
            merged.push_back(nums1[i++]);
        }
        while (j < nums2.size()) {
            merged.push_back(nums2[j++]);
        }
        
        int n = merged.size();
        if (n % 2 == 1) {
            return merged[n / 2];
        } else {
            return (merged[n / 2 - 1] + merged[n / 2]) / 2.0;
        }
    }
    
    // Space-optimized merge approach (O(m+n) time, O(1) space)
    double findMedianOptimizedMerge(std::vector<int>& nums1, std::vector<int>& nums2) {
        int total = nums1.size() + nums2.size();
        int target1 = (total - 1) / 2;
        int target2 = total / 2;
        
        int i = 0, j = 0, count = 0;
        int val1 = 0, val2 = 0;
        
        while (count <= target2) {
            int current;
            
            if (i >= nums1.size()) {
                current = nums2[j++];
            } else if (j >= nums2.size()) {
                current = nums1[i++];
            } else if (nums1[i] <= nums2[j]) {
                current = nums1[i++];
            } else {
                current = nums2[j++];
            }
            
            if (count == target1) val1 = current;
            if (count == target2) val2 = current;
            
            count++;
        }
        
        return (val1 + val2) / 2.0;
    }
};

// Extended binary search utilities for related problems
class BinarySearchUtils {
public:
    // Find kth element in two sorted arrays
    static int findKthElement(std::vector<int>& nums1, std::vector<int>& nums2, int k) {
        if (nums1.size() > nums2.size()) {
            return findKthElement(nums2, nums1, k);
        }
        
        int m = nums1.size();
        int n = nums2.size();
        
        if (m == 0) return nums2[k - 1];
        if (k == 1) return std::min(nums1[0], nums2[0]);
        
        int i = std::min(k / 2, m);
        int j = k - i;
        
        if (j > n) {
            j = n;
            i = k - j;
        }
        
        if (nums1[i - 1] < nums2[j - 1]) {
            std::vector<int> newNums1(nums1.begin() + i, nums1.end());
            return findKthElement(newNums1, nums2, k - i);
        } else {
            std::vector<int> newNums2(nums2.begin() + j, nums2.end());
            return findKthElement(nums1, newNums2, k - j);
        }
    }
    
    // Check if we can achieve median <= target with given partition
    static bool canAchieveMedian(std::vector<int>& nums1, std::vector<int>& nums2, double target) {
        int total = nums1.size() + nums2.size();
        int half = (total + 1) / 2;
        
        int count = 0;
        int i = 0, j = 0;
        
        // Count elements <= target
        while ((i < nums1.size() || j < nums2.size()) && count < half) {
            if (i >= nums1.size()) {
                if (nums2[j] <= target) count++;
                j++;
            } else if (j >= nums2.size()) {
                if (nums1[i] <= target) count++;
                i++;
            } else if (nums1[i] <= nums2[j]) {
                if (nums1[i] <= target) count++;
                i++;
            } else {
                if (nums2[j] <= target) count++;
                j++;
            }
        }
        
        return count >= half;
    }
};

// Performance testing utilities
class PerformanceTest {
public:
    static void testPerformance(Solution& solution) {
        std::cout << "Performance Test Results:" << std::endl;
        std::cout << "========================" << std::endl;
        
        // Generate large test arrays
        std::vector<int> large1, large2;
        for (int i = 0; i < 10000; i += 2) {
            large1.push_back(i);
        }
        for (int i = 1; i < 10000; i += 2) {
            large2.push_back(i);
        }
        
        // Test binary search approach
        auto start = std::chrono::high_resolution_clock::now();
        double result1 = solution.findMedianSortedArrays(large1, large2);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Binary Search: " << result1 << " (Time: " << duration1.count() << " μs)" << std::endl;
        
        // Test merge approach
        start = std::chrono::high_resolution_clock::now();
        double result2 = solution.findMedianOptimizedMerge(large1, large2);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Optimized Merge: " << result2 << " (Time: " << duration2.count() << " μs)" << std::endl;
    }
};

void testCase(Solution& solution, std::vector<int> nums1, std::vector<int> nums2, double expected) {
    std::cout << "nums1: [";
    for (int i = 0; i < nums1.size(); i++) {
        std::cout << nums1[i];
        if (i < nums1.size() - 1) std::cout << ",";
    }
    std::cout << "], nums2: [";
    for (int i = 0; i < nums2.size(); i++) {
        std::cout << nums2[i];
        if (i < nums2.size() - 1) std::cout << ",";
    }
    std::cout << "]" << std::endl;
    
    double result1 = solution.findMedianSortedArrays(nums1, nums2);
    double result2 = solution.findMedianBruteForce(nums1, nums2);
    double result3 = solution.findMedianOptimizedMerge(nums1, nums2);
    
    std::cout << "Expected: " << expected << std::endl;
    std::cout << "Binary Search: " << result1 << std::endl;
    std::cout << "Brute Force: " << result2 << std::endl;
    std::cout << "Optimized Merge: " << result3 << std::endl;
    std::cout << "All methods agree: " << (abs(result1 - result2) < 1e-9 && abs(result2 - result3) < 1e-9 ? "Yes" : "No") << std::endl;
    std::cout << "Correct: " << (abs(result1 - expected) < 1e-9 ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
}

int main() {
    Solution solution;
    
    std::cout << "Median of Two Sorted Arrays Test Cases:" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // Test case 1: Example from problem
    testCase(solution, {1, 3}, {2}, 2.0);
    
    // Test case 2: Even total length
    testCase(solution, {1, 2}, {3, 4}, 2.5);
    
    // Test case 3: One empty array
    testCase(solution, {}, {1}, 1.0);
    
    // Test case 4: Different sizes
    testCase(solution, {1, 3, 5}, {2, 4, 6, 7, 8}, 4.5);
    
    // Test case 5: No overlap
    testCase(solution, {1, 2, 3}, {4, 5, 6}, 3.5);
    
    // Test case 6: Complete overlap
    testCase(solution, {1, 1, 1}, {1, 1, 1}, 1.0);
    
    // Test case 7: Negative numbers
    testCase(solution, {-2, -1}, {3, 4}, 0.5);
    
    // Test utility functions
    std::cout << "Utility Function Tests:" << std::endl;
    std::cout << "======================" << std::endl;
    
    std::vector<int> test1 = {1, 3, 5};
    std::vector<int> test2 = {2, 4, 6};
    
    int kth = BinarySearchUtils::findKthElement(test1, test2, 3);
    std::cout << "3rd element in merged arrays: " << kth << std::endl;
    
    bool canAchieve = BinarySearchUtils::canAchieveMedian(test1, test2, 3.5);
    std::cout << "Can achieve median <= 3.5: " << (canAchieve ? "Yes" : "No") << std::endl;
    
    return 0;
}

/*
Algorithm Analysis:

1. Binary Search Approach (Optimal):
   - Time: O(log(min(m,n))) where m,n are array sizes
   - Space: O(1)
   - Key insight: Binary search on the smaller array for partition point

2. Merge Approach:
   - Time: O(m+n)
   - Space: O(m+n) for brute force, O(1) for optimized
   - Straightforward but doesn't meet O(log(m+n)) requirement

Key Insights:

1. Partition Strategy:
   - Divide both arrays such that left partition has (m+n+1)/2 elements
   - All elements in left <= all elements in right
   - If maxLeftX <= minRightY and maxLeftY <= minRightX, partition is correct

2. Binary Search Logic:
   - Search on smaller array to minimize complexity
   - Adjust partition based on cross-partition comparisons
   - Handle edge cases with INT_MIN/INT_MAX

3. Median Calculation:
   - Odd total: max of left partition
   - Even total: average of max(left) and min(right)

Edge Cases:
- One or both arrays empty
- Arrays of very different sizes
- Duplicate elements
- Negative numbers
- Single element arrays

Common Mistakes:
1. Not ensuring smaller array for binary search
2. Incorrect partition size calculation
3. Boundary condition errors (INT_MIN/MAX)
4. Integer overflow in median calculation
5. Not handling odd/even total length correctly

Applications:
- Statistics and data analysis
- Database query optimization
- Stream processing with limited memory
- Parallel computing workload balancing

Interview Tips:
1. Start with brute force, then optimize
2. Draw examples to understand partition logic
3. Handle edge cases carefully
4. Consider integer overflow
5. Test with various input sizes

Related Problems:
- Kth element in two sorted arrays
- Sliding window median
- Running median from data stream
- Median in row-wise sorted matrix

Optimization Techniques:
1. Always search on smaller array
2. Use integer arithmetic where possible
3. Early termination conditions
4. Efficient boundary handling
*/
