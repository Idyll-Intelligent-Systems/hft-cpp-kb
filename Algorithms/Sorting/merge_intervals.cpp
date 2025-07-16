/*
Problem: Merge Intervals (Medium)
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, 
and return an array of the non-overlapping intervals that cover all the intervals in the input.

Example:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].

Time Complexity: O(n log n) due to sorting
Space Complexity: O(n) for the result array
*/

#include <iostream>
#include <vector>
#include <algorithm>

class Solution {
public:
    std::vector<std::vector<int>> merge(std::vector<std::vector<int>>& intervals) {
        if (intervals.empty()) return {};
        
        // Sort intervals by start time
        std::sort(intervals.begin(), intervals.end(), 
                 [](const std::vector<int>& a, const std::vector<int>& b) {
                     return a[0] < b[0];
                 });
        
        std::vector<std::vector<int>> merged;
        
        for (const auto& interval : intervals) {
            // If merged is empty or current interval doesn't overlap with the last one
            if (merged.empty() || merged.back()[1] < interval[0]) {
                merged.push_back(interval);
            } else {
                // Merge with the last interval
                merged.back()[1] = std::max(merged.back()[1], interval[1]);
            }
        }
        
        return merged;
    }
    
    // Alternative approach: In-place modification (if allowed to modify input)
    std::vector<std::vector<int>> mergeInPlace(std::vector<std::vector<int>>& intervals) {
        if (intervals.empty()) return {};
        
        // Sort intervals by start time
        std::sort(intervals.begin(), intervals.end(), 
                 [](const std::vector<int>& a, const std::vector<int>& b) {
                     return a[0] < b[0];
                 });
        
        int index = 0;  // Index for the merged result
        
        for (int i = 1; i < intervals.size(); i++) {
            // If current interval overlaps with intervals[index]
            if (intervals[index][1] >= intervals[i][0]) {
                // Merge intervals
                intervals[index][1] = std::max(intervals[index][1], intervals[i][1]);
            } else {
                // No overlap, move to next position
                index++;
                intervals[index] = intervals[i];
            }
        }
        
        // Resize to keep only merged intervals
        intervals.resize(index + 1);
        return intervals;
    }
    
    // Using custom comparator struct (alternative syntax)
    struct IntervalComparator {
        bool operator()(const std::vector<int>& a, const std::vector<int>& b) {
            if (a[0] == b[0]) {
                return a[1] < b[1];  // If start times are equal, sort by end time
            }
            return a[0] < b[0];
        }
    };
    
    std::vector<std::vector<int>> mergeWithStruct(std::vector<std::vector<int>>& intervals) {
        if (intervals.empty()) return {};
        
        std::sort(intervals.begin(), intervals.end(), IntervalComparator());
        
        std::vector<std::vector<int>> result;
        result.push_back(intervals[0]);
        
        for (int i = 1; i < intervals.size(); i++) {
            auto& last = result.back();
            const auto& current = intervals[i];
            
            if (last[1] >= current[0]) {
                // Overlapping intervals - merge them
                last[1] = std::max(last[1], current[1]);
            } else {
                // Non-overlapping - add current interval
                result.push_back(current);
            }
        }
        
        return result;
    }
};

// Helper class for interval operations
class IntervalUtils {
public:
    // Check if two intervals overlap
    static bool doOverlap(const std::vector<int>& a, const std::vector<int>& b) {
        return std::max(a[0], b[0]) <= std::min(a[1], b[1]);
    }
    
    // Merge two overlapping intervals
    static std::vector<int> mergeTwo(const std::vector<int>& a, const std::vector<int>& b) {
        return {std::min(a[0], b[0]), std::max(a[1], b[1])};
    }
    
    // Get intersection of two intervals (if they overlap)
    static std::vector<int> getIntersection(const std::vector<int>& a, const std::vector<int>& b) {
        int start = std::max(a[0], b[0]);
        int end = std::min(a[1], b[1]);
        return start <= end ? std::vector<int>{start, end} : std::vector<int>{};
    }
    
    // Calculate total covered length after merging
    static int getTotalCoveredLength(const std::vector<std::vector<int>>& intervals) {
        Solution solution;
        auto merged = const_cast<Solution&>(solution).merge(
            const_cast<std::vector<std::vector<int>>&>(intervals));
        
        int totalLength = 0;
        for (const auto& interval : merged) {
            totalLength += interval[1] - interval[0];
        }
        return totalLength;
    }
};

// Advanced sorting algorithms for comparison
class AdvancedSorting {
public:
    // Merge sort implementation for intervals
    static void mergeSort(std::vector<std::vector<int>>& intervals, int left, int right) {
        if (left >= right) return;
        
        int mid = left + (right - left) / 2;
        mergeSort(intervals, left, mid);
        mergeSort(intervals, mid + 1, right);
        mergeIntervals(intervals, left, mid, right);
    }
    
private:
    static void mergeIntervals(std::vector<std::vector<int>>& intervals, 
                              int left, int mid, int right) {
        std::vector<std::vector<int>> temp(right - left + 1);
        int i = left, j = mid + 1, k = 0;
        
        while (i <= mid && j <= right) {
            if (intervals[i][0] <= intervals[j][0]) {
                temp[k++] = intervals[i++];
            } else {
                temp[k++] = intervals[j++];
            }
        }
        
        while (i <= mid) temp[k++] = intervals[i++];
        while (j <= right) temp[k++] = intervals[j++];
        
        for (int i = 0; i < k; i++) {
            intervals[left + i] = temp[i];
        }
    }
};

// Helper functions for testing
void printIntervals(const std::vector<std::vector<int>>& intervals, const std::string& label) {
    std::cout << label << ": [";
    for (int i = 0; i < intervals.size(); i++) {
        std::cout << "[" << intervals[i][0] << "," << intervals[i][1] << "]";
        if (i < intervals.size() - 1) std::cout << ",";
    }
    std::cout << "]" << std::endl;
}

int main() {
    Solution solution;
    
    std::cout << "Merge Intervals Test Cases:" << std::endl;
    std::cout << "===========================" << std::endl;
    
    // Test case 1: Basic overlapping intervals
    std::vector<std::vector<int>> test1 = {{1,3},{2,6},{8,10},{15,18}};
    printIntervals(test1, "Test 1 Input");
    auto result1 = solution.merge(test1);
    printIntervals(result1, "Test 1 Output");
    std::cout << std::endl;
    
    // Test case 2: All intervals overlap
    std::vector<std::vector<int>> test2 = {{1,4},{4,5}};
    printIntervals(test2, "Test 2 Input");
    auto result2 = solution.merge(test2);
    printIntervals(result2, "Test 2 Output");
    std::cout << std::endl;
    
    // Test case 3: No overlapping intervals
    std::vector<std::vector<int>> test3 = {{1,2},{3,4},{5,6}};
    printIntervals(test3, "Test 3 Input");
    auto result3 = solution.merge(test3);
    printIntervals(result3, "Test 3 Output");
    std::cout << std::endl;
    
    // Test case 4: Completely overlapping intervals
    std::vector<std::vector<int>> test4 = {{1,10},{2,3},{4,5},{6,7},{8,9}};
    printIntervals(test4, "Test 4 Input");
    auto result4 = solution.merge(test4);
    printIntervals(result4, "Test 4 Output");
    std::cout << std::endl;
    
    // Test case 5: Unsorted input
    std::vector<std::vector<int>> test5 = {{15,18},{2,6},{8,10},{1,3}};
    printIntervals(test5, "Test 5 Input (Unsorted)");
    auto result5 = solution.merge(test5);
    printIntervals(result5, "Test 5 Output");
    std::cout << std::endl;
    
    // Test case 6: Edge case - single interval
    std::vector<std::vector<int>> test6 = {{1,4}};
    printIntervals(test6, "Test 6 Input");
    auto result6 = solution.merge(test6);
    printIntervals(result6, "Test 6 Output");
    std::cout << std::endl;
    
    // Test utility functions
    std::cout << "Utility Function Tests:" << std::endl;
    std::cout << "======================" << std::endl;
    
    std::vector<int> interval_a = {1, 5};
    std::vector<int> interval_b = {3, 7};
    
    std::cout << "Interval A: [" << interval_a[0] << "," << interval_a[1] << "]" << std::endl;
    std::cout << "Interval B: [" << interval_b[0] << "," << interval_b[1] << "]" << std::endl;
    std::cout << "Do they overlap? " << (IntervalUtils::doOverlap(interval_a, interval_b) ? "Yes" : "No") << std::endl;
    
    auto merged_ab = IntervalUtils::mergeTwo(interval_a, interval_b);
    std::cout << "Merged: [" << merged_ab[0] << "," << merged_ab[1] << "]" << std::endl;
    
    auto intersection = IntervalUtils::getIntersection(interval_a, interval_b);
    if (!intersection.empty()) {
        std::cout << "Intersection: [" << intersection[0] << "," << intersection[1] << "]" << std::endl;
    }
    
    int totalLength = IntervalUtils::getTotalCoveredLength(test1);
    std::cout << "Total covered length for test1: " << totalLength << std::endl;
    
    return 0;
}

/*
Algorithm Analysis:

1. Core Algorithm:
   - Sort intervals by start time: O(n log n)
   - Linear scan to merge: O(n)
   - Total: O(n log n)

2. Space Complexity:
   - O(n) for output array (worst case: no merging needed)
   - O(log n) for sorting (if in-place)
   - O(1) additional space for merging logic

Key Insights:

1. Sorting Strategy:
   - Sort by start time to process intervals in order
   - If start times are equal, can sort by end time for consistency

2. Merging Logic:
   - Two intervals [a,b] and [c,d] overlap if b >= c (after sorting by start)
   - Merged interval: [min(a,c), max(b,d)] = [a, max(b,d)] (since a <= c)

3. Optimization Opportunities:
   - In-place modification if input can be modified
   - Early termination if no more overlaps possible
   - Custom sorting for specific interval patterns

Edge Cases:
- Empty input array
- Single interval
- All intervals overlap into one
- No overlapping intervals
- Identical intervals
- Adjacent intervals (touching but not overlapping)

Common Variations:
1. Insert Interval: Insert new interval and merge
2. Non-overlapping Intervals: Count minimum removals
3. Meeting Rooms: Check if person can attend all meetings
4. Activity Selection: Select maximum non-overlapping activities

Sorting Algorithm Comparison:
- Quick Sort: O(n log n) average, O(n²) worst case
- Merge Sort: O(n log n) guaranteed, stable
- Heap Sort: O(n log n) guaranteed, not stable
- STL sort: Typically introsort (hybrid of quicksort, heapsort, insertion sort)

Applications:
- Calendar scheduling systems
- Resource allocation
- Time slot management
- Network bandwidth allocation
- Task scheduling in operating systems

Interview Tips:
1. Always sort first unless told otherwise
2. Handle edge cases (empty, single interval)
3. Consider if input can be modified
4. Think about follow-up questions (insert interval, etc.)
5. Analyze time/space complexity clearly
*/
