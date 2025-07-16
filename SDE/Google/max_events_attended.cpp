/*
Google SDE Interview Problem 2: Maximum Number of Events That Can Be Attended (Hard)
You are given an array of events where events[i] = [startDayi, endDayi]. 
Every event i starts at startDayi and ends at endDayi.

You can attend an event i at any day d where startTimei <= d <= endTimei. 
You can only attend one event per day.

Return the maximum number of events you can attend.

Example:
Input: events = [[1,2],[2,3],[3,4]]
Output: 3

Input: events = [[1,4],[4,4],[2,2],[3,4],[1,1]]
Output: 4

Time Complexity: O(n log n + d log n) where n = events, d = max day
Space Complexity: O(n) for priority queue and data structures
*/

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <set>

class Solution {
public:
    // Approach 1: Greedy with Priority Queue (Most Optimal)
    int maxEvents(std::vector<std::vector<int>>& events) {
        // Sort events by start day
        std::sort(events.begin(), events.end());
        
        // Priority queue to store end days (min heap)
        std::priority_queue<int, std::vector<int>, std::greater<int>> pq;
        
        int eventIndex = 0;
        int maxDay = 0;
        int eventsAttended = 0;
        
        // Find maximum day to consider
        for (const auto& event : events) {
            maxDay = std::max(maxDay, event[1]);
        }
        
        // Process each day
        for (int day = 1; day <= maxDay; day++) {
            // Add all events that start on this day
            while (eventIndex < events.size() && events[eventIndex][0] == day) {
                pq.push(events[eventIndex][1]);
                eventIndex++;
            }
            
            // Remove events that have already ended
            while (!pq.empty() && pq.top() < day) {
                pq.pop();
            }
            
            // Attend the event that ends earliest (if any available)
            if (!pq.empty()) {
                pq.pop();
                eventsAttended++;
            }
        }
        
        return eventsAttended;
    }
    
    // Approach 2: Using Set for Available Days
    int maxEventsWithSet(std::vector<std::vector<int>>& events) {
        // Sort events by end day (attend events ending earlier first)
        std::sort(events.begin(), events.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
            return a[1] < b[1];
        });
        
        // Set to maintain available days
        std::set<int> availableDays;
        int maxDay = 0;
        
        // Initialize available days
        for (const auto& event : events) {
            maxDay = std::max(maxDay, event[1]);
        }
        
        for (int day = 1; day <= maxDay; day++) {
            availableDays.insert(day);
        }
        
        int eventsAttended = 0;
        
        for (const auto& event : events) {
            int start = event[0];
            int end = event[1];
            
            // Find the earliest available day in the event's range
            auto it = availableDays.lower_bound(start);
            
            if (it != availableDays.end() && *it <= end) {
                eventsAttended++;
                availableDays.erase(it);
            }
        }
        
        return eventsAttended;
    }
    
    // Approach 3: Sweep Line Algorithm
    int maxEventsSweepLine(std::vector<std::vector<int>>& events) {
        std::vector<std::pair<int, int>> timeline; // {day, type} where type: 1=start, -1=end
        
        for (const auto& event : events) {
            timeline.push_back({event[0], 1});     // Start of event
            timeline.push_back({event[1] + 1, -1}); // End of event (exclusive)
        }
        
        std::sort(timeline.begin(), timeline.end());
        
        int activeEvents = 0;
        int eventsAttended = 0;
        int lastDay = 0;
        
        for (const auto& point : timeline) {
            int day = point.first;
            int type = point.second;
            
            // Process all days between lastDay and current day
            if (activeEvents > 0 && day > lastDay + 1) {
                int availableDays = day - lastDay - 1;
                int canAttend = std::min(activeEvents, availableDays);
                eventsAttended += canAttend;
            }
            
            if (type == 1) {
                activeEvents++;
            } else {
                activeEvents--;
            }
            
            lastDay = day;
        }
        
        return eventsAttended;
    }
    
    // Approach 4: Advanced Greedy with Early Termination
    int maxEventsOptimized(std::vector<std::vector<int>>& events) {
        // Sort by start day, then by end day
        std::sort(events.begin(), events.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
            if (a[0] == b[0]) return a[1] < b[1];
            return a[0] < b[0];
        });
        
        std::priority_queue<int, std::vector<int>, std::greater<int>> pq;
        int eventIndex = 0;
        int eventsAttended = 0;
        
        // Process events day by day
        for (int day = 1; day <= 100000; day++) { // Given constraint: days <= 10^5
            // Add events starting today
            while (eventIndex < events.size() && events[eventIndex][0] == day) {
                pq.push(events[eventIndex][1]);
                eventIndex++;
            }
            
            // Remove expired events
            while (!pq.empty() && pq.top() < day) {
                pq.pop();
            }
            
            // Attend one event if available
            if (!pq.empty()) {
                pq.pop();
                eventsAttended++;
            }
            
            // Early termination: no more events to process
            if (eventIndex >= events.size() && pq.empty()) {
                break;
            }
        }
        
        return eventsAttended;
    }
};

// Test helper class
class EventSchedulerTest {
public:
    static void runTests() {
        Solution solution;
        
        std::cout << "Google Events Scheduling Tests:" << std::endl;
        std::cout << "===============================" << std::endl;
        
        // Test case 1
        std::vector<std::vector<int>> events1 = {{1,2},{2,3},{3,4}};
        std::cout << "Test 1: " << solution.maxEvents(events1) << " (Expected: 3)" << std::endl;
        
        // Test case 2
        std::vector<std::vector<int>> events2 = {{1,4},{4,4},{2,2},{3,4},{1,1}};
        std::cout << "Test 2: " << solution.maxEvents(events2) << " (Expected: 4)" << std::endl;
        
        // Test case 3 - Overlapping events
        std::vector<std::vector<int>> events3 = {{1,5},{1,5},{1,5},{2,3},{2,3}};
        std::cout << "Test 3: " << solution.maxEvents(events3) << " (Expected: 5)" << std::endl;
        
        // Test case 4 - Single day events
        std::vector<std::vector<int>> events4 = {{1,1},{2,2},{3,3},{4,4}};
        std::cout << "Test 4: " << solution.maxEvents(events4) << " (Expected: 4)" << std::endl;
        
        // Test case 5 - Large span event
        std::vector<std::vector<int>> events5 = {{1,100000}};
        std::cout << "Test 5: " << solution.maxEvents(events5) << " (Expected: 1)" << std::endl;
        
        // Performance comparison
        std::cout << "\nPerformance Comparison:" << std::endl;
        std::vector<std::vector<int>> largeEvents;
        for (int i = 1; i <= 1000; i++) {
            largeEvents.push_back({i, i + 100});
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        int result1 = solution.maxEvents(largeEvents);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        int result2 = solution.maxEventsWithSet(largeEvents);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Priority Queue approach: " << result1 << " events, " << duration1.count() << " μs" << std::endl;
        std::cout << "Set approach: " << result2 << " events, " << duration2.count() << " μs" << std::endl;
    }
};

int main() {
    EventSchedulerTest::runTests();
    return 0;
}

/*
Algorithm Analysis:

Approach 1: Greedy with Priority Queue
- Sort events by start day
- For each day, add starting events to min-heap (by end day)
- Remove expired events, attend earliest ending event
- Time: O(n log n + d log n), Space: O(n)

Approach 2: Greedy with Set
- Sort events by end day
- Maintain set of available days
- For each event, find earliest available day in range
- Time: O(n log n + n log d), Space: O(d)

Approach 3: Sweep Line
- Create timeline of start/end events
- Track active events and attend optimally
- Time: O(n log n), Space: O(n)

Key Insights:
1. Greedy works: attend events ending earlier first
2. Process events chronologically for optimal scheduling
3. Use appropriate data structures for efficient lookups
4. Handle overlapping events carefully

Google Interview Focus:
- Multiple solution approaches
- Time/space complexity analysis
- Edge case handling
- Code optimization and clean structure
- Scalability considerations

Edge Cases:
- Single event
- All events on same day
- Non-overlapping events
- Events with large date ranges
- Maximum constraints (10^5 days)

Optimization Techniques:
- Early termination when no events remain
- Efficient data structure selection
- Memory-conscious implementation
- Cache-friendly access patterns
*/
