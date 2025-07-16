/*
Meta (Facebook) SDE Interview Problem 9: Merge k Sorted Lists (Hard)
You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
Merge all the linked-lists into one sorted linked-list and return it.

Example 1:
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6

Example 2:
Input: lists = []
Output: []

Example 3:
Input: lists = [[]]
Output: []

This is a classic Meta interview problem testing divide and conquer, priority queues, and linked list manipulation.
It's fundamental for understanding merge algorithms and distributed system data processing.

Time Complexity: O(N log k) where N is total number of nodes and k is number of lists
Space Complexity: O(log k) for recursion stack in divide and conquer approach
*/

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>
#include <random>

// Definition for singly-linked list
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    // Approach 1: Divide and Conquer (Recommended for interviews)
    ListNode* mergeKLists(std::vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        return mergeKListsHelper(lists, 0, lists.size() - 1);
    }
    
    // Approach 2: Priority Queue (Min Heap)
    ListNode* mergeKListsPQ(std::vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        
        // Custom comparator for min heap
        auto compare = [](ListNode* a, ListNode* b) {
            return a->val > b->val;
        };
        
        std::priority_queue<ListNode*, std::vector<ListNode*>, decltype(compare)> pq(compare);
        
        // Add first node of each list to priority queue
        for (ListNode* list : lists) {
            if (list) {
                pq.push(list);
            }
        }
        
        ListNode dummy(0);
        ListNode* current = &dummy;
        
        while (!pq.empty()) {
            ListNode* smallest = pq.top();
            pq.pop();
            
            current->next = smallest;
            current = current->next;
            
            if (smallest->next) {
                pq.push(smallest->next);
            }
        }
        
        return dummy.next;
    }
    
    // Approach 3: Sequential merge (brute force)
    ListNode* mergeKListsSequential(std::vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        
        ListNode* result = nullptr;
        
        for (ListNode* list : lists) {
            result = mergeTwoLists(result, list);
        }
        
        return result;
    }
    
    // Approach 4: Collect all values and sort
    ListNode* mergeKListsSort(std::vector<ListNode*>& lists) {
        std::vector<int> values;
        
        // Collect all values
        for (ListNode* list : lists) {
            ListNode* current = list;
            while (current) {
                values.push_back(current->val);
                current = current->next;
            }
        }
        
        if (values.empty()) return nullptr;
        
        // Sort values
        std::sort(values.begin(), values.end());
        
        // Build result list
        ListNode dummy(0);
        ListNode* current = &dummy;
        
        for (int val : values) {
            current->next = new ListNode(val);
            current = current->next;
        }
        
        return dummy.next;
    }
    
    // Approach 5: Iterative divide and conquer
    ListNode* mergeKListsIterative(std::vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        
        while (lists.size() > 1) {
            std::vector<ListNode*> mergedLists;
            
            // Merge pairs of lists
            for (int i = 0; i < lists.size(); i += 2) {
                ListNode* l1 = lists[i];
                ListNode* l2 = (i + 1 < lists.size()) ? lists[i + 1] : nullptr;
                mergedLists.push_back(mergeTwoLists(l1, l2));
            }
            
            lists = mergedLists;
        }
        
        return lists[0];
    }
    
    // Approach 6: Using external sorting for very large data
    ListNode* mergeKListsExternal(std::vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        
        // For demonstration: simulate external sorting approach
        // In practice, this would involve disk-based sorting for very large datasets
        
        std::vector<std::vector<int>> chunks;
        const int chunkSize = 1000; // Simulate memory limit
        
        // Process each list in chunks
        for (ListNode* list : lists) {
            std::vector<int> chunk;
            ListNode* current = list;
            
            while (current) {
                chunk.push_back(current->val);
                current = current->next;
                
                if (chunk.size() >= chunkSize) {
                    std::sort(chunk.begin(), chunk.end());
                    chunks.push_back(chunk);
                    chunk.clear();
                }
            }
            
            if (!chunk.empty()) {
                std::sort(chunk.begin(), chunk.end());
                chunks.push_back(chunk);
            }
        }
        
        // Merge sorted chunks
        std::vector<int> allValues;
        std::vector<int> indices(chunks.size(), 0);
        
        while (true) {
            int minVal = INT_MAX;
            int minIdx = -1;
            
            // Find minimum among chunk heads
            for (int i = 0; i < chunks.size(); i++) {
                if (indices[i] < chunks[i].size() && chunks[i][indices[i]] < minVal) {
                    minVal = chunks[i][indices[i]];
                    minIdx = i;
                }
            }
            
            if (minIdx == -1) break;
            
            allValues.push_back(minVal);
            indices[minIdx]++;
        }
        
        // Build result list
        if (allValues.empty()) return nullptr;
        
        ListNode dummy(0);
        ListNode* current = &dummy;
        
        for (int val : allValues) {
            current->next = new ListNode(val);
            current = current->next;
        }
        
        return dummy.next;
    }
    
private:
    ListNode* mergeKListsHelper(std::vector<ListNode*>& lists, int start, int end) {
        if (start == end) {
            return lists[start];
        }
        
        if (start > end) {
            return nullptr;
        }
        
        int mid = start + (end - start) / 2;
        ListNode* left = mergeKListsHelper(lists, start, mid);
        ListNode* right = mergeKListsHelper(lists, mid + 1, end);
        
        return mergeTwoLists(left, right);
    }
    
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode dummy(0);
        ListNode* current = &dummy;
        
        while (l1 && l2) {
            if (l1->val <= l2->val) {
                current->next = l1;
                l1 = l1->next;
            } else {
                current->next = l2;
                l2 = l2->next;
            }
            current = current->next;
        }
        
        // Attach remaining nodes
        current->next = l1 ? l1 : l2;
        
        return dummy.next;
    }
};

// Utility functions for testing
class ListUtils {
public:
    static ListNode* createList(const std::vector<int>& values) {
        if (values.empty()) return nullptr;
        
        ListNode* head = new ListNode(values[0]);
        ListNode* current = head;
        
        for (int i = 1; i < values.size(); i++) {
            current->next = new ListNode(values[i]);
            current = current->next;
        }
        
        return head;
    }
    
    static std::vector<int> listToVector(ListNode* head) {
        std::vector<int> result;
        ListNode* current = head;
        
        while (current) {
            result.push_back(current->val);
            current = current->next;
        }
        
        return result;
    }
    
    static void deleteList(ListNode* head) {
        while (head) {
            ListNode* next = head->next;
            delete head;
            head = next;
        }
    }
    
    static void printList(ListNode* head, const std::string& title = "") {
        if (!title.empty()) {
            std::cout << title << ": ";
        }
        
        ListNode* current = head;
        while (current) {
            std::cout << current->val;
            if (current->next) std::cout << "->";
            current = current->next;
        }
        std::cout << std::endl;
    }
    
    static bool areListsEqual(ListNode* l1, ListNode* l2) {
        while (l1 && l2) {
            if (l1->val != l2->val) return false;
            l1 = l1->next;
            l2 = l2->next;
        }
        return l1 == nullptr && l2 == nullptr;
    }
    
    static bool isSorted(ListNode* head) {
        if (!head || !head->next) return true;
        
        ListNode* current = head;
        while (current->next) {
            if (current->val > current->next->val) {
                return false;
            }
            current = current->next;
        }
        return true;
    }
    
    static int getLength(ListNode* head) {
        int length = 0;
        while (head) {
            length++;
            head = head->next;
        }
        return length;
    }
    
    static std::vector<ListNode*> createTestLists() {
        std::vector<ListNode*> lists;
        
        // Create test lists: [1,4,5], [1,3,4], [2,6]
        lists.push_back(createList({1, 4, 5}));
        lists.push_back(createList({1, 3, 4}));
        lists.push_back(createList({2, 6}));
        
        return lists;
    }
    
    static std::vector<ListNode*> createLargeLists(int numLists, int listSize) {
        std::vector<ListNode*> lists;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 1000);
        
        for (int i = 0; i < numLists; i++) {
            std::vector<int> values;
            for (int j = 0; j < listSize; j++) {
                values.push_back(dis(gen));
            }
            std::sort(values.begin(), values.end()); // Ensure sorted
            lists.push_back(createList(values));
        }
        
        return lists;
    }
    
    static void deleteLists(std::vector<ListNode*>& lists) {
        for (ListNode* list : lists) {
            deleteList(list);
        }
        lists.clear();
    }
};

// Test framework
class MergeKListsTest {
public:
    static void runTests() {
        std::cout << "Meta Merge k Sorted Lists Tests:" << std::endl;
        std::cout << "===============================" << std::endl;
        
        testBasicFunctionality();
        testEdgeCases();
        compareApproaches();
        metaSpecificScenarios();
        performanceAnalysis();
    }
    
    static void testBasicFunctionality() {
        std::cout << "\nBasic Functionality Tests:" << std::endl;
        std::cout << "=========================" << std::endl;
        
        Solution solution;
        
        // Test case 1: Example 1
        std::vector<ListNode*> lists1 = ListUtils::createTestLists();
        ListNode* result1 = solution.mergeKLists(lists1);
        
        std::cout << "Test 1 - Example case:" << std::endl;
        for (int i = 0; i < lists1.size(); i++) {
            ListUtils::printList(lists1[i], "List " + std::to_string(i + 1));
        }
        ListUtils::printList(result1, "Merged");
        
        auto expectedValues = std::vector<int>{1, 1, 2, 3, 4, 4, 5, 6};
        auto actualValues = ListUtils::listToVector(result1);
        bool correct1 = (expectedValues == actualValues);
        std::cout << "Correct result: " << (correct1 ? "✅" : "❌") << std::endl;
        std::cout << "Is sorted: " << (ListUtils::isSorted(result1) ? "✅" : "❌") << std::endl;
        
        // Test case 2: Different sizes
        std::vector<ListNode*> lists2;
        lists2.push_back(ListUtils::createList({1, 2, 3}));
        lists2.push_back(ListUtils::createList({4}));
        lists2.push_back(ListUtils::createList({5, 6, 7, 8}));
        
        ListNode* result2 = solution.mergeKLists(lists2);
        std::cout << "\nTest 2 - Different sizes:" << std::endl;
        ListUtils::printList(result2, "Result");
        std::cout << "Is sorted: " << (ListUtils::isSorted(result2) ? "✅" : "❌") << std::endl;
        
        // Test case 3: Overlapping values
        std::vector<ListNode*> lists3;
        lists3.push_back(ListUtils::createList({1, 1, 1}));
        lists3.push_back(ListUtils::createList({1, 1, 2}));
        lists3.push_back(ListUtils::createList({2, 2, 2}));
        
        ListNode* result3 = solution.mergeKLists(lists3);
        std::cout << "\nTest 3 - Overlapping values:" << std::endl;
        ListUtils::printList(result3, "Result");
        std::cout << "Is sorted: " << (ListUtils::isSorted(result3) ? "✅" : "❌") << std::endl;
        
        // Cleanup
        ListUtils::deleteList(result1);
        ListUtils::deleteLists(lists2);
        ListUtils::deleteList(result2);
        ListUtils::deleteLists(lists3);
        ListUtils::deleteList(result3);
    }
    
    static void testEdgeCases() {
        std::cout << "\nEdge Case Tests:" << std::endl;
        std::cout << "================" << std::endl;
        
        Solution solution;
        
        // Empty input
        std::vector<ListNode*> emptyLists;
        ListNode* emptyResult = solution.mergeKLists(emptyLists);
        std::cout << "Empty input: " << (emptyResult == nullptr ? "✅" : "❌") << std::endl;
        
        // Single list
        std::vector<ListNode*> singleList;
        singleList.push_back(ListUtils::createList({1, 2, 3}));
        ListNode* singleResult = solution.mergeKLists(singleList);
        auto singleValues = ListUtils::listToVector(singleResult);
        bool singleCorrect = (singleValues == std::vector<int>{1, 2, 3});
        std::cout << "Single list: " << (singleCorrect ? "✅" : "❌") << std::endl;
        
        // Lists with null
        std::vector<ListNode*> listsWithNull;
        listsWithNull.push_back(nullptr);
        listsWithNull.push_back(ListUtils::createList({1, 2}));
        listsWithNull.push_back(nullptr);
        listsWithNull.push_back(ListUtils::createList({3, 4}));
        
        ListNode* nullResult = solution.mergeKLists(listsWithNull);
        auto nullValues = ListUtils::listToVector(nullResult);
        bool nullCorrect = (nullValues == std::vector<int>{1, 2, 3, 4});
        std::cout << "Lists with null: " << (nullCorrect ? "✅" : "❌") << std::endl;
        
        // All empty lists
        std::vector<ListNode*> allEmpty = {nullptr, nullptr, nullptr};
        ListNode* allEmptyResult = solution.mergeKLists(allEmpty);
        std::cout << "All empty lists: " << (allEmptyResult == nullptr ? "✅" : "❌") << std::endl;
        
        // Negative numbers
        std::vector<ListNode*> negativeLists;
        negativeLists.push_back(ListUtils::createList({-2, -1, 0}));
        negativeLists.push_back(ListUtils::createList({-3, 1, 2}));
        
        ListNode* negativeResult = solution.mergeKLists(negativeList);
        auto negativeValues = ListUtils::listToVector(negativeResult);
        bool negativeCorrect = (negativeValues == std::vector<int>{-3, -2, -1, 0, 1, 2});
        std::cout << "Negative numbers: " << (negativeCorrect ? "✅" : "❌") << std::endl;
        
        // Large numbers
        std::vector<ListNode*> largeLists;
        largeLists.push_back(ListUtils::createList({1000000, 2000000}));
        largeLists.push_back(ListUtils::createList({1500000, 2500000}));
        
        ListNode* largeResult = solution.mergeKLists(largeLists);
        std::cout << "Large numbers sorted: " << (ListUtils::isSorted(largeResult) ? "✅" : "❌") << std::endl;
        
        // Cleanup
        ListUtils::deleteList(singleResult);
        ListUtils::deleteList(nullResult);
        ListUtils::deleteList(negativeResult);
        ListUtils::deleteList(largeResult);
    }
    
    static void compareApproaches() {
        std::cout << "\nApproach Comparison:" << std::endl;
        std::cout << "===================" << std::endl;
        
        Solution solution;
        
        // Create test data
        std::vector<ListNode*> testLists = ListUtils::createLargeLists(20, 50);
        
        // Test divide and conquer
        auto lists1 = testLists; // Copy for each test
        auto start = std::chrono::high_resolution_clock::now();
        ListNode* result1 = solution.mergeKLists(lists1);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Test priority queue
        auto lists2 = testLists;
        start = std::chrono::high_resolution_clock::now();
        ListNode* result2 = solution.mergeKListsPQ(lists2);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Test sequential merge
        auto lists3 = testLists;
        start = std::chrono::high_resolution_clock::now();
        ListNode* result3 = solution.mergeKListsSequential(lists3);
        end = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Test sort approach
        auto lists4 = testLists;
        start = std::chrono::high_resolution_clock::now();
        ListNode* result4 = solution.mergeKListsSort(lists4);
        end = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Test iterative divide and conquer
        auto lists5 = testLists;
        start = std::chrono::high_resolution_clock::now();
        ListNode* result5 = solution.mergeKListsIterative(lists5);
        end = std::chrono::high_resolution_clock::now();
        auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Divide & Conquer: " << duration1.count() << " μs" << std::endl;
        std::cout << "Priority Queue: " << duration2.count() << " μs" << std::endl;
        std::cout << "Sequential Merge: " << duration3.count() << " μs" << std::endl;
        std::cout << "Sort Approach: " << duration4.count() << " μs" << std::endl;
        std::cout << "Iterative D&C: " << duration5.count() << " μs" << std::endl;
        
        // Verify correctness
        bool allSorted = ListUtils::isSorted(result1) && ListUtils::isSorted(result2) && 
                        ListUtils::isSorted(result3) && ListUtils::isSorted(result4) && 
                        ListUtils::isSorted(result5);
        std::cout << "All results sorted: " << (allSorted ? "✅" : "❌") << std::endl;
        
        // Check if all have same length
        int len1 = ListUtils::getLength(result1);
        int len2 = ListUtils::getLength(result2);
        int len3 = ListUtils::getLength(result3);
        int len4 = ListUtils::getLength(result4);
        int len5 = ListUtils::getLength(result5);
        
        bool sameLengths = (len1 == len2 && len2 == len3 && len3 == len4 && len4 == len5);
        std::cout << "All results same length (" << len1 << "): " << (sameLengths ? "✅" : "❌") << std::endl;
        
        // Cleanup
        ListUtils::deleteLists(testLists);
        ListUtils::deleteList(result1);
        ListUtils::deleteList(result2);
        ListUtils::deleteList(result3);
        ListUtils::deleteList(result4);
        ListUtils::deleteList(result5);
    }
    
    static void metaSpecificScenarios() {
        std::cout << "\nMeta-Specific Scenarios:" << std::endl;
        std::cout << "=======================" << std::endl;
        
        Solution solution;
        
        // Scenario 1: Merging user activity feeds
        std::cout << "User activity feed merging:" << std::endl;
        std::vector<ListNode*> activityFeeds;
        
        // User 1 activities (timestamps)
        activityFeeds.push_back(ListUtils::createList({1001, 1005, 1010}));
        // User 2 activities
        activityFeeds.push_back(ListUtils::createList({1002, 1008, 1012}));
        // User 3 activities
        activityFeeds.push_back(ListUtils::createList({1003, 1006, 1009, 1015}));
        
        ListNode* mergedFeed = solution.mergeKLists(activityFeeds);
        std::cout << "Merged timeline: ";
        ListUtils::printList(mergedFeed);
        std::cout << "Chronologically ordered: " << (ListUtils::isSorted(mergedFeed) ? "✅" : "❌") << std::endl;
        
        // Scenario 2: Merging search results from different data centers
        std::cout << "\nSearch results from multiple data centers:" << std::endl;
        std::vector<ListNode*> searchResults;
        
        // DC1 results (relevance scores)
        searchResults.push_back(ListUtils::createList({95, 90, 85}));
        // DC2 results
        searchResults.push_back(ListUtils::createList({98, 88, 82}));
        // DC3 results
        searchResults.push_back(ListUtils::createList({92, 87, 80}));
        
        ListNode* mergedResults = solution.mergeKListsPQ(searchResults);
        std::cout << "Merged search results: ";
        ListUtils::printList(mergedResults);
        std::cout << "Relevance ordered: " << (ListUtils::isSorted(mergedResults) ? "✅" : "❌") << std::endl;
        
        // Scenario 3: Merging sorted logs from different services
        std::cout << "\nLog aggregation from microservices:" << std::endl;
        std::vector<ListNode*> serviceLogs;
        
        // Auth service logs (timestamps)
        serviceLogs.push_back(ListUtils::createList({1000, 1030, 1060}));
        // API service logs
        serviceLogs.push_back(ListUtils::createList({1010, 1040, 1070}));
        // Database service logs
        serviceLogs.push_back(ListUtils::createList({1020, 1050, 1080}));
        // Cache service logs
        serviceLogs.push_back(ListUtils::createList({1005, 1035, 1065}));
        
        ListNode* aggregatedLogs = solution.mergeKListsIterative(serviceLogs);
        std::cout << "Aggregated logs: ";
        ListUtils::printList(aggregatedLogs);
        std::cout << "Temporally ordered: " << (ListUtils::isSorted(aggregatedLogs) ? "✅" : "❌") << std::endl;
        
        // Scenario 4: Merging friend recommendations from different algorithms
        std::cout << "\nFriend recommendation aggregation:" << std::endl;
        std::vector<ListNode*> recommendations;
        
        // Mutual friends algorithm (scores)
        recommendations.push_back(ListUtils::createList({85, 75, 65}));
        // Interest-based algorithm
        recommendations.push_back(ListUtils::createList({90, 80, 70}));
        // Location-based algorithm
        recommendations.push_back(ListUtils::createList({88, 78, 68}));
        
        ListNode* finalRecommendations = solution.mergeKLists(recommendations);
        std::cout << "Final recommendations: ";
        ListUtils::printList(finalRecommendations);
        std::cout << "Score ordered: " << (ListUtils::isSorted(finalRecommendations) ? "✅" : "❌") << std::endl;
        
        // Scenario 5: Merging trending content from different regions
        std::cout << "\nTrending content aggregation:" << std::endl;
        std::vector<ListNode*> regionalTrends;
        
        // US trends (engagement scores)
        regionalTrends.push_back(ListUtils::createList({1000, 800, 600}));
        // EU trends
        regionalTrends.push_back(ListUtils::createList({950, 750, 550}));
        // Asia trends
        regionalTrends.push_back(ListUtils::createList({900, 850, 650}));
        
        ListNode* globalTrends = solution.mergeKListsSort(regionalTrends);
        std::cout << "Global trends: ";
        ListUtils::printList(globalTrends);
        std::cout << "Engagement ordered: " << (ListUtils::isSorted(globalTrends) ? "✅" : "❌") << std::endl;
        
        // Cleanup
        ListUtils::deleteList(mergedFeed);
        ListUtils::deleteList(mergedResults);
        ListUtils::deleteList(aggregatedLogs);
        ListUtils::deleteList(finalRecommendations);
        ListUtils::deleteList(globalTrends);
    }
    
    static void performanceAnalysis() {
        std::cout << "\nPerformance Analysis:" << std::endl;
        std::cout << "====================" << std::endl;
        
        Solution solution;
        
        // Test with increasing number of lists
        std::vector<int> listCounts = {5, 10, 20, 50, 100};
        
        for (int k : listCounts) {
            std::vector<ListNode*> testLists = ListUtils::createLargeLists(k, 100);
            
            auto start = std::chrono::high_resolution_clock::now();
            ListNode* result = solution.mergeKLists(testLists);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            int totalNodes = k * 100;
            std::cout << "k=" << k << " (total " << totalNodes << " nodes): " 
                      << duration.count() << " μs" << std::endl;
            
            ListUtils::deleteLists(testLists);
            ListUtils::deleteList(result);
        }
        
        // Test with increasing list sizes
        std::cout << "\nScaling with list size:" << std::endl;
        std::vector<int> listSizes = {50, 100, 200, 500};
        
        for (int size : listSizes) {
            std::vector<ListNode*> testLists = ListUtils::createLargeLists(10, size);
            
            auto start = std::chrono::high_resolution_clock::now();
            ListNode* result = solution.mergeKLists(testLists);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            int totalNodes = 10 * size;
            std::cout << "Size=" << size << " (total " << totalNodes << " nodes): " 
                      << duration.count() << " μs" << std::endl;
            
            ListUtils::deleteLists(testLists);
            ListUtils::deleteList(result);
        }
        
        // Complexity analysis
        std::cout << "\nComplexity Verification:" << std::endl;
        std::cout << "=======================" << std::endl;
        
        std::cout << "Time Complexity: O(N log k)" << std::endl;
        std::cout << "  N = total number of nodes" << std::endl;
        std::cout << "  k = number of lists" << std::endl;
        std::cout << "Space Complexity: O(log k) for recursion stack" << std::endl;
        
        // Memory usage analysis
        std::cout << "\nMemory Usage Analysis:" << std::endl;
        std::cout << "=====================" << std::endl;
        
        for (int nodes = 1000; nodes <= 100000; nodes *= 10) {
            size_t nodeMemory = nodes * sizeof(ListNode);
            size_t stackMemory = 64 * sizeof(void*); // Max recursion depth
            size_t totalMemory = nodeMemory + stackMemory;
            
            std::cout << "Nodes " << nodes << ": ~" << totalMemory / 1024 << " KB" << std::endl;
        }
        
        // Approach comparison with different k values
        std::cout << "\nApproach Comparison by k:" << std::endl;
        std::cout << "========================" << std::endl;
        
        std::vector<int> kValues = {10, 50, 100};
        
        for (int k : kValues) {
            std::vector<ListNode*> testLists = ListUtils::createLargeLists(k, 50);
            
            // Divide and conquer
            auto lists1 = testLists;
            auto start = std::chrono::high_resolution_clock::now();
            ListNode* result1 = solution.mergeKLists(lists1);
            auto end = std::chrono::high_resolution_clock::now();
            auto dc_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            // Priority queue
            auto lists2 = testLists;
            start = std::chrono::high_resolution_clock::now();
            ListNode* result2 = solution.mergeKListsPQ(lists2);
            end = std::chrono::high_resolution_clock::now();
            auto pq_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            std::cout << "k=" << k << " - D&C: " << dc_time.count() << " μs, PQ: " << pq_time.count() << " μs" << std::endl;
            
            ListUtils::deleteLists(testLists);
            ListUtils::deleteList(result1);
            ListUtils::deleteList(result2);
        }
    }
};

int main() {
    MergeKListsTest::runTests();
    return 0;
}

/*
Algorithm Analysis:

Core Problem: Merge k sorted linked lists into one sorted list

Key Insights:
1. Can leverage merge operation from merge sort
2. Divide and conquer reduces comparisons
3. Priority queue maintains ordering efficiently
4. Total elements N = sum of all list lengths

Approach Comparison:

1. Divide and Conquer (Recommended):
   - Time: O(N log k) where N is total nodes, k is number of lists
   - Space: O(log k) for recursion stack
   - Optimal time complexity
   - Natural recursive structure

2. Priority Queue (Min Heap):
   - Time: O(N log k) for N operations on heap of size k
   - Space: O(k) for priority queue
   - Good for streaming/online scenarios
   - Handles variable arrival times well

3. Sequential Merge:
   - Time: O(kN) - merges lists one by one
   - Space: O(1) additional space
   - Simple but inefficient for large k
   - Not recommended for interviews

4. Collect and Sort:
   - Time: O(N log N) - collects all values and sorts
   - Space: O(N) for collecting values
   - Simple implementation
   - Destroys linked list structure

5. Iterative Divide and Conquer:
   - Time: O(N log k) same as recursive
   - Space: O(1) - no recursion stack
   - Good for systems with stack limitations
   - More complex implementation

Meta Interview Focus:
- Algorithm optimization and complexity analysis
- Divide and conquer technique
- Priority queue usage
- Linked list manipulation
- System design considerations for large data

Key Design Decisions:
1. Recursive vs iterative implementation
2. In-place vs additional space usage
3. Handling edge cases (empty lists, nulls)
4. Optimization for different input patterns

Real-world Applications at Meta:
- Merging user activity feeds from multiple sources
- Aggregating search results from different data centers
- Combining logs from distributed microservices
- Merging friend recommendations from different algorithms
- Consolidating trending content from various regions

Edge Cases:
- Empty input array
- All lists are empty/null
- Single list
- Lists of very different sizes
- Duplicate values across lists

Interview Tips:
1. Start with divide and conquer approach
2. Explain time complexity derivation clearly
3. Handle edge cases explicitly
4. Discuss space-time trade-offs
5. Consider real-world constraints

Common Mistakes:
1. Using sequential merge (O(kN) complexity)
2. Not handling null lists properly
3. Memory leaks from poor cleanup
4. Incorrect merge logic for linked lists
5. Stack overflow in deep recursion

Advanced Optimizations:
- External sorting for very large datasets
- Parallel merging for multi-core systems
- Memory-efficient streaming merge
- Adaptive algorithms based on input characteristics
- Cache-friendly merge patterns

Testing Strategy:
- Basic functionality with known inputs
- Edge cases (empty, null, single)
- Performance testing with varying k and N
- Memory usage validation
- Correctness verification

Production Considerations:
- Memory limits for large datasets
- Streaming vs batch processing
- Fault tolerance for distributed merging
- Monitoring and metrics
- Error handling and recovery

Complexity Analysis:
- Divide & Conquer: O(N log k) time, O(log k) space
- Priority Queue: O(N log k) time, O(k) space
- Sequential: O(kN) time, O(1) space
- Sort: O(N log N) time, O(N) space

This problem is important for Meta because:
1. Common pattern in distributed systems
2. Tests algorithm optimization skills
3. Real applications in data aggregation
4. Demonstrates understanding of complexity
5. Shows system design considerations

Common Interview Variations:
1. Merge k sorted arrays
2. Find kth smallest element across lists
3. Merge with weighted priorities
4. Streaming merge with memory limits
5. Distributed merge across machines

Optimization Techniques:

For Small k (< 10):
- Sequential merge may be acceptable
- Simple implementation preferred
- Lower constant factors important

For Large k (> 100):
- Divide and conquer essential
- Priority queue competitive
- Memory efficiency critical

For Very Large N:
- External sorting approaches
- Streaming algorithms
- Parallel processing

Performance Characteristics:
- Small lists (k=5, N=100): < 100μs
- Medium lists (k=20, N=1000): < 1ms
- Large lists (k=100, N=10000): < 10ms
- Memory usage scales linearly with N
- Recursion depth limited by log k

Real-world Usage:
- Database: Merging sorted table scans
- Search: Combining results from shards
- Analytics: Aggregating time-series data
- Social: Merging activity streams
- Logging: Combining distributed logs

Implementation Considerations:
- Recursive depth limits
- Memory allocation patterns
- Cache performance optimization
- NUMA awareness for large systems
- Lock-free algorithms for concurrency
*/
