/*
LeetCode 23: Merge k Sorted Lists
================================

Problem: You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
Merge all the linked-lists into one sorted linked-list and return it.

Example:
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]

This problem is frequently asked at Google and tests:
1. Divide and conquer algorithms
2. Priority queue/heap usage
3. Linked list manipulation
4. Time/space complexity optimization

Time Complexity: O(N log k) where N is total number of nodes
Space Complexity: O(log k) for recursion stack or O(k) for priority queue
*/

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

// Definition for singly-linked list
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class MergeKSortedLists {
public:
    // Solution 1: Divide and Conquer (Most Efficient)
    // Time: O(N log k), Space: O(log k)
    ListNode* mergeKLists(std::vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        return divideAndConquer(lists, 0, lists.size() - 1);
    }
    
private:
    ListNode* divideAndConquer(std::vector<ListNode*>& lists, int start, int end) {
        if (start == end) return lists[start];
        if (start > end) return nullptr;
        
        int mid = start + (end - start) / 2;
        ListNode* left = divideAndConquer(lists, start, mid);
        ListNode* right = divideAndConquer(lists, mid + 1, end);
        
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
        
        current->next = l1 ? l1 : l2;
        return dummy.next;
    }
    
public:
    // Solution 2: Priority Queue (Min-Heap)
    // Time: O(N log k), Space: O(k)
    ListNode* mergeKListsWithHeap(std::vector<ListNode*>& lists) {
        // Custom comparator for priority queue
        auto compare = [](ListNode* a, ListNode* b) {
            return a->val > b->val; // Min heap
        };
        
        std::priority_queue<ListNode*, std::vector<ListNode*>, decltype(compare)> pq(compare);
        
        // Add first node of each list to heap
        for (ListNode* list : lists) {
            if (list) pq.push(list);
        }
        
        ListNode dummy(0);
        ListNode* current = &dummy;
        
        while (!pq.empty()) {
            ListNode* node = pq.top();
            pq.pop();
            
            current->next = node;
            current = current->next;
            
            if (node->next) {
                pq.push(node->next);
            }
        }
        
        return dummy.next;
    }
    
    // Solution 3: Sequential Merging (Less Efficient)
    // Time: O(k*N), Space: O(1)
    ListNode* mergeKListsSequential(std::vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        
        ListNode* result = lists[0];
        for (int i = 1; i < lists.size(); i++) {
            result = mergeTwoLists(result, lists[i]);
        }
        return result;
    }
    
    // Solution 4: Iterative Divide and Conquer
    // Time: O(N log k), Space: O(1)
    ListNode* mergeKListsIterative(std::vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        
        while (lists.size() > 1) {
            std::vector<ListNode*> merged;
            
            for (int i = 0; i < lists.size(); i += 2) {
                ListNode* l1 = lists[i];
                ListNode* l2 = (i + 1 < lists.size()) ? lists[i + 1] : nullptr;
                merged.push_back(mergeTwoLists(l1, l2));
            }
            
            lists = merged;
        }
        
        return lists[0];
    }
};

// Utility functions for testing
class ListUtils {
public:
    // Create linked list from vector
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
    
    // Convert linked list to vector for easy printing
    static std::vector<int> listToVector(ListNode* head) {
        std::vector<int> result;
        while (head) {
            result.push_back(head->val);
            head = head->next;
        }
        return result;
    }
    
    // Print linked list
    static void printList(ListNode* head) {
        std::vector<int> values = listToVector(head);
        std::cout << "[";
        for (int i = 0; i < values.size(); i++) {
            std::cout << values[i];
            if (i < values.size() - 1) std::cout << ",";
        }
        std::cout << "]";
    }
    
    // Clean up memory
    static void deleteList(ListNode* head) {
        while (head) {
            ListNode* temp = head;
            head = head->next;
            delete temp;
        }
    }
};

// Performance testing
class PerformanceTester {
public:
    static void compareAlgorithms() {
        std::cout << "\nPerformance Comparison:\n";
        std::cout << "======================\n";
        
        std::vector<int> k_values = {2, 4, 8, 16, 32};
        std::vector<int> list_sizes = {100, 500, 1000};
        
        for (int k : k_values) {
            for (int size : list_sizes) {
                std::cout << "k=" << k << ", list_size=" << size << ":\n";
                
                // Create test data
                std::vector<ListNode*> lists;
                for (int i = 0; i < k; i++) {
                    std::vector<int> values;
                    for (int j = 0; j < size; j++) {
                        values.push_back(i * size + j * k); // Ensure sorted
                    }
                    lists.push_back(ListUtils::createList(values));
                }
                
                MergeKSortedLists solver;
                
                // Test different approaches (in production, you'd measure actual time)
                auto result1 = solver.mergeKLists(lists);
                std::cout << "  Divide & Conquer: O(N log k)\n";
                
                // Create fresh test data for heap approach
                std::vector<ListNode*> lists2;
                for (int i = 0; i < k; i++) {
                    std::vector<int> values;
                    for (int j = 0; j < size; j++) {
                        values.push_back(i * size + j * k);
                    }
                    lists2.push_back(ListUtils::createList(values));
                }
                
                auto result2 = solver.mergeKListsWithHeap(lists2);
                std::cout << "  Priority Queue: O(N log k)\n";
                
                // Cleanup
                ListUtils::deleteList(result1);
                ListUtils::deleteList(result2);
                
                std::cout << "\n";
            }
        }
    }
    
    static void memoryUsageAnalysis() {
        std::cout << "Memory Usage Analysis:\n";
        std::cout << "=====================\n";
        std::cout << "Divide & Conquer: O(log k) - recursion stack\n";
        std::cout << "Priority Queue: O(k) - heap size\n";
        std::cout << "Sequential: O(1) - constant extra space\n";
        std::cout << "Iterative D&C: O(1) - constant extra space\n\n";
    }
};

// Advanced variations and follow-ups
class AdvancedVariations {
public:
    // Merge k sorted arrays (not linked lists)
    static std::vector<int> mergeKSortedArrays(std::vector<std::vector<int>>& arrays) {
        auto compare = [&arrays](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return arrays[a.first][a.second] > arrays[b.first][b.second];
        };
        
        std::priority_queue<std::pair<int, int>, 
                           std::vector<std::pair<int, int>>, 
                           decltype(compare)> pq(compare);
        
        // Initialize heap with first element of each array
        for (int i = 0; i < arrays.size(); i++) {
            if (!arrays[i].empty()) {
                pq.push({i, 0}); // {array_index, element_index}
            }
        }
        
        std::vector<int> result;
        
        while (!pq.empty()) {
            auto [array_idx, elem_idx] = pq.top();
            pq.pop();
            
            result.push_back(arrays[array_idx][elem_idx]);
            
            if (elem_idx + 1 < arrays[array_idx].size()) {
                pq.push({array_idx, elem_idx + 1});
            }
        }
        
        return result;
    }
    
    // Merge k sorted lists with duplicates removed
    static ListNode* mergeKListsNoDuplicates(std::vector<ListNode*>& lists) {
        MergeKSortedLists solver;
        ListNode* merged = solver.mergeKLists(lists);
        
        if (!merged) return nullptr;
        
        ListNode* current = merged;
        while (current->next) {
            if (current->val == current->next->val) {
                ListNode* duplicate = current->next;
                current->next = current->next->next;
                delete duplicate;
            } else {
                current = current->next;
            }
        }
        
        return merged;
    }
    
    // Find kth smallest element across k sorted lists
    static int findKthSmallest(std::vector<ListNode*>& lists, int k) {
        auto compare = [](ListNode* a, ListNode* b) {
            return a->val > b->val;
        };
        
        std::priority_queue<ListNode*, std::vector<ListNode*>, decltype(compare)> pq(compare);
        
        for (ListNode* list : lists) {
            if (list) pq.push(list);
        }
        
        for (int i = 0; i < k && !pq.empty(); i++) {
            ListNode* node = pq.top();
            pq.pop();
            
            if (i == k - 1) return node->val;
            
            if (node->next) pq.push(node->next);
        }
        
        return -1; // k is larger than total elements
    }
};

// Interview follow-up questions and solutions
class InterviewFollowUps {
public:
    static void discussComplexityTradeoffs() {
        std::cout << "Algorithm Complexity Trade-offs:\n";
        std::cout << "================================\n\n";
        
        std::cout << "1. Divide & Conquer:\n";
        std::cout << "   - Time: O(N log k)\n";
        std::cout << "   - Space: O(log k) recursion\n";
        std::cout << "   - Pros: Optimal time, good cache locality\n";
        std::cout << "   - Cons: Recursion overhead\n\n";
        
        std::cout << "2. Priority Queue:\n";
        std::cout << "   - Time: O(N log k)\n";
        std::cout << "   - Space: O(k) heap\n";
        std::cout << "   - Pros: Intuitive, handles streaming data\n";
        std::cout << "   - Cons: Heap operations overhead\n\n";
        
        std::cout << "3. Sequential Merging:\n";
        std::cout << "   - Time: O(k*N)\n";
        std::cout << "   - Space: O(1)\n";
        std::cout << "   - Pros: Simple, minimal space\n";
        std::cout << "   - Cons: Poor time complexity\n\n";
    }
    
    static void realWorldApplications() {
        std::cout << "Real-World Applications:\n";
        std::cout << "=======================\n";
        std::cout << "1. External sorting (merge phase)\n";
        std::cout << "2. Database join operations\n";
        std::cout << "3. Log file merging\n";
        std::cout << "4. Time series data aggregation\n";
        std::cout << "5. Distributed system data merging\n";
        std::cout << "6. Search engine result merging\n\n";
    }
};

int main() {
    std::cout << "Merge k Sorted Lists - Google Interview Problem\n";
    std::cout << "===============================================\n";
    
    // Test case 1: Basic example
    std::vector<std::vector<int>> test_data = {
        {1, 4, 5},
        {1, 3, 4},
        {2, 6}
    };
    
    std::vector<ListNode*> lists;
    for (const auto& data : test_data) {
        lists.push_back(ListUtils::createList(data));
    }
    
    std::cout << "Test Case 1:\n";
    std::cout << "Input lists: ";
    for (int i = 0; i < lists.size(); i++) {
        ListUtils::printList(lists[i]);
        if (i < lists.size() - 1) std::cout << ", ";
    }
    std::cout << "\n";
    
    MergeKSortedLists solver;
    ListNode* result = solver.mergeKLists(lists);
    
    std::cout << "Merged result: ";
    ListUtils::printList(result);
    std::cout << "\n\n";
    
    // Test case 2: Empty lists
    std::vector<ListNode*> empty_lists = {nullptr, nullptr};
    ListNode* empty_result = solver.mergeKLists(empty_lists);
    std::cout << "Test Case 2 (empty lists): ";
    ListUtils::printList(empty_result);
    std::cout << "\n\n";
    
    // Test case 3: Single element lists
    std::vector<ListNode*> single_lists = {
        ListUtils::createList({1}),
        ListUtils::createList({2}),
        ListUtils::createList({3})
    };
    
    ListNode* single_result = solver.mergeKLists(single_lists);
    std::cout << "Test Case 3 (single elements): ";
    ListUtils::printList(single_result);
    std::cout << "\n\n";
    
    // Advanced variations
    std::cout << "Advanced Variation - Merge k sorted arrays:\n";
    std::vector<std::vector<int>> arrays = {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}};
    auto merged_array = AdvancedVariations::mergeKSortedArrays(arrays);
    std::cout << "Merged array: [";
    for (int i = 0; i < merged_array.size(); i++) {
        std::cout << merged_array[i];
        if (i < merged_array.size() - 1) std::cout << ",";
    }
    std::cout << "]\n\n";
    
    // Performance analysis
    PerformanceTester::compareAlgorithms();
    PerformanceTester::memoryUsageAnalysis();
    
    // Interview discussions
    InterviewFollowUps::discussComplexityTradeoffs();
    InterviewFollowUps::realWorldApplications();
    
    // Cleanup
    ListUtils::deleteList(result);
    ListUtils::deleteList(empty_result);
    ListUtils::deleteList(single_result);
    
    return 0;
}
