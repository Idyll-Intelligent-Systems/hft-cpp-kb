/*
Problem: Merge k Sorted Lists (Hard)
You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
Merge all the linked-lists into one sorted linked-list and return it.

Example:
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]

Time Complexity: O(N log k) where N is total number of nodes and k is number of lists
Space Complexity: O(log k) for recursion stack in divide and conquer approach
*/

#include <iostream>
#include <vector>
#include <queue>

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    // Approach 1: Divide and Conquer (Most Efficient)
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
    // Approach 2: Priority Queue (Min Heap)
    ListNode* mergeKListsWithPriorityQueue(std::vector<ListNode*>& lists) {
        auto compare = [](ListNode* a, ListNode* b) {
            return a->val > b->val;  // Min heap
        };
        
        std::priority_queue<ListNode*, std::vector<ListNode*>, decltype(compare)> pq(compare);
        
        // Add first node of each list to priority queue
        for (ListNode* list : lists) {
            if (list) pq.push(list);
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
    
    // Approach 3: Sequential Merge (Less Efficient)
    ListNode* mergeKListsSequential(std::vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        
        ListNode* result = lists[0];
        for (int i = 1; i < lists.size(); i++) {
            result = mergeTwoLists(result, lists[i]);
        }
        
        return result;
    }
};

// Helper functions for testing
ListNode* createList(const std::vector<int>& values) {
    if (values.empty()) return nullptr;
    
    ListNode* head = new ListNode(values[0]);
    ListNode* current = head;
    
    for (int i = 1; i < values.size(); i++) {
        current->next = new ListNode(values[i]);
        current = current->next;
    }
    
    return head;
}

void printList(ListNode* head) {
    while (head) {
        std::cout << head->val;
        if (head->next) std::cout << " -> ";
        head = head->next;
    }
    std::cout << std::endl;
}

void deleteList(ListNode* head) {
    while (head) {
        ListNode* temp = head;
        head = head->next;
        delete temp;
    }
}

int main() {
    Solution solution;
    
    // Test case 1
    std::vector<ListNode*> lists1 = {
        createList({1, 4, 5}),
        createList({1, 3, 4}),
        createList({2, 6})
    };
    
    std::cout << "Test 1 - Input lists:" << std::endl;
    for (ListNode* list : lists1) {
        printList(list);
    }
    
    ListNode* result1 = solution.mergeKLists(lists1);
    std::cout << "Merged result: ";
    printList(result1);
    std::cout << std::endl;
    
    // Test case 2: Empty lists
    std::vector<ListNode*> lists2 = {};
    ListNode* result2 = solution.mergeKLists(lists2);
    std::cout << "Test 2 - Empty input: ";
    printList(result2);
    std::cout << std::endl;
    
    // Test case 3: Single empty list
    std::vector<ListNode*> lists3 = {nullptr};
    ListNode* result3 = solution.mergeKLists(lists3);
    std::cout << "Test 3 - Single empty list: ";
    printList(result3);
    std::cout << std::endl;
    
    // Cleanup
    deleteList(result1);
    deleteList(result2);
    deleteList(result3);
    
    return 0;
}

/*
Algorithm Analysis:

1. Divide and Conquer Approach:
   - Time: O(N log k) where N = total nodes, k = number of lists
   - Space: O(log k) for recursion stack
   - Most efficient for large k

2. Priority Queue Approach:
   - Time: O(N log k)
   - Space: O(k) for priority queue
   - Good for understanding but slightly more overhead

3. Sequential Merge:
   - Time: O(N * k) - less efficient
   - Space: O(1)
   - Simple but not optimal for large k

Key Insights:
- Divide and conquer reduces the number of merge operations
- Each merge operation is O(n) where n is total nodes being merged
- Priority queue maintains the smallest element across all lists
- Dummy node simplifies list construction logic

Edge Cases:
- Empty input vector
- All lists are empty
- Single list (empty or non-empty)
- Lists of varying lengths
*/
