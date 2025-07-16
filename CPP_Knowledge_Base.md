# C++ Knowledge Base for Interviews & Online Assessments

## Table of Contents
1. [C++ Fundamentals](#cpp-fundamentals)
2. [Data Structures](#data-structures)
3. [Algorithms](#algorithms)
4. [HFT Company Questions](#hft-company-questions)
5. [Time & Space Complexity](#time-space-complexity)
6. [C++ STL Cheat Sheet](#cpp-stl-cheat-sheet)
7. [Interview Tips](#interview-tips)

---

## C++ Fundamentals

### Memory Management
```cpp
// Stack vs Heap allocation
int stackVar = 10;              // Stack - automatic cleanup
int* heapVar = new int(10);     // Heap - manual cleanup required
delete heapVar;                 // Don't forget to delete!

// Smart pointers (C++11+)
std::unique_ptr<int> smart_ptr = std::make_unique<int>(10);
std::shared_ptr<int> shared_ptr = std::make_shared<int>(10);
```

### Object-Oriented Programming
```cpp
class Base {
public:
    virtual void print() { std::cout << "Base"; }
    virtual ~Base() = default;  // Virtual destructor
};

class Derived : public Base {
public:
    void print() override { std::cout << "Derived"; }
};
```

### Move Semantics (C++11)
```cpp
class MyClass {
private:
    int* data;
public:
    // Move constructor
    MyClass(MyClass&& other) noexcept : data(other.data) {
        other.data = nullptr;
    }
    
    // Move assignment
    MyClass& operator=(MyClass&& other) noexcept {
        if (this != &other) {
            delete data;
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }
};
```

---

## Data Structures

### Array & Vector Operations
```cpp
#include <vector>
#include <algorithm>

std::vector<int> arr = {3, 1, 4, 1, 5};
std::sort(arr.begin(), arr.end());              // Sort
auto it = std::binary_search(arr.begin(), arr.end(), 4);  // Binary search
arr.erase(std::remove(arr.begin(), arr.end(), 1), arr.end()); // Remove element
```

### Stack & Queue
```cpp
#include <stack>
#include <queue>

std::stack<int> st;
st.push(1); st.push(2);
int top = st.top(); st.pop();

std::queue<int> q;
q.push(1); q.push(2);
int front = q.front(); q.pop();

std::priority_queue<int> pq;  // Max heap by default
std::priority_queue<int, std::vector<int>, std::greater<int>> min_pq; // Min heap
```

### Hash Map & Set
```cpp
#include <unordered_map>
#include <unordered_set>

std::unordered_map<int, std::string> map;
map[1] = "one";
if (map.find(1) != map.end()) { /* exists */ }

std::unordered_set<int> set;
set.insert(1);
if (set.count(1)) { /* exists */ }
```

### Trees
```cpp
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// Binary Tree Traversal
void inorder(TreeNode* root) {
    if (!root) return;
    inorder(root->left);
    std::cout << root->val << " ";
    inorder(root->right);
}
```

---

## Algorithms

### Sorting Algorithms

#### Quick Sort
```cpp
void quickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}
```

### Graph Algorithms

#### DFS & BFS
```cpp
// DFS
void dfs(std::vector<std::vector<int>>& graph, int node, std::vector<bool>& visited) {
    visited[node] = true;
    for (int neighbor : graph[node]) {
        if (!visited[neighbor]) {
            dfs(graph, neighbor, visited);
        }
    }
}

// BFS
void bfs(std::vector<std::vector<int>>& graph, int start) {
    std::queue<int> q;
    std::vector<bool> visited(graph.size(), false);
    q.push(start);
    visited[start] = true;
    
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        
        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}
```

### Dynamic Programming

#### Common Patterns
```cpp
// 1D DP - Fibonacci
int fibonacci(int n) {
    std::vector<int> dp(n + 1);
    dp[0] = 0; dp[1] = 1;
    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i-1] + dp[i-2];
    }
    return dp[n];
}

// 2D DP - Longest Common Subsequence
int lcs(const std::string& s1, const std::string& s2) {
    int m = s1.length(), n = s2.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = std::max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    return dp[m][n];
}
```

---

## HFT Company Questions

### Latency Optimization
```cpp
// Cache-friendly data structures
struct alignas(64) CacheLinePadded {  // Align to cache line
    int data;
    char padding[60];  // Prevent false sharing
};

// Branch prediction optimization
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

if (LIKELY(price > 0)) {
    // Hot path
}
```

### Order Book Implementation
```cpp
class OrderBook {
private:
    std::map<double, int> bids;   // Price -> Quantity
    std::map<double, int> asks;
    
public:
    void addBid(double price, int qty) {
        bids[price] += qty;
    }
    
    void addAsk(double price, int qty) {
        asks[price] += qty;
    }
    
    double getBestBid() {
        return bids.empty() ? 0.0 : bids.rbegin()->first;
    }
    
    double getBestAsk() {
        return asks.empty() ? 0.0 : asks.begin()->first;
    }
};
```

---

## Time & Space Complexity

### Big O Notation Cheat Sheet

| Operation | Array | Linked List | Hash Table | Binary Tree | Heap |
|-----------|-------|-------------|------------|-------------|------|
| Access    | O(1)  | O(n)        | O(1)*      | O(log n)    | O(1) |
| Search    | O(n)  | O(n)        | O(1)*      | O(log n)    | O(n) |
| Insert    | O(n)  | O(1)        | O(1)*      | O(log n)    | O(log n) |
| Delete    | O(n)  | O(1)        | O(1)*      | O(log n)    | O(log n) |

*Average case for hash table

### Common Algorithm Complexities
- **Sorting**: Quick Sort O(n log n), Merge Sort O(n log n), Heap Sort O(n log n)
- **Graph**: DFS/BFS O(V + E), Dijkstra O((V + E) log V)
- **Dynamic Programming**: Usually O(n²) or O(n³) depending on problem

---

## C++ STL Cheat Sheet

### Containers
```cpp
// Sequence Containers
std::vector<int> v;           // Dynamic array
std::deque<int> dq;           // Double-ended queue
std::list<int> lst;           // Doubly linked list
std::array<int, 10> arr;      // Fixed-size array

// Associative Containers
std::set<int> s;              // Sorted unique elements
std::multiset<int> ms;        // Sorted elements (duplicates allowed)
std::map<int, string> m;      // Sorted key-value pairs
std::multimap<int, string> mm; // Sorted key-value pairs (duplicate keys)

// Unordered Containers
std::unordered_set<int> us;   // Hash set
std::unordered_map<int, string> um; // Hash map
```

### Algorithms
```cpp
#include <algorithm>

// Sorting & Searching
std::sort(v.begin(), v.end());
std::binary_search(v.begin(), v.end(), target);
std::lower_bound(v.begin(), v.end(), target);
std::upper_bound(v.begin(), v.end(), target);

// Min/Max
auto min_it = std::min_element(v.begin(), v.end());
auto max_it = std::max_element(v.begin(), v.end());

// Permutations
std::next_permutation(v.begin(), v.end());
std::prev_permutation(v.begin(), v.end());

// Numeric
int sum = std::accumulate(v.begin(), v.end(), 0);
```

### Useful Iterators
```cpp
// Reverse iteration
for (auto it = v.rbegin(); it != v.rend(); ++it) { /* ... */ }

// Range-based for loop (C++11)
for (const auto& element : v) { /* ... */ }
for (auto& [key, value] : map) { /* ... */ }  // C++17 structured binding
```

---

## Interview Tips

### Problem-Solving Approach
1. **Clarify Requirements**: Ask about input constraints, edge cases
2. **Think Out Loud**: Explain your thought process
3. **Start Simple**: Brute force first, then optimize
4. **Test Your Code**: Walk through examples
5. **Analyze Complexity**: Discuss time and space complexity

### Common Patterns
- **Two Pointers**: For sorted arrays, palindromes
- **Sliding Window**: For substring problems
- **Binary Search**: For sorted data, optimization problems
- **DFS/BFS**: For tree/graph traversal
- **Dynamic Programming**: For optimization problems with overlapping subproblems

### Debugging Tips
```cpp
// Use assert for debugging
#include <cassert>
assert(index >= 0 && index < size);

// Print debugging
#ifdef DEBUG
#define DBG(x) std::cerr << #x << " = " << x << std::endl
#else
#define DBG(x)
#endif
```

---

## Quick Reference

### Input/Output
```cpp
#include <iostream>
#include <sstream>

// Fast I/O
std::ios_base::sync_with_stdio(false);
std::cin.tie(nullptr);

// String stream
std::stringstream ss("1 2 3");
int a, b, c;
ss >> a >> b >> c;
```

### Bit Manipulation
```cpp
// Common operations
int setBit(int n, int pos) { return n | (1 << pos); }
int clearBit(int n, int pos) { return n & ~(1 << pos); }
int toggleBit(int n, int pos) { return n ^ (1 << pos); }
bool checkBit(int n, int pos) { return (n & (1 << pos)) != 0; }

// Count set bits
int countBits(int n) { return __builtin_popcount(n); }
```

### Lambda Functions (C++11)
```cpp
auto lambda = [](int a, int b) -> int { return a + b; };
std::sort(v.begin(), v.end(), [](int a, int b) { return a > b; }); // Descending
```

---

*This knowledge base covers essential C++ concepts for technical interviews and competitive programming. Practice regularly and focus on understanding the underlying principles rather than memorizing code.*
