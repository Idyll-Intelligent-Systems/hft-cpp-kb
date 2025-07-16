/*
Meta (Facebook) SDE Interview Problem 1: Binary Tree Vertical Order Traversal (Hard)
Given the root of a binary tree, return the vertical order traversal of its nodes' values. 
(i.e., from top to bottom, column by column).

If two nodes are in the same row and column, the order should be from left to right.

Example 1:
Input: root = [3,9,20,null,null,15,7]
       3
     /   \
    9     20
         /  \
        15   7
Output: [[9],[3,15],[20],[7]]

Example 2:
Input: root = [3,9,8,4,0,1,7,null,null,null,2,5]
       3
     /   \
    9     8
   / \   / \
  4   0 1   7
     /     /
    null  2
         /
        5
Output: [[4],[9,5],[3,0,1],[8,2],[7]]

This is a classic Meta interview problem testing BFS/DFS, coordinate mapping, and data structure design.

Time Complexity: O(n log n) due to sorting, O(n) for BFS traversal
Space Complexity: O(n) for queue and result storage
*/

#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <unordered_map>
#include <algorithm>

// Definition for a binary tree node
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    // Approach 1: BFS with coordinate tracking (Most intuitive)
    std::vector<std::vector<int>> verticalOrder(TreeNode* root) {
        if (!root) return {};
        
        // Map: column -> vector of (row, value) pairs
        std::map<int, std::vector<std::pair<int, int>>> columnMap;
        
        // BFS with (node, column, row) information
        std::queue<std::tuple<TreeNode*, int, int>> q;
        q.push({root, 0, 0});
        
        while (!q.empty()) {
            auto [node, col, row] = q.front();
            q.pop();
            
            columnMap[col].push_back({row, node->val});
            
            if (node->left) {
                q.push({node->left, col - 1, row + 1});
            }
            if (node->right) {
                q.push({node->right, col + 1, row + 1});
            }
        }
        
        std::vector<std::vector<int>> result;
        for (auto& [col, nodes] : columnMap) {
            // Sort by row first, then by value (for same row, left to right)
            std::sort(nodes.begin(), nodes.end());
            
            std::vector<int> columnValues;
            for (auto& [row, val] : nodes) {
                columnValues.push_back(val);
            }
            result.push_back(columnValues);
        }
        
        return result;
    }
    
    // Approach 2: DFS with coordinate tracking
    std::vector<std::vector<int>> verticalOrderDFS(TreeNode* root) {
        if (!root) return {};
        
        std::map<int, std::vector<std::pair<int, int>>> columnMap;
        
        dfsHelper(root, 0, 0, columnMap);
        
        std::vector<std::vector<int>> result;
        for (auto& [col, nodes] : columnMap) {
            std::sort(nodes.begin(), nodes.end());
            
            std::vector<int> columnValues;
            for (auto& [row, val] : nodes) {
                columnValues.push_back(val);
            }
            result.push_back(columnValues);
        }
        
        return result;
    }
    
    // Approach 3: Optimized BFS with level-order processing
    std::vector<std::vector<int>> verticalOrderOptimized(TreeNode* root) {
        if (!root) return {};
        
        std::map<int, std::vector<int>> columnMap;
        std::queue<std::pair<TreeNode*, int>> q;
        q.push({root, 0});
        
        while (!q.empty()) {
            int size = q.size();
            std::map<int, std::vector<int>> levelMap;
            
            // Process current level
            for (int i = 0; i < size; i++) {
                auto [node, col] = q.front();
                q.pop();
                
                levelMap[col].push_back(node->val);
                
                if (node->left) {
                    q.push({node->left, col - 1});
                }
                if (node->right) {
                    q.push({node->right, col + 1});
                }
            }
            
            // Add level results to main map
            for (auto& [col, values] : levelMap) {
                for (int val : values) {
                    columnMap[col].push_back(val);
                }
            }
        }
        
        std::vector<std::vector<int>> result;
        for (auto& [col, values] : columnMap) {
            result.push_back(values);
        }
        
        return result;
    }
    
    // Approach 4: Two-pass approach (collect ranges first)
    std::vector<std::vector<int>> verticalOrderTwoPass(TreeNode* root) {
        if (!root) return {};
        
        // First pass: find column range
        int minCol = 0, maxCol = 0;
        findColumnRange(root, 0, minCol, maxCol);
        
        // Second pass: collect values
        std::vector<std::vector<int>> result(maxCol - minCol + 1);
        std::queue<std::pair<TreeNode*, int>> q;
        q.push({root, -minCol}); // Adjust to 0-based indexing
        
        while (!q.empty()) {
            int size = q.size();
            std::vector<std::pair<int, int>> levelNodes; // (column, value)
            
            for (int i = 0; i < size; i++) {
                auto [node, col] = q.front();
                q.pop();
                
                levelNodes.push_back({col, node->val});
                
                if (node->left) {
                    q.push({node->left, col - 1});
                }
                if (node->right) {
                    q.push({node->right, col + 1});
                }
            }
            
            // Sort by column for current level (left to right)
            std::sort(levelNodes.begin(), levelNodes.end());
            
            for (auto& [col, val] : levelNodes) {
                result[col].push_back(val);
            }
        }
        
        return result;
    }
    
    // Approach 5: Memory-optimized with custom data structure
    std::vector<std::vector<int>> verticalOrderMemoryOptimized(TreeNode* root) {
        if (!root) return {};
        
        struct NodeInfo {
            TreeNode* node;
            int col;
            int row;
        };
        
        std::vector<NodeInfo> allNodes;
        std::queue<NodeInfo> q;
        q.push({root, 0, 0});
        
        while (!q.empty()) {
            NodeInfo current = q.front();
            q.pop();
            
            allNodes.push_back(current);
            
            if (current.node->left) {
                q.push({current.node->left, current.col - 1, current.row + 1});
            }
            if (current.node->right) {
                q.push({current.node->right, current.col + 1, current.row + 1});
            }
        }
        
        // Sort by column, then by row, then by original order
        std::sort(allNodes.begin(), allNodes.end(), 
                 [](const NodeInfo& a, const NodeInfo& b) {
                     if (a.col != b.col) return a.col < b.col;
                     return a.row < b.row;
                 });
        
        std::vector<std::vector<int>> result;
        int currentCol = INT_MIN;
        
        for (const NodeInfo& info : allNodes) {
            if (info.col != currentCol) {
                result.push_back({});
                currentCol = info.col;
            }
            result.back().push_back(info.node->val);
        }
        
        return result;
    }
    
private:
    void dfsHelper(TreeNode* node, int col, int row, 
                   std::map<int, std::vector<std::pair<int, int>>>& columnMap) {
        if (!node) return;
        
        columnMap[col].push_back({row, node->val});
        
        dfsHelper(node->left, col - 1, row + 1, columnMap);
        dfsHelper(node->right, col + 1, row + 1, columnMap);
    }
    
    void findColumnRange(TreeNode* node, int col, int& minCol, int& maxCol) {
        if (!node) return;
        
        minCol = std::min(minCol, col);
        maxCol = std::max(maxCol, col);
        
        findColumnRange(node->left, col - 1, minCol, maxCol);
        findColumnRange(node->right, col + 1, minCol, maxCol);
    }
};

// Test framework
class VerticalOrderTest {
public:
    static void runTests() {
        Solution solution;
        
        std::cout << "Meta Binary Tree Vertical Order Tests:" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        testBasicCases(solution);
        testEdgeCases(solution);
        compareApproaches(solution);
        metaSpecificScenarios(solution);
        performanceAnalysis(solution);
    }
    
    static void testBasicCases(Solution& solution) {
        std::cout << "\nBasic Test Cases:" << std::endl;
        std::cout << "=================" << std::endl;
        
        // Test case 1: Example 1
        TreeNode* root1 = createTree1();
        auto result1 = solution.verticalOrder(root1);
        std::cout << "Test 1 - Example tree:" << std::endl;
        printResult(result1);
        std::cout << "Expected: [[9],[3,15],[20],[7]]" << std::endl;
        
        // Test case 2: Single node
        TreeNode* root2 = new TreeNode(1);
        auto result2 = solution.verticalOrder(root2);
        std::cout << "\nTest 2 - Single node:" << std::endl;
        printResult(result2);
        std::cout << "Expected: [[1]]" << std::endl;
        
        // Test case 3: Left skewed tree
        TreeNode* root3 = createLeftSkewedTree();
        auto result3 = solution.verticalOrder(root3);
        std::cout << "\nTest 3 - Left skewed:" << std::endl;
        printResult(result3);
        std::cout << "Expected: [[3],[2],[1]]" << std::endl;
        
        // Test case 4: Right skewed tree
        TreeNode* root4 = createRightSkewedTree();
        auto result4 = solution.verticalOrder(root4);
        std::cout << "\nTest 4 - Right skewed:" << std::endl;
        printResult(result4);
        std::cout << "Expected: [[1],[2],[3]]" << std::endl;
        
        // Cleanup
        deleteTree(root1);
        deleteTree(root2);
        deleteTree(root3);
        deleteTree(root4);
    }
    
    static void testEdgeCases(Solution& solution) {
        std::cout << "\nEdge Case Tests:" << std::endl;
        std::cout << "================" << std::endl;
        
        // Empty tree
        auto result1 = solution.verticalOrder(nullptr);
        std::cout << "Empty tree: ";
        printResult(result1);
        std::cout << " (Expected: [])" << std::endl;
        
        // Complex tree with same row/column nodes
        TreeNode* complexRoot = createComplexTree();
        auto result2 = solution.verticalOrder(complexRoot);
        std::cout << "Complex tree: ";
        printResult(result2);
        std::cout << std::endl;
        
        // Perfect binary tree
        TreeNode* perfectRoot = createPerfectBinaryTree();
        auto result3 = solution.verticalOrder(perfectRoot);
        std::cout << "Perfect binary tree: ";
        printResult(result3);
        std::cout << std::endl;
        
        deleteTree(complexRoot);
        deleteTree(perfectRoot);
    }
    
    static void compareApproaches(Solution& solution) {
        std::cout << "\nApproach Comparison:" << std::endl;
        std::cout << "===================" << std::endl;
        
        TreeNode* testRoot = createLargeTree(1000);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result1 = solution.verticalOrder(testRoot);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        auto result2 = solution.verticalOrderDFS(testRoot);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        auto result3 = solution.verticalOrderOptimized(testRoot);
        end = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        auto result4 = solution.verticalOrderMemoryOptimized(testRoot);
        end = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "BFS with coordinates: " << duration1.count() << " ms" << std::endl;
        std::cout << "DFS with coordinates: " << duration2.count() << " ms" << std::endl;
        std::cout << "Optimized BFS: " << duration3.count() << " ms" << std::endl;
        std::cout << "Memory optimized: " << duration4.count() << " ms" << std::endl;
        
        bool allMatch = (result1.size() == result2.size() && 
                        result2.size() == result3.size() && 
                        result3.size() == result4.size());
        std::cout << "All results same size: " << (allMatch ? "✅" : "❌") << std::endl;
        
        deleteTree(testRoot);
    }
    
    static void metaSpecificScenarios(Solution& solution) {
        std::cout << "\nMeta-Specific Scenarios:" << std::endl;
        std::cout << "=======================" << std::endl;
        
        // Scenario 1: Social network hierarchy
        TreeNode* socialRoot = createSocialNetworkTree();
        auto result1 = solution.verticalOrder(socialRoot);
        std::cout << "Social network hierarchy:" << std::endl;
        printResult(result1);
        
        // Scenario 2: Organizational chart
        TreeNode* orgRoot = createOrgChartTree();
        auto result2 = solution.verticalOrder(orgRoot);
        std::cout << "Organizational chart:" << std::endl;
        printResult(result2);
        
        // Scenario 3: Feed timeline structure
        TreeNode* feedRoot = createFeedTimelineTree();
        auto result3 = solution.verticalOrder(feedRoot);
        std::cout << "Feed timeline structure:" << std::endl;
        printResult(result3);
        
        deleteTree(socialRoot);
        deleteTree(orgRoot);
        deleteTree(feedRoot);
    }
    
    static void performanceAnalysis(Solution& solution) {
        std::cout << "\nPerformance Analysis:" << std::endl;
        std::cout << "====================" << std::endl;
        
        std::vector<int> sizes = {100, 500, 1000, 5000};
        
        for (int size : sizes) {
            TreeNode* testTree = createLargeTree(size);
            
            auto start = std::chrono::high_resolution_clock::now();
            auto result = solution.verticalOrder(testTree);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            std::cout << "Tree size " << size << ": " << duration.count() << " μs" << std::endl;
            
            deleteTree(testTree);
        }
        
        // Memory usage analysis
        std::cout << "\nMemory Usage Estimation:" << std::endl;
        std::cout << "========================" << std::endl;
        
        int nodeCount = 1000;
        size_t bfsMemory = nodeCount * (sizeof(TreeNode*) + sizeof(int) * 2) + // Queue
                          nodeCount * sizeof(std::pair<int, int>); // Map storage
        size_t dfsMemory = nodeCount * sizeof(void*) + // Call stack
                          nodeCount * sizeof(std::pair<int, int>); // Map storage
        
        std::cout << "BFS approach: ~" << bfsMemory / 1024 << " KB" << std::endl;
        std::cout << "DFS approach: ~" << dfsMemory / 1024 << " KB" << std::endl;
    }
    
    static TreeNode* createTree1() {
        // Tree from example 1
        TreeNode* root = new TreeNode(3);
        root->left = new TreeNode(9);
        root->right = new TreeNode(20);
        root->right->left = new TreeNode(15);
        root->right->right = new TreeNode(7);
        return root;
    }
    
    static TreeNode* createLeftSkewedTree() {
        TreeNode* root = new TreeNode(1);
        root->left = new TreeNode(2);
        root->left->left = new TreeNode(3);
        return root;
    }
    
    static TreeNode* createRightSkewedTree() {
        TreeNode* root = new TreeNode(1);
        root->right = new TreeNode(2);
        root->right->right = new TreeNode(3);
        return root;
    }
    
    static TreeNode* createComplexTree() {
        TreeNode* root = new TreeNode(1);
        root->left = new TreeNode(2);
        root->right = new TreeNode(3);
        root->left->left = new TreeNode(4);
        root->left->right = new TreeNode(5);
        root->right->left = new TreeNode(6);
        root->right->right = new TreeNode(7);
        root->left->left->left = new TreeNode(8);
        root->left->right->right = new TreeNode(9);
        return root;
    }
    
    static TreeNode* createPerfectBinaryTree() {
        TreeNode* root = new TreeNode(1);
        root->left = new TreeNode(2);
        root->right = new TreeNode(3);
        root->left->left = new TreeNode(4);
        root->left->right = new TreeNode(5);
        root->right->left = new TreeNode(6);
        root->right->right = new TreeNode(7);
        return root;
    }
    
    static TreeNode* createSocialNetworkTree() {
        // Simulate user connections
        TreeNode* root = new TreeNode(100); // User ID
        root->left = new TreeNode(50);
        root->right = new TreeNode(150);
        root->left->left = new TreeNode(25);
        root->left->right = new TreeNode(75);
        root->right->left = new TreeNode(125);
        root->right->right = new TreeNode(175);
        return root;
    }
    
    static TreeNode* createOrgChartTree() {
        // Simulate organizational structure
        TreeNode* root = new TreeNode(1); // CEO
        root->left = new TreeNode(2); // VP Engineering
        root->right = new TreeNode(3); // VP Marketing
        root->left->left = new TreeNode(4); // Director
        root->left->right = new TreeNode(5); // Director
        root->right->left = new TreeNode(6); // Manager
        return root;
    }
    
    static TreeNode* createFeedTimelineTree() {
        // Simulate feed post hierarchy
        TreeNode* root = new TreeNode(1001); // Post ID
        root->left = new TreeNode(1002); // Comment
        root->right = new TreeNode(1003); // Share
        root->left->left = new TreeNode(1004); // Reply
        root->left->right = new TreeNode(1005); // Like
        return root;
    }
    
    static TreeNode* createLargeTree(int nodeCount) {
        if (nodeCount <= 0) return nullptr;
        
        TreeNode* root = new TreeNode(1);
        std::queue<TreeNode*> q;
        q.push(root);
        
        int currentVal = 2;
        while (!q.empty() && currentVal <= nodeCount) {
            TreeNode* current = q.front();
            q.pop();
            
            if (currentVal <= nodeCount) {
                current->left = new TreeNode(currentVal++);
                q.push(current->left);
            }
            
            if (currentVal <= nodeCount) {
                current->right = new TreeNode(currentVal++);
                q.push(current->right);
            }
        }
        
        return root;
    }
    
    static void deleteTree(TreeNode* root) {
        if (!root) return;
        deleteTree(root->left);
        deleteTree(root->right);
        delete root;
    }
    
    static void printResult(const std::vector<std::vector<int>>& result) {
        std::cout << "[";
        for (int i = 0; i < result.size(); i++) {
            std::cout << "[";
            for (int j = 0; j < result[i].size(); j++) {
                std::cout << result[i][j];
                if (j < result[i].size() - 1) std::cout << ",";
            }
            std::cout << "]";
            if (i < result.size() - 1) std::cout << ",";
        }
        std::cout << "]";
    }
};

int main() {
    VerticalOrderTest::runTests();
    return 0;
}

/*
Algorithm Analysis:

Core Problem: Binary tree vertical order traversal with proper ordering

Key Insights:
1. Assign column coordinates: left child = parent_col - 1, right child = parent_col + 1
2. Assign row coordinates for same-column ordering
3. Use BFS to maintain level order, DFS with sorting for flexibility

Approach Comparison:

1. BFS with Coordinate Tracking (Recommended):
   - Time: O(n log n) due to sorting within columns
   - Space: O(n) for queue and maps
   - Natural level-order processing
   - Handles same-row ordering correctly

2. DFS with Coordinate Tracking:
   - Time: O(n log n) due to sorting
   - Space: O(h + n) where h is height for recursion
   - Simpler implementation
   - Requires sorting for correct order

3. Optimized BFS:
   - Time: O(n) for traversal + O(k log k) for column sorting
   - Processes level by level for natural ordering
   - More complex but potentially faster

4. Two-pass Approach:
   - First pass finds column range
   - Second pass collects values
   - Good for memory optimization

Meta Interview Focus:
- Tree traversal algorithms (BFS/DFS)
- Coordinate system design
- Data structure choice (map vs array)
- Memory vs time optimization
- Edge case handling

Key Design Decisions:
1. BFS vs DFS for traversal
2. Map vs vector for column storage
3. Sorting strategy for same-column nodes
4. Memory optimization techniques

Real-world Applications at Meta:
- Social graph visualization
- Organizational chart rendering
- Feed timeline structure
- Comment thread display
- Network analysis visualization

Edge Cases:
- Empty tree
- Single node
- Skewed trees (left/right)
- Multiple nodes in same position
- Very wide or deep trees

Interview Tips:
1. Start with coordinate assignment explanation
2. Choose BFS for natural level ordering
3. Discuss sorting requirements
4. Handle edge cases explicitly
5. Consider memory optimization

Common Mistakes:
1. Wrong coordinate calculation
2. Incorrect same-row ordering
3. Not handling empty tree
4. Memory leaks in tree creation
5. Poor sorting implementation

Advanced Optimizations:
- Custom comparator for efficient sorting
- Memory pool for node allocation
- Iterative vs recursive approaches
- Parallel processing for large trees
- Cache-friendly data structures

Testing Strategy:
- Basic functionality with known results
- Edge cases (empty, single node, skewed)
- Performance with large trees
- Meta-specific scenarios
- Memory usage analysis

Production Considerations:
- Thread safety for concurrent access
- Memory management and cleanup
- Input validation and error handling
- Scalability for large social graphs
- Visualization-friendly output format

Complexity Analysis:
- Best case: O(n) when no sorting needed within columns
- Average case: O(n log n) for typical trees
- Worst case: O(n log n) when many nodes in same column
- Space: O(n) for storing all nodes and coordinates

This problem is important for Meta because:
1. Tests tree traversal understanding
2. Requires coordinate system thinking
3. Has direct UI/visualization applications
4. Shows data structure design skills
5. Demonstrates optimization awareness
*/
