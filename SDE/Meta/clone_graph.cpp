/*
Meta (Facebook) SDE Interview Problem 5: Clone Graph (Medium-Hard)
Given a reference of a node in a connected undirected graph, return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

For simplicity, each node's value is the same as the node's index (1-indexed). 
For example, the first node with val = 1, the second node with val = 2, and so on. 
The graph is represented in the test case using an adjacency list.

Example 1:
Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]

Example 2:
Input: adjList = [[]]
Output: [[]]

This is a classic Meta interview problem testing graph traversal, deep copying, and handling of object references.
It's fundamental for understanding how to work with complex data structures and avoid infinite loops.

Time Complexity: O(V + E) where V is vertices and E is edges
Space Complexity: O(V) for the hash map storing cloned nodes
*/

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stack>
#include <algorithm>
#include <chrono>

// Definition for a Node
class Node {
public:
    int val;
    std::vector<Node*> neighbors;
    
    Node() {
        val = 0;
        neighbors = std::vector<Node*>();
    }
    
    Node(int _val) {
        val = _val;
        neighbors = std::vector<Node*>();
    }
    
    Node(int _val, std::vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};

class Solution {
public:
    // Approach 1: DFS with HashMap (Recommended)
    Node* cloneGraph(Node* node) {
        if (!node) return nullptr;
        
        std::unordered_map<Node*, Node*> cloneMap;
        return dfsClone(node, cloneMap);
    }
    
    // Approach 2: BFS with HashMap
    Node* cloneGraphBFS(Node* node) {
        if (!node) return nullptr;
        
        std::unordered_map<Node*, Node*> cloneMap;
        std::queue<Node*> queue;
        
        // Start with the first node
        cloneMap[node] = new Node(node->val);
        queue.push(node);
        
        while (!queue.empty()) {
            Node* current = queue.front();
            queue.pop();
            
            // Process all neighbors
            for (Node* neighbor : current->neighbors) {
                if (cloneMap.find(neighbor) == cloneMap.end()) {
                    // Create new clone if not exists
                    cloneMap[neighbor] = new Node(neighbor->val);
                    queue.push(neighbor);
                }
                
                // Add neighbor to current node's clone
                cloneMap[current]->neighbors.push_back(cloneMap[neighbor]);
            }
        }
        
        return cloneMap[node];
    }
    
    // Approach 3: Iterative DFS with Stack
    Node* cloneGraphIterativeDFS(Node* node) {
        if (!node) return nullptr;
        
        std::unordered_map<Node*, Node*> cloneMap;
        std::stack<Node*> stack;
        
        cloneMap[node] = new Node(node->val);
        stack.push(node);
        
        while (!stack.empty()) {
            Node* current = stack.top();
            stack.pop();
            
            for (Node* neighbor : current->neighbors) {
                if (cloneMap.find(neighbor) == cloneMap.end()) {
                    cloneMap[neighbor] = new Node(neighbor->val);
                    stack.push(neighbor);
                }
                
                cloneMap[current]->neighbors.push_back(cloneMap[neighbor]);
            }
        }
        
        return cloneMap[node];
    }
    
    // Approach 4: Two-pass approach (first create nodes, then connect)
    Node* cloneGraphTwoPass(Node* node) {
        if (!node) return nullptr;
        
        std::unordered_map<Node*, Node*> cloneMap;
        std::unordered_set<Node*> visited;
        
        // First pass: Create all nodes
        createNodes(node, cloneMap, visited);
        
        // Second pass: Connect the nodes
        visited.clear();
        connectNodes(node, cloneMap, visited);
        
        return cloneMap[node];
    }
    
    // Approach 5: Using node value as key optimization
    Node* cloneGraphValueKey(Node* node) {
        if (!node) return nullptr;
        
        std::unordered_map<int, Node*> valueToClone;
        std::unordered_set<int> visited;
        
        return dfsCloneByValue(node, valueToClone, visited);
    }
    
    // Approach 6: Memory-optimized with object pooling
    Node* cloneGraphPooled(Node* node) {
        if (!node) return nullptr;
        
        nodePool.clear();
        poolIndex = 0;
        
        std::unordered_map<Node*, Node*> cloneMap;
        return dfsClonePooled(node, cloneMap);
    }
    
private:
    Node* dfsClone(Node* node, std::unordered_map<Node*, Node*>& cloneMap) {
        if (cloneMap.find(node) != cloneMap.end()) {
            return cloneMap[node];
        }
        
        // Create clone of current node
        Node* clone = new Node(node->val);
        cloneMap[node] = clone;
        
        // Clone all neighbors recursively
        for (Node* neighbor : node->neighbors) {
            clone->neighbors.push_back(dfsClone(neighbor, cloneMap));
        }
        
        return clone;
    }
    
    void createNodes(Node* node, std::unordered_map<Node*, Node*>& cloneMap, 
                    std::unordered_set<Node*>& visited) {
        if (visited.find(node) != visited.end()) {
            return;
        }
        
        visited.insert(node);
        cloneMap[node] = new Node(node->val);
        
        for (Node* neighbor : node->neighbors) {
            createNodes(neighbor, cloneMap, visited);
        }
    }
    
    void connectNodes(Node* node, std::unordered_map<Node*, Node*>& cloneMap,
                     std::unordered_set<Node*>& visited) {
        if (visited.find(node) != visited.end()) {
            return;
        }
        
        visited.insert(node);
        
        for (Node* neighbor : node->neighbors) {
            cloneMap[node]->neighbors.push_back(cloneMap[neighbor]);
            connectNodes(neighbor, cloneMap, visited);
        }
    }
    
    Node* dfsCloneByValue(Node* node, std::unordered_map<int, Node*>& valueToClone,
                         std::unordered_set<int>& visited) {
        if (visited.find(node->val) != visited.end()) {
            return valueToClone[node->val];
        }
        
        visited.insert(node->val);
        Node* clone = new Node(node->val);
        valueToClone[node->val] = clone;
        
        for (Node* neighbor : node->neighbors) {
            clone->neighbors.push_back(dfsCloneByValue(neighbor, valueToClone, visited));
        }
        
        return clone;
    }
    
    // Object pool for memory optimization
    std::vector<Node*> nodePool;
    int poolIndex = 0;
    
    Node* getPooledNode(int val) {
        if (poolIndex >= nodePool.size()) {
            nodePool.push_back(new Node(val));
        } else {
            nodePool[poolIndex]->val = val;
            nodePool[poolIndex]->neighbors.clear();
        }
        return nodePool[poolIndex++];
    }
    
    Node* dfsClonePooled(Node* node, std::unordered_map<Node*, Node*>& cloneMap) {
        if (cloneMap.find(node) != cloneMap.end()) {
            return cloneMap[node];
        }
        
        Node* clone = getPooledNode(node->val);
        cloneMap[node] = clone;
        
        for (Node* neighbor : node->neighbors) {
            clone->neighbors.push_back(dfsClonePooled(neighbor, cloneMap));
        }
        
        return clone;
    }
};

// Graph utilities for testing
class GraphUtils {
public:
    static Node* createGraph(std::vector<std::vector<int>>& adjList) {
        if (adjList.empty()) return nullptr;
        
        std::vector<Node*> nodes(adjList.size());
        
        // Create all nodes first
        for (int i = 0; i < adjList.size(); i++) {
            nodes[i] = new Node(i + 1);
        }
        
        // Connect neighbors
        for (int i = 0; i < adjList.size(); i++) {
            for (int neighborVal : adjList[i]) {
                nodes[i]->neighbors.push_back(nodes[neighborVal - 1]);
            }
        }
        
        return nodes[0];
    }
    
    static void deleteGraph(Node* node) {
        if (!node) return;
        
        std::unordered_set<Node*> visited;
        deleteGraphHelper(node, visited);
    }
    
    static bool compareGraphs(Node* original, Node* cloned) {
        if (!original && !cloned) return true;
        if (!original || !cloned) return false;
        
        std::unordered_set<Node*> visitedOriginal, visitedCloned;
        return compareGraphsHelper(original, cloned, visitedOriginal, visitedCloned);
    }
    
    static void printGraph(Node* node, const std::string& title) {
        if (!node) {
            std::cout << title << ": Empty graph" << std::endl;
            return;
        }
        
        std::cout << title << ":" << std::endl;
        std::unordered_set<Node*> visited;
        printGraphHelper(node, visited);
        std::cout << std::endl;
    }
    
    static std::vector<std::vector<int>> graphToAdjList(Node* node) {
        if (!node) return {};
        
        std::unordered_map<Node*, int> nodeToIndex;
        std::vector<Node*> nodes;
        std::unordered_set<Node*> visited;
        
        // Collect all nodes
        collectNodes(node, nodes, visited);
        
        // Create mapping
        for (int i = 0; i < nodes.size(); i++) {
            nodeToIndex[nodes[i]] = i;
        }
        
        // Build adjacency list
        std::vector<std::vector<int>> adjList(nodes.size());
        for (int i = 0; i < nodes.size(); i++) {
            for (Node* neighbor : nodes[i]->neighbors) {
                adjList[i].push_back(nodeToIndex[neighbor] + 1); // 1-indexed
            }
        }
        
        return adjList;
    }
    
private:
    static void deleteGraphHelper(Node* node, std::unordered_set<Node*>& visited) {
        if (!node || visited.find(node) != visited.end()) {
            return;
        }
        
        visited.insert(node);
        
        for (Node* neighbor : node->neighbors) {
            deleteGraphHelper(neighbor, visited);
        }
        
        delete node;
    }
    
    static bool compareGraphsHelper(Node* original, Node* cloned,
                                   std::unordered_set<Node*>& visitedOriginal,
                                   std::unordered_set<Node*>& visitedCloned) {
        if (original == cloned) return false; // Should be different objects
        if (original->val != cloned->val) return false;
        if (original->neighbors.size() != cloned->neighbors.size()) return false;
        
        visitedOriginal.insert(original);
        visitedCloned.insert(cloned);
        
        // Check neighbors
        for (int i = 0; i < original->neighbors.size(); i++) {
            Node* origNeighbor = original->neighbors[i];
            Node* clonedNeighbor = cloned->neighbors[i];
            
            if (origNeighbor->val != clonedNeighbor->val) return false;
            
            if (visitedOriginal.find(origNeighbor) == visitedOriginal.end()) {
                if (!compareGraphsHelper(origNeighbor, clonedNeighbor, 
                                       visitedOriginal, visitedCloned)) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    static void printGraphHelper(Node* node, std::unordered_set<Node*>& visited) {
        if (!node || visited.find(node) != visited.end()) {
            return;
        }
        
        visited.insert(node);
        
        std::cout << "Node " << node->val << " -> [";
        for (int i = 0; i < node->neighbors.size(); i++) {
            std::cout << node->neighbors[i]->val;
            if (i < node->neighbors.size() - 1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
        
        for (Node* neighbor : node->neighbors) {
            printGraphHelper(neighbor, visited);
        }
    }
    
    static void collectNodes(Node* node, std::vector<Node*>& nodes, 
                            std::unordered_set<Node*>& visited) {
        if (!node || visited.find(node) != visited.end()) {
            return;
        }
        
        visited.insert(node);
        nodes.push_back(node);
        
        for (Node* neighbor : node->neighbors) {
            collectNodes(neighbor, nodes, visited);
        }
    }
};

// Test framework
class CloneGraphTest {
public:
    static void runTests() {
        std::cout << "Meta Clone Graph Tests:" << std::endl;
        std::cout << "======================" << std::endl;
        
        testBasicCases();
        testEdgeCases();
        compareApproaches();
        metaSpecificScenarios();
        performanceAnalysis();
    }
    
    static void testBasicCases() {
        std::cout << "\nBasic Test Cases:" << std::endl;
        std::cout << "=================" << std::endl;
        
        Solution solution;
        
        // Test case 1: Example 1 - 4-node cycle
        std::vector<std::vector<int>> adjList1 = {{2,4},{1,3},{2,4},{1,3}};
        Node* graph1 = GraphUtils::createGraph(adjList1);
        Node* cloned1 = solution.cloneGraph(graph1);
        
        std::cout << "Test 1 - 4-node cycle:" << std::endl;
        GraphUtils::printGraph(graph1, "Original");
        GraphUtils::printGraph(cloned1, "Cloned");
        std::cout << "Graphs match structure: " << (GraphUtils::compareGraphs(graph1, cloned1) ? "✅" : "❌") << std::endl;
        
        // Test case 2: Example 2 - Single node
        std::vector<std::vector<int>> adjList2 = {{}};
        Node* graph2 = GraphUtils::createGraph(adjList2);
        Node* cloned2 = solution.cloneGraph(graph2);
        
        std::cout << "\nTest 2 - Single node:" << std::endl;
        GraphUtils::printGraph(graph2, "Original");
        GraphUtils::printGraph(cloned2, "Cloned");
        std::cout << "Graphs match structure: " << (GraphUtils::compareGraphs(graph2, cloned2) ? "✅" : "❌") << std::endl;
        
        // Test case 3: Linear chain
        std::vector<std::vector<int>> adjList3 = {{2},{1,3},{2}};
        Node* graph3 = GraphUtils::createGraph(adjList3);
        Node* cloned3 = solution.cloneGraph(graph3);
        
        std::cout << "\nTest 3 - Linear chain:" << std::endl;
        GraphUtils::printGraph(graph3, "Original");
        GraphUtils::printGraph(cloned3, "Cloned");
        std::cout << "Graphs match structure: " << (GraphUtils::compareGraphs(graph3, cloned3) ? "✅" : "❌") << std::endl;
        
        // Cleanup
        GraphUtils::deleteGraph(graph1);
        GraphUtils::deleteGraph(cloned1);
        GraphUtils::deleteGraph(graph2);
        GraphUtils::deleteGraph(cloned2);
        GraphUtils::deleteGraph(graph3);
        GraphUtils::deleteGraph(cloned3);
    }
    
    static void testEdgeCases() {
        std::cout << "\nEdge Case Tests:" << std::endl;
        std::cout << "================" << std::endl;
        
        Solution solution;
        
        // Empty graph
        Node* clonedEmpty = solution.cloneGraph(nullptr);
        std::cout << "Empty graph: " << (clonedEmpty == nullptr ? "✅" : "❌") << std::endl;
        
        // Self-loop
        std::vector<std::vector<int>> adjListSelf = {{1}};
        Node* graphSelf = GraphUtils::createGraph(adjListSelf);
        Node* clonedSelf = solution.cloneGraph(graphSelf);
        
        std::cout << "Self-loop graph:" << std::endl;
        GraphUtils::printGraph(graphSelf, "Original");
        GraphUtils::printGraph(clonedSelf, "Cloned");
        std::cout << "Self-loop preserved: " << (GraphUtils::compareGraphs(graphSelf, clonedSelf) ? "✅" : "❌") << std::endl;
        
        // Complete graph (all nodes connected)
        std::vector<std::vector<int>> adjListComplete = {{2,3},{1,3},{1,2}};
        Node* graphComplete = GraphUtils::createGraph(adjListComplete);
        Node* clonedComplete = solution.cloneGraph(graphComplete);
        
        std::cout << "\nComplete graph:" << std::endl;
        GraphUtils::printGraph(graphComplete, "Original");
        GraphUtils::printGraph(clonedComplete, "Cloned");
        std::cout << "Complete graph cloned: " << (GraphUtils::compareGraphs(graphComplete, clonedComplete) ? "✅" : "❌") << std::endl;
        
        // Cleanup
        GraphUtils::deleteGraph(graphSelf);
        GraphUtils::deleteGraph(clonedSelf);
        GraphUtils::deleteGraph(graphComplete);
        GraphUtils::deleteGraph(clonedComplete);
    }
    
    static void compareApproaches() {
        std::cout << "\nApproach Comparison:" << std::endl;
        std::cout << "===================" << std::endl;
        
        Solution solution;
        
        // Create a complex graph for testing
        std::vector<std::vector<int>> adjList = {{2,3,4},{1,5},{1,6},{1},{2},{3}};
        Node* originalGraph = GraphUtils::createGraph(adjList);
        
        auto start = std::chrono::high_resolution_clock::now();
        Node* cloned1 = solution.cloneGraph(originalGraph);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        Node* cloned2 = solution.cloneGraphBFS(originalGraph);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        Node* cloned3 = solution.cloneGraphIterativeDFS(originalGraph);
        end = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        Node* cloned4 = solution.cloneGraphTwoPass(originalGraph);
        end = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        Node* cloned5 = solution.cloneGraphValueKey(originalGraph);
        end = std::chrono::high_resolution_clock::now();
        auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "DFS Recursive: " << duration1.count() << " μs" << std::endl;
        std::cout << "BFS: " << duration2.count() << " μs" << std::endl;
        std::cout << "DFS Iterative: " << duration3.count() << " μs" << std::endl;
        std::cout << "Two-pass: " << duration4.count() << " μs" << std::endl;
        std::cout << "Value-key: " << duration5.count() << " μs" << std::endl;
        
        // Verify all approaches produce correct results
        bool all_correct = GraphUtils::compareGraphs(originalGraph, cloned1) &&
                          GraphUtils::compareGraphs(originalGraph, cloned2) &&
                          GraphUtils::compareGraphs(originalGraph, cloned3) &&
                          GraphUtils::compareGraphs(originalGraph, cloned4) &&
                          GraphUtils::compareGraphs(originalGraph, cloned5);
        
        std::cout << "All approaches correct: " << (all_correct ? "✅" : "❌") << std::endl;
        
        // Cleanup
        GraphUtils::deleteGraph(originalGraph);
        GraphUtils::deleteGraph(cloned1);
        GraphUtils::deleteGraph(cloned2);
        GraphUtils::deleteGraph(cloned3);
        GraphUtils::deleteGraph(cloned4);
        GraphUtils::deleteGraph(cloned5);
    }
    
    static void metaSpecificScenarios() {
        std::cout << "\nMeta-Specific Scenarios:" << std::endl;
        std::cout << "=======================" << std::endl;
        
        Solution solution;
        
        // Scenario 1: Social network graph
        std::cout << "Social network user connections:" << std::endl;
        std::vector<std::vector<int>> socialNet = {{2,3,4},{1,3},{1,2,4,5},{1,3},{3}};
        Node* socialGraph = GraphUtils::createGraph(socialNet);
        Node* clonedSocial = solution.cloneGraph(socialGraph);
        
        GraphUtils::printGraph(socialGraph, "Original Social Network");
        GraphUtils::printGraph(clonedSocial, "Cloned Social Network");
        std::cout << "Social network cloned correctly: " << (GraphUtils::compareGraphs(socialGraph, clonedSocial) ? "✅" : "❌") << std::endl;
        
        // Scenario 2: Page dependency graph
        std::cout << "\nPage dependency graph:" << std::endl;
        std::vector<std::vector<int>> pageGraph = {{2},{3},{4,5},{},{3}};
        Node* pages = GraphUtils::createGraph(pageGraph);
        Node* clonedPages = solution.cloneGraph(pages);
        
        GraphUtils::printGraph(pages, "Original Page Dependencies");
        GraphUtils::printGraph(clonedPages, "Cloned Page Dependencies");
        std::cout << "Page dependencies cloned: " << (GraphUtils::compareGraphs(pages, clonedPages) ? "✅" : "❌") << std::endl;
        
        // Scenario 3: Friend recommendation graph
        std::cout << "\nFriend recommendation network:" << std::endl;
        std::vector<std::vector<int>> friendNet = {{2,3},{1,4},{1,4},{2,3}};
        Node* friends = GraphUtils::createGraph(friendNet);
        Node* clonedFriends = solution.cloneGraph(friends);
        
        GraphUtils::printGraph(friends, "Original Friend Network");
        GraphUtils::printGraph(clonedFriends, "Cloned Friend Network");
        std::cout << "Friend network cloned: " << (GraphUtils::compareGraphs(friends, clonedFriends) ? "✅" : "❌") << std::endl;
        
        // Cleanup
        GraphUtils::deleteGraph(socialGraph);
        GraphUtils::deleteGraph(clonedSocial);
        GraphUtils::deleteGraph(pages);
        GraphUtils::deleteGraph(clonedPages);
        GraphUtils::deleteGraph(friends);
        GraphUtils::deleteGraph(clonedFriends);
    }
    
    static void performanceAnalysis() {
        std::cout << "\nPerformance Analysis:" << std::endl;
        std::cout << "====================" << std::endl;
        
        Solution solution;
        
        // Test with increasing graph sizes
        std::vector<int> sizes = {10, 50, 100, 500};
        
        for (int size : sizes) {
            Node* testGraph = createLargeGraph(size);
            
            auto start = std::chrono::high_resolution_clock::now();
            Node* cloned = solution.cloneGraph(testGraph);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            std::cout << "Graph size " << size << ": " << duration.count() << " μs" << std::endl;
            
            GraphUtils::deleteGraph(testGraph);
            GraphUtils::deleteGraph(cloned);
        }
        
        // Memory usage analysis
        std::cout << "\nMemory Usage Analysis:" << std::endl;
        std::cout << "=====================" << std::endl;
        
        std::cout << "Space complexity: O(V) for hash map storage" << std::endl;
        std::cout << "Additional space: O(V) for recursion stack (DFS)" << std::endl;
        std::cout << "Clone storage: O(V + E) for new graph structure" << std::endl;
        
        // Estimate memory for different sizes
        for (int v = 100; v <= 10000; v *= 10) {
            int e = v * 3; // Assume average degree of 3
            size_t mapMemory = v * (sizeof(void*) * 2); // Hash map overhead
            size_t graphMemory = v * sizeof(Node) + e * sizeof(Node*); // Graph storage
            
            std::cout << "V=" << v << ", E=" << e << ": ~" << (mapMemory + graphMemory) / 1024 << " KB" << std::endl;
        }
        
        // Test deep recursion limits
        std::cout << "\nRecursion Depth Test:" << std::endl;
        std::cout << "====================" << std::endl;
        
        for (int depth = 100; depth <= 1000; depth += 200) {
            Node* linearGraph = createLinearGraph(depth);
            
            auto start = std::chrono::high_resolution_clock::now();
            Node* cloned = solution.cloneGraph(linearGraph);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            std::cout << "Linear depth " << depth << ": " << duration.count() << " μs" << std::endl;
            
            GraphUtils::deleteGraph(linearGraph);
            GraphUtils::deleteGraph(cloned);
        }
    }
    
    static Node* createLargeGraph(int size) {
        std::vector<Node*> nodes(size);
        
        // Create nodes
        for (int i = 0; i < size; i++) {
            nodes[i] = new Node(i + 1);
        }
        
        // Create connections (each node connects to next few nodes)
        for (int i = 0; i < size; i++) {
            for (int j = 1; j <= std::min(3, size - i - 1); j++) {
                nodes[i]->neighbors.push_back(nodes[i + j]);
                nodes[i + j]->neighbors.push_back(nodes[i]); // Undirected
            }
        }
        
        return nodes[0];
    }
    
    static Node* createLinearGraph(int length) {
        if (length == 0) return nullptr;
        
        std::vector<Node*> nodes(length);
        
        // Create nodes
        for (int i = 0; i < length; i++) {
            nodes[i] = new Node(i + 1);
        }
        
        // Create linear connections
        for (int i = 0; i < length - 1; i++) {
            nodes[i]->neighbors.push_back(nodes[i + 1]);
            nodes[i + 1]->neighbors.push_back(nodes[i]);
        }
        
        return nodes[0];
    }
};

int main() {
    CloneGraphTest::runTests();
    return 0;
}

/*
Algorithm Analysis:

Core Problem: Create a deep copy of an undirected graph with object references

Key Insights:
1. Need to avoid infinite loops in cyclic graphs
2. Must track already cloned nodes to maintain references
3. Each original node maps to exactly one cloned node
4. Preserve the exact graph structure and connectivity

Approach Comparison:

1. DFS with HashMap (Recommended):
   - Time: O(V + E) for visiting each node and edge once
   - Space: O(V) for recursion stack and hash map
   - Natural recursive implementation
   - Handles cycles automatically with memoization

2. BFS with HashMap:
   - Time: O(V + E) same complexity
   - Space: O(V) for queue and hash map
   - Iterative, avoids recursion depth issues
   - Level-by-level processing

3. Iterative DFS with Stack:
   - Time: O(V + E) same complexity
   - Space: O(V) for explicit stack and hash map
   - Combines DFS traversal with iterative approach
   - Good compromise between recursion and BFS

4. Two-Pass Approach:
   - First pass creates all nodes
   - Second pass connects neighbors
   - Separates concerns clearly
   - May be easier to understand

5. Value-Key Optimization:
   - Uses node value as hash key instead of pointer
   - Only works when values are unique
   - Slight memory optimization
   - More fragile to input assumptions

Meta Interview Focus:
- Graph traversal algorithms (DFS/BFS)
- Deep copying and object references
- Cycle detection and handling
- HashMap usage for memoization
- Memory management considerations

Key Design Decisions:
1. DFS vs BFS for graph traversal
2. Recursive vs iterative implementation
3. When to create vs when to reference cloned nodes
4. How to handle self-loops and cycles

Real-world Applications at Meta:
- Social network graph duplication
- Page dependency copying
- Friend recommendation graph backup
- Content relationship preservation
- User connection analysis

Edge Cases:
- Empty graph (null input)
- Single node with no neighbors
- Single node with self-loop
- Complete graph (all nodes connected)
- Linear chain graph

Interview Tips:
1. Start with DFS + HashMap approach
2. Explain cycle handling with visited tracking
3. Discuss when to create new nodes
4. Consider memory management
5. Verify deep copy properties

Common Mistakes:
1. Creating multiple copies of same node
2. Not handling cycles properly
3. Shallow copy instead of deep copy
4. Memory leaks in cleanup
5. Wrong reference connections

Advanced Optimizations:
- Object pooling for memory efficiency
- Parallel traversal for large graphs
- Streaming copy for memory-constrained environments
- Compressed representation for sparse graphs
- Incremental copying for dynamic graphs

Testing Strategy:
- Basic connected graphs
- Edge cases (empty, single node, self-loops)
- Complex structures (cycles, complete graphs)
- Performance with large graphs
- Memory usage validation

Production Considerations:
- Thread safety for concurrent access
- Memory limits for large graphs
- Error handling and validation
- Progress reporting for long operations
- Cancellation support

Complexity Analysis:
- Time: O(V + E) to visit each node and edge once
- Space: O(V) for hash map and recursion/queue
- Best case: O(V) for tree-like graphs
- Worst case: O(V + E) for dense graphs

This problem is important for Meta because:
1. Fundamental graph operation
2. Tests understanding of object references
3. Real applications in social networks
4. Demonstrates cycle handling skills
5. Shows memory management awareness

Common Interview Variations:
1. Clone directed graph
2. Clone graph with weighted edges
3. Clone graph with additional node properties
4. Return mapping from original to cloned nodes
5. Clone subgraph starting from multiple nodes

Graph Types and Handling:
- Tree: Simple DFS, no cycles
- DAG: DFS with visited set
- Cyclic: HashMap for cycle detection
- Complete: Dense adjacency handling
- Sparse: Efficient neighbor iteration

Performance Characteristics:
- Small graphs (< 100 nodes): < 1ms
- Medium graphs (< 1000 nodes): < 10ms
- Large graphs (< 10K nodes): < 100ms
- Memory scales linearly with graph size
- DFS may have stack overflow for very deep graphs

Real-world Usage:
- Social networks: Friend connection copying
- Web crawling: Site structure duplication
- Game development: Map/level cloning
- Database: Relationship graph backup
- AI/ML: Neural network structure copying

Memory Management:
- Automatic cleanup with smart pointers
- Manual cleanup requires careful traversal
- Object pooling for frequent operations
- Lazy copying for large graphs
- Reference counting for shared ownership
*/
