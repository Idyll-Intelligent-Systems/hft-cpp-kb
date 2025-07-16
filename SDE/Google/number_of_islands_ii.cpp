/*
Google SDE Interview Problem 5: Number of Islands II (Hard)
You are given an empty 2D binary grid which represents a 2D map where '1's represent land and '0's represent water. 
Initially all the grids are water.

Operation addLand(row, col) turns the water at position (row, col) into a land.

Return a list of integers representing the number of islands after each addLand operation. 
An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.

Example:
Input: m = 3, n = 3, positions = [[0,0],[0,1],[1,2],[2,1]]
Output: [1,1,2,3]
Explanation:
Initially, the 2d grid is filled with water.
0 0 0
0 0 0
0 0 0

Operation #1: addLand(0, 0) turns the water at grid[0][0] into a land.
1 0 0
0 0 0   Number of islands = 1
0 0 0

Operation #2: addLand(0, 1) turns the water at grid[0][1] into a land.
1 1 0
0 0 0   Number of islands = 1
0 0 0

Time Complexity: O(k * α(mn)) where k is operations, α is inverse Ackermann function
Space Complexity: O(mn) for Union-Find structure
*/

#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>

class UnionFind {
private:
    std::vector<int> parent;
    std::vector<int> rank;
    int count;
    
public:
    UnionFind(int n) : parent(n), rank(n, 0), count(0) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]); // Path compression
        }
        return parent[x];
    }
    
    void unite(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX != rootY) {
            // Union by rank
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
            count--;
        }
    }
    
    void addComponent() {
        count++;
    }
    
    int getCount() const {
        return count;
    }
};

class Solution {
public:
    std::vector<int> numIslands2(int m, int n, std::vector<std::vector<int>>& positions) {
        std::vector<int> result;
        UnionFind uf(m * n);
        std::vector<std::vector<bool>> isLand(m, std::vector<bool>(n, false));
        
        std::vector<std::pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        
        for (const auto& pos : positions) {
            int row = pos[0], col = pos[1];
            
            // Skip if already land
            if (isLand[row][col]) {
                result.push_back(uf.getCount());
                continue;
            }
            
            // Add new island
            isLand[row][col] = true;
            uf.addComponent();
            
            int currentId = row * n + col;
            
            // Check all four directions
            for (const auto& dir : directions) {
                int newRow = row + dir.first;
                int newCol = col + dir.second;
                
                if (newRow >= 0 && newRow < m && newCol >= 0 && newCol < n && isLand[newRow][newCol]) {
                    int neighborId = newRow * n + newCol;
                    uf.unite(currentId, neighborId);
                }
            }
            
            result.push_back(uf.getCount());
        }
        
        return result;
    }
    
    // Alternative approach with explicit island tracking
    std::vector<int> numIslands2Explicit(int m, int n, std::vector<std::vector<int>>& positions) {
        std::vector<int> result;
        std::vector<std::vector<int>> grid(m, std::vector<int>(n, 0));
        std::unordered_map<int, int> islandToRoot;
        std::vector<int> parent;
        int islandCount = 0;
        
        auto find = [&](int x) -> int {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        };
        
        auto unite = [&](int x, int y) -> bool {
            int rootX = find(x);
            int rootY = find(y);
            if (rootX != rootY) {
                parent[rootX] = rootY;
                islandCount--;
                return true;
            }
            return false;
        };
        
        for (const auto& pos : positions) {
            int row = pos[0], col = pos[1];
            
            if (grid[row][col] != 0) {
                result.push_back(islandCount);
                continue;
            }
            
            int islandId = parent.size();
            parent.push_back(islandId);
            grid[row][col] = islandId;
            islandCount++;
            
            // Check neighbors
            std::vector<std::pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
            for (const auto& dir : directions) {
                int newRow = row + dir.first;
                int newCol = col + dir.second;
                
                if (newRow >= 0 && newRow < m && newCol >= 0 && newCol < n && grid[newRow][newCol] != 0) {
                    unite(islandId, grid[newRow][newCol]);
                }
            }
            
            result.push_back(islandCount);
        }
        
        return result;
    }
    
    // Memory optimized version
    std::vector<int> numIslands2Optimized(int m, int n, std::vector<std::vector<int>>& positions) {
        std::vector<int> result;
        std::unordered_set<int> lands;
        std::vector<int> parent;
        std::vector<int> rank;
        int count = 0;
        
        auto find = [&](int x) -> int {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        };
        
        auto unite = [&](int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
            
            if (rootX != rootY) {
                if (rank[rootX] < rank[rootY]) {
                    parent[rootX] = rootY;
                } else if (rank[rootX] > rank[rootY]) {
                    parent[rootY] = rootX;
                } else {
                    parent[rootY] = rootX;
                    rank[rootX]++;
                }
                count--;
            }
        };
        
        for (const auto& pos : positions) {
            int row = pos[0], col = pos[1];
            int id = row * n + col;
            
            if (lands.count(id)) {
                result.push_back(count);
                continue;
            }
            
            lands.insert(id);
            
            // Initialize union-find for new land
            if (id >= parent.size()) {
                parent.resize(id + 1);
                rank.resize(id + 1, 0);
                for (int i = parent.size() - (id + 1 - parent.size()); i <= id; i++) {
                    parent[i] = i;
                }
            } else {
                parent[id] = id;
                rank[id] = 0;
            }
            
            count++;
            
            // Check four directions
            std::vector<std::pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
            for (const auto& dir : directions) {
                int newRow = row + dir.first;
                int newCol = col + dir.second;
                int neighborId = newRow * n + newCol;
                
                if (newRow >= 0 && newRow < m && newCol >= 0 && newCol < n && lands.count(neighborId)) {
                    unite(id, neighborId);
                }
            }
            
            result.push_back(count);
        }
        
        return result;
    }
};

// Comprehensive testing framework
class Islands2Test {
public:
    static void runTests() {
        Solution solution;
        
        std::cout << "Google Number of Islands II Tests:" << std::endl;
        std::cout << "==================================" << std::endl;
        
        // Test case 1: Basic example
        std::vector<std::vector<int>> positions1 = {{0,0},{0,1},{1,2},{2,1}};
        auto result1 = solution.numIslands2(3, 3, positions1);
        std::cout << "Test 1: ";
        printVector(result1);
        std::cout << " (Expected: [1,1,2,3])" << std::endl;
        
        // Test case 2: All positions same
        std::vector<std::vector<int>> positions2 = {{0,0},{0,0},{0,0}};
        auto result2 = solution.numIslands2(1, 1, positions2);
        std::cout << "Test 2: ";
        printVector(result2);
        std::cout << " (Expected: [1,1,1])" << std::endl;
        
        // Test case 3: Forming single large island
        std::vector<std::vector<int>> positions3 = {{0,0},{0,1},{0,2},{1,0},{1,1},{1,2}};
        auto result3 = solution.numIslands2(2, 3, positions3);
        std::cout << "Test 3: ";
        printVector(result3);
        std::cout << " (Expected: [1,1,1,2,1,1])" << std::endl;
        
        // Test case 4: No merging
        std::vector<std::vector<int>> positions4 = {{0,0},{2,2},{1,1},{0,2},{2,0}};
        auto result4 = solution.numIslands2(3, 3, positions4);
        std::cout << "Test 4: ";
        printVector(result4);
        std::cout << " (Expected: [1,2,3,4,5])" << std::endl;
        
        // Performance test
        performanceTest();
    }
    
    static void printVector(const std::vector<int>& vec) {
        std::cout << "[";
        for (int i = 0; i < vec.size(); i++) {
            std::cout << vec[i];
            if (i < vec.size() - 1) std::cout << ",";
        }
        std::cout << "]";
    }
    
    static void performanceTest() {
        std::cout << "\nPerformance Test:" << std::endl;
        std::cout << "=================" << std::endl;
        
        Solution solution;
        
        // Generate large test case
        int m = 1000, n = 1000;
        std::vector<std::vector<int>> largePositions;
        
        // Create positions that will form a grid pattern
        for (int i = 0; i < 1000; i++) {
            largePositions.push_back({i % m, (i * 7) % n});
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = solution.numIslands2(m, n, largePositions);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Processed " << largePositions.size() << " operations on " 
                  << m << "x" << n << " grid in " << duration.count() << " ms" << std::endl;
        std::cout << "Final island count: " << result.back() << std::endl;
        
        // Test optimized version
        start = std::chrono::high_resolution_clock::now();
        auto resultOpt = solution.numIslands2Optimized(m, n, largePositions);
        end = std::chrono::high_resolution_clock::now();
        auto durationOpt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Optimized version: " << durationOpt.count() << " ms" << std::endl;
        std::cout << "Results match: " << (result == resultOpt ? "Yes" : "No") << std::endl;
    }
    
    // Stress testing with edge cases
    static void stressTest() {
        Solution solution;
        
        std::cout << "\nStress Test:" << std::endl;
        std::cout << "============" << std::endl;
        
        // Test 1: Single row
        std::vector<std::vector<int>> singleRow = {{0,0},{0,1},{0,2},{0,3},{0,4}};
        auto result1 = solution.numIslands2(1, 5, singleRow);
        std::cout << "Single row: ";
        printVector(result1);
        std::cout << std::endl;
        
        // Test 2: Single column
        std::vector<std::vector<int>> singleCol = {{0,0},{1,0},{2,0},{3,0},{4,0}};
        auto result2 = solution.numIslands2(5, 1, singleCol);
        std::cout << "Single column: ";
        printVector(result2);
        std::cout << std::endl;
        
        // Test 3: Diagonal pattern
        std::vector<std::vector<int>> diagonal = {{0,0},{1,1},{2,2},{3,3}};
        auto result3 = solution.numIslands2(4, 4, diagonal);
        std::cout << "Diagonal: ";
        printVector(result3);
        std::cout << std::endl;
        
        // Test 4: Reverse diagonal
        std::vector<std::vector<int>> revDiagonal = {{0,3},{1,2},{2,1},{3,0}};
        auto result4 = solution.numIslands2(4, 4, revDiagonal);
        std::cout << "Reverse diagonal: ";
        printVector(result4);
        std::cout << std::endl;
    }
};

int main() {
    Islands2Test::runTests();
    Islands2Test::stressTest();
    return 0;
}

/*
Algorithm Analysis:

Core Algorithm: Union-Find (Disjoint Set Union)
1. Each land cell becomes a component
2. Union adjacent land cells when new land is added
3. Track total number of connected components

Key Optimizations:
1. Path Compression: Flatten tree structure during find()
2. Union by Rank: Attach smaller tree under larger tree
3. Coordinate Compression: Map 2D coordinates to 1D

Time Complexity:
- Per operation: O(α(mn)) where α is inverse Ackermann function
- Total: O(k * α(mn)) for k operations
- Practically constant time per operation

Space Complexity: O(mn) for Union-Find structure

Google Interview Focus:
- Union-Find algorithm understanding
- Path compression and union by rank optimizations
- Handling dynamic connectivity problems
- Memory usage optimization
- Edge case handling

Key Insights:
1. Use Union-Find for dynamic connectivity
2. Convert 2D coordinates to 1D for efficiency
3. Only check existing land neighbors
4. Handle duplicate positions gracefully

Edge Cases:
- Duplicate positions
- Single cell grid
- All positions forming one island
- No merging (all separate islands)
- Large sparse grids

Interview Tips:
1. Explain Union-Find concept clearly
2. Discuss optimization techniques
3. Handle coordinate conversion carefully
4. Consider memory vs time tradeoffs
5. Test with various patterns

Alternative Approaches:
- DFS/BFS for each operation (less efficient)
- Explicit graph representation
- Segment tree for 2D range queries
*/
