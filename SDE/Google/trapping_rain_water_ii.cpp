/*
Google SDE Interview Problem 9: Trapping Rain Water II (Hard)
Given an m x n integer matrix heightMap representing the height of each unit cell 
in a 2D elevation map, return the volume of water it can trap after raining.

Example:
Input: heightMap = [[1,4,3,1,3,2],[3,2,1,3,2,4],[2,3,3,2,3,1]]
Output: 4
Explanation: After the rain, water is trapped between the blocks.

The key insight is that water level at any cell is determined by the minimum height 
of the path to the boundary (like water flowing out).

Time Complexity: O(mn * log(mn)) where m,n are dimensions
Space Complexity: O(mn) for priority queue and visited array
*/

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

class Solution {
public:
    // Approach 1: Priority Queue with Dijkstra-like Algorithm (Optimal)
    int trapRainWater(std::vector<std::vector<int>>& heightMap) {
        if (heightMap.empty() || heightMap[0].empty()) {
            return 0;
        }
        
        int m = heightMap.size();
        int n = heightMap[0].size();
        
        // Priority queue to process cells from lowest to highest
        // pair<height, pair<row, col>>
        std::priority_queue<std::vector<int>, std::vector<std::vector<int>>, std::greater<std::vector<int>>> pq;
        
        std::vector<std::vector<bool>> visited(m, std::vector<bool>(n, false));
        
        // Add all boundary cells to priority queue
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    pq.push({heightMap[i][j], i, j});
                    visited[i][j] = true;
                }
            }
        }
        
        int waterTrapped = 0;
        std::vector<std::vector<int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        
        while (!pq.empty()) {
            auto current = pq.top();
            pq.pop();
            
            int height = current[0];
            int x = current[1];
            int y = current[2];
            
            // Check all four neighbors
            for (const auto& dir : directions) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny]) {
                    visited[nx][ny] = true;
                    
                    // Water level is at least the height of current cell
                    int waterLevel = std::max(height, heightMap[nx][ny]);
                    waterTrapped += waterLevel - heightMap[nx][ny];
                    
                    pq.push({waterLevel, nx, ny});
                }
            }
        }
        
        return waterTrapped;
    }
    
    // Approach 2: Modified Dijkstra with explicit water level tracking
    int trapRainWaterExplicit(std::vector<std::vector<int>>& heightMap) {
        if (heightMap.empty() || heightMap[0].empty()) return 0;
        
        int m = heightMap.size();
        int n = heightMap[0].size();
        
        // Track the minimum water level that can reach each cell
        std::vector<std::vector<int>> waterLevel(m, std::vector<int>(n, INT_MAX));
        std::priority_queue<std::vector<int>, std::vector<std::vector<int>>, std::greater<std::vector<int>>> pq;
        
        // Initialize boundary cells
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    waterLevel[i][j] = heightMap[i][j];
                    pq.push({heightMap[i][j], i, j});
                }
            }
        }
        
        std::vector<std::vector<int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        
        while (!pq.empty()) {
            auto current = pq.top();
            pq.pop();
            
            int level = current[0];
            int x = current[1];
            int y = current[2];
            
            if (level > waterLevel[x][y]) continue; // Already processed with better path
            
            for (const auto& dir : directions) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    int newLevel = std::max(level, heightMap[nx][ny]);
                    
                    if (newLevel < waterLevel[nx][ny]) {
                        waterLevel[nx][ny] = newLevel;
                        pq.push({newLevel, nx, ny});
                    }
                }
            }
        }
        
        int totalWater = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                totalWater += waterLevel[i][j] - heightMap[i][j];
            }
        }
        
        return totalWater;
    }
    
    // Approach 3: Union-Find based solution
    int trapRainWaterUnionFind(std::vector<std::vector<int>>& heightMap) {
        if (heightMap.empty() || heightMap[0].empty()) return 0;
        
        int m = heightMap.size();
        int n = heightMap[0].size();
        
        // Create sorted list of all cells
        std::vector<std::tuple<int, int, int>> cells; // height, row, col
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                cells.push_back({heightMap[i][j], i, j});
            }
        }
        
        std::sort(cells.begin(), cells.end());
        
        // Union-Find structure
        std::vector<int> parent(m * n);
        std::vector<int> minHeight(m * n);
        std::vector<bool> processed(m * n, false);
        
        for (int i = 0; i < m * n; i++) {
            parent[i] = i;
            minHeight[i] = INT_MAX;
        }
        
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
                parent[rootX] = rootY;
                minHeight[rootY] = std::min(minHeight[rootX], minHeight[rootY]);
            }
        };
        
        auto getId = [&](int r, int c) -> int {
            return r * n + c;
        };
        
        // Mark boundary cells
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    int id = getId(i, j);
                    minHeight[id] = heightMap[i][j];
                }
            }
        }
        
        std::vector<std::vector<int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int totalWater = 0;
        
        for (const auto& cell : cells) {
            int height = std::get<0>(cell);
            int x = std::get<1>(cell);
            int y = std::get<2>(cell);
            int id = getId(x, y);
            
            processed[id] = true;
            
            // Check neighbors and union if processed
            for (const auto& dir : directions) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    int neighborId = getId(nx, ny);
                    if (processed[neighborId]) {
                        unite(id, neighborId);
                    }
                }
            }
            
            int root = find(id);
            if (minHeight[root] > height) {
                minHeight[root] = height;
            }
            
            totalWater += std::max(0, minHeight[root] - height);
        }
        
        return totalWater;
    }
    
    // Approach 4: BFS with level-by-level processing
    int trapRainWaterBFS(std::vector<std::vector<int>>& heightMap) {
        if (heightMap.empty() || heightMap[0].empty()) return 0;
        
        int m = heightMap.size();
        int n = heightMap[0].size();
        
        std::vector<std::vector<int>> waterLevel(m, std::vector<int>(n, INT_MAX));
        std::queue<std::pair<int, int>> q;
        
        // Initialize boundary
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    waterLevel[i][j] = heightMap[i][j];
                    q.push({i, j});
                }
            }
        }
        
        std::vector<std::vector<int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        
        while (!q.empty()) {
            auto [x, y] = q.front();
            q.pop();
            
            for (const auto& dir : directions) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    int newLevel = std::max(waterLevel[x][y], heightMap[nx][ny]);
                    if (newLevel < waterLevel[nx][ny]) {
                        waterLevel[nx][ny] = newLevel;
                        q.push({nx, ny});
                    }
                }
            }
        }
        
        int totalWater = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                totalWater += waterLevel[i][j] - heightMap[i][j];
            }
        }
        
        return totalWater;
    }
};

// Comprehensive test framework
class RainWater2DTest {
public:
    static void runTests() {
        Solution solution;
        
        std::cout << "Google Trapping Rain Water II Tests:" << std::endl;
        std::cout << "====================================" << std::endl;
        
        testBasicCases(solution);
        testEdgeCases(solution);
        compareApproaches(solution);
        performanceTest(solution);
        visualTest(solution);
    }
    
    static void testBasicCases(Solution& solution) {
        std::cout << "\nBasic Test Cases:" << std::endl;
        std::cout << "=================" << std::endl;
        
        // Test case 1: Example from problem
        std::vector<std::vector<int>> heightMap1 = {
            {1,4,3,1,3,2},
            {3,2,1,3,2,4},
            {2,3,3,2,3,1}
        };
        int expected1 = 4;
        int result1 = solution.trapRainWater(heightMap1);
        std::cout << "Test 1: " << result1 << " (Expected: " << expected1 << ") " 
                  << (result1 == expected1 ? "✅" : "❌") << std::endl;
        
        // Test case 2: Simple bowl
        std::vector<std::vector<int>> heightMap2 = {
            {3,3,3},
            {3,1,3},
            {3,3,3}
        };
        int expected2 = 2;
        int result2 = solution.trapRainWater(heightMap2);
        std::cout << "Test 2: " << result2 << " (Expected: " << expected2 << ") " 
                  << (result2 == expected2 ? "✅" : "❌") << std::endl;
        
        // Test case 3: No water trapped
        std::vector<std::vector<int>> heightMap3 = {
            {1,2,3},
            {4,5,6},
            {7,8,9}
        };
        int expected3 = 0;
        int result3 = solution.trapRainWater(heightMap3);
        std::cout << "Test 3: " << result3 << " (Expected: " << expected3 << ") " 
                  << (result3 == expected3 ? "✅" : "❌") << std::endl;
        
        // Test case 4: Single cell
        std::vector<std::vector<int>> heightMap4 = {{5}};
        int expected4 = 0;
        int result4 = solution.trapRainWater(heightMap4);
        std::cout << "Test 4: " << result4 << " (Expected: " << expected4 << ") " 
                  << (result4 == expected4 ? "✅" : "❌") << std::endl;
        
        // Test case 5: Complex landscape
        std::vector<std::vector<int>> heightMap5 = {
            {5,5,5,5},
            {5,1,1,5},
            {5,1,5,5},
            {5,1,1,1}
        };
        int expected5 = 3;
        int result5 = solution.trapRainWater(heightMap5);
        std::cout << "Test 5: " << result5 << " (Expected: " << expected5 << ") " 
                  << (result5 == expected5 ? "✅" : "❌") << std::endl;
    }
    
    static void testEdgeCases(Solution& solution) {
        std::cout << "\nEdge Case Tests:" << std::endl;
        std::cout << "================" << std::endl;
        
        // Empty matrix
        std::vector<std::vector<int>> empty;
        int result1 = solution.trapRainWater(empty);
        std::cout << "Empty matrix: " << result1 << " (Expected: 0) " 
                  << (result1 == 0 ? "✅" : "❌") << std::endl;
        
        // Single row
        std::vector<std::vector<int>> singleRow = {{1,2,1,3,1}};
        int result2 = solution.trapRainWater(singleRow);
        std::cout << "Single row: " << result2 << " (Expected: 0) " 
                  << (result2 == 0 ? "✅" : "❌") << std::endl;
        
        // Single column
        std::vector<std::vector<int>> singleCol = {{1},{2},{1},{3},{1}};
        int result3 = solution.trapRainWater(singleCol);
        std::cout << "Single column: " << result3 << " (Expected: 0) " 
                  << (result3 == 0 ? "✅" : "❌") << std::endl;
        
        // All same height
        std::vector<std::vector<int>> sameHeight = {
            {2,2,2},
            {2,2,2},
            {2,2,2}
        };
        int result4 = solution.trapRainWater(sameHeight);
        std::cout << "Same height: " << result4 << " (Expected: 0) " 
                  << (result4 == 0 ? "✅" : "❌") << std::endl;
        
        // Deep pit
        std::vector<std::vector<int>> deepPit = {
            {10,10,10},
            {10,1,10},
            {10,10,10}
        };
        int result5 = solution.trapRainWater(deepPit);
        std::cout << "Deep pit: " << result5 << " (Expected: 9) " 
                  << (result5 == 9 ? "✅" : "❌") << std::endl;
    }
    
    static void compareApproaches(Solution& solution) {
        std::cout << "\nApproach Comparison:" << std::endl;
        std::cout << "===================" << std::endl;
        
        std::vector<std::vector<int>> testMap = {
            {1,4,3,1,3,2},
            {3,2,1,3,2,4},
            {2,3,3,2,3,1}
        };
        
        auto start = std::chrono::high_resolution_clock::now();
        int result1 = solution.trapRainWater(testMap);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        int result2 = solution.trapRainWaterExplicit(testMap);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        int result3 = solution.trapRainWaterBFS(testMap);
        end = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Priority Queue: " << result1 << " (" << duration1.count() << " μs)" << std::endl;
        std::cout << "Explicit Dijkstra: " << result2 << " (" << duration2.count() << " μs)" << std::endl;
        std::cout << "BFS: " << result3 << " (" << duration3.count() << " μs)" << std::endl;
        
        bool allMatch = (result1 == result2 && result2 == result3);
        std::cout << "All results match: " << (allMatch ? "✅" : "❌") << std::endl;
    }
    
    static void performanceTest(Solution& solution) {
        std::cout << "\nPerformance Test:" << std::endl;
        std::cout << "=================" << std::endl;
        
        // Generate large random height map
        int size = 100;
        std::vector<std::vector<int>> largeMap(size, std::vector<int>(size));
        
        srand(42); // Fixed seed for reproducible results
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                largeMap[i][j] = rand() % 100 + 1;
            }
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        int result = solution.trapRainWater(largeMap);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Large map (100x100): " << result << " units trapped" << std::endl;
        std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
        
        // Test with extreme values
        std::vector<std::vector<int>> extremeMap(50, std::vector<int>(50, 1000));
        for (int i = 10; i < 40; i++) {
            for (int j = 10; j < 40; j++) {
                extremeMap[i][j] = 1;
            }
        }
        
        start = std::chrono::high_resolution_clock::now();
        result = solution.trapRainWater(extremeMap);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Extreme height difference: " << result << " units trapped" << std::endl;
        std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
    }
    
    static void visualTest(Solution& solution) {
        std::cout << "\nVisual Test:" << std::endl;
        std::cout << "============" << std::endl;
        
        std::vector<std::vector<int>> heightMap = {
            {5,5,5,5,5},
            {5,1,1,1,5},
            {5,1,0,1,5},
            {5,1,1,1,5},
            {5,5,5,5,5}
        };
        
        std::cout << "Original height map:" << std::endl;
        printMatrix(heightMap);
        
        int waterTrapped = solution.trapRainWater(heightMap);
        std::cout << "Water trapped: " << waterTrapped << " units" << std::endl;
        
        // Calculate water levels
        std::vector<std::vector<int>> waterLevels = calculateWaterLevels(heightMap);
        std::cout << "Water levels:" << std::endl;
        printMatrix(waterLevels);
    }
    
    static void printMatrix(const std::vector<std::vector<int>>& matrix) {
        for (const auto& row : matrix) {
            for (int val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    static std::vector<std::vector<int>> calculateWaterLevels(const std::vector<std::vector<int>>& heightMap) {
        // This is a simplified version to show water levels
        Solution solution;
        int m = heightMap.size();
        int n = heightMap[0].size();
        
        std::vector<std::vector<int>> waterLevel(m, std::vector<int>(n, INT_MAX));
        std::priority_queue<std::vector<int>, std::vector<std::vector<int>>, std::greater<std::vector<int>>> pq;
        
        // Initialize boundary
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    waterLevel[i][j] = heightMap[i][j];
                    pq.push({heightMap[i][j], i, j});
                }
            }
        }
        
        std::vector<std::vector<int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        
        while (!pq.empty()) {
            auto current = pq.top();
            pq.pop();
            
            int level = current[0];
            int x = current[1];
            int y = current[2];
            
            if (level > waterLevel[x][y]) continue;
            
            for (const auto& dir : directions) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    int newLevel = std::max(level, heightMap[nx][ny]);
                    if (newLevel < waterLevel[nx][ny]) {
                        waterLevel[nx][ny] = newLevel;
                        pq.push({newLevel, nx, ny});
                    }
                }
            }
        }
        
        return waterLevel;
    }
};

int main() {
    RainWater2DTest::runTests();
    return 0;
}

/*
Algorithm Analysis:

Core Algorithm: Modified Dijkstra's Algorithm with Priority Queue

Key Insight: Water flows to the boundary, so work inward from boundaries using minimum heights

Time Complexity: O(mn * log(mn))
- Each cell is processed once
- Priority queue operations are O(log(mn))
- Total cells: mn

Space Complexity: O(mn)
- Priority queue can contain up to mn elements
- Visited array: mn booleans
- Additional space for directions

Algorithm Steps:
1. Add all boundary cells to priority queue with their heights
2. Process cells from lowest to highest height
3. For each cell, check unvisited neighbors
4. Water level = max(current_height, neighbor_ground_height)
5. Trapped water = water_level - ground_height

Google Interview Focus:
- 2D extension of classic 1D problem
- Priority queue usage for graph algorithms
- Dijkstra's algorithm modification
- Boundary condition handling
- Space-time complexity analysis

Key Optimizations:
1. Use priority queue for optimal processing order
2. Mark visited cells to avoid reprocessing
3. Process from boundaries inward (minimum water escape path)
4. Early termination when possible

Alternative Approaches:
1. Union-Find: Group cells by connectivity and water levels
2. BFS: Level-by-level processing (less optimal)
3. Simulation: Iterative water level raising
4. Dynamic Programming: 4-directional min-path calculation

Real-world Applications:
- Watershed analysis in geography
- Flood simulation and prediction
- 3D terrain water drainage
- Urban planning for water management
- Game development for realistic water physics

Edge Cases:
- Empty or single-cell grids
- All cells at same height
- No water can be trapped
- Very deep pits
- Linear arrangements

Interview Tips:
1. Start with 1D version understanding
2. Explain boundary cell initialization
3. Discuss why priority queue is necessary
4. Handle edge cases explicitly
5. Analyze time/space complexity

Common Mistakes:
1. Not initializing boundary cells correctly
2. Wrong priority queue ordering
3. Forgetting to mark cells as visited
4. Incorrect water level calculation
5. Not handling single-row/column cases

Advanced Optimizations:
- Memory pool for priority queue nodes
- Parallel processing for independent regions
- Approximation algorithms for very large grids
- GPU acceleration for massive terrains
- Incremental updates for dynamic terrain

Testing Strategy:
- Basic functionality with known results
- Edge cases and boundary conditions
- Performance with large inputs
- Visual verification of water levels
- Cross-validation between approaches

Complexity Analysis:
- Best case: O(mn) when all cells are at boundary level
- Average case: O(mn log(mn)) for typical terrain
- Worst case: O(mn log(mn)) when processing order matters most
- Space: Always O(mn) regardless of input characteristics
*/
