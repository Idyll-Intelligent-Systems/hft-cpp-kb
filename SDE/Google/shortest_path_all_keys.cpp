/*
Google SDE Interview Problem 10: Shortest Path to Get All Keys (Hard)
You are given an m x n grid grid where:
- '.' is an empty cell
- '#' is a wall
- '@' is the starting point
- Lowercase letters represent keys
- Uppercase letters represent locks

We start at the starting point and want to reach any cell that has all the keys.
Return the length of the shortest such path. If it's impossible, return -1.

Example:
Input: grid = ["@.a.#","###.#","b.A.B"]
Output: 8
Explanation: Starting from @, we need to collect keys 'a' and 'b', then reach any cell.

This is a classic BFS problem with state compression using bitmasks to track collected keys.

Time Complexity: O(mn * 2^k) where m,n are grid dimensions, k is number of keys
Space Complexity: O(mn * 2^k) for the visited state tracking
*/

#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <unordered_set>

class Solution {
public:
    // Approach 1: BFS with Bitmask State Compression (Optimal)
    int shortestPathAllKeys(std::vector<std::string>& grid) {
        if (grid.empty() || grid[0].empty()) {
            return -1;
        }
        
        int m = grid.size();
        int n = grid[0].size();
        
        // Find starting position and count total keys
        int startX = -1, startY = -1;
        int totalKeys = 0;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '@') {
                    startX = i;
                    startY = j;
                } else if (grid[i][j] >= 'a' && grid[i][j] <= 'f') {
                    totalKeys++;
                }
            }
        }
        
        if (totalKeys == 0) return 0; // No keys to collect
        
        // BFS with state: (row, col, keys_bitmask)
        // State encoding: state = row * n * (1<<totalKeys) + col * (1<<totalKeys) + keys
        std::queue<std::tuple<int, int, int, int>> q; // x, y, keys, steps
        std::unordered_set<int> visited;
        
        q.push({startX, startY, 0, 0});
        visited.insert(encodeState(startX, startY, 0, n, totalKeys));
        
        std::vector<std::vector<int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int allKeysMask = (1 << totalKeys) - 1;
        
        while (!q.empty()) {
            auto [x, y, keys, steps] = q.front();
            q.pop();
            
            // Check if we have all keys
            if (keys == allKeysMask) {
                return steps;
            }
            
            // Explore all four directions
            for (const auto& dir : directions) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                
                if (nx < 0 || nx >= m || ny < 0 || ny >= n || grid[nx][ny] == '#') {
                    continue;
                }
                
                char cell = grid[nx][ny];
                int newKeys = keys;
                
                // Check if it's a key
                if (cell >= 'a' && cell <= 'f') {
                    int keyIndex = cell - 'a';
                    newKeys |= (1 << keyIndex);
                }
                // Check if it's a lock
                else if (cell >= 'A' && cell <= 'F') {
                    int lockIndex = cell - 'A';
                    if (!(keys & (1 << lockIndex))) {
                        continue; // Don't have the key for this lock
                    }
                }
                
                int newState = encodeState(nx, ny, newKeys, n, totalKeys);
                if (visited.find(newState) == visited.end()) {
                    visited.insert(newState);
                    q.push({nx, ny, newKeys, steps + 1});
                }
            }
        }
        
        return -1; // Impossible to collect all keys
    }
    
    // Approach 2: BFS with 3D visited array (more memory but cleaner)
    int shortestPathAllKeysArray(std::vector<std::string>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        
        int startX = -1, startY = -1, totalKeys = 0;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '@') {
                    startX = i;
                    startY = j;
                } else if (grid[i][j] >= 'a' && grid[i][j] <= 'f') {
                    totalKeys++;
                }
            }
        }
        
        if (totalKeys == 0) return 0;
        
        // 3D visited array: visited[x][y][keyMask]
        std::vector<std::vector<std::vector<bool>>> visited(
            m, std::vector<std::vector<bool>>(n, std::vector<bool>(1 << totalKeys, false))
        );
        
        std::queue<std::tuple<int, int, int, int>> q;
        q.push({startX, startY, 0, 0});
        visited[startX][startY][0] = true;
        
        std::vector<std::vector<int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int allKeysMask = (1 << totalKeys) - 1;
        
        while (!q.empty()) {
            auto [x, y, keys, steps] = q.front();
            q.pop();
            
            if (keys == allKeysMask) {
                return steps;
            }
            
            for (const auto& dir : directions) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                
                if (nx < 0 || nx >= m || ny < 0 || ny >= n || grid[nx][ny] == '#') {
                    continue;
                }
                
                char cell = grid[nx][ny];
                int newKeys = keys;
                
                if (cell >= 'a' && cell <= 'f') {
                    newKeys |= (1 << (cell - 'a'));
                } else if (cell >= 'A' && cell <= 'F') {
                    if (!(keys & (1 << (cell - 'A')))) {
                        continue;
                    }
                }
                
                if (!visited[nx][ny][newKeys]) {
                    visited[nx][ny][newKeys] = true;
                    q.push({nx, ny, newKeys, steps + 1});
                }
            }
        }
        
        return -1;
    }
    
    // Approach 3: A* with Heuristic (for optimization)
    int shortestPathAllKeysAStar(std::vector<std::string>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        
        int startX = -1, startY = -1, totalKeys = 0;
        std::vector<std::pair<int, int>> keyPositions;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '@') {
                    startX = i;
                    startY = j;
                } else if (grid[i][j] >= 'a' && grid[i][j] <= 'f') {
                    keyPositions.push_back({i, j});
                    totalKeys++;
                }
            }
        }
        
        if (totalKeys == 0) return 0;
        
        // Priority queue for A*: (f_score, steps, x, y, keys)
        auto cmp = [](const std::tuple<int, int, int, int, int>& a, 
                     const std::tuple<int, int, int, int, int>& b) {
            return std::get<0>(a) > std::get<0>(b);
        };
        
        std::priority_queue<std::tuple<int, int, int, int, int>, 
                           std::vector<std::tuple<int, int, int, int, int>>, 
                           decltype(cmp)> pq(cmp);
        
        std::unordered_set<int> visited;
        
        int heuristic = calculateHeuristic(startX, startY, 0, keyPositions);
        pq.push({heuristic, 0, startX, startY, 0});
        visited.insert(encodeState(startX, startY, 0, n, totalKeys));
        
        std::vector<std::vector<int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int allKeysMask = (1 << totalKeys) - 1;
        
        while (!pq.empty()) {
            auto [f_score, steps, x, y, keys] = pq.top();
            pq.pop();
            
            if (keys == allKeysMask) {
                return steps;
            }
            
            for (const auto& dir : directions) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                
                if (nx < 0 || nx >= m || ny < 0 || ny >= n || grid[nx][ny] == '#') {
                    continue;
                }
                
                char cell = grid[nx][ny];
                int newKeys = keys;
                
                if (cell >= 'a' && cell <= 'f') {
                    newKeys |= (1 << (cell - 'a'));
                } else if (cell >= 'A' && cell <= 'F') {
                    if (!(keys & (1 << (cell - 'A')))) {
                        continue;
                    }
                }
                
                int newState = encodeState(nx, ny, newKeys, n, totalKeys);
                if (visited.find(newState) == visited.end()) {
                    visited.insert(newState);
                    int h = calculateHeuristic(nx, ny, newKeys, keyPositions);
                    pq.push({steps + 1 + h, steps + 1, nx, ny, newKeys});
                }
            }
        }
        
        return -1;
    }
    
    // Approach 4: Bidirectional BFS (advanced optimization)
    int shortestPathAllKeysBidirectional(std::vector<std::string>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        
        int startX = -1, startY = -1, totalKeys = 0;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '@') {
                    startX = i;
                    startY = j;
                } else if (grid[i][j] >= 'a' && grid[i][j] <= 'f') {
                    totalKeys++;
                }
            }
        }
        
        if (totalKeys == 0) return 0;
        
        // This is complex for this problem since we don't have a fixed target
        // Instead, we'll use the standard BFS approach
        return shortestPathAllKeys(grid);
    }
    
private:
    int encodeState(int x, int y, int keys, int n, int totalKeys) {
        return x * n * (1 << totalKeys) + y * (1 << totalKeys) + keys;
    }
    
    int calculateHeuristic(int x, int y, int keys, const std::vector<std::pair<int, int>>& keyPositions) {
        int minDist = 0;
        
        // Find minimum distance to any uncollected key
        for (int i = 0; i < keyPositions.size(); i++) {
            if (!(keys & (1 << i))) {
                int dist = abs(x - keyPositions[i].first) + abs(y - keyPositions[i].second);
                if (minDist == 0 || dist < minDist) {
                    minDist = dist;
                }
            }
        }
        
        return minDist;
    }
};

// Comprehensive test framework
class ShortestPathKeysTest {
public:
    static void runTests() {
        Solution solution;
        
        std::cout << "Google Shortest Path All Keys Tests:" << std::endl;
        std::cout << "====================================" << std::endl;
        
        testBasicCases(solution);
        testEdgeCases(solution);
        compareApproaches(solution);
        performanceTest(solution);
        complexScenarioTest(solution);
    }
    
    static void testBasicCases(Solution& solution) {
        std::cout << "\nBasic Test Cases:" << std::endl;
        std::cout << "=================" << std::endl;
        
        // Test case 1: Example from problem
        std::vector<std::string> grid1 = {"@.a.#","###.#","b.A.B"};
        int expected1 = 8;
        int result1 = solution.shortestPathAllKeys(grid1);
        std::cout << "Test 1: " << result1 << " (Expected: " << expected1 << ") " 
                  << (result1 == expected1 ? "✅" : "❌") << std::endl;
        
        // Test case 2: Simple case
        std::vector<std::string> grid2 = {"@..aA","..B#.","....b"};
        int expected2 = 6;
        int result2 = solution.shortestPathAllKeys(grid2);
        std::cout << "Test 2: " << result2 << " (Expected: " << expected2 << ") " 
                  << (result2 == expected2 ? "✅" : "❌") << std::endl;
        
        // Test case 3: No keys
        std::vector<std::string> grid3 = {"@..."};
        int expected3 = 0;
        int result3 = solution.shortestPathAllKeys(grid3);
        std::cout << "Test 3: " << result3 << " (Expected: " << expected3 << ") " 
                  << (result3 == expected3 ? "✅" : "❌") << std::endl;
        
        // Test case 4: Impossible
        std::vector<std::string> grid4 = {"@.a","###","..A"};
        int expected4 = -1;
        int result4 = solution.shortestPathAllKeys(grid4);
        std::cout << "Test 4: " << result4 << " (Expected: " << expected4 << ") " 
                  << (result4 == expected4 ? "✅" : "❌") << std::endl;
        
        // Test case 5: Single key
        std::vector<std::string> grid5 = {"@.a"};
        int expected5 = 2;
        int result5 = solution.shortestPathAllKeys(grid5);
        std::cout << "Test 5: " << result5 << " (Expected: " << expected5 << ") " 
                  << (result5 == expected5 ? "✅" : "❌") << std::endl;
    }
    
    static void testEdgeCases(Solution& solution) {
        std::cout << "\nEdge Case Tests:" << std::endl;
        std::cout << "================" << std::endl;
        
        // Key and lock at same position (conceptually)
        std::vector<std::string> edge1 = {"@aA"};
        int result1 = solution.shortestPathAllKeys(edge1);
        std::cout << "Key and lock inline: " << result1 << std::endl;
        
        // Multiple paths to same key
        std::vector<std::string> edge2 = {"@.a.","....","....","...a"};
        int result2 = solution.shortestPathAllKeys(edge2);
        std::cout << "Multiple paths: " << result2 << std::endl;
        
        // Complex lock dependencies
        std::vector<std::string> edge3 = {"@abcdef","ABCDEF","......","......"};
        int result3 = solution.shortestPathAllKeys(edge3);
        std::cout << "Complex dependencies: " << result3 << std::endl;
        
        // Maze-like structure
        std::vector<std::string> edge4 = {
            "@.#a",
            "###.",
            "b#.A",
            "...B"
        };
        int result4 = solution.shortestPathAllKeys(edge4);
        std::cout << "Maze structure: " << result4 << std::endl;
    }
    
    static void compareApproaches(Solution& solution) {
        std::cout << "\nApproach Comparison:" << std::endl;
        std::cout << "===================" << std::endl;
        
        std::vector<std::string> testGrid = {"@.a.#","###.#","b.A.B"};
        
        auto start = std::chrono::high_resolution_clock::now();
        int result1 = solution.shortestPathAllKeys(testGrid);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        int result2 = solution.shortestPathAllKeysArray(testGrid);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        int result3 = solution.shortestPathAllKeysAStar(testGrid);
        end = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Hash set approach: " << result1 << " (" << duration1.count() << " μs)" << std::endl;
        std::cout << "3D array approach: " << result2 << " (" << duration2.count() << " μs)" << std::endl;
        std::cout << "A* approach: " << result3 << " (" << duration3.count() << " μs)" << std::endl;
        
        bool allMatch = (result1 == result2 && result2 == result3);
        std::cout << "All results match: " << (allMatch ? "✅" : "❌") << std::endl;
    }
    
    static void performanceTest(Solution& solution) {
        std::cout << "\nPerformance Test:" << std::endl;
        std::cout << "=================" << std::endl;
        
        // Generate larger grid with multiple keys
        std::vector<std::string> largeGrid = {
            "@......a",
            "........",
            "........",
            "b.......",
            "........",
            "........",
            "c.......",
            "ABC....."
        };
        
        auto start = std::chrono::high_resolution_clock::now();
        int result = solution.shortestPathAllKeys(largeGrid);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Large grid result: " << result << std::endl;
        std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
        
        // Test with maximum keys (6 keys)
        std::vector<std::string> maxKeysGrid = {
            "@abcdef",
            ".......",
            "ABCDEF.",
            "......."
        };
        
        start = std::chrono::high_resolution_clock::now();
        result = solution.shortestPathAllKeys(maxKeysGrid);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Max keys (6) result: " << result << std::endl;
        std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
    }
    
    static void complexScenarioTest(Solution& solution) {
        std::cout << "\nComplex Scenario Tests:" << std::endl;
        std::cout << "=======================" << std::endl;
        
        // Scenario 1: Key needed to reach another key
        std::vector<std::string> scenario1 = {
            "@.A.b",
            ".....",
            "a...."
        };
        int result1 = solution.shortestPathAllKeys(scenario1);
        std::cout << "Key dependency scenario: " << result1 << std::endl;
        
        // Scenario 2: Backtracking required
        std::vector<std::string> scenario2 = {
            "@a.A.b",
            "#####.",
            "......"
        };
        int result2 = solution.shortestPathAllKeys(scenario2);
        std::cout << "Backtracking scenario: " << result2 << std::endl;
        
        // Scenario 3: Multiple valid end positions
        std::vector<std::string> scenario3 = {
            "@.....",
            ".a.b..",
            "......"
        };
        int result3 = solution.shortestPathAllKeys(scenario3);
        std::cout << "Multiple end positions: " << result3 << std::endl;
        
        // Scenario 4: Optimal vs suboptimal paths
        std::vector<std::string> scenario4 = {
            "@.....a",
            ".......",
            ".......",
            "b......"
        };
        int result4 = solution.shortestPathAllKeys(scenario4);
        std::cout << "Path optimization: " << result4 << std::endl;
    }
    
    static void stressTest(Solution& solution) {
        std::cout << "\nStress Test:" << std::endl;
        std::cout << "============" << std::endl;
        
        // Create a complex maze with all 6 keys
        std::vector<std::string> stressGrid = {
            "@..#..a#..b",
            "..###.###..",
            "..#.....#..",
            "A.#..c..#.B",
            "..#.....#..",
            "..###.###..",
            "d..#..C#..e",
            "...#...#...",
            "D..#...#..E",
            "...#...#...",
            "...#.f.#..F"
        };
        
        auto start = std::chrono::high_resolution_clock::now();
        int result = solution.shortestPathAllKeys(stressGrid);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Complex maze result: " << result << std::endl;
        std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
        std::cout << "State space size: " << stressGrid.size() * stressGrid[0].size() * (1 << 6) << std::endl;
    }
};

int main() {
    ShortestPathKeysTest::runTests();
    ShortestPathKeysTest::stressTest(Solution{});
    return 0;
}

/*
Algorithm Analysis:

Core Algorithm: BFS with State Compression using Bitmasks

Key Insight: State = (position, collected_keys_bitmask)
- Position: (row, col) coordinates
- Keys: Bitmask where bit i represents whether key i is collected

Time Complexity: O(mn * 2^k)
- m,n: Grid dimensions
- k: Number of keys (at most 6)
- Each state (position + key combination) visited once

Space Complexity: O(mn * 2^k)
- Visited states tracking
- BFS queue space

State Compression Technique:
- Use bitmask to represent collected keys
- Bit i = 1 means key i is collected
- Total states per position: 2^k

Google Interview Focus:
- BFS with state compression
- Bitmask manipulation
- Graph traversal with constraints
- Dynamic programming concepts
- Optimization techniques

Key Optimizations:
1. Bitmask for efficient key tracking
2. State encoding to reduce memory
3. Early termination when all keys collected
4. Pruning invalid moves (locked doors)

Alternative Approaches:
1. DFS with memoization (less optimal)
2. A* with heuristic (better for large grids)
3. Dijkstra's algorithm (overkill for unweighted)
4. Bidirectional BFS (complex for multiple targets)

Real-world Applications:
- Game pathfinding with inventory constraints
- Robot navigation with capability requirements
- Network routing with security clearances
- Puzzle game mechanics
- Access control systems

Edge Cases:
- No keys to collect
- Impossible to reach all keys
- Keys behind locks requiring other keys
- Multiple paths to same objective
- Starting position optimization

Interview Tips:
1. Identify state representation early
2. Explain bitmask usage clearly
3. Handle lock/key mechanics correctly
4. Discuss state space size
5. Consider optimization opportunities

Common Mistakes:
1. Wrong bitmask manipulation
2. Not handling locked doors properly
3. Incorrect state encoding/decoding
4. Missing boundary checks
5. Forgetting to mark states as visited

Advanced Optimizations:
- A* with admissible heuristic
- State space pruning
- Parallel BFS for independent regions
- Memory-efficient state representation
- Cache-aware traversal patterns

Complexity Analysis:
- Best case: O(mn) when keys are on direct path
- Average case: O(mn * 2^k) for typical mazes
- Worst case: O(mn * 2^k) when exploring full state space
- Space: Always O(mn * 2^k) for complete state tracking

Testing Strategy:
- Basic functionality with known solutions
- Edge cases and impossible scenarios
- Performance with maximum state space
- Complex maze navigation
- Stress testing with all features
*/
