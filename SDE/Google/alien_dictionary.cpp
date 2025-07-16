/*
Google SDE Interview Problem 1: Alien Dictionary (Hard)
There is a new alien language that uses the English alphabet. However, the order among the letters is unknown to you.
You are given a list of strings words from the alien language's dictionary, where the strings in words are sorted 
lexicographically by the rules of this new language.

Return a string of the unique letters in the new alien language sorted in lexicographically increasing order by the 
new language's rules. If there is no solution, return "". If there are multiple solutions, return any of them.

Example:
Input: words = ["wrt","wrf","er","ett","rftt"]
Output: "wertf"
Explanation: The order is "w" -> "e" -> "r" -> "t" -> "f"

Time Complexity: O(V + E) where V is number of unique characters, E is number of edges
Space Complexity: O(V + E) for adjacency list and data structures
*/

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>

class Solution {
public:
    std::string alienOrder(std::vector<std::string>& words) {
        // Build adjacency list and calculate in-degrees
        std::unordered_map<char, std::unordered_set<char>> graph;
        std::unordered_map<char, int> indegree;
        
        // Initialize all characters with indegree 0
        for (const std::string& word : words) {
            for (char c : word) {
                indegree[c] = 0;
            }
        }
        
        // Build graph by comparing adjacent words
        for (int i = 0; i < words.size() - 1; i++) {
            const std::string& word1 = words[i];
            const std::string& word2 = words[i + 1];
            
            // Check for invalid case: word1 is prefix of word2 but longer
            if (word1.length() > word2.length() && word1.substr(0, word2.length()) == word2) {
                return "";
            }
            
            // Find first differing character
            for (int j = 0; j < std::min(word1.length(), word2.length()); j++) {
                if (word1[j] != word2[j]) {
                    char from = word1[j];
                    char to = word2[j];
                    
                    // Only add edge if it doesn't exist (avoid duplicates)
                    if (graph[from].find(to) == graph[from].end()) {
                        graph[from].insert(to);
                        indegree[to]++;
                    }
                    break;
                }
            }
        }
        
        // Topological sort using Kahn's algorithm
        std::queue<char> queue;
        for (const auto& pair : indegree) {
            if (pair.second == 0) {
                queue.push(pair.first);
            }
        }
        
        std::string result;
        while (!queue.empty()) {
            char current = queue.front();
            queue.pop();
            result += current;
            
            // Process neighbors
            for (char neighbor : graph[current]) {
                indegree[neighbor]--;
                if (indegree[neighbor] == 0) {
                    queue.push(neighbor);
                }
            }
        }
        
        // Check if all characters are processed (no cycle)
        return result.length() == indegree.size() ? result : "";
    }
    
    // Alternative DFS approach
    std::string alienOrderDFS(std::vector<std::string>& words) {
        std::unordered_map<char, std::unordered_set<char>> graph;
        std::unordered_set<char> allChars;
        
        // Initialize
        for (const std::string& word : words) {
            for (char c : word) {
                allChars.insert(c);
            }
        }
        
        // Build graph
        for (int i = 0; i < words.size() - 1; i++) {
            const std::string& word1 = words[i];
            const std::string& word2 = words[i + 1];
            
            if (word1.length() > word2.length() && word1.substr(0, word2.length()) == word2) {
                return "";
            }
            
            for (int j = 0; j < std::min(word1.length(), word2.length()); j++) {
                if (word1[j] != word2[j]) {
                    graph[word1[j]].insert(word2[j]);
                    break;
                }
            }
        }
        
        // DFS with cycle detection
        std::unordered_map<char, int> state; // 0: unvisited, 1: visiting, 2: visited
        std::string result;
        
        for (char c : allChars) {
            if (state[c] == 0) {
                if (!dfs(c, graph, state, result)) {
                    return "";
                }
            }
        }
        
        std::reverse(result.begin(), result.end());
        return result;
    }
    
private:
    bool dfs(char node, std::unordered_map<char, std::unordered_set<char>>& graph,
             std::unordered_map<char, int>& state, std::string& result) {
        state[node] = 1; // Mark as visiting
        
        for (char neighbor : graph[node]) {
            if (state[neighbor] == 1) { // Back edge - cycle detected
                return false;
            }
            if (state[neighbor] == 0 && !dfs(neighbor, graph, state, result)) {
                return false;
            }
        }
        
        state[node] = 2; // Mark as visited
        result += node;
        return true;
    }
};

int main() {
    Solution solution;
    
    // Test case 1
    std::vector<std::string> words1 = {"wrt", "wrf", "er", "ett", "rftt"};
    std::cout << "Test 1: " << solution.alienOrder(words1) << " (Expected: wertf)" << std::endl;
    
    // Test case 2
    std::vector<std::string> words2 = {"z", "x"};
    std::cout << "Test 2: " << solution.alienOrder(words2) << " (Expected: zx)" << std::endl;
    
    // Test case 3 - Invalid case
    std::vector<std::string> words3 = {"abc", "ab"};
    std::cout << "Test 3: " << solution.alienOrder(words3) << " (Expected: empty)" << std::endl;
    
    // Test case 4 - Cycle
    std::vector<std::string> words4 = {"z", "x", "z"};
    std::cout << "Test 4: " << solution.alienOrder(words4) << " (Expected: empty)" << std::endl;
    
    return 0;
}

/*
Algorithm Analysis:

1. Graph Construction:
   - Compare adjacent words to find character ordering
   - Build directed graph where edge u->v means u comes before v
   - Handle edge cases (prefix validation)

2. Topological Sort:
   - Use Kahn's algorithm (BFS) or DFS
   - Detect cycles which indicate invalid ordering
   - Return empty string if cycle exists

Key Insights:
- Only compare adjacent words in the dictionary
- First differing character gives us ordering constraint
- Invalid if word1 is longer prefix of word2
- Result length should equal number of unique characters

Edge Cases:
- Empty input
- Single character words
- Prefix relationships
- Cycles in dependencies
- All characters same

Google Interview Tips:
1. Ask about duplicate characters in result
2. Clarify behavior for invalid inputs
3. Discuss both BFS and DFS approaches
4. Consider memory optimization
5. Handle Unicode vs ASCII characters
*/
