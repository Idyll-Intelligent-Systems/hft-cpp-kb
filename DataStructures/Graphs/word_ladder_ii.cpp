/*
Problem: Word Ladder II (Hard)
A transformation sequence from word beginWord to word endWord using a dictionary wordList 
is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:
- Every adjacent pair of words differs by exactly one letter
- Every si for 1 <= i <= k is in wordList
- sk == endWord

Return all the shortest transformation sequences from beginWord to endWord.

Example:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: [["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]

Time Complexity: O(N * M^2 * 26) for BFS + O(paths) for DFS
Space Complexity: O(N * M) for adjacency list and visited set
*/

#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <queue>

class Solution {
public:
    std::vector<std::vector<std::string>> findLadders(
        const std::string& beginWord, 
        const std::string& endWord, 
        std::vector<std::string>& wordList) {
        
        std::unordered_set<std::string> wordSet(wordList.begin(), wordList.end());
        std::vector<std::vector<std::string>> result;
        
        if (wordSet.find(endWord) == wordSet.end()) {
            return result;
        }
        
        // BFS to build adjacency graph and find shortest distance
        std::unordered_map<std::string, std::vector<std::string>> adjList;
        std::unordered_set<std::string> visited;
        std::queue<std::string> q;
        
        q.push(beginWord);
        visited.insert(beginWord);
        bool found = false;
        
        while (!q.empty() && !found) {
            int levelSize = q.size();
            std::unordered_set<std::string> currentLevelVisited;
            
            for (int i = 0; i < levelSize; i++) {
                std::string currentWord = q.front();
                q.pop();
                
                // Try all possible one-character changes
                for (int j = 0; j < currentWord.length(); j++) {
                    char originalChar = currentWord[j];
                    
                    for (char c = 'a'; c <= 'z'; c++) {
                        if (c == originalChar) continue;
                        
                        currentWord[j] = c;
                        
                        if (currentWord == endWord) {
                            found = true;
                            adjList[currentWord].push_back(endWord);
                        }
                        
                        if (wordSet.find(currentWord) != wordSet.end() && 
                            visited.find(currentWord) == visited.end()) {
                            
                            if (currentLevelVisited.find(currentWord) == currentLevelVisited.end()) {
                                currentLevelVisited.insert(currentWord);
                                q.push(currentWord);
                            }
                            
                            adjList[currentWord].push_back(endWord);
                        }
                    }
                    
                    currentWord[j] = originalChar; // Restore original character
                }
            }
            
            // Mark all words from current level as visited
            for (const std::string& word : currentLevelVisited) {
                visited.insert(word);
            }
        }
        
        // DFS to construct all shortest paths
        std::vector<std::string> path = {beginWord};
        dfs(beginWord, endWord, adjList, path, result);
        
        return result;
    }
    
private:
    void dfs(const std::string& currentWord, 
             const std::string& endWord,
             const std::unordered_map<std::string, std::vector<std::string>>& adjList,
             std::vector<std::string>& path,
             std::vector<std::vector<std::string>>& result) {
        
        if (currentWord == endWord) {
            result.push_back(path);
            return;
        }
        
        if (adjList.find(currentWord) == adjList.end()) {
            return;
        }
        
        for (const std::string& neighbor : adjList.at(currentWord)) {
            path.push_back(neighbor);
            dfs(neighbor, endWord, adjList, path, result);
            path.pop_back();
        }
    }
    
public:
    // Alternative approach using bidirectional BFS (more efficient)
    std::vector<std::vector<std::string>> findLaddersBidirectional(
        const std::string& beginWord, 
        const std::string& endWord, 
        std::vector<std::string>& wordList) {
        
        std::unordered_set<std::string> wordSet(wordList.begin(), wordList.end());
        if (wordSet.find(endWord) == wordSet.end()) {
            return {};
        }
        
        std::unordered_set<std::string> beginSet = {beginWord};
        std::unordered_set<std::string> endSet = {endWord};
        std::unordered_map<std::string, std::vector<std::string>> adjList;
        std::vector<std::vector<std::string>> result;
        
        if (buildAdjListBidirectional(beginSet, endSet, wordSet, adjList, false)) {
            std::vector<std::string> path = {beginWord};
            dfs(beginWord, endWord, adjList, path, result);
        }
        
        return result;
    }
    
private:
    bool buildAdjListBidirectional(
        std::unordered_set<std::string>& beginSet,
        std::unordered_set<std::string>& endSet,
        std::unordered_set<std::string>& wordSet,
        std::unordered_map<std::string, std::vector<std::string>>& adjList,
        bool reversed) {
        
        if (beginSet.empty()) return false;
        
        // Always search from the smaller set
        if (beginSet.size() > endSet.size()) {
            return buildAdjListBidirectional(endSet, beginSet, wordSet, adjList, !reversed);
        }
        
        // Remove words from wordSet to avoid cycles
        for (const std::string& word : beginSet) {
            wordSet.erase(word);
        }
        for (const std::string& word : endSet) {
            wordSet.erase(word);
        }
        
        std::unordered_set<std::string> nextSet;
        bool found = false;
        
        for (const std::string& word : beginSet) {
            std::string currentWord = word;
            
            for (int i = 0; i < currentWord.length(); i++) {
                char originalChar = currentWord[i];
                
                for (char c = 'a'; c <= 'z'; c++) {
                    if (c == originalChar) continue;
                    
                    currentWord[i] = c;
                    
                    if (endSet.find(currentWord) != endSet.end()) {
                        found = true;
                        if (!reversed) {
                            adjList[word].push_back(currentWord);
                        } else {
                            adjList[currentWord].push_back(word);
                        }
                    } else if (wordSet.find(currentWord) != wordSet.end()) {
                        nextSet.insert(currentWord);
                        if (!reversed) {
                            adjList[word].push_back(currentWord);
                        } else {
                            adjList[currentWord].push_back(word);
                        }
                    }
                }
                
                currentWord[i] = originalChar;
            }
        }
        
        return found || buildAdjListBidirectional(nextSet, endSet, wordSet, adjList, reversed);
    }
};

// Helper function to check if two words differ by exactly one character
bool isOneEditDistance(const std::string& word1, const std::string& word2) {
    if (word1.length() != word2.length()) return false;
    
    int diffCount = 0;
    for (int i = 0; i < word1.length(); i++) {
        if (word1[i] != word2[i]) {
            diffCount++;
            if (diffCount > 1) return false;
        }
    }
    
    return diffCount == 1;
}

// Helper function to print all paths
void printPaths(const std::vector<std::vector<std::string>>& paths) {
    for (const auto& path : paths) {
        for (int i = 0; i < path.size(); i++) {
            std::cout << path[i];
            if (i < path.size() - 1) std::cout << " -> ";
        }
        std::cout << std::endl;
    }
}

int main() {
    Solution solution;
    
    // Test case 1
    std::string beginWord1 = "hit";
    std::string endWord1 = "cog";
    std::vector<std::string> wordList1 = {"hot", "dot", "dog", "lot", "log", "cog"};
    
    std::cout << "Test 1:" << std::endl;
    std::cout << "Begin: " << beginWord1 << ", End: " << endWord1 << std::endl;
    std::cout << "Word List: ";
    for (const std::string& word : wordList1) {
        std::cout << word << " ";
    }
    std::cout << std::endl;
    
    auto result1 = solution.findLadders(beginWord1, endWord1, wordList1);
    std::cout << "All shortest paths:" << std::endl;
    printPaths(result1);
    std::cout << std::endl;
    
    // Test case 2: No path exists
    std::string beginWord2 = "hit";
    std::string endWord2 = "cog";
    std::vector<std::string> wordList2 = {"hot", "dot", "dog", "lot", "log"};
    
    std::cout << "Test 2 (No path):" << std::endl;
    std::cout << "Begin: " << beginWord2 << ", End: " << endWord2 << std::endl;
    
    auto result2 = solution.findLadders(beginWord2, endWord2, wordList2);
    std::cout << "Number of paths found: " << result2.size() << std::endl;
    std::cout << std::endl;
    
    // Test case 3: Single transformation
    std::string beginWord3 = "a";
    std::string endWord3 = "c";
    std::vector<std::string> wordList3 = {"a", "b", "c"};
    
    std::cout << "Test 3 (Single transformation):" << std::endl;
    auto result3 = solution.findLadders(beginWord3, endWord3, wordList3);
    std::cout << "All shortest paths:" << std::endl;
    printPaths(result3);
    
    return 0;
}

/*
Algorithm Analysis:

1. Standard BFS + DFS Approach:
   - BFS: Build adjacency list layer by layer to ensure shortest paths
   - DFS: Reconstruct all possible shortest paths using the adjacency list
   - Time: O(N * M^2 * 26) for BFS + O(paths) for DFS
   - Space: O(N * M) for adjacency list and data structures

2. Bidirectional BFS (Optimization):
   - Search from both ends simultaneously
   - Reduces search space significantly
   - Especially effective when branching factor is high
   - Time complexity remains similar but with better constants

Key Insights:
1. Use level-by-level BFS to ensure all paths found are shortest
2. Don't mark words as visited until the entire level is processed
3. Build adjacency list during BFS to guide DFS path reconstruction
4. Bidirectional search can dramatically reduce search space

Edge Cases:
- endWord not in wordList
- No transformation possible
- beginWord equals endWord
- Single character transformations
- Multiple paths of same length

Optimization Techniques:
1. Bidirectional BFS: Search from smaller frontier
2. Early termination when target is found
3. Level-wise processing to avoid longer paths
4. Efficient string comparison and character substitution

Applications:
- Word games and puzzles
- Shortest path problems in unweighted graphs
- State space search problems
- Genetic algorithm applications
*/
