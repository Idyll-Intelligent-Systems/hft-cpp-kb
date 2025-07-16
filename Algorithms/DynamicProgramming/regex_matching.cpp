/*
Problem: Regular Expression Matching (Hard)
Given an input string s and a pattern p, implement regular expression matching with 
support for '.' and '*' where:
- '.' Matches any single character
- '*' Matches zero or more of the preceding element

The matching should cover the entire input string (not partial).

Example:
Input: s = "aa", p = "a*"
Output: true (because '*' means zero or more of the preceding element 'a')

Input: s = "mississippi", p = "mis*is*p*."
Output: false

Time Complexity: O(m * n) where m = len(s), n = len(p)
Space Complexity: O(m * n) for the DP table
*/

#include <iostream>
#include <vector>
#include <string>

class Solution {
public:
    // Approach 1: Top-Down Dynamic Programming (Memoization)
    bool isMatch(const std::string& s, const std::string& p) {
        std::vector<std::vector<int>> memo(s.length() + 1, std::vector<int>(p.length() + 1, -1));
        return isMatchHelper(s, p, 0, 0, memo);
    }
    
private:
    bool isMatchHelper(const std::string& s, const std::string& p, int i, int j, 
                       std::vector<std::vector<int>>& memo) {
        // Base case: pattern is exhausted
        if (j == p.length()) {
            return i == s.length();
        }
        
        // Check memoization
        if (memo[i][j] != -1) {
            return memo[i][j] == 1;
        }
        
        bool result = false;
        
        // Check if current characters match
        bool firstMatch = (i < s.length()) && 
                         (p[j] == s[i] || p[j] == '.');
        
        // Handle '*' case
        if (j + 1 < p.length() && p[j + 1] == '*') {
            // Two options:
            // 1. Skip the pattern (0 occurrences of preceding character)
            // 2. Use the pattern if first character matches
            result = isMatchHelper(s, p, i, j + 2, memo) ||
                    (firstMatch && isMatchHelper(s, p, i + 1, j, memo));
        } else {
            // No '*', so characters must match and continue
            result = firstMatch && isMatchHelper(s, p, i + 1, j + 1, memo);
        }
        
        memo[i][j] = result ? 1 : 0;
        return result;
    }
    
public:
    // Approach 2: Bottom-Up Dynamic Programming
    bool isMatchBottomUp(const std::string& s, const std::string& p) {
        int m = s.length();
        int n = p.length();
        
        // dp[i][j] represents if s[0...i-1] matches p[0...j-1]
        std::vector<std::vector<bool>> dp(m + 1, std::vector<bool>(n + 1, false));
        
        // Base case: empty string matches empty pattern
        dp[0][0] = true;
        
        // Handle patterns like a*, a*b*, a*b*c* that can match empty string
        for (int j = 2; j <= n; j += 2) {
            if (p[j - 1] == '*') {
                dp[0][j] = dp[0][j - 2];
            }
        }
        
        // Fill the DP table
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                char sc = s[i - 1];  // Current character in string
                char pc = p[j - 1];  // Current character in pattern
                
                if (pc == '*') {
                    // Look at the character before '*'
                    char prevChar = p[j - 2];
                    
                    // Option 1: Skip the pattern (0 occurrences)
                    dp[i][j] = dp[i][j - 2];
                    
                    // Option 2: Use the pattern if characters match
                    if (prevChar == sc || prevChar == '.') {
                        dp[i][j] = dp[i][j] || dp[i - 1][j];
                    }
                } else {
                    // Regular character or '.'
                    if (pc == sc || pc == '.') {
                        dp[i][j] = dp[i - 1][j - 1];
                    }
                    // If characters don't match, dp[i][j] remains false
                }
            }
        }
        
        return dp[m][n];
    }
    
    // Approach 3: Space-Optimized Bottom-Up DP
    bool isMatchSpaceOptimized(const std::string& s, const std::string& p) {
        int m = s.length();
        int n = p.length();
        
        // Only need current and previous row
        std::vector<bool> prev(n + 1, false);
        std::vector<bool> curr(n + 1, false);
        
        // Base case
        prev[0] = true;
        
        // Handle patterns that can match empty string
        for (int j = 2; j <= n; j += 2) {
            if (p[j - 1] == '*') {
                prev[j] = prev[j - 2];
            }
        }
        
        for (int i = 1; i <= m; i++) {
            curr[0] = false;  // Non-empty string can't match empty pattern
            
            for (int j = 1; j <= n; j++) {
                char sc = s[i - 1];
                char pc = p[j - 1];
                
                if (pc == '*') {
                    char prevChar = p[j - 2];
                    curr[j] = curr[j - 2];  // Skip pattern
                    
                    if (prevChar == sc || prevChar == '.') {
                        curr[j] = curr[j] || prev[j];  // Use pattern
                    }
                } else {
                    if (pc == sc || pc == '.') {
                        curr[j] = prev[j - 1];
                    } else {
                        curr[j] = false;
                    }
                }
            }
            
            prev = curr;
        }
        
        return prev[n];
    }
};

// Test helper function
void testCase(Solution& solution, const std::string& s, const std::string& p, bool expected) {
    bool result1 = solution.isMatch(s, p);
    bool result2 = solution.isMatchBottomUp(s, p);
    bool result3 = solution.isMatchSpaceOptimized(s, p);
    
    std::cout << "s=\"" << s << "\", p=\"" << p << "\"" << std::endl;
    std::cout << "Expected: " << (expected ? "true" : "false") << std::endl;
    std::cout << "Memoization: " << (result1 ? "true" : "false") << std::endl;
    std::cout << "Bottom-up: " << (result2 ? "true" : "false") << std::endl;
    std::cout << "Space-optimized: " << (result3 ? "true" : "false") << std::endl;
    std::cout << "All methods agree: " << (result1 == result2 && result2 == result3 ? "Yes" : "No") << std::endl;
    std::cout << "Correct: " << (result1 == expected ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
}

int main() {
    Solution solution;
    
    std::cout << "Regular Expression Matching Test Cases:" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // Test case 1: Basic '*' usage
    testCase(solution, "aa", "a*", true);
    
    // Test case 2: '.' and '*' combination
    testCase(solution, "ab", ".*", true);
    
    // Test case 3: Complex pattern
    testCase(solution, "mississippi", "mis*is*p*.", false);
    
    // Test case 4: Empty string and pattern
    testCase(solution, "", "", true);
    
    // Test case 5: Empty string with pattern
    testCase(solution, "", "a*", true);
    
    // Test case 6: String with empty pattern
    testCase(solution, "a", "", false);
    
    // Test case 7: Single character match
    testCase(solution, "a", "a", true);
    
    // Test case 8: Single character mismatch
    testCase(solution, "a", "b", false);
    
    // Test case 9: '.' matching
    testCase(solution, "a", ".", true);
    
    // Test case 10: Complex case with multiple '*'
    testCase(solution, "aab", "c*a*b", true);
    
    // Test case 11: No match case
    testCase(solution, "ab", "a", false);
    
    // Test case 12: '*' at beginning
    testCase(solution, "bbbba", ".*a*a", true);
    
    return 0;
}

/*
Algorithm Analysis:

1. Top-Down DP (Memoization):
   - Time: O(m * n) where m = len(s), n = len(p)
   - Space: O(m * n) for memoization table + O(m + n) for recursion stack
   - Advantages: Intuitive recursive thinking, only computes needed subproblems
   - Good for: Understanding the recursive structure

2. Bottom-Up DP:
   - Time: O(m * n)
   - Space: O(m * n) for DP table
   - Advantages: No recursion overhead, clear iterative structure
   - Good for: Production code, easier to optimize

3. Space-Optimized DP:
   - Time: O(m * n)
   - Space: O(n) - only need current and previous row
   - Advantages: Minimal memory usage
   - Good for: Memory-constrained environments

Key Insights:

1. State Definition:
   - dp[i][j] = whether s[0...i-1] matches p[0...j-1]

2. Transition Cases:
   - Normal character: dp[i][j] = dp[i-1][j-1] if s[i-1] == p[j-1]
   - '.': dp[i][j] = dp[i-1][j-1] (always matches)
   - '*': Two options:
     a) Zero occurrences: dp[i][j] = dp[i][j-2]
     b) One or more: dp[i][j] = dp[i-1][j] if prev char matches

3. Base Cases:
   - Empty string matches empty pattern: dp[0][0] = true
   - Patterns like "a*b*c*" can match empty string

Edge Cases:
- Empty string and pattern
- Pattern with only '*' (invalid input)
- Consecutive '*' characters
- Pattern ending with '*'
- String longer/shorter than pattern

Common Mistakes:
1. Forgetting that '*' refers to the preceding character
2. Not handling empty string cases properly
3. Index confusion between 0-based and 1-based
4. Not considering both options for '*' (zero vs one+ occurrences)

Applications:
- Text editors (find and replace)
- Shell pattern matching
- Database query optimization
- Compiler lexical analysis
- Bioinformatics sequence matching

Interview Tips:
1. Start with recursive solution, then optimize
2. Draw small examples to understand transitions
3. Handle base cases carefully
4. Consider space optimization if asked
5. Test with edge cases
*/
