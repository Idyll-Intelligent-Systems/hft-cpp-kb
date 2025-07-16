/*
Meta (Facebook) SDE Interview Problem 6: Minimum Window Substring (Hard)
Given two strings s and t, return the minimum window substring of s such that every character in t 
(including duplicates) is included in the window. If there is no such window, return the empty string "".

The testcases will be generated such that the answer is unique.

Example 1:
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.

Example 2:
Input: s = "a", t = "a"
Output: "a"

Example 3:
Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.

This is a classic Meta interview problem testing sliding window technique, hash maps, and two-pointer approach.
It's fundamental for understanding optimization problems and character frequency tracking.

Time Complexity: O(|s| + |t|) where |s| and |t| are lengths of strings
Space Complexity: O(|s| + |t|) for hash maps storing character frequencies
*/

#include <iostream>
#include <string>
#include <unordered_map>
#include <climits>
#include <algorithm>
#include <chrono>
#include <vector>

class Solution {
public:
    // Approach 1: Sliding Window with Hash Map (Recommended)
    std::string minWindow(std::string s, std::string t) {
        if (s.empty() || t.empty() || s.length() < t.length()) {
            return "";
        }
        
        // Count characters in t
        std::unordered_map<char, int> target;
        for (char c : t) {
            target[c]++;
        }
        
        int required = target.size(); // Number of unique characters in t
        int formed = 0; // Number of unique characters matched with desired frequency
        
        std::unordered_map<char, int> window;
        int left = 0, right = 0;
        
        // Result tracking
        int minLen = INT_MAX;
        int minLeft = 0;
        
        while (right < s.length()) {
            // Expand window
            char rightChar = s[right];
            window[rightChar]++;
            
            // Check if current character's frequency matches target
            if (target.find(rightChar) != target.end() && 
                window[rightChar] == target[rightChar]) {
                formed++;
            }
            
            // Try to contract window
            while (left <= right && formed == required) {
                // Update result if current window is smaller
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    minLeft = left;
                }
                
                // Contract from left
                char leftChar = s[left];
                window[leftChar]--;
                
                if (target.find(leftChar) != target.end() && 
                    window[leftChar] < target[leftChar]) {
                    formed--;
                }
                
                left++;
            }
            
            right++;
        }
        
        return minLen == INT_MAX ? "" : s.substr(minLeft, minLen);
    }
    
    // Approach 2: Optimized sliding window with filtered string
    std::string minWindowOptimized(std::string s, std::string t) {
        if (s.empty() || t.empty()) return "";
        
        std::unordered_map<char, int> target;
        for (char c : t) {
            target[c]++;
        }
        
        // Create filtered string with only relevant characters
        std::vector<std::pair<int, char>> filtered;
        for (int i = 0; i < s.length(); i++) {
            if (target.find(s[i]) != target.end()) {
                filtered.push_back({i, s[i]});
            }
        }
        
        if (filtered.empty()) return "";
        
        int required = target.size();
        int formed = 0;
        std::unordered_map<char, int> window;
        
        int left = 0, right = 0;
        int minLen = INT_MAX;
        int minLeft = 0;
        
        while (right < filtered.size()) {
            char rightChar = filtered[right].second;
            window[rightChar]++;
            
            if (window[rightChar] == target[rightChar]) {
                formed++;
            }
            
            while (left <= right && formed == required) {
                int currentLen = filtered[right].first - filtered[left].first + 1;
                if (currentLen < minLen) {
                    minLen = currentLen;
                    minLeft = filtered[left].first;
                }
                
                char leftChar = filtered[left].second;
                window[leftChar]--;
                if (window[leftChar] < target[leftChar]) {
                    formed--;
                }
                left++;
            }
            
            right++;
        }
        
        return minLen == INT_MAX ? "" : s.substr(minLeft, minLen);
    }
    
    // Approach 3: Array-based counting (for ASCII characters)
    std::string minWindowArray(std::string s, std::string t) {
        if (s.empty() || t.empty()) return "";
        
        // Use arrays for ASCII characters
        int target[128] = {0};
        int window[128] = {0};
        
        int required = 0;
        for (char c : t) {
            if (target[c] == 0) required++;
            target[c]++;
        }
        
        int formed = 0;
        int left = 0, right = 0;
        int minLen = INT_MAX;
        int minLeft = 0;
        
        while (right < s.length()) {
            char rightChar = s[right];
            window[rightChar]++;
            
            if (target[rightChar] > 0 && window[rightChar] == target[rightChar]) {
                formed++;
            }
            
            while (formed == required) {
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    minLeft = left;
                }
                
                char leftChar = s[left];
                window[leftChar]--;
                
                if (target[leftChar] > 0 && window[leftChar] < target[leftChar]) {
                    formed--;
                }
                
                left++;
            }
            
            right++;
        }
        
        return minLen == INT_MAX ? "" : s.substr(minLeft, minLen);
    }
    
    // Approach 4: Two-pass approach for analysis
    std::string minWindowTwoPass(std::string s, std::string t) {
        if (s.empty() || t.empty()) return "";
        
        // First pass: find all possible windows
        std::vector<std::pair<int, int>> validWindows;
        findAllValidWindows(s, t, validWindows);
        
        if (validWindows.empty()) return "";
        
        // Second pass: find minimum window
        int minLen = INT_MAX;
        int bestStart = 0;
        
        for (const auto& window : validWindows) {
            int len = window.second - window.first + 1;
            if (len < minLen) {
                minLen = len;
                bestStart = window.first;
            }
        }
        
        return s.substr(bestStart, minLen);
    }
    
    // Approach 5: Template-based sliding window
    template<typename Container>
    std::string minWindowTemplate(const std::string& s, const std::string& t) {
        Container target, window;
        
        // Build target frequency map
        for (char c : t) {
            target[c]++;
        }
        
        int required = target.size();
        int formed = 0;
        int left = 0, right = 0;
        int minLen = INT_MAX;
        int minLeft = 0;
        
        while (right < s.length()) {
            char rightChar = s[right];
            window[rightChar]++;
            
            if (target.count(rightChar) && window[rightChar] == target[rightChar]) {
                formed++;
            }
            
            while (formed == required && left <= right) {
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    minLeft = left;
                }
                
                char leftChar = s[left];
                window[leftChar]--;
                
                if (target.count(leftChar) && window[leftChar] < target[leftChar]) {
                    formed--;
                }
                
                left++;
            }
            
            right++;
        }
        
        return minLen == INT_MAX ? "" : s.substr(minLeft, minLen);
    }
    
private:
    void findAllValidWindows(const std::string& s, const std::string& t, 
                            std::vector<std::pair<int, int>>& windows) {
        std::unordered_map<char, int> target;
        for (char c : t) {
            target[c]++;
        }
        
        for (int i = 0; i < s.length(); i++) {
            std::unordered_map<char, int> window;
            int required = target.size();
            int formed = 0;
            
            for (int j = i; j < s.length(); j++) {
                char c = s[j];
                window[c]++;
                
                if (target.count(c) && window[c] == target[c]) {
                    formed++;
                }
                
                if (formed == required) {
                    windows.push_back({i, j});
                    break;
                }
            }
        }
    }
};

// Test framework
class MinWindowSubstringTest {
public:
    static void runTests() {
        std::cout << "Meta Minimum Window Substring Tests:" << std::endl;
        std::cout << "===================================" << std::endl;
        
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
        
        // Test case 1: Example 1
        std::string result1 = solution.minWindow("ADOBECODEBANC", "ABC");
        std::cout << "Test 1 - s='ADOBECODEBANC', t='ABC': '" << result1 << "'" << std::endl;
        std::cout << "Expected: 'BANC'" << std::endl;
        std::cout << "Correct: " << (result1 == "BANC" ? "✅" : "❌") << std::endl;
        
        // Test case 2: Example 2
        std::string result2 = solution.minWindow("a", "a");
        std::cout << "\nTest 2 - s='a', t='a': '" << result2 << "'" << std::endl;
        std::cout << "Expected: 'a'" << std::endl;
        std::cout << "Correct: " << (result2 == "a" ? "✅" : "❌") << std::endl;
        
        // Test case 3: Example 3
        std::string result3 = solution.minWindow("a", "aa");
        std::cout << "\nTest 3 - s='a', t='aa': '" << result3 << "'" << std::endl;
        std::cout << "Expected: ''" << std::endl;
        std::cout << "Correct: " << (result3 == "" ? "✅" : "❌") << std::endl;
        
        // Test case 4: Multiple occurrences
        std::string result4 = solution.minWindow("ABAACBAB", "ABC");
        std::cout << "\nTest 4 - s='ABAACBAB', t='ABC': '" << result4 << "'" << std::endl;
        std::cout << "Expected: 'ACB' or similar valid window" << std::endl;
        
        // Test case 5: Duplicate characters in t
        std::string result5 = solution.minWindow("AABBCC", "ABC");
        std::cout << "\nTest 5 - s='AABBCC', t='ABC': '" << result5 << "'" << std::endl;
        std::cout << "Expected: Some valid window containing A, B, C" << std::endl;
    }
    
    static void testEdgeCases() {
        std::cout << "\nEdge Case Tests:" << std::endl;
        std::cout << "================" << std::endl;
        
        Solution solution;
        
        // Empty strings
        std::string result1 = solution.minWindow("", "a");
        std::cout << "Empty s: '" << result1 << "' (Expected: '')" << std::endl;
        
        std::string result2 = solution.minWindow("a", "");
        std::cout << "Empty t: '" << result2 << "' (Expected: '')" << std::endl;
        
        // s shorter than t
        std::string result3 = solution.minWindow("ab", "abc");
        std::cout << "s shorter than t: '" << result3 << "' (Expected: '')" << std::endl;
        
        // No valid window
        std::string result4 = solution.minWindow("abc", "def");
        std::cout << "No valid window: '" << result4 << "' (Expected: '')" << std::endl;
        
        // Entire string is minimum window
        std::string result5 = solution.minWindow("abc", "abc");
        std::cout << "Entire string: '" << result5 << "' (Expected: 'abc')" << std::endl;
        
        // Case sensitivity
        std::string result6 = solution.minWindow("Aa", "A");
        std::cout << "Case sensitive: '" << result6 << "' (Expected: 'A')" << std::endl;
        
        // Large duplicates
        std::string result7 = solution.minWindow("aaaaaaaaaa", "aa");
        std::cout << "Large duplicates: '" << result7 << "' (Expected: 'aa')" << std::endl;
    }
    
    static void compareApproaches() {
        std::cout << "\nApproach Comparison:" << std::endl;
        std::cout << "===================" << std::endl;
        
        Solution solution;
        std::string testS = "ADOBECODEBANC";
        std::string testT = "ABC";
        
        auto start = std::chrono::high_resolution_clock::now();
        std::string result1 = solution.minWindow(testS, testT);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        std::string result2 = solution.minWindowOptimized(testS, testT);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        std::string result3 = solution.minWindowArray(testS, testT);
        end = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        std::string result4 = solution.minWindowTwoPass(testS, testT);
        end = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        std::string result5 = solution.minWindowTemplate<std::unordered_map<char, int>>(testS, testT);
        end = std::chrono::high_resolution_clock::now();
        auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Standard sliding window: " << duration1.count() << " μs -> '" << result1 << "'" << std::endl;
        std::cout << "Optimized filtering: " << duration2.count() << " μs -> '" << result2 << "'" << std::endl;
        std::cout << "Array-based counting: " << duration3.count() << " μs -> '" << result3 << "'" << std::endl;
        std::cout << "Two-pass approach: " << duration4.count() << " μs -> '" << result4 << "'" << std::endl;
        std::cout << "Template approach: " << duration5.count() << " μs -> '" << result5 << "'" << std::endl;
        
        // Verify all approaches produce correct results
        bool isConsistent = (result1.length() == result2.length() && 
                           result2.length() == result3.length() && 
                           result3.length() == result4.length() && 
                           result4.length() == result5.length());
        
        std::cout << "All approaches consistent length: " << (isConsistent ? "✅" : "❌") << std::endl;
        
        // Verify all results are valid
        std::cout << "All results valid: " << (isValidWindow(testS, result1, testT) && 
                                             isValidWindow(testS, result2, testT) && 
                                             isValidWindow(testS, result3, testT) && 
                                             isValidWindow(testS, result4, testT) && 
                                             isValidWindow(testS, result5, testT) ? "✅" : "❌") << std::endl;
    }
    
    static void metaSpecificScenarios() {
        std::cout << "\nMeta-Specific Scenarios:" << std::endl;
        std::cout << "=======================" << std::endl;
        
        Solution solution;
        
        // Scenario 1: Search query matching
        std::cout << "Search query optimization:" << std::endl;
        std::string searchContent = "facebook meta social network platform";
        std::string queryTerms = "meta";
        std::string result1 = solution.minWindow(searchContent, queryTerms);
        std::cout << "Content: '" << searchContent << "'" << std::endl;
        std::cout << "Query: '" << queryTerms << "'" << std::endl;
        std::cout << "Minimum match: '" << result1 << "'" << std::endl;
        
        // Scenario 2: Ad keyword matching
        std::cout << "\nAd keyword matching:" << std::endl;
        std::string adContent = "buy shoes online best deals";
        std::string keywords = "shoe";
        std::string result2 = solution.minWindow(adContent, keywords);
        std::cout << "Ad content: '" << adContent << "'" << std::endl;
        std::cout << "Keywords: '" << keywords << "'" << std::endl;
        std::cout << "Match: '" << result2 << "'" << std::endl;
        
        // Scenario 3: Hashtag analysis
        std::cout << "\nHashtag content analysis:" << std::endl;
        std::string postContent = "love this amazing food photography";
        std::string hashtags = "photo";
        std::string result3 = solution.minWindow(postContent, hashtags);
        std::cout << "Post: '" << postContent << "'" << std::endl;
        std::cout << "Hashtag chars: '" << hashtags << "'" << std::endl;
        std::cout << "Relevant span: '" << result3 << "'" << std::endl;
        
        // Scenario 4: Content moderation
        std::cout << "\nContent moderation keyword detection:" << std::endl;
        std::string userContent = "this is a great product";
        std::string moderationKeywords = "great";
        std::string result4 = solution.minWindow(userContent, moderationKeywords);
        std::cout << "User content: '" << userContent << "'" << std::endl;
        std::cout << "Moderation pattern: '" << moderationKeywords << "'" << std::endl;
        std::cout << "Detection span: '" << result4 << "'" << std::endl;
        
        // Scenario 5: Username validation
        std::cout << "\nUsername character requirement:" << std::endl;
        std::string username = "john_doe_123";
        std::string requirements = "joe";
        std::string result5 = solution.minWindow(username, requirements);
        std::cout << "Username: '" << username << "'" << std::endl;
        std::cout << "Required chars: '" << requirements << "'" << std::endl;
        std::cout << "Minimum span: '" << result5 << "'" << std::endl;
    }
    
    static void performanceAnalysis() {
        std::cout << "\nPerformance Analysis:" << std::endl;
        std::cout << "====================" << std::endl;
        
        Solution solution;
        
        // Test with increasing string lengths
        std::vector<std::pair<int, int>> testSizes = {{100, 5}, {1000, 10}, {5000, 20}, {10000, 50}};
        
        for (const auto& sizes : testSizes) {
            int sLen = sizes.first;
            int tLen = sizes.second;
            
            std::string testS = generateTestString(sLen, "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
            std::string testT = generateTestString(tLen, "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
            
            auto start = std::chrono::high_resolution_clock::now();
            std::string result = solution.minWindow(testS, testT);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            std::cout << "s=" << sLen << ", t=" << tLen << ": " << duration.count() << " μs" 
                      << ", result_len=" << result.length() << std::endl;
        }
        
        // Complexity analysis
        std::cout << "\nComplexity Verification:" << std::endl;
        std::cout << "=======================" << std::endl;
        
        std::vector<int> stringLengths = {1000, 2000, 4000, 8000};
        std::vector<long long> timings;
        
        for (int len : stringLengths) {
            std::string testS = generateTestString(len, "ABCDEFGH");
            std::string testT = "ABCD";
            
            auto start = std::chrono::high_resolution_clock::now();
            solution.minWindow(testS, testT);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            timings.push_back(duration.count());
            std::cout << "n=" << len << ": " << duration.count() << " μs" << std::endl;
        }
        
        // Check linear scaling
        std::cout << "\nScaling ratios (should be ~2x for linear):" << std::endl;
        for (int i = 1; i < timings.size(); i++) {
            double ratio = (double)timings[i] / timings[i-1];
            std::cout << "n=" << stringLengths[i] << "/n=" << stringLengths[i-1] 
                      << " ratio: " << ratio << "x" << std::endl;
        }
        
        // Memory usage analysis
        std::cout << "\nMemory Usage Analysis:" << std::endl;
        std::cout << "=====================" << std::endl;
        
        std::cout << "Space complexity: O(|s| + |t|) for hash maps" << std::endl;
        std::cout << "Hash map overhead: ~32 bytes per entry" << std::endl;
        std::cout << "String storage: 1 byte per character" << std::endl;
        
        // Estimate memory for different sizes
        for (int len = 1000; len <= 100000; len *= 10) {
            size_t hashMapMemory = 256 * 32; // Worst case: all ASCII characters
            size_t stringMemory = len; // Input string
            size_t totalMemory = hashMapMemory + stringMemory;
            
            std::cout << "String length " << len << ": ~" << totalMemory / 1024 << " KB" << std::endl;
        }
    }
    
    static bool isValidWindow(const std::string& s, const std::string& window, const std::string& t) {
        if (window.empty() && !t.empty()) return false;
        
        std::unordered_map<char, int> targetCount, windowCount;
        
        for (char c : t) {
            targetCount[c]++;
        }
        
        for (char c : window) {
            windowCount[c]++;
        }
        
        for (const auto& pair : targetCount) {
            if (windowCount[pair.first] < pair.second) {
                return false;
            }
        }
        
        return true;
    }
    
    static std::string generateTestString(int length, const std::string& alphabet) {
        std::string result;
        result.reserve(length);
        
        for (int i = 0; i < length; i++) {
            result += alphabet[i % alphabet.length()];
        }
        
        return result;
    }
};

int main() {
    MinWindowSubstringTest::runTests();
    return 0;
}

/*
Algorithm Analysis:

Core Problem: Find minimum substring containing all characters from target string

Key Insights:
1. Sliding window technique with two pointers (left, right)
2. Expand window until all target characters are included
3. Contract window while maintaining validity to find minimum
4. Use hash maps to track character frequencies

Approach Comparison:

1. Standard Sliding Window (Recommended):
   - Time: O(|s| + |t|) - each character visited at most twice
   - Space: O(|s| + |t|) for hash maps
   - Two pointers: expand right, contract left
   - Optimal for general case

2. Optimized with Filtering:
   - Pre-filter string to only relevant characters
   - Reduces iterations for sparse target characters
   - Better performance when |t| << |s|
   - Same complexity but better constants

3. Array-based Counting:
   - Uses fixed arrays instead of hash maps
   - Faster for ASCII-only strings
   - O(1) character lookup time
   - Limited to fixed character sets

4. Two-pass Approach:
   - First pass finds all valid windows
   - Second pass finds minimum
   - Higher complexity O(|s|²) in worst case
   - Easier to understand but inefficient

5. Template-based:
   - Generic implementation for different containers
   - Allows switching between hash map and array
   - Same performance with compile-time optimization
   - Good for code reusability

Meta Interview Focus:
- Sliding window technique mastery
- Hash map usage for frequency tracking
- Two-pointer approach optimization
- String manipulation and substring operations
- Space-time complexity trade-offs

Key Design Decisions:
1. When to expand vs contract the window
2. How to track character frequency matches
3. Handling duplicate characters in target
4. Optimizing for different input characteristics

Real-world Applications at Meta:
- Search query matching and highlighting
- Content analysis and keyword extraction
- Ad targeting based on content keywords
- Hashtag and mention detection
- Spam detection pattern matching

Edge Cases:
- Empty strings (s or t)
- No valid window exists
- Entire string is minimum window
- Single character strings
- Case sensitivity handling

Interview Tips:
1. Start with sliding window approach
2. Explain frequency tracking clearly
3. Handle character counting correctly
4. Optimize window contraction logic
5. Consider edge cases thoroughly

Common Mistakes:
1. Wrong window expansion/contraction logic
2. Incorrect frequency matching conditions
3. Not handling duplicate characters
4. Poor boundary condition handling
5. Inefficient substring extraction

Advanced Optimizations:
- Character filtering for sparse targets
- Early termination when impossible
- Memory-efficient frequency tracking
- Parallel processing for multiple targets
- Streaming algorithms for large data

Testing Strategy:
- Basic functionality with known results
- Edge cases (empty, no solution, entire string)
- Performance with varying string sizes
- Memory usage validation
- Correctness verification

Production Considerations:
- Unicode and multi-byte character support
- Memory limits for large strings
- Timeout handling for long operations
- Input validation and sanitization
- Internationalization support

Complexity Analysis:
- Time: O(|s| + |t|) amortized
- Space: O(|s| + |t|) for frequency maps
- Best case: O(|s|) when immediate match
- Worst case: O(|s|) even with many retractions

This problem is important for Meta because:
1. Fundamental sliding window technique
2. Real applications in search and content analysis
3. Tests optimization and algorithm design skills
4. Common pattern in string processing
5. Demonstrates hash map usage mastery

Common Interview Variations:
1. Return all minimum windows
2. Case-insensitive matching
3. Find window with at most k distinct characters
4. Minimum window with character order preserved
5. Maximum window with all characters

Sliding Window Template:
1. Initialize left and right pointers
2. Expand right pointer until condition met
3. Contract left pointer while maintaining condition
4. Update result during contraction
5. Continue until right pointer exhausted

Performance Characteristics:
- Small strings (< 1K): < 1ms
- Medium strings (< 10K): < 10ms
- Large strings (< 100K): < 100ms
- Memory scales linearly with input size
- Very efficient for most practical inputs

Real-world Usage:
- Text editors: Find and highlight functionality
- Search engines: Query term highlighting
- Data analysis: Pattern extraction
- Security: Malware signature detection
- NLP: Named entity recognition

Optimization Techniques:
- Early exit when target impossible
- Character frequency pre-computation
- Minimal substring copying
- Cache-friendly memory access
- SIMD operations for large alphabets
*/
