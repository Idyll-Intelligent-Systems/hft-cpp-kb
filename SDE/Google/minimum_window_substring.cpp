/*
Google SDE Interview Problem 8: Minimum Window Substring (Hard)
Given two strings s and t of lengths m and n respectively, return the minimum window 
in s which will contain all the characters in t. If there is no such window in s 
that covers all characters in t, return the empty string "".

Note that If there is such a window, it is guaranteed that there will always be 
only one unique minimum window in s.

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

Time Complexity: O(|s| + |t|) where |s| and |t| are the lengths of strings
Space Complexity: O(|s| + |t|) for the hashmap and window tracking
*/

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <climits>

class Solution {
public:
    // Approach 1: Sliding Window with HashMap (Optimal)
    std::string minWindow(std::string s, std::string t) {
        if (s.empty() || t.empty() || s.length() < t.length()) {
            return "";
        }
        
        // Count characters in t
        std::unordered_map<char, int> tCount;
        for (char c : t) {
            tCount[c]++;
        }
        
        int left = 0, right = 0;
        int minLen = INT_MAX;
        int minStart = 0;
        int required = tCount.size(); // Number of unique characters in t
        int formed = 0; // Number of unique characters in current window with desired frequency
        
        std::unordered_map<char, int> windowCount;
        
        while (right < s.length()) {
            // Expand window by including character at right
            char rightChar = s[right];
            windowCount[rightChar]++;
            
            // Check if current character's frequency matches the required frequency
            if (tCount.count(rightChar) && windowCount[rightChar] == tCount[rightChar]) {
                formed++;
            }
            
            // Try to contract window from left
            while (formed == required && left <= right) {
                char leftChar = s[left];
                
                // Update minimum window if current is smaller
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    minStart = left;
                }
                
                // Remove leftmost character from window
                windowCount[leftChar]--;
                if (tCount.count(leftChar) && windowCount[leftChar] < tCount[leftChar]) {
                    formed--;
                }
                
                left++;
            }
            
            right++;
        }
        
        return minLen == INT_MAX ? "" : s.substr(minStart, minLen);
    }
    
    // Approach 2: Optimized with Character Array (for ASCII)
    std::string minWindowOptimized(std::string s, std::string t) {
        if (s.empty() || t.empty()) return "";
        
        // Use arrays for ASCII characters (faster than unordered_map)
        std::vector<int> tCount(128, 0);
        std::vector<int> windowCount(128, 0);
        
        int uniqueChars = 0;
        for (char c : t) {
            if (tCount[c] == 0) uniqueChars++;
            tCount[c]++;
        }
        
        int left = 0, right = 0;
        int minLen = INT_MAX;
        int minStart = 0;
        int formed = 0;
        
        while (right < s.length()) {
            char rightChar = s[right];
            windowCount[rightChar]++;
            
            if (tCount[rightChar] > 0 && windowCount[rightChar] == tCount[rightChar]) {
                formed++;
            }
            
            while (formed == uniqueChars) {
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    minStart = left;
                }
                
                char leftChar = s[left];
                windowCount[leftChar]--;
                if (tCount[leftChar] > 0 && windowCount[leftChar] < tCount[leftChar]) {
                    formed--;
                }
                left++;
            }
            
            right++;
        }
        
        return minLen == INT_MAX ? "" : s.substr(minStart, minLen);
    }
    
    // Approach 3: Two-pass algorithm with preprocessing
    std::string minWindowTwoPass(std::string s, std::string t) {
        if (s.empty() || t.empty()) return "";
        
        // First pass: identify all possible starting positions
        std::unordered_map<char, int> tCount;
        for (char c : t) {
            tCount[c]++;
        }
        
        std::vector<std::pair<int, char>> filteredS;
        for (int i = 0; i < s.length(); i++) {
            if (tCount.count(s[i])) {
                filteredS.push_back({i, s[i]});
            }
        }
        
        if (filteredS.empty()) return "";
        
        // Second pass: sliding window on filtered array
        int left = 0, right = 0;
        int minLen = INT_MAX;
        int minStart = 0;
        int required = tCount.size();
        int formed = 0;
        
        std::unordered_map<char, int> windowCount;
        
        while (right < filteredS.size()) {
            char rightChar = filteredS[right].second;
            windowCount[rightChar]++;
            
            if (windowCount[rightChar] == tCount[rightChar]) {
                formed++;
            }
            
            while (formed == required && left <= right) {
                int windowLen = filteredS[right].first - filteredS[left].first + 1;
                if (windowLen < minLen) {
                    minLen = windowLen;
                    minStart = filteredS[left].first;
                }
                
                char leftChar = filteredS[left].second;
                windowCount[leftChar]--;
                if (windowCount[leftChar] < tCount[leftChar]) {
                    formed--;
                }
                left++;
            }
            
            right++;
        }
        
        return minLen == INT_MAX ? "" : s.substr(minStart, minLen);
    }
    
    // Approach 4: Template matching with early termination
    std::string minWindowTemplate(std::string s, std::string t) {
        std::unordered_map<char, int> need;
        std::unordered_map<char, int> window;
        
        for (char c : t) {
            need[c]++;
        }
        
        int left = 0, right = 0;
        int valid = 0; // Number of characters that satisfy the condition
        int start = 0, len = INT_MAX;
        
        while (right < s.size()) {
            char c = s[right];
            right++;
            
            if (need.count(c)) {
                window[c]++;
                if (window[c] == need[c]) {
                    valid++;
                }
            }
            
            while (valid == need.size()) {
                if (right - left < len) {
                    start = left;
                    len = right - left;
                }
                
                char d = s[left];
                left++;
                
                if (need.count(d)) {
                    if (window[d] == need[d]) {
                        valid--;
                    }
                    window[d]--;
                }
            }
        }
        
        return len == INT_MAX ? "" : s.substr(start, len);
    }
    
    // Approach 5: Memory optimized for large strings
    std::string minWindowMemoryOptimized(std::string s, std::string t) {
        if (s.length() < t.length()) return "";
        
        std::vector<int> need(256, 0);
        int needCnt = 0;
        
        // Build need array
        for (char c : t) {
            if (need[c] == 0) needCnt++;
            need[c]++;
        }
        
        int left = 0;
        int minLen = INT_MAX, minStart = 0;
        int matchCnt = 0;
        
        for (int right = 0; right < s.length(); right++) {
            // Extend window
            if (need[s[right]] > 0) {
                need[s[right]]--;
                if (need[s[right]] == 0) matchCnt++;
            } else {
                need[s[right]]--;
            }
            
            // Contract window
            while (matchCnt == needCnt) {
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    minStart = left;
                }
                
                if (need[s[left]] == 0) {
                    matchCnt--;
                }
                need[s[left]]++;
                left++;
            }
        }
        
        return minLen == INT_MAX ? "" : s.substr(minStart, minLen);
    }
};

// Comprehensive test framework
class MinWindowTest {
public:
    static void runTests() {
        Solution solution;
        
        std::cout << "Google Minimum Window Substring Tests:" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        testBasicCases(solution);
        testEdgeCases(solution);
        compareApproaches(solution);
        performanceTest(solution);
        stressTest(solution);
    }
    
    static void testBasicCases(Solution& solution) {
        std::cout << "\nBasic Test Cases:" << std::endl;
        std::cout << "=================" << std::endl;
        
        std::vector<std::pair<std::pair<std::string, std::string>, std::string>> testCases = {
            {{"ADOBECODEBANC", "ABC"}, "BANC"},
            {{"a", "a"}, "a"},
            {{"a", "aa"}, ""},
            {{"ab", "b"}, "b"},
            {{"abc", "cba"}, "abc"},
            {{"ADOBECODEBANC", "AABC"}, "ADOBECODEBA"},
            {{"abcdef", "ace"}, "abcde"},
            {{"pwwkew", "wke"}, "kew"}
        };
        
        for (const auto& testCase : testCases) {
            std::string s = testCase.first.first;
            std::string t = testCase.first.second;
            std::string expected = testCase.second;
            std::string result = solution.minWindow(s, t);
            
            std::cout << "s=\"" << s << "\", t=\"" << t << "\"" << std::endl;
            std::cout << "Result: \"" << result << "\"" << std::endl;
            std::cout << "Expected: \"" << expected << "\"" << std::endl;
            std::cout << "Status: " << (result == expected ? "✅ PASS" : "❌ FAIL") << std::endl;
            std::cout << "---" << std::endl;
        }
    }
    
    static void testEdgeCases(Solution& solution) {
        std::cout << "\nEdge Case Tests:" << std::endl;
        std::cout << "================" << std::endl;
        
        std::vector<std::pair<std::pair<std::string, std::string>, std::string>> edgeCases = {
            {{"", "a"}, ""},
            {{"a", ""}, ""},
            {{"", ""}, ""},
            {{"bba", "ab"}, "ba"},
            {{"aa", "aa"}, "aa"},
            {{"a", "b"}, ""},
            {{"ab", "A"}, ""},  // Case sensitive
            {{"AaA", "A"}, "A"},
            {{"aaflslflsldkalskaaa", "aaa"}, "aaa"},
            {{"ADOBECODEBANC", "ABCC"}, ""}  // Not enough C's
        };
        
        for (const auto& testCase : edgeCases) {
            std::string s = testCase.first.first;
            std::string t = testCase.first.second;
            std::string expected = testCase.second;
            std::string result = solution.minWindow(s, t);
            
            std::cout << "s=\"" << s << "\", t=\"" << t << "\"" << std::endl;
            std::cout << "Result: \"" << result << "\" -> " 
                      << (result == expected ? "✅" : "❌") << std::endl;
        }
    }
    
    static void compareApproaches(Solution& solution) {
        std::cout << "\nApproach Comparison:" << std::endl;
        std::cout << "===================" << std::endl;
        
        std::string s = "ADOBECODEBANC";
        std::string t = "ABC";
        
        auto start = std::chrono::high_resolution_clock::now();
        std::string result1 = solution.minWindow(s, t);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        std::string result2 = solution.minWindowOptimized(s, t);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        std::string result3 = solution.minWindowTwoPass(s, t);
        end = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        std::string result4 = solution.minWindowTemplate(s, t);
        end = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        std::string result5 = solution.minWindowMemoryOptimized(s, t);
        end = std::chrono::high_resolution_clock::now();
        auto duration5 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        std::cout << "HashMap approach: \"" << result1 << "\" (" << duration1.count() << " ns)" << std::endl;
        std::cout << "Array approach: \"" << result2 << "\" (" << duration2.count() << " ns)" << std::endl;
        std::cout << "Two-pass approach: \"" << result3 << "\" (" << duration3.count() << " ns)" << std::endl;
        std::cout << "Template approach: \"" << result4 << "\" (" << duration4.count() << " ns)" << std::endl;
        std::cout << "Memory optimized: \"" << result5 << "\" (" << duration5.count() << " ns)" << std::endl;
        
        bool allMatch = (result1 == result2 && result2 == result3 && result3 == result4 && result4 == result5);
        std::cout << "All results match: " << (allMatch ? "✅" : "❌") << std::endl;
    }
    
    static void performanceTest(Solution& solution) {
        std::cout << "\nPerformance Test:" << std::endl;
        std::cout << "=================" << std::endl;
        
        // Generate large test strings
        std::string largeS = generateRandomString(100000, "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
        std::string largeT = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        
        auto start = std::chrono::high_resolution_clock::now();
        std::string result = solution.minWindow(largeS, largeT);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Large string test (100K chars):" << std::endl;
        std::cout << "Result length: " << result.length() << std::endl;
        std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
        
        // Test with repeated characters
        std::string repeatedS = std::string(50000, 'A') + "BC" + std::string(50000, 'A');
        std::string repeatedT = "ABC";
        
        start = std::chrono::high_resolution_clock::now();
        result = solution.minWindow(repeatedS, repeatedT);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "\nRepeated character test:" << std::endl;
        std::cout << "Result: \"" << result << "\"" << std::endl;
        std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
    }
    
    static void stressTest(Solution& solution) {
        std::cout << "\nStress Test:" << std::endl;
        std::cout << "============" << std::endl;
        
        // Test 1: Worst case - pattern at the end
        std::string s1 = std::string(10000, 'A') + "BC";
        std::string t1 = "ABC";
        
        auto start = std::chrono::high_resolution_clock::now();
        std::string result1 = solution.minWindow(s1, t1);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Pattern at end: " << duration1.count() << " μs" << std::endl;
        
        // Test 2: Multiple overlapping windows
        std::string s2 = "AABBCCAABBCCAABBCC";
        std::string t2 = "ABC";
        
        start = std::chrono::high_resolution_clock::now();
        std::string result2 = solution.minWindow(s2, t2);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        std::cout << "Overlapping windows result: \"" << result2 << "\" (" 
                  << duration2.count() << " ns)" << std::endl;
        
        // Test 3: Very long pattern
        std::string longPattern = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        std::string longString = longPattern + std::string(1000, 'Z') + longPattern;
        
        start = std::chrono::high_resolution_clock::now();
        std::string result3 = solution.minWindow(longString, longPattern);
        end = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Long pattern: " << duration3.count() << " μs" << std::endl;
        std::cout << "Result length: " << result3.length() << std::endl;
    }
    
    static std::string generateRandomString(int length, const std::string& charset) {
        std::string result;
        result.reserve(length);
        
        for (int i = 0; i < length; i++) {
            result += charset[rand() % charset.length()];
        }
        
        return result;
    }
};

int main() {
    srand(time(nullptr));
    MinWindowTest::runTests();
    return 0;
}

/*
Algorithm Analysis:

Core Algorithm: Sliding Window with Two Pointers

Key Insight: Use expandable window to find all valid windows, then contract to find minimum

Time Complexity: O(|s| + |t|)
- Each character in s is visited at most twice (once by right pointer, once by left)
- HashMap operations are O(1) on average

Space Complexity: O(|s| + |t|)
- HashMap to store character frequencies
- Additional space for window tracking

Sliding Window Pattern:
1. Expand window (right pointer) until valid
2. Contract window (left pointer) while maintaining validity
3. Track minimum valid window throughout

Google Interview Focus:
- Sliding window technique mastery
- HashMap frequency counting
- Two-pointer optimization
- Edge case handling
- Multiple solution approaches

Key Optimizations:
1. Use array instead of HashMap for ASCII (faster access)
2. Filter source string to only relevant characters
3. Early termination when impossible
4. Template method for reusable sliding window

Critical Edge Cases:
- Empty strings
- Target longer than source
- No valid window exists
- Case sensitivity
- Duplicate characters in target

Alternative Approaches:
1. Brute force: O(|s|²) - check all substrings
2. Two-pass with filtering: O(|s|) but with preprocessing
3. Template sliding window: Generic reusable pattern
4. Memory optimized: Use arrays instead of maps

Real-world Applications:
- DNA sequence matching
- Text pattern search
- Resource allocation windows
- Network packet analysis
- Bioinformatics applications

Interview Tips:
1. Start with brute force explanation
2. Optimize to sliding window
3. Handle edge cases explicitly
4. Discuss time/space tradeoffs
5. Code clean, readable solution

Common Mistakes:
1. Not handling duplicate characters correctly
2. Wrong window contraction condition
3. Off-by-one errors in substring
4. Not updating minimum properly
5. Forgetting case sensitivity

Advanced Optimizations:
- Rolling hash for quick validity check
- Parallel processing for multiple patterns
- Cache-friendly memory access patterns
- SIMD for character counting
- Incremental frequency updates

Testing Strategy:
- Basic functionality verification
- Edge case coverage
- Performance benchmarking
- Stress testing with large inputs
- Cross-validation between approaches
*/
