/*
Microsoft SDE Interview Problem 1: Longest Palindromic Substring (Hard)
Given a string s, return the longest palindromic substring in s.

Example 1:
Input: s = "babad"
Output: "bab"
Note: "aba" is also a valid answer.

Example 2:
Input: s = "cbbd"
Output: "bb"

This is a classic Microsoft interview problem that tests string manipulation,
dynamic programming, and optimization techniques.

Time Complexity: O(n) with Manacher's algorithm, O(n²) with expand around centers
Space Complexity: O(1) with expand around centers, O(n) with Manacher's
*/

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

class Solution {
public:
    // Approach 1: Expand Around Centers (Most Common in Interviews)
    std::string longestPalindrome(std::string s) {
        if (s.empty()) return "";
        
        int start = 0, maxLen = 1;
        
        for (int i = 0; i < s.length(); i++) {
            // Check for odd length palindromes
            int len1 = expandAroundCenter(s, i, i);
            // Check for even length palindromes
            int len2 = expandAroundCenter(s, i, i + 1);
            
            int len = std::max(len1, len2);
            if (len > maxLen) {
                maxLen = len;
                start = i - (len - 1) / 2;
            }
        }
        
        return s.substr(start, maxLen);
    }
    
    // Approach 2: Dynamic Programming
    std::string longestPalindromeDP(std::string s) {
        int n = s.length();
        if (n == 0) return "";
        
        std::vector<std::vector<bool>> dp(n, std::vector<bool>(n, false));
        int start = 0, maxLen = 1;
        
        // Every single character is a palindrome
        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
        }
        
        // Check for palindromes of length 2
        for (int i = 0; i < n - 1; i++) {
            if (s[i] == s[i + 1]) {
                dp[i][i + 1] = true;
                start = i;
                maxLen = 2;
            }
        }
        
        // Check for palindromes of length 3 and more
        for (int len = 3; len <= n; len++) {
            for (int i = 0; i < n - len + 1; i++) {
                int j = i + len - 1;
                
                if (s[i] == s[j] && dp[i + 1][j - 1]) {
                    dp[i][j] = true;
                    start = i;
                    maxLen = len;
                }
            }
        }
        
        return s.substr(start, maxLen);
    }
    
    // Approach 3: Manacher's Algorithm (Optimal O(n))
    std::string longestPalindromeManacher(std::string s) {
        if (s.empty()) return "";
        
        // Preprocess string: "abc" -> "^#a#b#c#$"
        std::string processed = "^#";
        for (char c : s) {
            processed += c;
            processed += "#";
        }
        processed += "$";
        
        int n = processed.length();
        std::vector<int> P(n, 0); // P[i] = radius of palindrome centered at i
        int center = 0, right = 0;
        
        for (int i = 1; i < n - 1; i++) {
            int mirror = 2 * center - i;
            
            if (i < right) {
                P[i] = std::min(right - i, P[mirror]);
            }
            
            // Try to expand palindrome centered at i
            while (processed[i + P[i] + 1] == processed[i - P[i] - 1]) {
                P[i]++;
            }
            
            // If palindrome centered at i extends past right, adjust center and right
            if (i + P[i] > right) {
                center = i;
                right = i + P[i];
            }
        }
        
        // Find the longest palindrome
        int maxLen = 0, centerIndex = 0;
        for (int i = 1; i < n - 1; i++) {
            if (P[i] > maxLen) {
                maxLen = P[i];
                centerIndex = i;
            }
        }
        
        int start = (centerIndex - maxLen) / 2;
        return s.substr(start, maxLen);
    }
    
    // Approach 4: Brute Force (for comparison)
    std::string longestPalindromeBruteForce(std::string s) {
        int n = s.length();
        std::string longest = "";
        
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                std::string substr = s.substr(i, j - i + 1);
                if (isPalindrome(substr) && substr.length() > longest.length()) {
                    longest = substr;
                }
            }
        }
        
        return longest;
    }
    
    // Approach 5: Optimized with early termination
    std::string longestPalindromeOptimized(std::string s) {
        if (s.empty()) return "";
        
        int n = s.length();
        int start = 0, maxLen = 1;
        
        for (int i = 0; i < n; i++) {
            // Skip if remaining characters can't form a longer palindrome
            if (n - i <= maxLen / 2) break;
            
            int left = i, right = i;
            
            // Skip duplicates
            while (right < n - 1 && s[right] == s[right + 1]) {
                right++;
            }
            
            // Next center point
            i = right;
            
            // Expand around center
            while (left > 0 && right < n - 1 && s[left - 1] == s[right + 1]) {
                left--;
                right++;
            }
            
            int len = right - left + 1;
            if (len > maxLen) {
                maxLen = len;
                start = left;
            }
        }
        
        return s.substr(start, maxLen);
    }
    
private:
    int expandAroundCenter(const std::string& s, int left, int right) {
        while (left >= 0 && right < s.length() && s[left] == s[right]) {
            left--;
            right++;
        }
        return right - left - 1;
    }
    
    bool isPalindrome(const std::string& s) {
        int left = 0, right = s.length() - 1;
        while (left < right) {
            if (s[left] != s[right]) return false;
            left++;
            right--;
        }
        return true;
    }
};

// Comprehensive test framework
class LongestPalindromeTest {
public:
    static void runTests() {
        Solution solution;
        
        std::cout << "Microsoft Longest Palindromic Substring Tests:" << std::endl;
        std::cout << "=============================================" << std::endl;
        
        testBasicCases(solution);
        testEdgeCases(solution);
        compareApproaches(solution);
        performanceTest(solution);
        microsoftSpecificTest(solution);
    }
    
    static void testBasicCases(Solution& solution) {
        std::cout << "\nBasic Test Cases:" << std::endl;
        std::cout << "=================" << std::endl;
        
        std::vector<std::pair<std::string, std::vector<std::string>>> testCases = {
            {"babad", {"bab", "aba"}},
            {"cbbd", {"bb"}},
            {"racecar", {"racecar"}},
            {"abc", {"a", "b", "c"}},
            {"abcdcba", {"abcdcba"}},
            {"noon", {"noon"}},
            {"abcdef", {"a", "b", "c", "d", "e", "f"}}
        };
        
        for (const auto& testCase : testCases) {
            std::string input = testCase.first;
            std::vector<std::string> validOutputs = testCase.second;
            std::string result = solution.longestPalindrome(input);
            
            bool isValid = std::find(validOutputs.begin(), validOutputs.end(), result) != validOutputs.end();
            
            std::cout << "Input: \"" << input << "\"" << std::endl;
            std::cout << "Result: \"" << result << "\"" << std::endl;
            std::cout << "Status: " << (isValid ? "✅ PASS" : "❌ FAIL") << std::endl;
            std::cout << "---" << std::endl;
        }
    }
    
    static void testEdgeCases(Solution& solution) {
        std::cout << "\nEdge Case Tests:" << std::endl;
        std::cout << "================" << std::endl;
        
        // Empty string
        std::string result1 = solution.longestPalindrome("");
        std::cout << "Empty string: \"" << result1 << "\" " 
                  << (result1 == "" ? "✅" : "❌") << std::endl;
        
        // Single character
        std::string result2 = solution.longestPalindrome("a");
        std::cout << "Single char: \"" << result2 << "\" " 
                  << (result2 == "a" ? "✅" : "❌") << std::endl;
        
        // All same characters
        std::string result3 = solution.longestPalindrome("aaaa");
        std::cout << "All same: \"" << result3 << "\" " 
                  << (result3 == "aaaa" ? "✅" : "❌") << std::endl;
        
        // No palindrome longer than 1
        std::string result4 = solution.longestPalindrome("abcdefg");
        std::cout << "No long palindrome: \"" << result4 << "\" " 
                  << (result4.length() == 1 ? "✅" : "❌") << std::endl;
        
        // Entire string is palindrome
        std::string result5 = solution.longestPalindrome("abccba");
        std::cout << "Entire palindrome: \"" << result5 << "\" " 
                  << (result5 == "abccba" ? "✅" : "❌") << std::endl;
    }
    
    static void compareApproaches(Solution& solution) {
        std::cout << "\nApproach Comparison:" << std::endl;
        std::cout << "===================" << std::endl;
        
        std::string testString = "bababcabcdcbababa";
        
        auto start = std::chrono::high_resolution_clock::now();
        std::string result1 = solution.longestPalindrome(testString);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        std::string result2 = solution.longestPalindromeDP(testString);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        std::string result3 = solution.longestPalindromeManacher(testString);
        end = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        std::string result4 = solution.longestPalindromeOptimized(testString);
        end = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Expand Around Centers: \"" << result1 << "\" (" << duration1.count() << " μs)" << std::endl;
        std::cout << "Dynamic Programming: \"" << result2 << "\" (" << duration2.count() << " μs)" << std::endl;
        std::cout << "Manacher's Algorithm: \"" << result3 << "\" (" << duration3.count() << " μs)" << std::endl;
        std::cout << "Optimized Expand: \"" << result4 << "\" (" << duration4.count() << " μs)" << std::endl;
        
        bool allMatch = (result1.length() == result2.length() && 
                        result2.length() == result3.length() && 
                        result3.length() == result4.length());
        std::cout << "All results same length: " << (allMatch ? "✅" : "❌") << std::endl;
    }
    
    static void performanceTest(Solution& solution) {
        std::cout << "\nPerformance Test:" << std::endl;
        std::cout << "=================" << std::endl;
        
        // Generate large test string
        std::string largeString = generateTestString(10000);
        
        auto start = std::chrono::high_resolution_clock::now();
        std::string result = solution.longestPalindrome(largeString);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Large string (10K chars):" << std::endl;
        std::cout << "Longest palindrome length: " << result.length() << std::endl;
        std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
        
        // Test Manacher's performance
        start = std::chrono::high_resolution_clock::now();
        std::string resultManacher = solution.longestPalindromeManacher(largeString);
        end = std::chrono::high_resolution_clock::now();
        auto durationManacher = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Manacher's algorithm: " << durationManacher.count() << " ms" << std::endl;
        std::cout << "Results match: " << (result.length() == resultManacher.length() ? "✅" : "❌") << std::endl;
    }
    
    static void microsoftSpecificTest(Solution& solution) {
        std::cout << "\nMicrosoft-Specific Scenarios:" << std::endl;
        std::cout << "=============================" << std::endl;
        
        // Test case 1: Product names (common in Microsoft interviews)
        std::string product = "Windows2020swodniW";
        std::string result1 = solution.longestPalindrome(product);
        std::cout << "Product name test: \"" << result1 << "\"" << std::endl;
        
        // Test case 2: Binary string
        std::string binary = "1001001100110011001001";
        std::string result2 = solution.longestPalindrome(binary);
        std::cout << "Binary string: \"" << result2 << "\"" << std::endl;
        
        // Test case 3: DNA sequence
        std::string dna = "ATCGATCGTAGCGAT";
        std::string result3 = solution.longestPalindrome(dna);
        std::cout << "DNA sequence: \"" << result3 << "\"" << std::endl;
        
        // Test case 4: Unicode characters
        std::string unicode = "café éfac";
        std::string result4 = solution.longestPalindrome(unicode);
        std::cout << "Unicode test: \"" << result4 << "\"" << std::endl;
        
        // Test case 5: Very long palindrome in middle
        std::string longPalindrome = "abc" + std::string(1000, 'x') + "cba";
        auto start = std::chrono::high_resolution_clock::now();
        std::string result5 = solution.longestPalindrome(longPalindrome);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Long palindrome performance: " << result5.length() 
                  << " chars in " << duration.count() << " μs" << std::endl;
    }
    
    static std::string generateTestString(int length) {
        std::string result;
        result.reserve(length);
        
        srand(42); // Fixed seed for reproducible results
        for (int i = 0; i < length; i++) {
            result += 'a' + (rand() % 26);
        }
        
        // Insert a known palindrome
        if (length > 100) {
            std::string palindrome = "abcdefghijklmnopqponmlkjihgfedcba";
            int pos = length / 2 - palindrome.length() / 2;
            result.replace(pos, palindrome.length(), palindrome);
        }
        
        return result;
    }
    
    static void stressTest(Solution& solution) {
        std::cout << "\nStress Test:" << std::endl;
        std::cout << "============" << std::endl;
        
        // Test worst case for different algorithms
        std::string worstCase = std::string(1000, 'a') + "b" + std::string(1000, 'a');
        
        auto start = std::chrono::high_resolution_clock::now();
        std::string result1 = solution.longestPalindrome(worstCase);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        std::string result2 = solution.longestPalindromeManacher(worstCase);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Worst case (2001 chars):" << std::endl;
        std::cout << "Expand Around Centers: " << duration1.count() << " ms" << std::endl;
        std::cout << "Manacher's: " << duration2.count() << " ms" << std::endl;
        std::cout << "Results match: " << (result1.length() == result2.length() ? "✅" : "❌") << std::endl;
        
        // Test with all same characters
        std::string allSame(5000, 'z');
        start = std::chrono::high_resolution_clock::now();
        std::string result3 = solution.longestPalindrome(allSame);
        end = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "All same chars (5000): " << duration3.count() << " ms" << std::endl;
        std::cout << "Result length: " << result3.length() << std::endl;
    }
};

int main() {
    LongestPalindromeTest::runTests();
    LongestPalindromeTest::stressTest(Solution{});
    return 0;
}

/*
Algorithm Analysis:

Core Problem: Find the longest contiguous palindromic substring

Approach Comparison:

1. Expand Around Centers (O(n²)):
   - Most intuitive and commonly asked in interviews
   - Easy to implement and understand
   - Good balance of time/space complexity
   - Preferred approach for Microsoft interviews

2. Dynamic Programming (O(n²), O(n²) space):
   - Bottom-up approach building solution
   - Clear state transitions
   - Higher space complexity
   - Good for explaining thought process

3. Manacher's Algorithm (O(n)):
   - Optimal time complexity
   - Complex to implement correctly
   - Shows advanced algorithmic knowledge
   - Rarely expected in interviews

4. Brute Force (O(n³)):
   - Simple but inefficient
   - Good starting point for optimization
   - Useful for small inputs only

Microsoft Interview Focus:
- String manipulation proficiency
- Multiple approach discussion
- Optimization thinking
- Edge case handling
- Real-world application scenarios

Key Optimizations:
1. Skip duplicate characters when expanding
2. Early termination when remaining chars insufficient
3. Handle even/odd length palindromes efficiently
4. Avoid redundant substring creation

Real-world Applications:
- DNA sequence analysis
- Text processing and pattern recognition
- Data compression algorithms
- Bioinformatics research
- Natural language processing

Edge Cases:
- Empty strings
- Single characters
- All identical characters
- No palindromes longer than 1
- Entire string is palindrome

Interview Tips:
1. Start with brute force explanation
2. Optimize to expand around centers
3. Discuss Manacher's if time permits
4. Handle edge cases explicitly
5. Consider memory vs time tradeoffs

Common Mistakes:
1. Not handling even-length palindromes
2. Off-by-one errors in substring extraction
3. Inefficient string operations
4. Missing edge case validation
5. Incorrect center expansion logic

Advanced Considerations:
- Unicode and multibyte character handling
- Memory-efficient implementations
- Parallel processing opportunities
- Stream processing for very large inputs
- Approximate algorithms for performance

Testing Strategy:
- Basic functionality verification
- Edge case coverage
- Performance benchmarking
- Microsoft-specific scenarios
- Cross-validation between approaches

Complexity Summary:
- Time: O(n) to O(n³) depending on approach
- Space: O(1) to O(n²) depending on approach
- Practical choice: Expand around centers for interviews
- Production choice: Manacher's for performance-critical applications
*/
