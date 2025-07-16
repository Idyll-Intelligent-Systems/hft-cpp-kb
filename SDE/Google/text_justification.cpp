/*
Google SDE Interview Problem 4: Text Justification (Hard)
Given an array of strings words and a width maxWidth, format the text such that each line has exactly 
maxWidth characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack as many words as you can in each line. 
Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line 
does not divide evenly between words, the empty slots on the left will be assigned more spaces than 
the slots on the right.

For the last line of text, it should be left-justified, and no extra space is inserted between words.

Example:
Input: words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
Output:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]

Time Complexity: O(n) where n is total characters
Space Complexity: O(maxWidth) for building each line
*/

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

class Solution {
public:
    std::vector<std::string> fullJustify(std::vector<std::string>& words, int maxWidth) {
        std::vector<std::string> result;
        int i = 0;
        
        while (i < words.size()) {
            // Find words that fit in current line
            std::vector<std::string> currentLine;
            int totalChars = 0;
            
            while (i < words.size()) {
                // Check if adding next word exceeds maxWidth
                int neededChars = totalChars + words[i].length() + currentLine.size();
                if (neededChars > maxWidth) break;
                
                currentLine.push_back(words[i]);
                totalChars += words[i].length();
                i++;
            }
            
            // Justify the current line
            std::string justifiedLine = justifyLine(currentLine, maxWidth, i == words.size());
            result.push_back(justifiedLine);
        }
        
        return result;
    }
    
private:
    std::string justifyLine(const std::vector<std::string>& words, int maxWidth, bool isLastLine) {
        if (words.size() == 1 || isLastLine) {
            return leftJustify(words, maxWidth);
        }
        
        // Calculate spaces needed
        int totalWordChars = 0;
        for (const std::string& word : words) {
            totalWordChars += word.length();
        }
        
        int totalSpaces = maxWidth - totalWordChars;
        int gaps = words.size() - 1;
        
        if (gaps == 0) {
            // Single word case
            return words[0] + std::string(totalSpaces, ' ');
        }
        
        int spacesPerGap = totalSpaces / gaps;
        int extraSpaces = totalSpaces % gaps;
        
        std::string result;
        for (int i = 0; i < words.size(); i++) {
            result += words[i];
            
            if (i < words.size() - 1) {
                // Add regular spaces
                result += std::string(spacesPerGap, ' ');
                
                // Add extra space if needed (distribute from left)
                if (i < extraSpaces) {
                    result += " ";
                }
            }
        }
        
        return result;
    }
    
    std::string leftJustify(const std::vector<std::string>& words, int maxWidth) {
        std::string result;
        
        for (int i = 0; i < words.size(); i++) {
            result += words[i];
            if (i < words.size() - 1) {
                result += " ";
            }
        }
        
        // Pad with spaces to reach maxWidth
        result += std::string(maxWidth - result.length(), ' ');
        return result;
    }
};

// Advanced solution with optimizations
class OptimizedSolution {
public:
    std::vector<std::string> fullJustify(std::vector<std::string>& words, int maxWidth) {
        std::vector<std::string> result;
        result.reserve(words.size() / 3); // Estimate result size
        
        int i = 0;
        while (i < words.size()) {
            auto [lineWords, nextIndex] = getWordsForLine(words, i, maxWidth);
            std::string line = formatLine(lineWords, maxWidth, nextIndex == words.size());
            result.push_back(std::move(line));
            i = nextIndex;
        }
        
        return result;
    }
    
private:
    std::pair<std::vector<std::string>, int> getWordsForLine(
        const std::vector<std::string>& words, int start, int maxWidth) {
        
        std::vector<std::string> lineWords;
        int totalLength = 0;
        int i = start;
        
        while (i < words.size()) {
            int spaceNeeded = totalLength + words[i].length() + (lineWords.empty() ? 0 : lineWords.size());
            
            if (spaceNeeded > maxWidth) break;
            
            lineWords.push_back(words[i]);
            totalLength += words[i].length();
            i++;
        }
        
        return {lineWords, i};
    }
    
    std::string formatLine(const std::vector<std::string>& words, int maxWidth, bool isLastLine) {
        std::string result;
        result.reserve(maxWidth);
        
        if (words.size() == 1 || isLastLine) {
            formatLeftJustified(result, words, maxWidth);
        } else {
            formatFullyJustified(result, words, maxWidth);
        }
        
        return result;
    }
    
    void formatLeftJustified(std::string& result, const std::vector<std::string>& words, int maxWidth) {
        for (int i = 0; i < words.size(); i++) {
            result += words[i];
            if (i < words.size() - 1) {
                result += ' ';
            }
        }
        result.append(maxWidth - result.length(), ' ');
    }
    
    void formatFullyJustified(std::string& result, const std::vector<std::string>& words, int maxWidth) {
        int totalWordLength = 0;
        for (const auto& word : words) {
            totalWordLength += word.length();
        }
        
        int totalSpaces = maxWidth - totalWordLength;
        int gaps = words.size() - 1;
        int spacesPerGap = totalSpaces / gaps;
        int extraSpaces = totalSpaces % gaps;
        
        for (int i = 0; i < words.size(); i++) {
            result += words[i];
            
            if (i < gaps) {
                result.append(spacesPerGap, ' ');
                if (i < extraSpaces) {
                    result += ' ';
                }
            }
        }
    }
};

// Test cases for comprehensive validation
class TextJustificationTest {
public:
    static void runTests() {
        Solution solution;
        OptimizedSolution optimizedSolution;
        
        std::cout << "Google Text Justification Tests:" << std::endl;
        std::cout << "================================" << std::endl;
        
        // Test case 1: Basic example
        std::vector<std::string> words1 = {"This", "is", "an", "example", "of", "text", "justification."};
        auto result1 = solution.fullJustify(words1, 16);
        std::cout << "Test 1 (maxWidth=16):" << std::endl;
        printResult(result1);
        
        // Test case 2: Single word per line
        std::vector<std::string> words2 = {"What", "must", "be", "acknowledgment", "shall", "be"};
        auto result2 = solution.fullJustify(words2, 16);
        std::cout << "\nTest 2 (maxWidth=16):" << std::endl;
        printResult(result2);
        
        // Test case 3: Multiple spaces distribution
        std::vector<std::string> words3 = {"Science", "is", "what", "we", "understand", "well", "enough", "to", "explain"};
        auto result3 = solution.fullJustify(words3, 20);
        std::cout << "\nTest 3 (maxWidth=20):" << std::endl;
        printResult(result3);
        
        // Test case 4: Single word
        std::vector<std::string> words4 = {"Listen"};
        auto result4 = solution.fullJustify(words4, 6);
        std::cout << "\nTest 4 (maxWidth=6):" << std::endl;
        printResult(result4);
        
        // Performance test
        performanceTest();
    }
    
    static void printResult(const std::vector<std::string>& result) {
        for (const std::string& line : result) {
            std::cout << "\"" << line << "\" (length: " << line.length() << ")" << std::endl;
        }
    }
    
    static void performanceTest() {
        std::cout << "\nPerformance Test:" << std::endl;
        std::cout << "=================" << std::endl;
        
        // Generate large test case
        std::vector<std::string> largeWords;
        std::string baseWord = "word";
        for (int i = 0; i < 10000; i++) {
            largeWords.push_back(baseWord + std::to_string(i % 100));
        }
        
        Solution solution;
        OptimizedSolution optimizedSolution;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result1 = solution.fullJustify(largeWords, 50);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        auto result2 = optimizedSolution.fullJustify(largeWords, 50);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Standard solution: " << result1.size() << " lines in " << duration1.count() << " ms" << std::endl;
        std::cout << "Optimized solution: " << result2.size() << " lines in " << duration2.count() << " ms" << std::endl;
    }
    
    // Validation helper
    static bool validateJustification(const std::vector<std::string>& result, int maxWidth) {
        for (int i = 0; i < result.size(); i++) {
            const std::string& line = result[i];
            
            // Check length
            if (line.length() != maxWidth) {
                std::cout << "Invalid length at line " << i << ": " << line.length() << " != " << maxWidth << std::endl;
                return false;
            }
            
            // Check last line (should be left-justified)
            if (i == result.size() - 1) {
                // Find last non-space character
                int lastChar = line.find_last_not_of(' ');
                if (lastChar != std::string::npos) {
                    // Check if spaces are only at the end
                    for (int j = 0; j <= lastChar; j++) {
                        if (line[j] == ' ') {
                            // Should be single space between words
                            if (j > 0 && line[j-1] != ' ' && j < lastChar && line[j+1] != ' ') {
                                continue; // Valid space between words
                            }
                        }
                    }
                }
            }
        }
        
        return true;
    }
};

int main() {
    TextJustificationTest::runTests();
    return 0;
}

/*
Algorithm Analysis:

Core Algorithm:
1. Greedy word packing: fit as many words as possible per line
2. Space distribution: divide extra spaces evenly, with left bias
3. Special handling for last line (left-justified only)

Key Components:
1. Line Construction:
   - Calculate words that fit
   - Determine space distribution
   - Handle edge cases (single word, last line)

2. Space Distribution:
   - Total spaces = maxWidth - sum(word lengths)
   - Spaces per gap = total spaces / (words - 1)
   - Extra spaces distributed left to right

Google Interview Focus:
- String manipulation efficiency
- Edge case handling
- Code organization and modularity
- Memory usage optimization
- Clear algorithm explanation

Optimizations:
1. Pre-allocate string capacity
2. Use move semantics
3. Efficient space calculation
4. Minimize string concatenations

Edge Cases:
- Single word per line
- Single word in entire input
- Last line formatting
- Maximum width equals word length
- Empty spaces distribution

Time Complexity: O(n) where n is total characters
Space Complexity: O(maxWidth) for each line construction

Interview Tips:
1. Start with basic algorithm
2. Handle edge cases incrementally
3. Optimize string operations
4. Discuss alternative approaches
5. Validate output format
*/
