/*
Google SDE Interview Problem 7: Valid Number (Hard)
A valid number can be split up into these components (in order):
1. A decimal number or an integer.
2. (Optional) An 'e' or 'E', followed by an integer.

A decimal number can be split up into these components (in order):
1. (Optional) A sign character ('+' or '-').
2. One of the following formats:
   - One or more digits, followed by a dot '.'.
   - One or more digits, followed by a dot '.', followed by one or more digits.
   - A dot '.', followed by one or more digits.

An integer can be split up into these components (in order):
1. (Optional) A sign character ('+' or '-').
2. One or more digits.

Examples:
Valid: "0", "3", "0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"
Invalid: "abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"

Time Complexity: O(n) where n is length of string
Space Complexity: O(1) constant space
*/

#include <iostream>
#include <string>
#include <regex>
#include <vector>

class Solution {
public:
    // Approach 1: State Machine (Most Robust)
    bool isNumber(std::string s) {
        enum State {
            STATE_INITIAL,
            STATE_SIGN,
            STATE_INTEGER,
            STATE_POINT,
            STATE_POINT_WITHOUT_INT,
            STATE_FRACTION,
            STATE_EXP,
            STATE_EXP_SIGN,
            STATE_EXP_NUMBER,
            STATE_END
        };
        
        std::vector<std::vector<State>> transitions = {
            // STATE_INITIAL
            {STATE_END, STATE_SIGN, STATE_INTEGER, STATE_POINT_WITHOUT_INT, STATE_END, STATE_END, STATE_END, STATE_END, STATE_END},
            // STATE_SIGN
            {STATE_END, STATE_END, STATE_INTEGER, STATE_POINT_WITHOUT_INT, STATE_END, STATE_END, STATE_END, STATE_END, STATE_END},
            // STATE_INTEGER
            {STATE_END, STATE_END, STATE_INTEGER, STATE_POINT, STATE_END, STATE_EXP, STATE_EXP, STATE_END, STATE_END},
            // STATE_POINT
            {STATE_END, STATE_END, STATE_FRACTION, STATE_END, STATE_END, STATE_EXP, STATE_EXP, STATE_END, STATE_END},
            // STATE_POINT_WITHOUT_INT
            {STATE_END, STATE_END, STATE_FRACTION, STATE_END, STATE_END, STATE_END, STATE_END, STATE_END, STATE_END},
            // STATE_FRACTION
            {STATE_END, STATE_END, STATE_FRACTION, STATE_END, STATE_END, STATE_EXP, STATE_EXP, STATE_END, STATE_END},
            // STATE_EXP
            {STATE_END, STATE_EXP_SIGN, STATE_EXP_NUMBER, STATE_END, STATE_END, STATE_END, STATE_END, STATE_END, STATE_END},
            // STATE_EXP_SIGN
            {STATE_END, STATE_END, STATE_EXP_NUMBER, STATE_END, STATE_END, STATE_END, STATE_END, STATE_END, STATE_END},
            // STATE_EXP_NUMBER
            {STATE_END, STATE_END, STATE_EXP_NUMBER, STATE_END, STATE_END, STATE_END, STATE_END, STATE_END, STATE_END}
        };
        
        auto getCharType = [](char c) -> int {
            if (c == ' ') return 0;
            if (c == '+' || c == '-') return 1;
            if (c >= '0' && c <= '9') return 2;
            if (c == '.') return 3;
            if (c == 'e' || c == 'E') return 5;
            return 8; // Invalid character
        };
        
        State state = STATE_INITIAL;
        
        for (char c : s) {
            int charType = getCharType(c);
            if (charType == 8 || state >= transitions.size()) return false;
            state = transitions[state][charType];
            if (state == STATE_END) return false;
        }
        
        return state == STATE_INTEGER || state == STATE_POINT || 
               state == STATE_FRACTION || state == STATE_EXP_NUMBER;
    }
    
    // Approach 2: Manual Parsing (More Intuitive)
    bool isNumberManual(std::string s) {
        int i = 0;
        int n = s.length();
        
        // Skip leading spaces
        while (i < n && s[i] == ' ') i++;
        if (i == n) return false;
        
        // Check sign
        if (i < n && (s[i] == '+' || s[i] == '-')) i++;
        
        bool isNumeric = false;
        
        // Check integer part
        while (i < n && isdigit(s[i])) {
            i++;
            isNumeric = true;
        }
        
        // Check decimal point and fractional part
        if (i < n && s[i] == '.') {
            i++;
            while (i < n && isdigit(s[i])) {
                i++;
                isNumeric = true;
            }
        }
        
        // Must have seen at least one digit so far
        if (!isNumeric) return false;
        
        // Check exponent part
        if (i < n && (s[i] == 'e' || s[i] == 'E')) {
            i++;
            if (i < n && (s[i] == '+' || s[i] == '-')) i++;
            
            bool hasExpDigits = false;
            while (i < n && isdigit(s[i])) {
                i++;
                hasExpDigits = true;
            }
            
            if (!hasExpDigits) return false;
        }
        
        // Skip trailing spaces
        while (i < n && s[i] == ' ') i++;
        
        return i == n;
    }
    
    // Approach 3: Regex (Concise but less efficient)
    bool isNumberRegex(std::string s) {
        std::regex pattern(R"(^\s*[+-]?((\d+\.?\d*)|(\.\d+))([eE][+-]?\d+)?\s*$)");
        return std::regex_match(s, pattern);
    }
    
    // Approach 4: Comprehensive Parser with Detailed Error Tracking
    bool isNumberDetailed(std::string s) {
        struct Parser {
            std::string str;
            int pos;
            
            Parser(const std::string& s) : str(s), pos(0) {}
            
            void skipSpaces() {
                while (pos < str.length() && str[pos] == ' ') pos++;
            }
            
            bool parseSign() {
                if (pos < str.length() && (str[pos] == '+' || str[pos] == '-')) {
                    pos++;
                    return true;
                }
                return false;
            }
            
            bool parseDigits() {
                int start = pos;
                while (pos < str.length() && isdigit(str[pos])) {
                    pos++;
                }
                return pos > start;
            }
            
            bool parseNumber() {
                skipSpaces();
                
                // Optional sign
                parseSign();
                
                bool hasInteger = parseDigits();
                
                bool hasFraction = false;
                if (pos < str.length() && str[pos] == '.') {
                    pos++; // consume '.'
                    hasFraction = parseDigits();
                }
                
                // Must have either integer part or fraction part
                if (!hasInteger && !hasFraction) {
                    return false;
                }
                
                // Optional exponent
                if (pos < str.length() && (str[pos] == 'e' || str[pos] == 'E')) {
                    pos++; // consume 'e' or 'E'
                    parseSign(); // optional sign in exponent
                    if (!parseDigits()) { // exponent must have digits
                        return false;
                    }
                }
                
                skipSpaces();
                return pos == str.length();
            }
        };
        
        Parser parser(s);
        return parser.parseNumber();
    }
    
    // Approach 5: Character-by-character validation
    bool isNumberCharByChar(std::string s) {
        if (s.empty()) return false;
        
        int i = 0, n = s.length();
        
        // Trim spaces
        while (i < n && s[i] == ' ') i++;
        while (n > 0 && s[n-1] == ' ') n--;
        
        if (i >= n) return false;
        
        // Parse sign
        if (s[i] == '+' || s[i] == '-') i++;
        
        bool seenDigit = false, seenDot = false, seenE = false;
        
        for (; i < n; i++) {
            char c = s[i];
            
            if (isdigit(c)) {
                seenDigit = true;
            } else if (c == '.') {
                if (seenDot || seenE) return false;
                seenDot = true;
            } else if (c == 'e' || c == 'E') {
                if (seenE || !seenDigit) return false;
                seenE = true;
                seenDigit = false; // Reset for exponent part
            } else if (c == '+' || c == '-') {
                if (i == 0 || (s[i-1] != 'e' && s[i-1] != 'E')) return false;
            } else {
                return false; // Invalid character
            }
        }
        
        return seenDigit;
    }
};

// Comprehensive test framework
class ValidNumberTest {
public:
    static void runTests() {
        Solution solution;
        
        std::cout << "Google Valid Number Tests:" << std::endl;
        std::cout << "=========================" << std::endl;
        
        testValidNumbers(solution);
        testInvalidNumbers(solution);
        compareApproaches(solution);
        performanceTest(solution);
        edgeCaseTest(solution);
    }
    
    static void testValidNumbers(Solution& solution) {
        std::cout << "\nValid Numbers Test:" << std::endl;
        std::cout << "==================" << std::endl;
        
        std::vector<std::string> validNumbers = {
            "0", "3", "0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", 
            "3e+7", "+6e-1", "53.5e93", "-123.456e789", "46.e3",
            " 0.1 ", "3.14159", "1.", ".1", "1e5", "6e-1", "99e2"
        };
        
        for (const std::string& num : validNumbers) {
            bool result = solution.isNumber(num);
            std::cout << "\"" << num << "\" -> " << (result ? "Valid" : "Invalid");
            if (!result) std::cout << " (ERROR!)";
            std::cout << std::endl;
        }
    }
    
    static void testInvalidNumbers(Solution& solution) {
        std::cout << "\nInvalid Numbers Test:" << std::endl;
        std::cout << "====================" << std::endl;
        
        std::vector<std::string> invalidNumbers = {
            "abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53",
            ".", ".e1", "e", "E", "e+", "1e+", "+", "-", " ", "",
            "1 2", " 1 2 ", "1..2", "1.2.3", "1e2e3", "+-1", "1+2"
        };
        
        for (const std::string& num : invalidNumbers) {
            bool result = solution.isNumber(num);
            std::cout << "\"" << num << "\" -> " << (result ? "Valid" : "Invalid");
            if (result) std::cout << " (ERROR!)";
            std::cout << std::endl;
        }
    }
    
    static void compareApproaches(Solution& solution) {
        std::cout << "\nApproach Comparison:" << std::endl;
        std::cout << "===================" << std::endl;
        
        std::vector<std::string> testCases = {
            "123", "12.3", "12.3e4", "-12.3e-4", ".123", "123.", "abc"
        };
        
        for (const std::string& test : testCases) {
            bool result1 = solution.isNumber(test);
            bool result2 = solution.isNumberManual(test);
            bool result3 = solution.isNumberRegex(test);
            bool result4 = solution.isNumberDetailed(test);
            bool result5 = solution.isNumberCharByChar(test);
            
            std::cout << "\"" << test << "\" -> ";
            std::cout << "SM:" << result1 << " MP:" << result2 << " RE:" << result3 
                      << " DT:" << result4 << " CC:" << result5;
            
            if (!(result1 == result2 && result2 == result3 && result3 == result4 && result4 == result5)) {
                std::cout << " (MISMATCH!)";
            }
            std::cout << std::endl;
        }
    }
    
    static void performanceTest(Solution& solution) {
        std::cout << "\nPerformance Test:" << std::endl;
        std::cout << "=================" << std::endl;
        
        std::vector<std::string> perfTestCases;
        for (int i = 0; i < 10000; i++) {
            perfTestCases.push_back("123.456e" + std::to_string(i % 100));
        }
        
        // Test state machine approach
        auto start = std::chrono::high_resolution_clock::now();
        for (const auto& test : perfTestCases) {
            solution.isNumber(test);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Test manual parsing approach
        start = std::chrono::high_resolution_clock::now();
        for (const auto& test : perfTestCases) {
            solution.isNumberManual(test);
        }
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Test regex approach
        start = std::chrono::high_resolution_clock::now();
        for (const auto& test : perfTestCases) {
            solution.isNumberRegex(test);
        }
        end = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "State Machine: " << duration1.count() << " ms" << std::endl;
        std::cout << "Manual Parsing: " << duration2.count() << " ms" << std::endl;
        std::cout << "Regex: " << duration3.count() << " ms" << std::endl;
    }
    
    static void edgeCaseTest(Solution& solution) {
        std::cout << "\nEdge Case Test:" << std::endl;
        std::cout << "===============" << std::endl;
        
        std::vector<std::pair<std::string, bool>> edgeCases = {
            {"0e0", true},
            {"1e0", true},
            {"1e00", true},
            {".e1", false},
            {"e", false},
            {".", false},
            {" 005047e+6", true},
            {" .-4", false},
            {" +.", false},
            {" 46.e3", true},
            {" .2e81", true},
            {" --6", false},
            {" -+3", false},
            {" 2-1", false},
            {" +0e-", false},
            {" e9", false}
        };
        
        for (const auto& testCase : edgeCases) {
            bool result = solution.isNumber(testCase.first);
            std::cout << "\"" << testCase.first << "\" -> " << (result ? "Valid" : "Invalid");
            std::cout << " (Expected: " << (testCase.second ? "Valid" : "Invalid") << ")";
            if (result != testCase.second) {
                std::cout << " ❌";
            } else {
                std::cout << " ✅";
            }
            std::cout << std::endl;
        }
    }
    
    static void stressTest(Solution& solution) {
        std::cout << "\nStress Test:" << std::endl;
        std::cout << "============" << std::endl;
        
        // Test very long numbers
        std::string longNumber = "1";
        for (int i = 0; i < 1000; i++) {
            longNumber += "0";
        }
        longNumber += ".";
        for (int i = 0; i < 1000; i++) {
            longNumber += "0";
        }
        longNumber += "e1000";
        
        auto start = std::chrono::high_resolution_clock::now();
        bool result = solution.isNumber(longNumber);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Long number (4000+ chars): " << (result ? "Valid" : "Invalid") 
                  << " in " << duration.count() << " μs" << std::endl;
        
        // Test many spaces
        std::string manySpaces = std::string(1000, ' ') + "123.45" + std::string(1000, ' ');
        start = std::chrono::high_resolution_clock::now();
        result = solution.isNumber(manySpaces);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Many spaces (2000+ chars): " << (result ? "Valid" : "Invalid") 
                  << " in " << duration.count() << " μs" << std::endl;
    }
};

int main() {
    ValidNumberTest::runTests();
    ValidNumberTest::stressTest(Solution{});
    return 0;
}

/*
Algorithm Analysis:

Core Challenge: Parse and validate complex number format with multiple optional components

Approach Comparison:

1. State Machine (Recommended for interviews):
   - Most robust and maintainable
   - Clear state transitions
   - Easy to debug and extend
   - Time: O(n), Space: O(1)

2. Manual Parsing:
   - More intuitive to understand
   - Sequential validation
   - Good for explaining logic
   - Time: O(n), Space: O(1)

3. Regular Expression:
   - Concise but less efficient
   - Hard to debug edge cases
   - Not recommended for production
   - Time: O(n), Space: O(1)

4. Character-by-character:
   - Flag-based tracking
   - Good balance of clarity and efficiency
   - Easy to implement
   - Time: O(n), Space: O(1)

Google Interview Focus:
- String parsing and validation
- State machine design
- Edge case handling
- Multiple solution approaches
- Performance analysis

Key Design Decisions:
1. How to handle leading/trailing spaces
2. State transition logic for complex rules
3. Error detection and early termination
4. Code maintainability vs performance

Edge Cases to Consider:
- Empty string and spaces only
- Multiple signs (--6, -+3)
- Multiple dots (1.2.3)
- Missing digits (.e1, e3)
- Invalid exponent (1e, 1e+)
- Very long numbers

Common Mistakes:
1. Forgetting to validate exponent digits
2. Not handling signs in exponent
3. Allowing multiple decimal points
4. Missing digit validation after decimal
5. Not trimming spaces properly

Interview Tips:
1. Start with simple case analysis
2. Build state machine incrementally
3. Test edge cases thoroughly
4. Discuss alternative approaches
5. Consider real-world requirements

Real-world Applications:
- JSON number validation
- Calculator input parsing
- Scientific notation processing
- Financial data validation
- Configuration file parsing

Optimization Opportunities:
1. Early termination on invalid characters
2. Compile-time regex for known patterns
3. SIMD for bulk validation
4. State machine compression
5. Branch prediction optimization

Testing Strategy:
- Valid number formats
- Invalid number formats
- Edge cases and boundary conditions
- Performance with large inputs
- Cross-validation between approaches
*/
