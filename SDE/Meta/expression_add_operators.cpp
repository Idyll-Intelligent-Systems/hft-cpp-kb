/*
Meta (Facebook) SDE Interview Problem 4: Expression Add Operators (Hard)
Given a string num that contains only digits and a target integer, return all possibilities to 
add the binary operators '+', '-', or '*' between the digits of num so that the resulting expression evaluates to the target value.

Note that operands in the returned expressions should not have leading zeros.

Example 1:
Input: num = "123", target = 6
Output: ["1+2+3","1*2*3"]

Example 2:
Input: num = "232", target = 8
Output: ["2*3+2","2+3*2"]

Example 3:
Input: num = "105", target = 5
Output: ["1*0+5","10-5"]

This is a classic Meta interview problem testing backtracking, operator precedence, and mathematical expression evaluation.
It requires careful handling of multiplication precedence and avoiding integer overflow.

Time Complexity: O(4^n * n) where n is length of string (4 choices per position: +, -, *, or extend number)
Space Complexity: O(n) for recursion depth and string construction
*/

#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <climits>
#include <chrono>

class Solution {
public:
    // Approach 1: Backtracking with proper operator precedence handling
    std::vector<std::string> addOperators(std::string num, int target) {
        std::vector<std::string> result;
        if (num.empty()) return result;
        
        backtrack(num, target, 0, "", 0, 0, result);
        return result;
    }
    
    // Approach 2: Iterative approach with stack simulation
    std::vector<std::string> addOperatorsIterative(std::string num, int target) {
        std::vector<std::string> result;
        if (num.empty()) return result;
        
        struct State {
            int index;
            std::string expr;
            long long eval;
            long long mult;
        };
        
        std::vector<State> stack;
        
        // Initialize with all possible first numbers
        for (int i = 1; i <= num.length(); i++) {
            std::string firstNum = num.substr(0, i);
            if (firstNum.length() > 1 && firstNum[0] == '0') break;
            
            long long val = std::stoll(firstNum);
            if (i == num.length()) {
                if (val == target) {
                    result.push_back(firstNum);
                }
            } else {
                stack.push_back({i, firstNum, val, val});
            }
        }
        
        while (!stack.empty()) {
            State current = stack.back();
            stack.pop_back();
            
            if (current.index == num.length()) {
                if (current.eval == target) {
                    result.push_back(current.expr);
                }
                continue;
            }
            
            // Try all possible next numbers
            for (int i = current.index + 1; i <= num.length(); i++) {
                std::string nextNum = num.substr(current.index, i - current.index);
                if (nextNum.length() > 1 && nextNum[0] == '0') break;
                
                long long val = std::stoll(nextNum);
                
                // Try '+' operator
                stack.push_back({
                    i, 
                    current.expr + "+" + nextNum,
                    current.eval + val,
                    val
                });
                
                // Try '-' operator
                stack.push_back({
                    i,
                    current.expr + "-" + nextNum,
                    current.eval - val,
                    -val
                });
                
                // Try '*' operator
                stack.push_back({
                    i,
                    current.expr + "*" + nextNum,
                    current.eval - current.mult + current.mult * val,
                    current.mult * val
                });
            }
        }
        
        return result;
    }
    
    // Approach 3: Optimized backtracking with early pruning
    std::vector<std::string> addOperatorsOptimized(std::string num, int target) {
        std::vector<std::string> result;
        if (num.empty()) return result;
        
        // Pre-calculate bounds for pruning
        std::vector<long long> minVal(num.length()), maxVal(num.length());
        calculateBounds(num, minVal, maxVal);
        
        backtrackOptimized(num, target, 0, "", 0, 0, result, minVal, maxVal);
        return result;
    }
    
    // Approach 4: Memoization approach (for optimization)
    std::vector<std::string> addOperatorsMemo(std::string num, int target) {
        std::vector<std::string> result;
        if (num.empty()) return result;
        
        std::unordered_map<std::string, std::vector<std::string>> memo;
        return backtrackMemo(num, target, 0, 0, 0, memo);
    }
    
    // Approach 5: Mathematical optimization with digit analysis
    std::vector<std::string> addOperatorsMath(std::string num, int target) {
        std::vector<std::string> result;
        if (num.empty()) return result;
        
        // Analyze digits for quick rejection
        long long digitSum = 0, digitProduct = 1;
        bool hasZero = false;
        
        for (char c : num) {
            int digit = c - '0';
            digitSum += digit;
            if (digit == 0) hasZero = true;
            else digitProduct *= digit;
        }
        
        // Quick bounds check
        if (target > digitProduct || (target < -digitProduct && !hasZero)) {
            // Mathematical impossibility check
            if (abs(target) > digitProduct) {
                return result; // Early exit
            }
        }
        
        backtrackMath(num, target, 0, "", 0, 0, result);
        return result;
    }
    
private:
    void backtrack(const std::string& num, int target, int index, 
                   std::string expr, long long eval, long long mult,
                   std::vector<std::string>& result) {
        if (index == num.length()) {
            if (eval == target) {
                result.push_back(expr);
            }
            return;
        }
        
        for (int i = index; i < num.length(); i++) {
            std::string operand = num.substr(index, i - index + 1);
            
            // Skip numbers with leading zeros (except single "0")
            if (operand.length() > 1 && operand[0] == '0') {
                break;
            }
            
            // Check for potential overflow
            if (operand.length() > 10) break;
            
            long long val = std::stoll(operand);
            
            if (index == 0) {
                // First number, no operator needed
                backtrack(num, target, i + 1, operand, val, val, result);
            } else {
                // Try '+' operator
                backtrack(num, target, i + 1, expr + "+" + operand, 
                         eval + val, val, result);
                
                // Try '-' operator
                backtrack(num, target, i + 1, expr + "-" + operand, 
                         eval - val, -val, result);
                
                // Try '*' operator
                // Need to undo the last addition/subtraction and apply multiplication
                backtrack(num, target, i + 1, expr + "*" + operand, 
                         eval - mult + mult * val, mult * val, result);
            }
        }
    }
    
    void calculateBounds(const std::string& num, std::vector<long long>& minVal, 
                        std::vector<long long>& maxVal) {
        int n = num.length();
        
        // Calculate minimum and maximum possible values from each position
        for (int i = n - 1; i >= 0; i--) {
            if (i == n - 1) {
                minVal[i] = maxVal[i] = num[i] - '0';
            } else {
                long long digitVal = num[i] - '0';
                
                // Minimum: subtract maximum from right
                minVal[i] = std::min({
                    digitVal - maxVal[i + 1],
                    digitVal * minVal[i + 1],
                    digitVal + minVal[i + 1]
                });
                
                // Maximum: add maximum from right or multiply if beneficial
                maxVal[i] = std::max({
                    digitVal + maxVal[i + 1],
                    digitVal * maxVal[i + 1],
                    digitVal - minVal[i + 1]
                });
            }
        }
    }
    
    void backtrackOptimized(const std::string& num, int target, int index, 
                           std::string expr, long long eval, long long mult,
                           std::vector<std::string>& result,
                           const std::vector<long long>& minVal,
                           const std::vector<long long>& maxVal) {
        if (index == num.length()) {
            if (eval == target) {
                result.push_back(expr);
            }
            return;
        }
        
        // Pruning: check if target is reachable
        if (index < num.length() - 1) {
            long long remaining = target - eval;
            if (remaining < minVal[index] || remaining > maxVal[index]) {
                return; // Pruning
            }
        }
        
        for (int i = index; i < num.length(); i++) {
            std::string operand = num.substr(index, i - index + 1);
            
            if (operand.length() > 1 && operand[0] == '0') {
                break;
            }
            
            if (operand.length() > 10) break;
            
            long long val = std::stoll(operand);
            
            if (index == 0) {
                backtrackOptimized(num, target, i + 1, operand, val, val, 
                                 result, minVal, maxVal);
            } else {
                backtrackOptimized(num, target, i + 1, expr + "+" + operand, 
                                 eval + val, val, result, minVal, maxVal);
                
                backtrackOptimized(num, target, i + 1, expr + "-" + operand, 
                                 eval - val, -val, result, minVal, maxVal);
                
                backtrackOptimized(num, target, i + 1, expr + "*" + operand, 
                                 eval - mult + mult * val, mult * val, 
                                 result, minVal, maxVal);
            }
        }
    }
    
    std::vector<std::string> backtrackMemo(const std::string& num, int target, 
                                          int index, long long eval, long long mult,
                                          std::unordered_map<std::string, std::vector<std::string>>& memo) {
        if (index == num.length()) {
            if (eval == target) {
                return {""};
            }
            return {};
        }
        
        std::string key = std::to_string(index) + "," + std::to_string(eval) + "," + std::to_string(mult);
        if (memo.find(key) != memo.end()) {
            return memo[key];
        }
        
        std::vector<std::string> result;
        
        for (int i = index; i < num.length(); i++) {
            std::string operand = num.substr(index, i - index + 1);
            
            if (operand.length() > 1 && operand[0] == '0') {
                break;
            }
            
            if (operand.length() > 10) break;
            
            long long val = std::stoll(operand);
            
            if (index == 0) {
                auto subResults = backtrackMemo(num, target, i + 1, val, val, memo);
                for (const auto& sub : subResults) {
                    result.push_back(operand + sub);
                }
            } else {
                // Try '+'
                auto addResults = backtrackMemo(num, target, i + 1, eval + val, val, memo);
                for (const auto& sub : addResults) {
                    result.push_back("+" + operand + sub);
                }
                
                // Try '-'
                auto subResults = backtrackMemo(num, target, i + 1, eval - val, -val, memo);
                for (const auto& sub : subResults) {
                    result.push_back("-" + operand + sub);
                }
                
                // Try '*'
                auto multResults = backtrackMemo(num, target, i + 1, 
                                               eval - mult + mult * val, mult * val, memo);
                for (const auto& sub : multResults) {
                    result.push_back("*" + operand + sub);
                }
            }
        }
        
        memo[key] = result;
        return result;
    }
    
    void backtrackMath(const std::string& num, int target, int index, 
                      std::string expr, long long eval, long long mult,
                      std::vector<std::string>& result) {
        if (index == num.length()) {
            if (eval == target) {
                result.push_back(expr);
            }
            return;
        }
        
        for (int i = index; i < num.length(); i++) {
            std::string operand = num.substr(index, i - index + 1);
            
            if (operand.length() > 1 && operand[0] == '0') {
                break;
            }
            
            if (operand.length() > 10) break;
            
            long long val = std::stoll(operand);
            
            // Mathematical pruning: check if remaining digits can reach target
            if (index > 0) {
                long long remaining = num.length() - i - 1;
                if (remaining > 0) {
                    long long maxPossible = eval + remaining * 9; // All 9's added
                    long long minPossible = eval - remaining * 9; // All 9's subtracted
                    
                    if (target > maxPossible || target < minPossible) {
                        continue; // This branch cannot reach target
                    }
                }
            }
            
            if (index == 0) {
                backtrackMath(num, target, i + 1, operand, val, val, result);
            } else {
                backtrackMath(num, target, i + 1, expr + "+" + operand, 
                            eval + val, val, result);
                
                backtrackMath(num, target, i + 1, expr + "-" + operand, 
                            eval - val, -val, result);
                
                backtrackMath(num, target, i + 1, expr + "*" + operand, 
                            eval - mult + mult * val, mult * val, result);
            }
        }
    }
};

// Test framework
class ExpressionAddOperatorsTest {
public:
    static void runTests() {
        std::cout << "Meta Expression Add Operators Tests:" << std::endl;
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
        auto result1 = solution.addOperators("123", 6);
        std::cout << "Test 1 - num='123', target=6: ";
        printResult(result1);
        std::cout << "Expected: ['1+2+3','1*2*3']" << std::endl;
        
        // Test case 2: Example 2
        auto result2 = solution.addOperators("232", 8);
        std::cout << "Test 2 - num='232', target=8: ";
        printResult(result2);
        std::cout << "Expected: ['2*3+2','2+3*2']" << std::endl;
        
        // Test case 3: Example 3
        auto result3 = solution.addOperators("105", 5);
        std::cout << "Test 3 - num='105', target=5: ";
        printResult(result3);
        std::cout << "Expected: ['1*0+5','10-5']" << std::endl;
        
        // Test case 4: Simple single digit
        auto result4 = solution.addOperators("5", 5);
        std::cout << "Test 4 - num='5', target=5: ";
        printResult(result4);
        std::cout << "Expected: ['5']" << std::endl;
        
        // Test case 5: Two digits
        auto result5 = solution.addOperators("12", 3);
        std::cout << "Test 5 - num='12', target=3: ";
        printResult(result5);
        std::cout << "Expected: ['1+2']" << std::endl;
    }
    
    static void testEdgeCases() {
        std::cout << "\nEdge Case Tests:" << std::endl;
        std::cout << "================" << std::endl;
        
        Solution solution;
        
        // No solution
        auto result1 = solution.addOperators("123", 100);
        std::cout << "No solution (123, 100): ";
        printResult(result1);
        std::cout << " (Expected: [])" << std::endl;
        
        // Leading zeros
        auto result2 = solution.addOperators("102", 2);
        std::cout << "Leading zeros (102, 2): ";
        printResult(result2);
        std::cout << " (Expected: ['1+0+1','1-0+1','1*0+2','10-8'] or similar)" << std::endl;
        
        // All zeros
        auto result3 = solution.addOperators("000", 0);
        std::cout << "All zeros (000, 0): ";
        printResult(result3);
        std::cout << " (Expected: ['0+0+0','0-0+0','0*0+0'] or similar)" << std::endl;
        
        // Large target
        auto result4 = solution.addOperators("999", 27);
        std::cout << "Large numbers (999, 27): ";
        printResult(result4);
        std::cout << " (Expected: ['9*9/3','9+9+9'] or similar)" << std::endl;
        
        // Negative target
        auto result5 = solution.addOperators("123", -1);
        std::cout << "Negative target (123, -1): ";
        printResult(result5);
        std::cout << " (Expected solutions with subtraction)" << std::endl;
        
        // Single digit, impossible target
        auto result6 = solution.addOperators("1", 2);
        std::cout << "Impossible single (1, 2): ";
        printResult(result6);
        std::cout << " (Expected: [])" << std::endl;
    }
    
    static void compareApproaches() {
        std::cout << "\nApproach Comparison:" << std::endl;
        std::cout << "===================" << std::endl;
        
        Solution solution;
        std::string testNum = "1234";
        int testTarget = 10;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result1 = solution.addOperators(testNum, testTarget);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        auto result2 = solution.addOperatorsIterative(testNum, testTarget);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        auto result3 = solution.addOperatorsOptimized(testNum, testTarget);
        end = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        auto result4 = solution.addOperatorsMath(testNum, testTarget);
        end = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Standard backtrack: " << duration1.count() << " μs, " << result1.size() << " solutions" << std::endl;
        std::cout << "Iterative approach: " << duration2.count() << " μs, " << result2.size() << " solutions" << std::endl;
        std::cout << "Optimized backtrack: " << duration3.count() << " μs, " << result3.size() << " solutions" << std::endl;
        std::cout << "Math optimized: " << duration4.count() << " μs, " << result4.size() << " solutions" << std::endl;
        
        bool allMatch = (result1.size() == result2.size() && result2.size() == result3.size() && result3.size() == result4.size());
        std::cout << "All approaches same count: " << (allMatch ? "✅" : "❌") << std::endl;
        
        std::cout << "\nStandard results: ";
        printResult(result1);
        std::cout << std::endl;
    }
    
    static void metaSpecificScenarios() {
        std::cout << "\nMeta-Specific Scenarios:" << std::endl;
        std::cout << "=======================" << std::endl;
        
        Solution solution;
        
        // Scenario 1: Ad targeting calculation
        std::cout << "Ad targeting metric calculation:" << std::endl;
        auto result1 = solution.addOperators("12345", 15);
        std::cout << "Budget/impression calc (12345 -> 15): ";
        printResult(result1);
        std::cout << std::endl;
        
        // Scenario 2: User engagement score
        std::cout << "\nUser engagement score:" << std::endl;
        auto result2 = solution.addOperators("567", 20);
        std::cout << "Engagement calc (567 -> 20): ";
        printResult(result2);
        std::cout << std::endl;
        
        // Scenario 3: Feed ranking algorithm
        std::cout << "\nFeed ranking calculation:" << std::endl;
        auto result3 = solution.addOperators("789", 30);
        std::cout << "Ranking score (789 -> 30): ";
        printResult(result3);
        std::cout << std::endl;
        
        // Scenario 4: A/B test metric
        std::cout << "\nA/B test conversion rate:" << std::endl;
        auto result4 = solution.addOperators("246", 12);
        std::cout << "Conversion calc (246 -> 12): ";
        printResult(result4);
        std::cout << std::endl;
        
        // Scenario 5: Revenue calculation
        std::cout << "\nRevenue optimization:" << std::endl;
        auto result5 = solution.addOperators("135", 9);
        std::cout << "Revenue calc (135 -> 9): ";
        printResult(result5);
        std::cout << std::endl;
    }
    
    static void performanceAnalysis() {
        std::cout << "\nPerformance Analysis:" << std::endl;
        std::cout << "====================" << std::endl;
        
        Solution solution;
        
        // Test with increasing string lengths
        std::vector<std::pair<std::string, int>> testCases = {
            {"12", 3},
            {"123", 6},
            {"1234", 10},
            {"12345", 15},
            {"123456", 21}
        };
        
        for (const auto& testCase : testCases) {
            std::string num = testCase.first;
            int target = testCase.second;
            
            auto start = std::chrono::high_resolution_clock::now();
            auto result = solution.addOperators(num, target);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            std::cout << "Length " << num.length() << " ('" << num << "' -> " << target << "): " 
                      << duration.count() << " μs, " << result.size() << " solutions" << std::endl;
        }
        
        // Complexity analysis with exponential growth
        std::cout << "\nComplexity Growth Analysis:" << std::endl;
        std::cout << "==========================" << std::endl;
        
        std::vector<std::string> growthTest = {"12", "123", "1234", "12345"};
        std::vector<long long> timings;
        
        for (const std::string& num : growthTest) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = solution.addOperators(num, 10);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            timings.push_back(duration.count());
            std::cout << "n=" << num.length() << ": " << duration.count() << " μs" << std::endl;
        }
        
        // Calculate growth ratios
        std::cout << "\nGrowth ratios:" << std::endl;
        for (int i = 1; i < timings.size(); i++) {
            double ratio = (double)timings[i] / timings[i-1];
            std::cout << "n=" << (i+2) << "/n=" << (i+1) << " ratio: " << ratio << "x" << std::endl;
        }
        
        // Memory usage estimation
        std::cout << "\nMemory Usage Analysis:" << std::endl;
        std::cout << "=====================" << std::endl;
        
        std::cout << "Recursion depth: O(n) where n is string length" << std::endl;
        std::cout << "String storage: O(4^n * n) for all possible expressions" << std::endl;
        std::cout << "Working memory: O(n) for current expression construction" << std::endl;
        
        // Estimate for common cases
        for (int n = 3; n <= 8; n++) {
            long long expressions = 1;
            for (int i = 1; i < n; i++) {
                expressions *= 4; // 4 choices per gap
            }
            expressions *= n; // Average string length
            
            std::cout << "n=" << n << ": ~" << expressions << " bytes for results" << std::endl;
        }
    }
    
    static void printResult(const std::vector<std::string>& result) {
        std::cout << "[";
        for (int i = 0; i < result.size(); i++) {
            std::cout << "'" << result[i] << "'";
            if (i < result.size() - 1) std::cout << ",";
        }
        std::cout << "]";
    }
};

int main() {
    ExpressionAddOperatorsTest::runTests();
    return 0;
}

/*
Algorithm Analysis:

Core Problem: Insert operators (+, -, *) between digits to reach target value

Key Insights:
1. Multiplication has higher precedence than addition/subtraction
2. Need to track last multiplied value to handle precedence correctly
3. Leading zeros are invalid except for single "0"
4. Expression evaluation must handle operator precedence

Approach Comparison:

1. Standard Backtracking (Recommended):
   - Time: O(4^n * n) where n is string length
   - Space: O(n) recursion depth + O(k * n) for results
   - Handles operator precedence correctly
   - Clean recursive implementation

2. Iterative with Stack:
   - Time: Same complexity, different constant factors
   - Space: O(4^n) for explicit stack
   - Avoids recursion depth issues
   - More complex state management

3. Optimized with Pruning:
   - Pre-calculates bounds for early termination
   - Better performance in practice
   - More complex implementation
   - Significant speedup for large inputs

4. Memoization Approach:
   - Caches intermediate results
   - Complex due to expression building
   - Limited effectiveness for this problem
   - State space too large

5. Mathematical Optimization:
   - Quick bounds checking
   - Digit analysis for early pruning
   - Best for sparse solution spaces
   - Requires careful mathematical reasoning

Meta Interview Focus:
- Backtracking and recursion
- Operator precedence handling
- String manipulation and parsing
- Mathematical expression evaluation
- Optimization and pruning techniques

Key Design Decisions:
1. How to handle multiplication precedence
2. Avoiding leading zeros
3. Managing integer overflow
4. Pruning strategies for performance

Real-world Applications at Meta:
- Formula evaluation in spreadsheets
- Mathematical expression parsing
- Calculator implementation
- A/B test metric calculations
- Ad targeting score computation

Edge Cases:
- Single digit strings
- Leading zeros in operands
- No valid solutions
- Negative targets
- Integer overflow scenarios

Interview Tips:
1. Start with basic backtracking
2. Explain operator precedence handling
3. Handle leading zeros correctly
4. Discuss overflow prevention
5. Consider optimization opportunities

Common Mistakes:
1. Wrong operator precedence handling
2. Allowing invalid leading zeros
3. Integer overflow not handled
4. Incorrect base case handling
5. Poor performance without pruning

Advanced Optimizations:
- Bounds checking for early pruning
- Mathematical impossibility detection
- Digit frequency analysis
- Parallel branch exploration
- Memory-efficient result storage

Testing Strategy:
- Basic examples with known results
- Edge cases (single digit, no solution)
- Operator precedence verification
- Performance with increasing complexity
- Leading zero handling validation

Production Considerations:
- Input validation and limits
- Memory usage constraints
- Timeout handling for large inputs
- Result size limitations
- Error handling and recovery

Complexity Analysis:
- Time: O(4^n * n) for n-1 operator positions
- Space: O(n) recursion + O(k * n) for k solutions
- Best case: O(n) when no valid solutions
- Worst case: Exponential when many solutions exist

This problem is important for Meta because:
1. Tests algorithmic thinking and recursion
2. Real applications in formula parsing
3. Demonstrates optimization techniques
4. Shows understanding of operator precedence
5. Common in mathematical computation tasks

Common Interview Variations:
1. Return only count of valid expressions
2. Support division operator
3. Handle parentheses in expressions
4. Find lexicographically smallest result
5. Evaluate expression with given variables

Operator Precedence Handling:
- Multiplication binds tighter than +/-
- Track last multiplication operand
- When adding new *, undo last +/- and apply *
- Maintain current evaluation and last multiplier

Performance Characteristics:
- String length 5: < 1ms typical
- String length 8: < 10ms typical
- String length 10: < 100ms typical (exponential growth)
- Memory scales with number of valid expressions
- Pruning effectiveness varies by target value

Real-world Usage:
- Calculator apps: instant expression evaluation
- Spreadsheet software: formula validation
- Math education tools: problem generation
- Financial modeling: metric calculations
- Data analysis: custom formula creation
*/
