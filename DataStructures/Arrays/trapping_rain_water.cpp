/*
Problem: Trapping Rain Water (Hard)
Given n non-negative integers representing an elevation map where the width of each bar is 1, 
compute how much water it can trap after raining.

Example:
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6

Time Complexity: O(n)
Space Complexity: O(1)
*/

#include <iostream>
#include <vector>
#include <algorithm>

class Solution {
public:
    int trap(std::vector<int>& height) {
        if (height.empty()) return 0;
        
        int left = 0, right = height.size() - 1;
        int leftMax = 0, rightMax = 0;
        int waterTrapped = 0;
        
        while (left < right) {
            if (height[left] < height[right]) {
                if (height[left] >= leftMax) {
                    leftMax = height[left];
                } else {
                    waterTrapped += leftMax - height[left];
                }
                left++;
            } else {
                if (height[right] >= rightMax) {
                    rightMax = height[right];
                } else {
                    waterTrapped += rightMax - height[right];
                }
                right--;
            }
        }
        
        return waterTrapped;
    }
    
    // Alternative O(n) space solution using prefix/suffix arrays
    int trapWithExtraSpace(std::vector<int>& height) {
        int n = height.size();
        if (n <= 2) return 0;
        
        std::vector<int> leftMax(n), rightMax(n);
        
        leftMax[0] = height[0];
        for (int i = 1; i < n; i++) {
            leftMax[i] = std::max(leftMax[i-1], height[i]);
        }
        
        rightMax[n-1] = height[n-1];
        for (int i = n-2; i >= 0; i--) {
            rightMax[i] = std::max(rightMax[i+1], height[i]);
        }
        
        int waterTrapped = 0;
        for (int i = 0; i < n; i++) {
            waterTrapped += std::min(leftMax[i], rightMax[i]) - height[i];
        }
        
        return waterTrapped;
    }
};

int main() {
    Solution solution;
    
    // Test case 1
    std::vector<int> height1 = {0,1,0,2,1,0,1,3,2,1,2,1};
    std::cout << "Test 1: " << solution.trap(height1) << " (Expected: 6)" << std::endl;
    
    // Test case 2
    std::vector<int> height2 = {3,0,2,0,4};
    std::cout << "Test 2: " << solution.trap(height2) << " (Expected: 7)" << std::endl;
    
    // Test case 3
    std::vector<int> height3 = {4,2,0,3,2,5};
    std::cout << "Test 3: " << solution.trap(height3) << " (Expected: 9)" << std::endl;
    
    return 0;
}

/*
Key Insights:
1. Two-pointer approach is optimal for space complexity
2. Water level at any position is min(leftMax, rightMax) - height[i]
3. We can calculate leftMax and rightMax on the fly using two pointers
4. Move the pointer with smaller height to ensure we have seen the true maximum

Edge Cases:
- Empty array or single element
- All elements are the same
- Strictly increasing or decreasing array
*/
