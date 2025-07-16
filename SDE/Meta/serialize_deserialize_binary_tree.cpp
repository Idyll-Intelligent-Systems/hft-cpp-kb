/*
Meta (Facebook) SDE Interview Problem 7: Serialize and Deserialize Binary Tree (Hard)
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be 
stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in 
the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your 
serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be 
serialized to a string and this string can be deserialized to the original tree structure.

Example:
    1
   / \
  2   3
     / \
    4   5

Input: root = [1,2,3,null,null,4,5]
Serialized: "1,2,null,null,3,4,null,null,5,null,null"
Deserialized: [1,2,3,null,null,4,5]

This is a classic Meta interview problem testing tree traversal, string manipulation, and data structure design.
It's fundamental for understanding serialization protocols and distributed systems.

Time Complexity: O(n) for both serialization and deserialization
Space Complexity: O(n) for storing the serialized string and recursion stack
*/

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <queue>
#include <stack>
#include <algorithm>
#include <chrono>

// Definition for a binary tree node
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Codec {
public:
    // Approach 1: Pre-order DFS with recursion (Recommended)
    std::string serialize(TreeNode* root) {
        std::string result;
        serializeHelper(root, result);
        return result;
    }
    
    TreeNode* deserialize(std::string data) {
        std::istringstream iss(data);
        return deserializeHelper(iss);
    }
    
    // Approach 2: Level-order BFS serialization
    std::string serializeBFS(TreeNode* root) {
        if (!root) return "null";
        
        std::string result;
        std::queue<TreeNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();
            
            if (node) {
                result += std::to_string(node->val) + ",";
                q.push(node->left);
                q.push(node->right);
            } else {
                result += "null,";
            }
        }
        
        // Remove trailing comma and nulls
        if (!result.empty() && result.back() == ',') {
            result.pop_back();
        }
        
        return result;
    }
    
    TreeNode* deserializeBFS(std::string data) {
        if (data == "null" || data.empty()) return nullptr;
        
        std::vector<std::string> tokens = split(data, ',');
        if (tokens.empty()) return nullptr;
        
        TreeNode* root = new TreeNode(std::stoi(tokens[0]));
        std::queue<TreeNode*> q;
        q.push(root);
        
        int i = 1;
        while (!q.empty() && i < tokens.size()) {
            TreeNode* node = q.front();
            q.pop();
            
            // Left child
            if (i < tokens.size() && tokens[i] != "null") {
                node->left = new TreeNode(std::stoi(tokens[i]));
                q.push(node->left);
            }
            i++;
            
            // Right child
            if (i < tokens.size() && tokens[i] != "null") {
                node->right = new TreeNode(std::stoi(tokens[i]));
                q.push(node->right);
            }
            i++;
        }
        
        return root;
    }
    
    // Approach 3: Iterative DFS with stack
    std::string serializeIterative(TreeNode* root) {
        std::string result;
        std::stack<TreeNode*> stk;
        stk.push(root);
        
        while (!stk.empty()) {
            TreeNode* node = stk.top();
            stk.pop();
            
            if (node) {
                result += std::to_string(node->val) + ",";
                stk.push(node->right); // Push right first (LIFO)
                stk.push(node->left);
            } else {
                result += "null,";
            }
        }
        
        if (!result.empty() && result.back() == ',') {
            result.pop_back();
        }
        
        return result;
    }
    
    TreeNode* deserializeIterative(std::string data) {
        if (data.empty()) return nullptr;
        
        std::vector<std::string> tokens = split(data, ',');
        if (tokens.empty() || tokens[0] == "null") return nullptr;
        
        int index = 0;
        TreeNode* root = new TreeNode(std::stoi(tokens[index++]));
        std::stack<TreeNode*> stk;
        stk.push(root);
        
        while (!stk.empty() && index < tokens.size()) {
            TreeNode* node = stk.top();
            stk.pop();
            
            // Process left child
            if (index < tokens.size()) {
                if (tokens[index] != "null") {
                    node->left = new TreeNode(std::stoi(tokens[index]));
                    stk.push(node->left);
                }
                index++;
            }
            
            // Process right child
            if (index < tokens.size()) {
                if (tokens[index] != "null") {
                    node->right = new TreeNode(std::stoi(tokens[index]));
                    stk.push(node->right);
                }
                index++;
            }
        }
        
        return root;
    }
    
    // Approach 4: Compact binary encoding
    std::string serializeBinary(TreeNode* root) {
        std::vector<int> preorder, inorder;
        getPreorder(root, preorder);
        getInorder(root, inorder);
        
        std::string result;
        result += std::to_string(preorder.size()) + ",";
        
        for (int val : preorder) {
            result += std::to_string(val) + ",";
        }
        
        for (int val : inorder) {
            result += std::to_string(val) + ",";
        }
        
        if (!result.empty() && result.back() == ',') {
            result.pop_back();
        }
        
        return result;
    }
    
    TreeNode* deserializeBinary(std::string data) {
        if (data.empty()) return nullptr;
        
        std::vector<std::string> tokens = split(data, ',');
        if (tokens.empty()) return nullptr;
        
        int size = std::stoi(tokens[0]);
        if (size == 0) return nullptr;
        
        std::vector<int> preorder(size), inorder(size);
        
        for (int i = 0; i < size; i++) {
            preorder[i] = std::stoi(tokens[1 + i]);
            inorder[i] = std::stoi(tokens[1 + size + i]);
        }
        
        return buildTree(preorder, inorder);
    }
    
    // Approach 5: JSON-like format
    std::string serializeJSON(TreeNode* root) {
        if (!root) return "null";
        
        std::string result = "{";
        result += "\"val\":" + std::to_string(root->val) + ",";
        result += "\"left\":" + serializeJSON(root->left) + ",";
        result += "\"right\":" + serializeJSON(root->right);
        result += "}";
        
        return result;
    }
    
    TreeNode* deserializeJSON(std::string data) {
        if (data == "null") return nullptr;
        
        // Simple JSON parser for this specific format
        int pos = 0;
        return parseJSONNode(data, pos);
    }
    
private:
    void serializeHelper(TreeNode* root, std::string& result) {
        if (!root) {
            result += "null,";
            return;
        }
        
        result += std::to_string(root->val) + ",";
        serializeHelper(root->left, result);
        serializeHelper(root->right, result);
    }
    
    TreeNode* deserializeHelper(std::istringstream& iss) {
        std::string val;
        if (!std::getline(iss, val, ',')) {
            return nullptr;
        }
        
        if (val == "null") {
            return nullptr;
        }
        
        TreeNode* root = new TreeNode(std::stoi(val));
        root->left = deserializeHelper(iss);
        root->right = deserializeHelper(iss);
        
        return root;
    }
    
    std::vector<std::string> split(const std::string& str, char delimiter) {
        std::vector<std::string> tokens;
        std::istringstream iss(str);
        std::string token;
        
        while (std::getline(iss, token, delimiter)) {
            tokens.push_back(token);
        }
        
        return tokens;
    }
    
    void getPreorder(TreeNode* root, std::vector<int>& result) {
        if (!root) return;
        result.push_back(root->val);
        getPreorder(root->left, result);
        getPreorder(root->right, result);
    }
    
    void getInorder(TreeNode* root, std::vector<int>& result) {
        if (!root) return;
        getInorder(root->left, result);
        result.push_back(root->val);
        getInorder(root->right, result);
    }
    
    TreeNode* buildTree(std::vector<int>& preorder, std::vector<int>& inorder) {
        if (preorder.empty() || inorder.empty()) return nullptr;
        
        return buildTreeHelper(preorder, 0, preorder.size() - 1,
                              inorder, 0, inorder.size() - 1);
    }
    
    TreeNode* buildTreeHelper(std::vector<int>& preorder, int preStart, int preEnd,
                             std::vector<int>& inorder, int inStart, int inEnd) {
        if (preStart > preEnd || inStart > inEnd) return nullptr;
        
        TreeNode* root = new TreeNode(preorder[preStart]);
        
        // Find root in inorder
        int rootIdx = inStart;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == root->val) {
                rootIdx = i;
                break;
            }
        }
        
        int leftSize = rootIdx - inStart;
        
        root->left = buildTreeHelper(preorder, preStart + 1, preStart + leftSize,
                                   inorder, inStart, rootIdx - 1);
        root->right = buildTreeHelper(preorder, preStart + leftSize + 1, preEnd,
                                    inorder, rootIdx + 1, inEnd);
        
        return root;
    }
    
    TreeNode* parseJSONNode(const std::string& data, int& pos) {
        if (pos >= data.length()) return nullptr;
        
        if (data.substr(pos, 4) == "null") {
            pos += 4;
            return nullptr;
        }
        
        // Skip '{'
        pos++;
        
        // Parse "val":value
        while (pos < data.length() && data[pos] != ':') pos++;
        pos++; // Skip ':'
        
        int valStart = pos;
        while (pos < data.length() && data[pos] != ',' && data[pos] != '}') pos++;
        int val = std::stoi(data.substr(valStart, pos - valStart));
        
        TreeNode* node = new TreeNode(val);
        
        // Parse left child
        while (pos < data.length() && data[pos] != ':') pos++;
        pos++; // Skip ':'
        node->left = parseJSONNode(data, pos);
        
        // Parse right child
        while (pos < data.length() && data[pos] != ':') pos++;
        pos++; // Skip ':'
        node->right = parseJSONNode(data, pos);
        
        // Skip '}'
        while (pos < data.length() && data[pos] != '}') pos++;
        pos++;
        
        return node;
    }
};

// Tree utilities for testing
class TreeUtils {
public:
    static TreeNode* createExampleTree() {
        TreeNode* root = new TreeNode(1);
        root->left = new TreeNode(2);
        root->right = new TreeNode(3);
        root->right->left = new TreeNode(4);
        root->right->right = new TreeNode(5);
        return root;
    }
    
    static TreeNode* createComplexTree() {
        TreeNode* root = new TreeNode(1);
        root->left = new TreeNode(2);
        root->right = new TreeNode(3);
        root->left->left = new TreeNode(4);
        root->left->right = new TreeNode(5);
        root->right->left = new TreeNode(6);
        root->right->right = new TreeNode(7);
        root->left->left->left = new TreeNode(8);
        root->left->left->right = new TreeNode(9);
        return root;
    }
    
    static TreeNode* createSkewedTree() {
        TreeNode* root = new TreeNode(1);
        root->right = new TreeNode(2);
        root->right->right = new TreeNode(3);
        root->right->right->right = new TreeNode(4);
        return root;
    }
    
    static void deleteTree(TreeNode* root) {
        if (!root) return;
        deleteTree(root->left);
        deleteTree(root->right);
        delete root;
    }
    
    static bool areTreesEqual(TreeNode* t1, TreeNode* t2) {
        if (!t1 && !t2) return true;
        if (!t1 || !t2) return false;
        
        return (t1->val == t2->val) && 
               areTreesEqual(t1->left, t2->left) && 
               areTreesEqual(t1->right, t2->right);
    }
    
    static void printInorder(TreeNode* root, std::vector<int>& result) {
        if (!root) return;
        printInorder(root->left, result);
        result.push_back(root->val);
        printInorder(root->right, result);
    }
    
    static void printPreorder(TreeNode* root, std::vector<int>& result) {
        if (!root) return;
        result.push_back(root->val);
        printPreorder(root->left, result);
        printPreorder(root->right, result);
    }
    
    static void printTree(TreeNode* root, const std::string& title) {
        std::cout << title << std::endl;
        std::vector<int> inorder, preorder;
        printInorder(root, inorder);
        printPreorder(root, preorder);
        
        std::cout << "Inorder: ";
        for (int val : inorder) std::cout << val << " ";
        std::cout << std::endl;
        
        std::cout << "Preorder: ";
        for (int val : preorder) std::cout << val << " ";
        std::cout << std::endl;
    }
    
    static TreeNode* createLargeTree(int levels) {
        if (levels <= 0) return nullptr;
        
        TreeNode* root = new TreeNode(1);
        std::queue<TreeNode*> q;
        q.push(root);
        
        int currentLevel = 1;
        int nodeVal = 2;
        
        while (currentLevel < levels && !q.empty()) {
            int levelSize = q.size();
            
            for (int i = 0; i < levelSize && currentLevel < levels; i++) {
                TreeNode* node = q.front();
                q.pop();
                
                node->left = new TreeNode(nodeVal++);
                node->right = new TreeNode(nodeVal++);
                
                q.push(node->left);
                q.push(node->right);
            }
            
            currentLevel++;
        }
        
        return root;
    }
};

// Test framework
class SerializeDeserializeTest {
public:
    static void runTests() {
        std::cout << "Meta Serialize and Deserialize Binary Tree Tests:" << std::endl;
        std::cout << "================================================" << std::endl;
        
        testBasicCases();
        testEdgeCases();
        compareApproaches();
        metaSpecificScenarios();
        performanceAnalysis();
    }
    
    static void testBasicCases() {
        std::cout << "\nBasic Test Cases:" << std::endl;
        std::cout << "=================" << std::endl;
        
        Codec codec;
        
        // Test case 1: Example tree
        TreeNode* tree1 = TreeUtils::createExampleTree();
        std::string serialized1 = codec.serialize(tree1);
        TreeNode* deserialized1 = codec.deserialize(serialized1);
        
        std::cout << "Test 1 - Example tree:" << std::endl;
        std::cout << "Serialized: " << serialized1 << std::endl;
        TreeUtils::printTree(tree1, "Original:");
        TreeUtils::printTree(deserialized1, "Deserialized:");
        std::cout << "Trees equal: " << (TreeUtils::areTreesEqual(tree1, deserialized1) ? "✅" : "❌") << std::endl;
        
        // Test case 2: Complex tree
        TreeNode* tree2 = TreeUtils::createComplexTree();
        std::string serialized2 = codec.serialize(tree2);
        TreeNode* deserialized2 = codec.deserialize(serialized2);
        
        std::cout << "\nTest 2 - Complex tree:" << std::endl;
        std::cout << "Serialized length: " << serialized2.length() << std::endl;
        std::cout << "Trees equal: " << (TreeUtils::areTreesEqual(tree2, deserialized2) ? "✅" : "❌") << std::endl;
        
        // Test case 3: Skewed tree
        TreeNode* tree3 = TreeUtils::createSkewedTree();
        std::string serialized3 = codec.serialize(tree3);
        TreeNode* deserialized3 = codec.deserialize(serialized3);
        
        std::cout << "\nTest 3 - Skewed tree:" << std::endl;
        std::cout << "Serialized: " << serialized3 << std::endl;
        std::cout << "Trees equal: " << (TreeUtils::areTreesEqual(tree3, deserialized3) ? "✅" : "❌") << std::endl;
        
        // Cleanup
        TreeUtils::deleteTree(tree1);
        TreeUtils::deleteTree(deserialized1);
        TreeUtils::deleteTree(tree2);
        TreeUtils::deleteTree(deserialized2);
        TreeUtils::deleteTree(tree3);
        TreeUtils::deleteTree(deserialized3);
    }
    
    static void testEdgeCases() {
        std::cout << "\nEdge Case Tests:" << std::endl;
        std::cout << "================" << std::endl;
        
        Codec codec;
        
        // Empty tree
        TreeNode* emptyTree = nullptr;
        std::string serializedEmpty = codec.serialize(emptyTree);
        TreeNode* deserializedEmpty = codec.deserialize(serializedEmpty);
        
        std::cout << "Empty tree: " << std::endl;
        std::cout << "Serialized: '" << serializedEmpty << "'" << std::endl;
        std::cout << "Deserialized null: " << (deserializedEmpty == nullptr ? "✅" : "❌") << std::endl;
        
        // Single node
        TreeNode* singleNode = new TreeNode(42);
        std::string serializedSingle = codec.serialize(singleNode);
        TreeNode* deserializedSingle = codec.deserialize(serializedSingle);
        
        std::cout << "\nSingle node:" << std::endl;
        std::cout << "Serialized: " << serializedSingle << std::endl;
        std::cout << "Trees equal: " << (TreeUtils::areTreesEqual(singleNode, deserializedSingle) ? "✅" : "❌") << std::endl;
        
        // Only left child
        TreeNode* leftOnly = new TreeNode(1);
        leftOnly->left = new TreeNode(2);
        leftOnly->left->left = new TreeNode(3);
        
        std::string serializedLeft = codec.serialize(leftOnly);
        TreeNode* deserializedLeft = codec.deserialize(serializedLeft);
        
        std::cout << "\nLeft-only tree:" << std::endl;
        std::cout << "Serialized: " << serializedLeft << std::endl;
        std::cout << "Trees equal: " << (TreeUtils::areTreesEqual(leftOnly, deserializedLeft) ? "✅" : "❌") << std::endl;
        
        // Negative values
        TreeNode* negativeTree = new TreeNode(-1);
        negativeTree->left = new TreeNode(-2);
        negativeTree->right = new TreeNode(-3);
        
        std::string serializedNeg = codec.serialize(negativeTree);
        TreeNode* deserializedNeg = codec.deserialize(serializedNeg);
        
        std::cout << "\nNegative values:" << std::endl;
        std::cout << "Serialized: " << serializedNeg << std::endl;
        std::cout << "Trees equal: " << (TreeUtils::areTreesEqual(negativeTree, deserializedNeg) ? "✅" : "❌") << std::endl;
        
        // Cleanup
        TreeUtils::deleteTree(singleNode);
        TreeUtils::deleteTree(deserializedSingle);
        TreeUtils::deleteTree(leftOnly);
        TreeUtils::deleteTree(deserializedLeft);
        TreeUtils::deleteTree(negativeTree);
        TreeUtils::deleteTree(deserializedNeg);
    }
    
    static void compareApproaches() {
        std::cout << "\nApproach Comparison:" << std::endl;
        std::cout << "===================" << std::endl;
        
        Codec codec;
        TreeNode* testTree = TreeUtils::createLargeTree(10);
        
        // Test different serialization methods
        auto start = std::chrono::high_resolution_clock::now();
        std::string result1 = codec.serialize(testTree);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        std::string result2 = codec.serializeBFS(testTree);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        std::string result3 = codec.serializeIterative(testTree);
        end = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        std::string result4 = codec.serializeBinary(testTree);
        end = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        std::string result5 = codec.serializeJSON(testTree);
        end = std::chrono::high_resolution_clock::now();
        auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "DFS Preorder: " << duration1.count() << " μs, length: " << result1.length() << std::endl;
        std::cout << "BFS Level-order: " << duration2.count() << " μs, length: " << result2.length() << std::endl;
        std::cout << "Iterative DFS: " << duration3.count() << " μs, length: " << result3.length() << std::endl;
        std::cout << "Binary encoding: " << duration4.count() << " μs, length: " << result4.length() << std::endl;
        std::cout << "JSON format: " << duration5.count() << " μs, length: " << result5.length() << std::endl;
        
        // Test deserialization
        start = std::chrono::high_resolution_clock::now();
        TreeNode* tree1 = codec.deserialize(result1);
        end = std::chrono::high_resolution_clock::now();
        auto deserDuration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        TreeNode* tree2 = codec.deserializeBFS(result2);
        end = std::chrono::high_resolution_clock::now();
        auto deserDuration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "\nDeserialization times:" << std::endl;
        std::cout << "DFS: " << deserDuration1.count() << " μs" << std::endl;
        std::cout << "BFS: " << deserDuration2.count() << " μs" << std::endl;
        
        // Verify correctness
        bool correct1 = TreeUtils::areTreesEqual(testTree, tree1);
        bool correct2 = TreeUtils::areTreesEqual(testTree, tree2);
        
        std::cout << "\nCorrectness verification:" << std::endl;
        std::cout << "DFS correct: " << (correct1 ? "✅" : "❌") << std::endl;
        std::cout << "BFS correct: " << (correct2 ? "✅" : "❌") << std::endl;
        
        // Cleanup
        TreeUtils::deleteTree(testTree);
        TreeUtils::deleteTree(tree1);
        TreeUtils::deleteTree(tree2);
    }
    
    static void metaSpecificScenarios() {
        std::cout << "\nMeta-Specific Scenarios:" << std::endl;
        std::cout << "=======================" << std::endl;
        
        Codec codec;
        
        // Scenario 1: Social network hierarchy serialization
        std::cout << "Social network user hierarchy:" << std::endl;
        TreeNode* socialHierarchy = new TreeNode(1); // CEO
        socialHierarchy->left = new TreeNode(2); // VP Engineering
        socialHierarchy->right = new TreeNode(3); // VP Marketing
        socialHierarchy->left->left = new TreeNode(4); // Director
        socialHierarchy->left->right = new TreeNode(5); // Director
        
        std::string socialSerialized = codec.serialize(socialHierarchy);
        std::cout << "Hierarchy serialized: " << socialSerialized << std::endl;
        
        TreeNode* socialDeserialized = codec.deserialize(socialSerialized);
        std::cout << "Restoration successful: " << (TreeUtils::areTreesEqual(socialHierarchy, socialDeserialized) ? "✅" : "❌") << std::endl;
        
        // Scenario 2: Content dependency tree
        std::cout << "\nContent dependency tree:" << std::endl;
        TreeNode* contentTree = new TreeNode(100); // Main page
        contentTree->left = new TreeNode(200); // CSS dependency
        contentTree->right = new TreeNode(300); // JS dependency
        contentTree->left->left = new TreeNode(400); // Font file
        contentTree->right->left = new TreeNode(500); // Library
        
        std::string contentSerialized = codec.serializeBFS(contentTree);
        std::cout << "Content tree (BFS): " << contentSerialized << std::endl;
        
        TreeNode* contentDeserialized = codec.deserializeBFS(contentSerialized);
        std::cout << "Content tree restored: " << (TreeUtils::areTreesEqual(contentTree, contentDeserialized) ? "✅" : "❌") << std::endl;
        
        // Scenario 3: Permission tree structure
        std::cout << "\nPermission tree structure:" << std::endl;
        TreeNode* permissionTree = new TreeNode(1); // Admin
        permissionTree->left = new TreeNode(2); // Moderator
        permissionTree->right = new TreeNode(3); // User
        permissionTree->left->left = new TreeNode(4); // Content mod
        
        std::string permissionJSON = codec.serializeJSON(permissionTree);
        std::cout << "Permission JSON: " << permissionJSON.substr(0, 50) << "..." << std::endl;
        
        TreeNode* permissionDeserialized = codec.deserializeJSON(permissionJSON);
        std::cout << "Permission tree restored: " << (TreeUtils::areTreesEqual(permissionTree, permissionDeserialized) ? "✅" : "❌") << std::endl;
        
        // Cleanup
        TreeUtils::deleteTree(socialHierarchy);
        TreeUtils::deleteTree(socialDeserialized);
        TreeUtils::deleteTree(contentTree);
        TreeUtils::deleteTree(contentDeserialized);
        TreeUtils::deleteTree(permissionTree);
        TreeUtils::deleteTree(permissionDeserialized);
    }
    
    static void performanceAnalysis() {
        std::cout << "\nPerformance Analysis:" << std::endl;
        std::cout << "====================" << std::endl;
        
        Codec codec;
        
        // Test with increasing tree sizes
        std::vector<int> treeLevels = {5, 8, 10, 12};
        
        for (int level : treeLevels) {
            TreeNode* testTree = TreeUtils::createLargeTree(level);
            int nodeCount = (1 << level) - 1; // 2^level - 1 nodes
            
            // Serialization performance
            auto start = std::chrono::high_resolution_clock::now();
            std::string serialized = codec.serialize(testTree);
            auto end = std::chrono::high_resolution_clock::now();
            auto serializeTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            // Deserialization performance
            start = std::chrono::high_resolution_clock::now();
            TreeNode* deserialized = codec.deserialize(serialized);
            end = std::chrono::high_resolution_clock::now();
            auto deserializeTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            std::cout << "Level " << level << " (" << nodeCount << " nodes):" << std::endl;
            std::cout << "  Serialize: " << serializeTime.count() << " μs" << std::endl;
            std::cout << "  Deserialize: " << deserializeTime.count() << " μs" << std::endl;
            std::cout << "  String length: " << serialized.length() << std::endl;
            std::cout << "  Correctness: " << (TreeUtils::areTreesEqual(testTree, deserialized) ? "✅" : "❌") << std::endl;
            
            TreeUtils::deleteTree(testTree);
            TreeUtils::deleteTree(deserialized);
        }
        
        // Memory usage analysis
        std::cout << "\nMemory Usage Analysis:" << std::endl;
        std::cout << "=====================" << std::endl;
        
        for (int nodes = 100; nodes <= 10000; nodes *= 10) {
            size_t nodeMemory = nodes * sizeof(TreeNode);
            size_t stringMemory = nodes * 10; // Estimate ~10 chars per node
            size_t stackMemory = 64 * sizeof(void*); // Max recursion depth
            
            std::cout << "Nodes " << nodes << ": ~" << (nodeMemory + stringMemory + stackMemory) / 1024 << " KB" << std::endl;
        }
        
        // Compression analysis
        std::cout << "\nCompression Analysis:" << std::endl;
        std::cout << "====================" << std::endl;
        
        TreeNode* compressTree = TreeUtils::createLargeTree(8);
        
        std::string dfsResult = codec.serialize(compressTree);
        std::string bfsResult = codec.serializeBFS(compressTree);
        std::string binaryResult = codec.serializeBinary(compressTree);
        std::string jsonResult = codec.serializeJSON(compressTree);
        
        std::cout << "DFS format: " << dfsResult.length() << " chars" << std::endl;
        std::cout << "BFS format: " << bfsResult.length() << " chars" << std::endl;
        std::cout << "Binary format: " << binaryResult.length() << " chars" << std::endl;
        std::cout << "JSON format: " << jsonResult.length() << " chars" << std::endl;
        
        std::cout << "\nCompression ratios (vs DFS):" << std::endl;
        std::cout << "BFS: " << (double)bfsResult.length() / dfsResult.length() << "x" << std::endl;
        std::cout << "Binary: " << (double)binaryResult.length() / dfsResult.length() << "x" << std::endl;
        std::cout << "JSON: " << (double)jsonResult.length() / dfsResult.length() << "x" << std::endl;
        
        TreeUtils::deleteTree(compressTree);
    }
};

int main() {
    SerializeDeserializeTest::runTests();
    return 0;
}

/*
Algorithm Analysis:

Core Problem: Convert binary tree to string and back while preserving structure

Key Insights:
1. Need to handle null nodes explicitly in serialization
2. Traversal order determines serialization format
3. Deserialization must reverse the serialization process exactly
4. Must handle edge cases like empty trees and single nodes

Approach Comparison:

1. DFS Preorder (Recommended):
   - Time: O(n) for both operations
   - Space: O(n) for string + O(h) for recursion
   - Natural recursive implementation
   - Compact representation with null markers

2. BFS Level-order:
   - Time: O(n) for both operations
   - Space: O(n) for queue and string
   - Intuitive level-by-level processing
   - Good for visualization and debugging

3. Iterative DFS:
   - Time: O(n) for both operations
   - Space: O(n) for stack and string
   - Avoids recursion depth limits
   - More complex state management

4. Binary Encoding (Preorder + Inorder):
   - Time: O(n log n) due to tree reconstruction
   - Space: O(n) for both traversals
   - Most compact when no nulls needed
   - Only works for trees with unique values

5. JSON Format:
   - Time: O(n) for serialization, complex parsing
   - Space: O(n) with high constant factor
   - Human-readable and self-describing
   - Compatible with web standards

Meta Interview Focus:
- Tree traversal algorithms (DFS/BFS)
- String manipulation and parsing
- Recursion and state management
- Data structure design principles
- Serialization protocol considerations

Key Design Decisions:
1. Traversal order (preorder, level-order, etc.)
2. Null node representation
3. Delimiter choice and escaping
4. Memory vs readability trade-offs

Real-world Applications at Meta:
- Social graph persistence and transmission
- Content hierarchy serialization
- Configuration tree storage
- Permission structure encoding
- Cache invalidation tree representation

Edge Cases:
- Empty tree (null root)
- Single node tree
- Highly unbalanced trees
- Trees with negative values
- Very deep trees (stack overflow)

Interview Tips:
1. Start with DFS preorder approach
2. Explain null node handling clearly
3. Discuss delimiter choice rationale
4. Handle edge cases explicitly
5. Consider different format trade-offs

Common Mistakes:
1. Not handling null nodes properly
2. Incorrect parsing logic for deserialization
3. Memory leaks in tree construction
4. Wrong delimiter causing parsing errors
5. Not considering negative numbers

Advanced Optimizations:
- Compact binary encoding for space efficiency
- Streaming serialization for large trees
- Compression techniques for repeated patterns
- Parallel processing for very large trees
- Custom formats for specific use cases

Testing Strategy:
- Basic functionality with known trees
- Edge cases (empty, single node, skewed)
- Round-trip correctness verification
- Performance with large trees
- Format comparison and analysis

Production Considerations:
- Version compatibility for format changes
- Error handling and recovery
- Memory limits for large structures
- Network transmission efficiency
- Cross-language compatibility

Complexity Analysis:
- Time: O(n) for both serialization and deserialization
- Space: O(n) for string storage + O(h) for recursion
- Best case: O(n) for balanced trees
- Worst case: O(n) even for skewed trees

This problem is important for Meta because:
1. Fundamental for distributed systems
2. Essential for data persistence
3. Tests tree algorithm understanding
4. Real applications in social graph storage
5. Demonstrates serialization design skills

Common Interview Variations:
1. Serialize N-ary tree
2. Binary serialization instead of string
3. Space-optimized encoding
4. Streaming serialization/deserialization
5. Serialize with additional node properties

Serialization Formats:

1. Preorder with nulls: "1,2,null,null,3,4,null,null,5,null,null"
2. Level-order: "1,2,3,null,null,4,5"
3. Binary: preorder + inorder arrays
4. JSON: {"val":1,"left":{"val":2...},"right":...}
5. Custom: application-specific encoding

Performance Characteristics:
- Small trees (< 100 nodes): < 1ms
- Medium trees (< 1K nodes): < 10ms
- Large trees (< 10K nodes): < 100ms
- Memory scales linearly with tree size
- String length typically 3-5x node count

Real-world Usage:
- Database: Tree index persistence
- Caching: Social graph serialization
- APIs: Hierarchical data transmission
- Backup: Configuration tree storage
- Analytics: Decision tree export/import

Format Comparison:
- DFS: Compact, easy to implement
- BFS: Intuitive, good for debugging
- Binary: Space-efficient for unique values
- JSON: Human-readable, web-compatible
- Custom: Optimized for specific needs
*/
