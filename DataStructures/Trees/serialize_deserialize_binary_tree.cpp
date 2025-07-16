/*
Problem: Serialize and Deserialize Binary Tree (Hard)
Design an algorithm to serialize and deserialize a binary tree. There is no restriction 
on how your serialization/deserialization algorithm should work. You just need to ensure 
that a binary tree can be serialized to a string and this string can be deserialized to 
the original tree structure.

Example:
Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]

Time Complexity: O(n) for both serialize and deserialize
Space Complexity: O(n) for storing the serialized string and recursion stack
*/

#include <iostream>
#include <string>
#include <sstream>
#include <queue>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Codec {
public:
    // Approach 1: Preorder Traversal with Recursion
    std::string serialize(TreeNode* root) {
        std::string result;
        serializeHelper(root, result);
        return result;
    }
    
    TreeNode* deserialize(const std::string& data) {
        std::istringstream iss(data);
        return deserializeHelper(iss);
    }
    
private:
    void serializeHelper(TreeNode* node, std::string& result) {
        if (!node) {
            result += "null,";
            return;
        }
        
        result += std::to_string(node->val) + ",";
        serializeHelper(node->left, result);
        serializeHelper(node->right, result);
    }
    
    TreeNode* deserializeHelper(std::istringstream& iss) {
        std::string val;
        if (!std::getline(iss, val, ',') || val == "null") {
            return nullptr;
        }
        
        TreeNode* node = new TreeNode(std::stoi(val));
        node->left = deserializeHelper(iss);
        node->right = deserializeHelper(iss);
        
        return node;
    }
    
public:
    // Approach 2: Level Order Traversal (BFS)
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
        
        return result;
    }
    
    TreeNode* deserializeBFS(const std::string& data) {
        if (data == "null") return nullptr;
        
        std::istringstream iss(data);
        std::string val;
        std::getline(iss, val, ',');
        
        TreeNode* root = new TreeNode(std::stoi(val));
        std::queue<TreeNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();
            
            // Process left child
            if (std::getline(iss, val, ',')) {
                if (val != "null") {
                    node->left = new TreeNode(std::stoi(val));
                    q.push(node->left);
                }
            }
            
            // Process right child
            if (std::getline(iss, val, ',')) {
                if (val != "null") {
                    node->right = new TreeNode(std::stoi(val));
                    q.push(node->right);
                }
            }
        }
        
        return root;
    }
};

// Helper functions for testing
TreeNode* buildSampleTree() {
    /*
    Build tree:
        1
       / \
      2   3
         / \
        4   5
    */
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->right->left = new TreeNode(4);
    root->right->right = new TreeNode(5);
    return root;
}

void inorderTraversal(TreeNode* root) {
    if (!root) return;
    inorderTraversal(root->left);
    std::cout << root->val << " ";
    inorderTraversal(root->right);
}

void deleteTree(TreeNode* root) {
    if (!root) return;
    deleteTree(root->left);
    deleteTree(root->right);
    delete root;
}

bool areTreesEqual(TreeNode* t1, TreeNode* t2) {
    if (!t1 && !t2) return true;
    if (!t1 || !t2) return false;
    return t1->val == t2->val && 
           areTreesEqual(t1->left, t2->left) && 
           areTreesEqual(t1->right, t2->right);
}

int main() {
    Codec codec;
    
    // Test case 1: Normal tree
    TreeNode* originalTree = buildSampleTree();
    std::cout << "Original tree (inorder): ";
    inorderTraversal(originalTree);
    std::cout << std::endl;
    
    // Test preorder serialization/deserialization
    std::string serialized = codec.serialize(originalTree);
    std::cout << "Serialized (preorder): " << serialized << std::endl;
    
    TreeNode* deserializedTree = codec.deserialize(serialized);
    std::cout << "Deserialized tree (inorder): ";
    inorderTraversal(deserializedTree);
    std::cout << std::endl;
    
    std::cout << "Trees are equal: " << (areTreesEqual(originalTree, deserializedTree) ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
    
    // Test BFS serialization/deserialization
    std::string serializedBFS = codec.serializeBFS(originalTree);
    std::cout << "Serialized (BFS): " << serializedBFS << std::endl;
    
    TreeNode* deserializedBFS = codec.deserializeBFS(serializedBFS);
    std::cout << "Deserialized BFS tree (inorder): ";
    inorderTraversal(deserializedBFS);
    std::cout << std::endl;
    
    std::cout << "BFS Trees are equal: " << (areTreesEqual(originalTree, deserializedBFS) ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
    
    // Test case 2: Empty tree
    TreeNode* emptyTree = nullptr;
    std::string emptySerialized = codec.serialize(emptyTree);
    std::cout << "Empty tree serialized: " << emptySerialized << std::endl;
    
    TreeNode* emptyDeserialized = codec.deserialize(emptySerialized);
    std::cout << "Empty tree deserialized: " << (emptyDeserialized == nullptr ? "null" : "not null") << std::endl;
    std::cout << std::endl;
    
    // Test case 3: Single node tree
    TreeNode* singleNode = new TreeNode(42);
    std::string singleSerialized = codec.serialize(singleNode);
    std::cout << "Single node serialized: " << singleSerialized << std::endl;
    
    TreeNode* singleDeserialized = codec.deserialize(singleSerialized);
    std::cout << "Single node deserialized value: " << singleDeserialized->val << std::endl;
    
    // Cleanup
    deleteTree(originalTree);
    deleteTree(deserializedTree);
    deleteTree(deserializedBFS);
    deleteTree(emptyDeserialized);
    deleteTree(singleNode);
    deleteTree(singleDeserialized);
    
    return 0;
}

/*
Algorithm Analysis:

1. Preorder Traversal Approach:
   - Time: O(n) for both serialize and deserialize
   - Space: O(n) for recursion stack (worst case for skewed tree)
   - Advantages: Natural recursive structure, compact representation
   - Uses preorder: root -> left -> right

2. Level Order Traversal (BFS) Approach:
   - Time: O(n) for both operations
   - Space: O(w) where w is maximum width of tree for queue
   - Advantages: Iterative, easier to understand for some
   - More intuitive for those familiar with level-order traversal

Key Design Decisions:
1. Use comma as delimiter for easy parsing
2. Use "null" to represent null nodes (maintains tree structure)
3. For preorder: process root first, then recursively process children
4. For BFS: use queue to process nodes level by level

Edge Cases Handled:
- Empty tree (null root)
- Single node tree
- Skewed trees (left-heavy or right-heavy)
- Trees with negative values
- Large trees

Optimization Considerations:
- Could use more compact representations (binary encoding)
- Could optimize string concatenation with StringBuilder equivalent
- Could use iterative preorder to avoid recursion stack overflow
*/
