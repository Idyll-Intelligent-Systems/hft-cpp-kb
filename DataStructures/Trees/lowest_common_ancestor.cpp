/*
Advanced Tree Problem: Lowest Common Ancestor in Binary Tree
===========================================================

Problem: Given a binary tree and two nodes p and q, find their lowest common ancestor (LCA).
The LCA is defined as the lowest node that has both p and q as descendants 
(where we allow a node to be a descendant of itself).

This problem is crucial in trading systems for:
- Order book tree structures
- Market data hierarchical organization
- Risk management tree traversals
- Portfolio hierarchy management

Multiple solutions with different trade-offs:
1. Recursive DFS - O(n) time, O(h) space
2. Iterative with parent pointers - O(n) time, O(n) space
3. Path comparison - O(n) time, O(h) space
4. Optimized for multiple queries - O(n) preprocessing, O(1) query
*/

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <queue>
#include <algorithm>

// Definition for a binary tree node
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class LowestCommonAncestor {
public:
    // Solution 1: Recursive DFS (Most Common)
    // Time: O(n), Space: O(h) where h is height
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root || root == p || root == q) {
            return root;
        }
        
        TreeNode* left = lowestCommonAncestor(root->left, p, q);
        TreeNode* right = lowestCommonAncestor(root->right, p, q);
        
        if (left && right) {
            return root; // Found LCA
        }
        
        return left ? left : right;
    }
    
    // Solution 2: Using Parent Pointers
    // Time: O(n), Space: O(n)
    TreeNode* lowestCommonAncestorWithParents(TreeNode* root, TreeNode* p, TreeNode* q) {
        // Build parent map
        std::unordered_map<TreeNode*, TreeNode*> parent;
        std::queue<TreeNode*> queue;
        queue.push(root);
        parent[root] = nullptr;
        
        while (!parent.count(p) || !parent.count(q)) {
            TreeNode* node = queue.front();
            queue.pop();
            
            if (node->left) {
                parent[node->left] = node;
                queue.push(node->left);
            }
            if (node->right) {
                parent[node->right] = node;
                queue.push(node->right);
            }
        }
        
        // Find ancestors of p
        std::unordered_set<TreeNode*> ancestors;
        while (p) {
            ancestors.insert(p);
            p = parent[p];
        }
        
        // Find first common ancestor
        while (q) {
            if (ancestors.count(q)) {
                return q;
            }
            q = parent[q];
        }
        
        return nullptr;
    }
    
    // Solution 3: Path Comparison
    // Time: O(n), Space: O(h)
    TreeNode* lowestCommonAncestorPaths(TreeNode* root, TreeNode* p, TreeNode* q) {
        std::vector<TreeNode*> pathP, pathQ;
        
        if (!findPath(root, p, pathP) || !findPath(root, q, pathQ)) {
            return nullptr;
        }
        
        int i = 0;
        while (i < pathP.size() && i < pathQ.size() && pathP[i] == pathQ[i]) {
            i++;
        }
        
        return pathP[i-1];
    }
    
private:
    bool findPath(TreeNode* root, TreeNode* target, std::vector<TreeNode*>& path) {
        if (!root) return false;
        
        path.push_back(root);
        
        if (root == target) return true;
        
        if (findPath(root->left, target, path) || findPath(root->right, target, path)) {
            return true;
        }
        
        path.pop_back();
        return false;
    }
    
public:
    // Solution 4: Optimized for Multiple Queries (Preprocessing)
    class LCAPreprocessor {
    private:
        std::vector<std::vector<TreeNode*>> up;
        std::unordered_map<TreeNode*, int> depth;
        int LOG;
        
    public:
        LCAPreprocessor(TreeNode* root) {
            if (!root) return;
            
            LOG = 20; // log2(max_nodes)
            up.assign(LOG, std::vector<TreeNode*>());
            
            dfs(root, nullptr);
            
            for (int j = 1; j < LOG; j++) {
                for (auto& [node, d] : depth) {
                    if (up[j-1].size() > d && up[j-1][d]) {
                        if (up[j].size() <= d) up[j].resize(d + 1);
                        TreeNode* parent = up[j-1][d];
                        auto it = depth.find(parent);
                        if (it != depth.end() && up[j-1].size() > it->second) {
                            up[j][d] = up[j-1][it->second];
                        }
                    }
                }
            }
        }
        
        void dfs(TreeNode* node, TreeNode* parent) {
            if (up[0].size() <= depth.size()) up[0].resize(depth.size() + 1);
            up[0][depth.size()] = parent;
            depth[node] = depth.size();
            
            if (node->left) dfs(node->left, node);
            if (node->right) dfs(node->right, node);
        }
        
        TreeNode* lca(TreeNode* p, TreeNode* q) {
            if (depth[p] < depth[q]) std::swap(p, q);
            
            int diff = depth[p] - depth[q];
            for (int i = 0; i < LOG; i++) {
                if ((diff >> i) & 1) {
                    p = up[i][depth[p]];
                }
            }
            
            if (p == q) return p;
            
            for (int i = LOG - 1; i >= 0; i--) {
                if (up[i][depth[p]] != up[i][depth[q]]) {
                    p = up[i][depth[p]];
                    q = up[i][depth[q]];
                }
            }
            
            return up[0][depth[p]];
        }
    };
};

// Trading-specific applications
class TradingTreeApplications {
public:
    // Order book tree structure
    struct OrderBookNode {
        double price;
        int quantity;
        int order_count;
        OrderBookNode* left;
        OrderBookNode* right;
        
        OrderBookNode(double p, int q) : price(p), quantity(q), order_count(1), left(nullptr), right(nullptr) {}
    };
    
    // Find LCA in order book tree for risk calculations
    static OrderBookNode* findPriceRangeLCA(OrderBookNode* root, double price1, double price2) {
        if (!root) return nullptr;
        
        if (root->price >= std::min(price1, price2) && root->price <= std::max(price1, price2)) {
            return root;
        }
        
        if (root->price > std::max(price1, price2)) {
            return findPriceRangeLCA(root->left, price1, price2);
        } else {
            return findPriceRangeLCA(root->right, price1, price2);
        }
    }
    
    // Portfolio hierarchy for risk aggregation
    struct PortfolioNode {
        std::string name;
        double exposure;
        double var;
        std::vector<PortfolioNode*> children;
        PortfolioNode* parent;
        
        PortfolioNode(const std::string& n, double exp = 0.0) 
            : name(n), exposure(exp), var(0.0), parent(nullptr) {}
    };
    
    // Find common risk aggregation level
    static PortfolioNode* findRiskAggregationLevel(PortfolioNode* node1, PortfolioNode* node2) {
        std::unordered_set<PortfolioNode*> ancestors;
        
        // Collect ancestors of node1
        PortfolioNode* current = node1;
        while (current) {
            ancestors.insert(current);
            current = current->parent;
        }
        
        // Find first common ancestor of node2
        current = node2;
        while (current) {
            if (ancestors.count(current)) {
                return current;
            }
            current = current->parent;
        }
        
        return nullptr;
    }
    
    // Market data tree for efficient range queries
    struct MarketDataNode {
        std::string symbol;
        double price;
        long long timestamp;
        MarketDataNode* left;
        MarketDataNode* right;
        
        MarketDataNode(const std::string& sym, double p, long long ts) 
            : symbol(sym), price(p), timestamp(ts), left(nullptr), right(nullptr) {}
    };
    
    // Find LCA for time range queries
    static MarketDataNode* findTimeRangeLCA(MarketDataNode* root, long long start_time, long long end_time) {
        if (!root) return nullptr;
        
        if (root->timestamp >= start_time && root->timestamp <= end_time) {
            return root;
        }
        
        MarketDataNode* left_result = findTimeRangeLCA(root->left, start_time, end_time);
        MarketDataNode* right_result = findTimeRangeLCA(root->right, start_time, end_time);
        
        if (left_result && right_result) return root;
        return left_result ? left_result : right_result;
    }
};

// Performance testing and comparison
class PerformanceTester {
public:
    static void testLCAMethods() {
        std::cout << "LCA Methods Performance Comparison:\n";
        std::cout << "===================================\n\n";
        
        // Create test tree
        TreeNode* root = createTestTree();
        LowestCommonAncestor lca;
        
        // Test nodes
        TreeNode* p = root->left->left;    // Node 4
        TreeNode* q = root->left->right;   // Node 5
        
        std::cout << "Test Tree Structure:\n";
        std::cout << "       3\n";
        std::cout << "      / \\\n";
        std::cout << "     5   1\n";
        std::cout << "    / \\ / \\\n";
        std::cout << "   6  2 0  8\n";
        std::cout << "     / \\\n";
        std::cout << "    7   4\n\n";
        
        std::cout << "Finding LCA of nodes " << p->val << " and " << q->val << ":\n\n";
        
        // Method 1: Recursive
        auto start = std::chrono::high_resolution_clock::now();
        TreeNode* result1 = lca.lowestCommonAncestor(root, p, q);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        std::cout << "Recursive DFS: LCA = " << result1->val << " (Time: " << duration1.count() << " ns)\n";
        
        // Method 2: Parent pointers
        start = std::chrono::high_resolution_clock::now();
        TreeNode* result2 = lca.lowestCommonAncestorWithParents(root, p, q);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        std::cout << "Parent Pointers: LCA = " << result2->val << " (Time: " << duration2.count() << " ns)\n";
        
        // Method 3: Path comparison
        start = std::chrono::high_resolution_clock::now();
        TreeNode* result3 = lca.lowestCommonAncestorPaths(root, p, q);
        end = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        std::cout << "Path Comparison: LCA = " << result3->val << " (Time: " << duration3.count() << " ns)\n\n";
        
        std::cout << "Space Complexity Comparison:\n";
        std::cout << "Recursive DFS: O(h) - recursion stack\n";
        std::cout << "Parent Pointers: O(n) - parent map + ancestor set\n";
        std::cout << "Path Comparison: O(h) - path vectors\n";
        std::cout << "Preprocessed: O(n log n) - binary lifting table\n\n";
    }
    
    static TreeNode* createTestTree() {
        TreeNode* root = new TreeNode(3);
        root->left = new TreeNode(5);
        root->right = new TreeNode(1);
        root->left->left = new TreeNode(6);
        root->left->right = new TreeNode(2);
        root->right->left = new TreeNode(0);
        root->right->right = new TreeNode(8);
        root->left->right->left = new TreeNode(7);
        root->left->right->right = new TreeNode(4);
        
        return root;
    }
};

// Real-world trading scenarios
class TradingScenarios {
public:
    static void orderBookLCAExample() {
        std::cout << "Trading Application: Order Book Price Range Analysis\n";
        std::cout << "===================================================\n";
        
        // Create order book tree (BST by price)
        TradingTreeApplications::OrderBookNode* orderBook = nullptr;
        orderBook = insertOrder(orderBook, 100.50, 1000);
        orderBook = insertOrder(orderBook, 100.25, 500);
        orderBook = insertOrder(orderBook, 100.75, 800);
        orderBook = insertOrder(orderBook, 100.10, 300);
        orderBook = insertOrder(orderBook, 100.40, 600);
        orderBook = insertOrder(orderBook, 100.60, 400);
        orderBook = insertOrder(orderBook, 100.90, 700);
        
        std::cout << "Order Book Structure (prices):\n";
        printOrderBook(orderBook, 0);
        std::cout << "\n";
        
        // Find LCA for price range analysis
        double price1 = 100.25, price2 = 100.75;
        auto lca_node = TradingTreeApplications::findPriceRangeLCA(orderBook, price1, price2);
        
        std::cout << "Price range analysis between $" << price1 << " and $" << price2 << ":\n";
        std::cout << "LCA price level: $" << lca_node->price << "\n";
        std::cout << "This represents the key price level for risk aggregation\n\n";
    }
    
    static TradingTreeApplications::OrderBookNode* insertOrder(
        TradingTreeApplications::OrderBookNode* root, double price, int quantity) {
        if (!root) return new TradingTreeApplications::OrderBookNode(price, quantity);
        
        if (price < root->price) {
            root->left = insertOrder(root->left, price, quantity);
        } else if (price > root->price) {
            root->right = insertOrder(root->right, price, quantity);
        } else {
            root->quantity += quantity;
            root->order_count++;
        }
        
        return root;
    }
    
    static void printOrderBook(TradingTreeApplications::OrderBookNode* root, int depth) {
        if (!root) return;
        
        printOrderBook(root->right, depth + 1);
        
        for (int i = 0; i < depth; i++) std::cout << "  ";
        std::cout << "$" << std::fixed << std::setprecision(2) << root->price 
                  << " (" << root->quantity << ")\n";
        
        printOrderBook(root->left, depth + 1);
    }
    
    static void portfolioRiskExample() {
        std::cout << "Trading Application: Portfolio Risk Hierarchy\n";
        std::cout << "============================================\n";
        
        // Create portfolio hierarchy
        auto root = new TradingTreeApplications::PortfolioNode("Total Portfolio");
        auto equities = new TradingTreeApplications::PortfolioNode("Equities");
        auto bonds = new TradingTreeApplications::PortfolioNode("Bonds");
        auto us_equities = new TradingTreeApplications::PortfolioNode("US Equities");
        auto intl_equities = new TradingTreeApplications::PortfolioNode("International Equities");
        auto tech_stocks = new TradingTreeApplications::PortfolioNode("Tech Stocks");
        auto financial_stocks = new TradingTreeApplications::PortfolioNode("Financial Stocks");
        
        // Set up hierarchy
        equities->parent = root;
        bonds->parent = root;
        us_equities->parent = equities;
        intl_equities->parent = equities;
        tech_stocks->parent = us_equities;
        financial_stocks->parent = us_equities;
        
        std::cout << "Portfolio Hierarchy:\n";
        std::cout << "Total Portfolio\n";
        std::cout << "├── Equities\n";
        std::cout << "│   ├── US Equities\n";
        std::cout << "│   │   ├── Tech Stocks\n";
        std::cout << "│   │   └── Financial Stocks\n";
        std::cout << "│   └── International Equities\n";
        std::cout << "└── Bonds\n\n";
        
        // Find common risk aggregation level
        auto common_level = TradingTreeApplications::findRiskAggregationLevel(tech_stocks, financial_stocks);
        std::cout << "Risk aggregation level for Tech and Financial stocks: " << common_level->name << "\n";
        
        common_level = TradingTreeApplications::findRiskAggregationLevel(tech_stocks, intl_equities);
        std::cout << "Risk aggregation level for Tech stocks and International equities: " << common_level->name << "\n";
        
        common_level = TradingTreeApplications::findRiskAggregationLevel(equities, bonds);
        std::cout << "Risk aggregation level for Equities and Bonds: " << common_level->name << "\n\n";
    }
};

int main() {
    std::cout << "Lowest Common Ancestor in Trading Systems\n";
    std::cout << "=========================================\n\n";
    
    // Performance testing
    PerformanceTester::testLCAMethods();
    
    // Trading applications
    TradingScenarios::orderBookLCAExample();
    TradingScenarios::portfolioRiskExample();
    
    // Algorithm complexity summary
    std::cout << "Algorithm Selection Guidelines:\n";
    std::cout << "==============================\n";
    std::cout << "• Single query: Use recursive DFS (simplest, O(n) time, O(h) space)\n";
    std::cout << "• Multiple queries on same tree: Use preprocessing (O(n log n) space, O(1) query)\n";
    std::cout << "• Memory constrained: Use iterative approaches\n";
    std::cout << "• Need path information: Use path comparison method\n\n";
    
    std::cout << "Trading System Applications:\n";
    std::cout << "• Order book price range analysis\n";
    std::cout << "• Portfolio risk hierarchy navigation\n";
    std::cout << "• Market data temporal range queries\n";
    std::cout << "• Trade execution tree optimization\n";
    
    return 0;
}
