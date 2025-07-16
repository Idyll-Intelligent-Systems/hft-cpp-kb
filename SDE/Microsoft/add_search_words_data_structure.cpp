/*
Microsoft SDE Interview Problem 2: Design Add and Search Words Data Structure (Hard)
Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the WordDictionary class:
- WordDictionary() Initializes the object.
- void addWord(word) Adds word to the data structure, it can be matched later.
- bool search(word) Returns true if there is any string in the data structure that matches word or false otherwise. 
  word may contain dots '.' where dots can be matched with any letter.

Example:
Input: ["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
       [[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output: [null,null,null,null,false,true,true,true]

Time Complexity: O(n) for addWord, O(n * 26^k) for search in worst case where k is dots
Space Complexity: O(ALPHABET_SIZE * N * M) where N is number of words, M is average length
*/

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

// Approach 1: Trie with DFS for wildcard search
class WordDictionary {
private:
    struct TrieNode {
        std::unordered_map<char, std::unique_ptr<TrieNode>> children;
        bool isEndOfWord;
        
        TrieNode() : isEndOfWord(false) {}
    };
    
    std::unique_ptr<TrieNode> root;
    
    bool searchHelper(const std::string& word, int index, TrieNode* node) {
        if (index == word.length()) {
            return node->isEndOfWord;
        }
        
        char ch = word[index];
        
        if (ch == '.') {
            // Wildcard: try all possible children
            for (auto& [key, child] : node->children) {
                if (searchHelper(word, index + 1, child.get())) {
                    return true;
                }
            }
            return false;
        } else {
            // Regular character
            if (node->children.find(ch) == node->children.end()) {
                return false;
            }
            return searchHelper(word, index + 1, node->children[ch].get());
        }
    }
    
public:
    WordDictionary() {
        root = std::make_unique<TrieNode>();
    }
    
    void addWord(std::string word) {
        TrieNode* current = root.get();
        
        for (char ch : word) {
            if (current->children.find(ch) == current->children.end()) {
                current->children[ch] = std::make_unique<TrieNode>();
            }
            current = current->children[ch].get();
        }
        
        current->isEndOfWord = true;
    }
    
    bool search(std::string word) {
        return searchHelper(word, 0, root.get());
    }
};

// Approach 2: Optimized with array-based Trie (for ASCII)
class WordDictionaryOptimized {
private:
    struct TrieNode {
        std::vector<TrieNode*> children;
        bool isEndOfWord;
        
        TrieNode() : children(26, nullptr), isEndOfWord(false) {}
        
        ~TrieNode() {
            for (TrieNode* child : children) {
                delete child;
            }
        }
    };
    
    TrieNode* root;
    
    bool searchHelper(const std::string& word, int index, TrieNode* node) {
        if (index == word.length()) {
            return node->isEndOfWord;
        }
        
        char ch = word[index];
        
        if (ch == '.') {
            for (int i = 0; i < 26; i++) {
                if (node->children[i] && searchHelper(word, index + 1, node->children[i])) {
                    return true;
                }
            }
            return false;
        } else {
            int idx = ch - 'a';
            if (!node->children[idx]) {
                return false;
            }
            return searchHelper(word, index + 1, node->children[idx]);
        }
    }
    
public:
    WordDictionaryOptimized() {
        root = new TrieNode();
    }
    
    ~WordDictionaryOptimized() {
        delete root;
    }
    
    void addWord(std::string word) {
        TrieNode* current = root;
        
        for (char ch : word) {
            int idx = ch - 'a';
            if (!current->children[idx]) {
                current->children[idx] = new TrieNode();
            }
            current = current->children[idx];
        }
        
        current->isEndOfWord = true;
    }
    
    bool search(std::string word) {
        return searchHelper(word, 0, root);
    }
};

// Approach 3: Length-based grouping for optimization
class WordDictionaryLengthGrouped {
private:
    std::unordered_map<int, std::vector<std::string>> wordsByLength;
    
    bool matches(const std::string& pattern, const std::string& word) {
        if (pattern.length() != word.length()) return false;
        
        for (int i = 0; i < pattern.length(); i++) {
            if (pattern[i] != '.' && pattern[i] != word[i]) {
                return false;
            }
        }
        return true;
    }
    
public:
    WordDictionaryLengthGrouped() {}
    
    void addWord(std::string word) {
        wordsByLength[word.length()].push_back(word);
    }
    
    bool search(std::string word) {
        int len = word.length();
        if (wordsByLength.find(len) == wordsByLength.end()) {
            return false;
        }
        
        for (const std::string& candidate : wordsByLength[len]) {
            if (matches(word, candidate)) {
                return true;
            }
        }
        return false;
    }
};

// Approach 4: Hybrid approach with prefix optimization
class WordDictionaryHybrid {
private:
    struct TrieNode {
        std::unordered_map<char, TrieNode*> children;
        std::vector<std::string> wordsEndingHere;
        bool isEndOfWord;
        
        TrieNode() : isEndOfWord(false) {}
        
        ~TrieNode() {
            for (auto& [key, child] : children) {
                delete child;
            }
        }
    };
    
    TrieNode* root;
    
    bool searchHelper(const std::string& word, int index, TrieNode* node) {
        if (index == word.length()) {
            return node->isEndOfWord;
        }
        
        char ch = word[index];
        
        if (ch == '.') {
            // If we have many dots remaining, switch to brute force on stored words
            int dotsRemaining = 0;
            for (int i = index; i < word.length(); i++) {
                if (word[i] == '.') dotsRemaining++;
            }
            
            if (dotsRemaining > 3) { // Threshold for switching strategies
                std::string suffix = word.substr(index);
                for (const std::string& storedWord : node->wordsEndingHere) {
                    if (matchesSuffix(suffix, storedWord, 0)) {
                        return true;
                    }
                }
                return false;
            }
            
            for (auto& [key, child] : node->children) {
                if (searchHelper(word, index + 1, child)) {
                    return true;
                }
            }
            return false;
        } else {
            if (node->children.find(ch) == node->children.end()) {
                return false;
            }
            return searchHelper(word, index + 1, node->children[ch]);
        }
    }
    
    bool matchesSuffix(const std::string& pattern, const std::string& word, int offset) {
        if (pattern.length() != word.length() - offset) return false;
        
        for (int i = 0; i < pattern.length(); i++) {
            if (pattern[i] != '.' && pattern[i] != word[offset + i]) {
                return false;
            }
        }
        return true;
    }
    
    void collectAllWords(TrieNode* node, std::string& currentWord, std::vector<std::string>& result) {
        if (node->isEndOfWord) {
            result.push_back(currentWord);
        }
        
        for (auto& [ch, child] : node->children) {
            currentWord.push_back(ch);
            collectAllWords(child, currentWord, result);
            currentWord.pop_back();
        }
    }
    
public:
    WordDictionaryHybrid() {
        root = new TrieNode();
    }
    
    ~WordDictionaryHybrid() {
        delete root;
    }
    
    void addWord(std::string word) {
        TrieNode* current = root;
        
        for (char ch : word) {
            if (current->children.find(ch) == current->children.end()) {
                current->children[ch] = new TrieNode();
            }
            current = current->children[ch];
            
            // Store word at each node for hybrid search
            current->wordsEndingHere.push_back(word);
        }
        
        current->isEndOfWord = true;
    }
    
    bool search(std::string word) {
        return searchHelper(word, 0, root);
    }
};

// Test framework
class WordDictionaryTest {
public:
    static void runTests() {
        std::cout << "Microsoft Add and Search Words Tests:" << std::endl;
        std::cout << "====================================" << std::endl;
        
        testBasicFunctionality();
        testEdgeCases();
        performanceComparison();
        microsoftScenarios();
    }
    
    static void testBasicFunctionality() {
        std::cout << "\nBasic Functionality Test:" << std::endl;
        std::cout << "=========================" << std::endl;
        
        WordDictionary wd;
        
        wd.addWord("bad");
        wd.addWord("dad");
        wd.addWord("mad");
        
        std::cout << "Added words: bad, dad, mad" << std::endl;
        
        std::vector<std::pair<std::string, bool>> searches = {
            {"pad", false},
            {"bad", true},
            {".ad", true},
            {"b..", true},
            {"...", true},
            {"....", false},
            {".", false}
        };
        
        for (const auto& [query, expected] : searches) {
            bool result = wd.search(query);
            std::cout << "search(\"" << query << "\"): " << (result ? "true" : "false")
                      << " (Expected: " << (expected ? "true" : "false") << ") "
                      << (result == expected ? "✅" : "❌") << std::endl;
        }
    }
    
    static void testEdgeCases() {
        std::cout << "\nEdge Cases Test:" << std::endl;
        std::cout << "================" << std::endl;
        
        WordDictionary wd;
        
        // Empty dictionary
        std::cout << "Empty dictionary search: " << (wd.search("a") ? "true" : "false") 
                  << " (Expected: false) " << (!wd.search("a") ? "✅" : "❌") << std::endl;
        
        // Single character words
        wd.addWord("a");
        std::cout << "Single char 'a' search: " << (wd.search("a") ? "true" : "false") 
                  << " (Expected: true) " << (wd.search("a") ? "✅" : "❌") << std::endl;
        std::cout << "Single char '.' search: " << (wd.search(".") ? "true" : "false") 
                  << " (Expected: true) " << (wd.search(".") ? "✅" : "❌") << std::endl;
        
        // All dots
        wd.addWord("abc");
        std::cout << "All dots '...' search: " << (wd.search("...") ? "true" : "false") 
                  << " (Expected: true) " << (wd.search("...") ? "✅" : "❌") << std::endl;
        
        // Very long words
        std::string longWord(100, 'x');
        wd.addWord(longWord);
        std::string longPattern(100, '.');
        std::cout << "Long word pattern match: " << (wd.search(longPattern) ? "true" : "false") 
                  << " (Expected: true) " << (wd.search(longPattern) ? "✅" : "❌") << std::endl;
    }
    
    static void performanceComparison() {
        std::cout << "\nPerformance Comparison:" << std::endl;
        std::cout << "======================" << std::endl;
        
        std::vector<std::string> words;
        for (int i = 0; i < 1000; i++) {
            words.push_back(generateRandomWord(5 + i % 10));
        }
        
        // Test Trie approach
        auto start = std::chrono::high_resolution_clock::now();
        WordDictionary wd1;
        for (const std::string& word : words) {
            wd1.addWord(word);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Test optimized approach
        start = std::chrono::high_resolution_clock::now();
        WordDictionaryOptimized wd2;
        for (const std::string& word : words) {
            wd2.addWord(word);
        }
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Test length-grouped approach
        start = std::chrono::high_resolution_clock::now();
        WordDictionaryLengthGrouped wd3;
        for (const std::string& word : words) {
            wd3.addWord(word);
        }
        end = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Adding 1000 words:" << std::endl;
        std::cout << "Trie: " << duration1.count() << " ms" << std::endl;
        std::cout << "Optimized Trie: " << duration2.count() << " ms" << std::endl;
        std::cout << "Length Grouped: " << duration3.count() << " ms" << std::endl;
        
        // Test search performance
        std::vector<std::string> searchQueries = {
            ".....",
            "a....",
            "..b..",
            "...c.",
            "....d"
        };
        
        start = std::chrono::high_resolution_clock::now();
        for (const std::string& query : searchQueries) {
            wd1.search(query);
        }
        end = std::chrono::high_resolution_clock::now();
        duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        for (const std::string& query : searchQueries) {
            wd2.search(query);
        }
        end = std::chrono::high_resolution_clock::now();
        duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "\nWildcard search performance:" << std::endl;
        std::cout << "Trie: " << duration1.count() << " μs" << std::endl;
        std::cout << "Optimized Trie: " << duration2.count() << " μs" << std::endl;
    }
    
    static void microsoftScenarios() {
        std::cout << "\nMicrosoft-Specific Scenarios:" << std::endl;
        std::cout << "=============================" << std::endl;
        
        WordDictionary wd;
        
        // Scenario 1: Product names
        wd.addWord("windows");
        wd.addWord("office");
        wd.addWord("azure");
        wd.addWord("teams");
        wd.addWord("outlook");
        
        std::cout << "Product search '.ffice': " << (wd.search(".ffice") ? "true" : "false") << std::endl;
        std::cout << "Product search 'a....': " << (wd.search("a....") ? "true" : "false") << std::endl;
        std::cout << "Product search '....s': " << (wd.search("....s") ? "true" : "false") << std::endl;
        
        // Scenario 2: File extensions
        WordDictionary fileExt;
        fileExt.addWord("txt");
        fileExt.addWord("doc");
        fileExt.addWord("pdf");
        fileExt.addWord("exe");
        fileExt.addWord("dll");
        
        std::cout << "File ext '.x.': " << (fileExt.search(".x.") ? "true" : "false") << std::endl;
        std::cout << "File ext '...': " << (fileExt.search("...") ? "true" : "false") << std::endl;
        
        // Scenario 3: User permissions
        WordDictionary permissions;
        permissions.addWord("read");
        permissions.addWord("write");
        permissions.addWord("execute");
        permissions.addWord("admin");
        
        std::cout << "Permission '.ead': " << (permissions.search(".ead") ? "true" : "false") << std::endl;
        std::cout << "Permission '....e': " << (permissions.search("....e") ? "true" : "false") << std::endl;
    }
    
    static std::string generateRandomWord(int length) {
        std::string word;
        word.reserve(length);
        
        for (int i = 0; i < length; i++) {
            word += 'a' + (rand() % 26);
        }
        
        return word;
    }
    
    static void stressTest() {
        std::cout << "\nStress Test:" << std::endl;
        std::cout << "============" << std::endl;
        
        WordDictionary wd;
        
        // Add many words
        const int NUM_WORDS = 10000;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < NUM_WORDS; i++) {
            wd.addWord(generateRandomWord(8));
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Added " << NUM_WORDS << " words in " << duration.count() << " ms" << std::endl;
        
        // Test worst-case search (all dots)
        start = std::chrono::high_resolution_clock::now();
        bool result = wd.search("........");
        end = std::chrono::high_resolution_clock::now();
        auto searchDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Worst-case search (8 dots): " << searchDuration.count() << " ms" << std::endl;
        std::cout << "Result: " << (result ? "true" : "false") << std::endl;
    }
};

int main() {
    srand(42); // Fixed seed for reproducible results
    WordDictionaryTest::runTests();
    WordDictionaryTest::stressTest();
    return 0;
}

/*
Algorithm Analysis:

Core Data Structure: Trie (Prefix Tree) with DFS for wildcard matching

Key Components:
1. TrieNode with children map and end-of-word flag
2. addWord: Standard trie insertion O(m) where m is word length
3. search: DFS with backtracking for '.' wildcards

Time Complexity:
- addWord: O(m) where m is word length
- search: O(n * 26^k) worst case, where n is word length, k is number of dots
- Average case much better due to pruning

Space Complexity: O(ALPHABET_SIZE * N * M)
- N: number of words
- M: average word length
- Each node can have up to 26 children

Microsoft Interview Focus:
- Trie data structure design and implementation
- DFS with backtracking for wildcard matching
- Memory optimization techniques
- Performance analysis under different scenarios
- Real-world application design

Design Decisions:
1. HashMap vs Array for children (flexibility vs performance)
2. Recursive vs Iterative DFS
3. Memory vs time tradeoffs
4. Hybrid approaches for optimization

Optimization Strategies:
1. Array-based children for ASCII (faster access)
2. Length-based grouping for fewer wildcard cases
3. Hybrid approach switching strategies based on wildcard density
4. Prefix caching for common patterns

Real-world Applications:
- Autocomplete with partial matching
- Spell checkers with wildcard support
- Pattern matching in text editors
- Database query optimization
- File system search

Edge Cases:
- Empty dictionary searches
- Single character words and patterns
- All wildcard patterns
- Very long words
- No matching patterns

Interview Tips:
1. Start with basic trie structure
2. Explain DFS approach for wildcards
3. Discuss optimization opportunities
4. Handle edge cases explicitly
5. Consider real-world performance

Common Mistakes:
1. Not handling end-of-word flag correctly
2. Incorrect DFS backtracking
3. Memory leaks in C++ implementation
4. Not optimizing for common cases
5. Poor wildcard handling

Advanced Optimizations:
- Compressed tries (radix trees)
- Parallel search for multiple patterns
- Cache-aware memory layout
- Approximate matching algorithms
- Incremental updates

Testing Strategy:
- Basic functionality verification
- Edge case coverage
- Performance benchmarking
- Microsoft-specific scenarios
- Stress testing with large datasets

Production Considerations:
- Thread safety for concurrent access
- Persistence and serialization
- Memory usage monitoring
- Query optimization
- Scalability planning
*/
