/*
Google SDE Interview Problem 6: Design Search Autocomplete System (Hard)
Design a search autocomplete system for a search engine. 

Users can input a sentence (at least one word and end with a special character '#'). 
For each character they type except '#', return the top 3 historical hot sentences 
that have the same prefix as the part of the sentence already typed.

Requirements:
1. The hot degree for a sentence is defined as the number of times a user typed the exactly same sentence before.
2. The returned top 3 hot sentences should be sorted by hot degree (descending). 
3. If several sentences have the same degree of hot, use ASCII-code order (smaller one appears first).
4. If less than 3 hot sentences exist, return as many as you can.
5. When the input is a special character '#', it means the sentence ends, and in this case, you need to record the current sentence.

Example:
Input: sentences = ["i love you", "island","ironman", "i love leetcode"], times = [5,3,2,2]
AutocompleteSystem(sentences, times);
input('i') -> output ["i love you", "island","i love leetcode"]
input(' ') -> output ["i love you","i love leetcode"]
input('a') -> output []
input('#') -> record "i a" and output []

Time Complexity: O(p + q log q) per query where p is prefix length, q is matching sentences
Space Complexity: O(ALPHABET_SIZE * N * M) where N is number of sentences, M is average length
*/

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <algorithm>

// Trie node for efficient prefix searching
struct TrieNode {
    std::unordered_map<char, TrieNode*> children;
    std::unordered_map<std::string, int> sentences; // sentence -> frequency
    
    TrieNode() {}
    
    ~TrieNode() {
        for (auto& pair : children) {
            delete pair.second;
        }
    }
};

class AutocompleteSystem {
private:
    TrieNode* root;
    TrieNode* currentNode;
    std::string currentInput;
    std::unordered_map<std::string, int> sentenceFreq;
    
    // Comparator for priority queue (min-heap based on our criteria)
    struct SentenceComparator {
        bool operator()(const std::pair<std::string, int>& a, 
                       const std::pair<std::string, int>& b) {
            if (a.second != b.second) {
                return a.second > b.second; // Lower frequency has higher priority (min-heap)
            }
            return a.first < b.first; // Lexicographically smaller has higher priority
        }
    };
    
public:
    AutocompleteSystem(std::vector<std::string>& sentences, std::vector<int>& times) {
        root = new TrieNode();
        currentNode = root;
        currentInput = "";
        
        // Build initial trie and frequency map
        for (int i = 0; i < sentences.size(); i++) {
            insertSentence(sentences[i], times[i]);
        }
    }
    
    ~AutocompleteSystem() {
        delete root;
    }
    
    std::vector<std::string> input(char c) {
        if (c == '#') {
            // End of sentence - record it
            if (!currentInput.empty()) {
                insertSentence(currentInput, 1);
            }
            currentInput = "";
            currentNode = root;
            return {};
        }
        
        currentInput += c;
        
        // Navigate trie
        if (currentNode && currentNode->children.count(c)) {
            currentNode = currentNode->children[c];
            return getTop3Sentences();
        } else {
            currentNode = nullptr; // No valid path
            return {};
        }
    }
    
private:
    void insertSentence(const std::string& sentence, int freq) {
        sentenceFreq[sentence] += freq;
        
        TrieNode* node = root;
        for (char c : sentence) {
            if (!node->children[c]) {
                node->children[c] = new TrieNode();
            }
            node = node->children[c];
            node->sentences[sentence] += freq;
        }
    }
    
    std::vector<std::string> getTop3Sentences() {
        if (!currentNode) return {};
        
        // Use min-heap to maintain top 3
        std::priority_queue<std::pair<std::string, int>, 
                           std::vector<std::pair<std::string, int>>, 
                           SentenceComparator> minHeap;
        
        for (const auto& pair : currentNode->sentences) {
            minHeap.push({pair.first, pair.second});
            if (minHeap.size() > 3) {
                minHeap.pop();
            }
        }
        
        // Extract results and reverse (since we used min-heap)
        std::vector<std::string> result;
        while (!minHeap.empty()) {
            result.push_back(minHeap.top().first);
            minHeap.pop();
        }
        
        std::reverse(result.begin(), result.end());
        return result;
    }
};

// Alternative implementation with different data structure approach
class AutocompleteSystemV2 {
private:
    std::unordered_map<std::string, int> frequencies;
    std::string currentInput;
    
public:
    AutocompleteSystemV2(std::vector<std::string>& sentences, std::vector<int>& times) {
        currentInput = "";
        for (int i = 0; i < sentences.size(); i++) {
            frequencies[sentences[i]] = times[i];
        }
    }
    
    std::vector<std::string> input(char c) {
        if (c == '#') {
            frequencies[currentInput]++;
            currentInput = "";
            return {};
        }
        
        currentInput += c;
        
        // Find all sentences with current prefix
        std::vector<std::pair<std::string, int>> candidates;
        for (const auto& pair : frequencies) {
            if (pair.first.length() >= currentInput.length() && 
                pair.first.substr(0, currentInput.length()) == currentInput) {
                candidates.push_back(pair);
            }
        }
        
        // Sort by frequency (desc) then lexicographically (asc)
        std::sort(candidates.begin(), candidates.end(), 
                 [](const auto& a, const auto& b) {
                     if (a.second != b.second) {
                         return a.second > b.second;
                     }
                     return a.first < b.first;
                 });
        
        // Return top 3
        std::vector<std::string> result;
        for (int i = 0; i < std::min(3, (int)candidates.size()); i++) {
            result.push_back(candidates[i].first);
        }
        
        return result;
    }
};

// Optimized version with suffix array for faster prefix matching
class AutocompleteSystemV3 {
private:
    struct SentenceInfo {
        std::string sentence;
        int frequency;
        
        SentenceInfo(const std::string& s, int f) : sentence(s), frequency(f) {}
    };
    
    std::vector<SentenceInfo> sentences;
    std::string currentInput;
    
    // Binary search for prefix range
    std::pair<int, int> findPrefixRange(const std::string& prefix) {
        int left = 0, right = sentences.size();
        
        // Find first position >= prefix
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (sentences[mid].sentence >= prefix) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        int start = left;
        
        // Find first position > prefix
        right = sentences.size();
        std::string nextPrefix = prefix;
        if (!nextPrefix.empty()) {
            nextPrefix.back()++;
        } else {
            nextPrefix = "\x7f"; // DEL character
        }
        
        left = 0;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (sentences[mid].sentence < nextPrefix) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        int end = left;
        
        return {start, end};
    }
    
public:
    AutocompleteSystemV3(std::vector<std::string>& sentenceList, std::vector<int>& times) {
        currentInput = "";
        
        // Build sentence list
        for (int i = 0; i < sentenceList.size(); i++) {
            sentences.emplace_back(sentenceList[i], times[i]);
        }
        
        // Sort lexicographically for binary search
        std::sort(sentences.begin(), sentences.end(), 
                 [](const SentenceInfo& a, const SentenceInfo& b) {
                     return a.sentence < b.sentence;
                 });
    }
    
    std::vector<std::string> input(char c) {
        if (c == '#') {
            // Add new sentence
            auto it = std::lower_bound(sentences.begin(), sentences.end(), 
                                     SentenceInfo(currentInput, 0),
                                     [](const SentenceInfo& a, const SentenceInfo& b) {
                                         return a.sentence < b.sentence;
                                     });
            
            if (it != sentences.end() && it->sentence == currentInput) {
                it->frequency++;
            } else {
                sentences.insert(it, SentenceInfo(currentInput, 1));
            }
            
            currentInput = "";
            return {};
        }
        
        currentInput += c;
        
        // Find sentences with current prefix
        std::vector<std::pair<std::string, int>> candidates;
        auto range = findPrefixRange(currentInput);
        
        for (int i = range.first; i < range.second; i++) {
            if (sentences[i].sentence.length() >= currentInput.length() &&
                sentences[i].sentence.substr(0, currentInput.length()) == currentInput) {
                candidates.push_back({sentences[i].sentence, sentences[i].frequency});
            }
        }
        
        // Sort by frequency (desc) then lexicographically (asc)
        std::sort(candidates.begin(), candidates.end(),
                 [](const auto& a, const auto& b) {
                     if (a.second != b.second) {
                         return a.second > b.second;
                     }
                     return a.first < b.first;
                 });
        
        // Return top 3
        std::vector<std::string> result;
        for (int i = 0; i < std::min(3, (int)candidates.size()); i++) {
            result.push_back(candidates[i].first);
        }
        
        return result;
    }
};

// Test framework
class AutocompleteTest {
public:
    static void runTests() {
        std::cout << "Google Search Autocomplete System Tests:" << std::endl;
        std::cout << "=======================================" << std::endl;
        
        testBasicFunctionality();
        testEdgeCases();
        performanceTest();
    }
    
    static void testBasicFunctionality() {
        std::cout << "\nTest 1: Basic Functionality" << std::endl;
        std::cout << "============================" << std::endl;
        
        std::vector<std::string> sentences = {"i love you", "island","ironman", "i love leetcode"};
        std::vector<int> times = {5,3,2,2};
        
        AutocompleteSystem ac(sentences, times);
        
        auto result1 = ac.input('i');
        std::cout << "input('i'): ";
        printVector(result1);
        std::cout << " (Expected: [\"i love you\", \"island\", \"i love leetcode\"])" << std::endl;
        
        auto result2 = ac.input(' ');
        std::cout << "input(' '): ";
        printVector(result2);
        std::cout << " (Expected: [\"i love you\", \"i love leetcode\"])" << std::endl;
        
        auto result3 = ac.input('a');
        std::cout << "input('a'): ";
        printVector(result3);
        std::cout << " (Expected: [])" << std::endl;
        
        auto result4 = ac.input('#');
        std::cout << "input('#'): ";
        printVector(result4);
        std::cout << " (Expected: [])" << std::endl;
        
        // Test the newly added sentence
        auto result5 = ac.input('i');
        auto result6 = ac.input(' ');
        auto result7 = ac.input('a');
        std::cout << "After adding 'i a', input('i',' ','a'): ";
        printVector(result7);
        std::cout << " (Should include 'i a')" << std::endl;
    }
    
    static void testEdgeCases() {
        std::cout << "\nTest 2: Edge Cases" << std::endl;
        std::cout << "==================" << std::endl;
        
        // Test with empty initial data
        std::vector<std::string> emptySentences;
        std::vector<int> emptyTimes;
        AutocompleteSystem ac1(emptySentences, emptyTimes);
        
        auto result1 = ac1.input('h');
        std::cout << "Empty system input('h'): ";
        printVector(result1);
        std::cout << " (Expected: [])" << std::endl;
        
        ac1.input('e');
        ac1.input('l');
        ac1.input('l');
        ac1.input('o');
        ac1.input('#');
        
        auto result2 = ac1.input('h');
        std::cout << "After adding 'hello', input('h'): ";
        printVector(result2);
        std::cout << " (Expected: [\"hello\"])" << std::endl;
        
        // Test with same frequency sentences
        std::vector<std::string> sentences2 = {"abc", "abd", "abe"};
        std::vector<int> times2 = {3, 3, 3};
        AutocompleteSystem ac2(sentences2, times2);
        
        auto result3 = ac2.input('a');
        auto result4 = ac2.input('b');
        std::cout << "Same frequency test input('a','b'): ";
        printVector(result4);
        std::cout << " (Expected: [\"abc\", \"abd\", \"abe\"])" << std::endl;
    }
    
    static void performanceTest() {
        std::cout << "\nPerformance Test" << std::endl;
        std::cout << "================" << std::endl;
        
        // Generate large dataset
        std::vector<std::string> largeSentences;
        std::vector<int> largeTimes;
        
        for (int i = 0; i < 1000; i++) {
            std::string sentence = "sentence" + std::to_string(i);
            largeSentences.push_back(sentence);
            largeTimes.push_back(i % 100 + 1);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        AutocompleteSystem ac(largeSentences, largeTimes);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Initialization with 1000 sentences: " << duration.count() << " ms" << std::endl;
        
        // Test query performance
        start = std::chrono::high_resolution_clock::now();
        auto result = ac.input('s');
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Query 's' performance: " << duration.count() << " μs" << std::endl;
        std::cout << "Results count: " << result.size() << std::endl;
        
        // Compare different implementations
        std::cout << "\nComparing implementations:" << std::endl;
        
        std::vector<std::string> testSentences = {"apple", "application", "apply", "april"};
        std::vector<int> testTimes = {10, 8, 6, 4};
        
        AutocompleteSystemV2 ac2(testSentences, testTimes);
        AutocompleteSystemV3 ac3(testSentences, testTimes);
        
        start = std::chrono::high_resolution_clock::now();
        auto result1 = ac.input('a');
        auto dur1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - start);
        
        start = std::chrono::high_resolution_clock::now();
        auto result2 = ac2.input('a');
        auto dur2 = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - start);
        
        start = std::chrono::high_resolution_clock::now();
        auto result3 = ac3.input('a');
        auto dur3 = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - start);
        
        std::cout << "Trie implementation: " << dur1.count() << " ns" << std::endl;
        std::cout << "Hash map implementation: " << dur2.count() << " ns" << std::endl;
        std::cout << "Sorted array implementation: " << dur3.count() << " ns" << std::endl;
    }
    
    static void printVector(const std::vector<std::string>& vec) {
        std::cout << "[";
        for (int i = 0; i < vec.size(); i++) {
            std::cout << "\"" << vec[i] << "\"";
            if (i < vec.size() - 1) std::cout << ", ";
        }
        std::cout << "]";
    }
    
    static void stressTest() {
        std::cout << "\nStress Test" << std::endl;
        std::cout << "===========" << std::endl;
        
        std::vector<std::string> sentences = {"hello world", "hello", "help", "helicopter"};
        std::vector<int> times = {5, 10, 3, 1};
        
        AutocompleteSystem ac(sentences, times);
        
        // Test progressive input
        std::cout << "Progressive input test:" << std::endl;
        std::string testInput = "hello world is amazing";
        
        for (char c : testInput) {
            if (c == ' ') std::cout << " -> ";
            auto result = ac.input(c);
            if (c != ' ') {
                std::cout << "'" << c << "': ";
                printVector(result);
                std::cout << std::endl;
            }
        }
        
        ac.input('#'); // End sentence
        
        // Test if new sentence is recorded
        auto finalResult = ac.input('h');
        std::cout << "After recording new sentence, 'h': ";
        printVector(finalResult);
        std::cout << std::endl;
    }
};

int main() {
    AutocompleteTest::runTests();
    AutocompleteTest::stressTest();
    return 0;
}

/*
Algorithm Analysis:

Core Data Structure: Trie + HashMap for efficient prefix search and frequency tracking

Key Components:
1. TrieNode: Stores children and sentence frequencies at each prefix
2. Current state tracking: currentNode and currentInput
3. Priority queue for top-k selection

Time Complexity:
- Insert: O(L) where L is sentence length
- Query: O(P + K log K) where P is prefix length, K is matching sentences
- Space: O(ALPHABET_SIZE * N * M) for trie

Space Optimizations:
1. Store frequencies only at nodes where sentences exist
2. Use unordered_map for sparse character sets
3. Compress common prefixes

Google Interview Focus:
- Trie data structure design and implementation
- Efficient top-k selection algorithms
- String prefix matching optimization
- Memory usage in large-scale systems
- Real-time query processing

Key Design Decisions:
1. Trie vs Hash Map vs Sorted Array tradeoffs
2. How to handle frequency updates
3. Memory vs query time optimization
4. Handling special characters and edge cases

Alternative Approaches:
1. Hash Map with linear search (simpler, less memory)
2. Suffix Array with binary search (good for read-heavy)
3. Radix Tree for compressed storage
4. External sorting for very large datasets

Performance Considerations:
- Cache locality in trie traversal
- Memory fragmentation with many small nodes
- Query batching for better throughput
- Prefix compression for space efficiency

Edge Cases:
- Empty input system
- Duplicate sentence addition
- All sentences with same frequency
- Very long sentences
- Special characters in input

Real-world Optimizations:
1. LRU cache for hot prefixes
2. Parallel processing for multiple queries
3. Incremental updates without full rebuild
4. Persistence and recovery mechanisms

Interview Tips:
1. Start with simple hash map approach
2. Discuss trie benefits for prefix matching
3. Explain top-k selection strategies
4. Consider memory vs latency tradeoffs
5. Design for scale and real-world usage
*/
