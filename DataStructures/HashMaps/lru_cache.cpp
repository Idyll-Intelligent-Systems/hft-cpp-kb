/*
Problem: LRU Cache (Medium/Hard)
Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:
- LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
- int get(int key) Return the value of the key if the key exists, otherwise return -1.
- void put(int key, int value) Update the value of the key if the key exists. 
  Otherwise, add the key-value pair to the cache. If the number of keys exceeds 
  the capacity from this operation, evict the least recently used key.

The functions get and put must each run in O(1) average time complexity.

Example:
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
*/

#include <iostream>
#include <unordered_map>

class LRUCache {
private:
    struct Node {
        int key;
        int value;
        Node* prev;
        Node* next;
        
        Node(int k, int v) : key(k), value(v), prev(nullptr), next(nullptr) {}
    };
    
    int capacity;
    std::unordered_map<int, Node*> cache;  // key -> node mapping
    Node* head;  // Dummy head (most recently used end)
    Node* tail;  // Dummy tail (least recently used end)
    
    // Add node right after head (mark as most recently used)
    void addNode(Node* node) {
        node->prev = head;
        node->next = head->next;
        
        head->next->prev = node;
        head->next = node;
    }
    
    // Remove an existing node from the linked list
    void removeNode(Node* node) {
        Node* prevNode = node->prev;
        Node* nextNode = node->next;
        
        prevNode->next = nextNode;
        nextNode->prev = prevNode;
    }
    
    // Move node to head (mark as most recently used)
    void moveToHead(Node* node) {
        removeNode(node);
        addNode(node);
    }
    
    // Remove the last node (least recently used)
    Node* removeTail() {
        Node* lastNode = tail->prev;
        removeNode(lastNode);
        return lastNode;
    }
    
public:
    LRUCache(int capacity) : capacity(capacity) {
        head = new Node(0, 0);
        tail = new Node(0, 0);
        
        head->next = tail;
        tail->prev = head;
    }
    
    ~LRUCache() {
        // Clean up all nodes
        Node* current = head;
        while (current) {
            Node* next = current->next;
            delete current;
            current = next;
        }
    }
    
    int get(int key) {
        if (cache.find(key) != cache.end()) {
            Node* node = cache[key];
            // Move the accessed node to head (most recently used)
            moveToHead(node);
            return node->value;
        }
        return -1;
    }
    
    void put(int key, int value) {
        if (cache.find(key) != cache.end()) {
            // Update existing key
            Node* node = cache[key];
            node->value = value;
            moveToHead(node);
        } else {
            // Add new key-value pair
            Node* newNode = new Node(key, value);
            
            if (cache.size() >= capacity) {
                // Remove LRU item
                Node* tail_node = removeTail();
                cache.erase(tail_node->key);
                delete tail_node;
            }
            
            cache[key] = newNode;
            addNode(newNode);
        }
    }
    
    // Helper function to print current cache state (for debugging)
    void printCache() {
        std::cout << "Cache state (MRU -> LRU): ";
        Node* current = head->next;
        while (current != tail) {
            std::cout << "(" << current->key << "," << current->value << ") ";
            current = current->next;
        }
        std::cout << std::endl;
    }
};

// Alternative implementation using STL list (less efficient but simpler)
#include <list>

class LRUCacheSTL {
private:
    int capacity;
    std::list<std::pair<int, int>> cache_list;  // key-value pairs
    std::unordered_map<int, std::list<std::pair<int, int>>::iterator> cache_map;
    
public:
    LRUCacheSTL(int capacity) : capacity(capacity) {}
    
    int get(int key) {
        if (cache_map.find(key) == cache_map.end()) {
            return -1;
        }
        
        // Move to front (most recently used)
        auto it = cache_map[key];
        int value = it->second;
        cache_list.erase(it);
        cache_list.push_front({key, value});
        cache_map[key] = cache_list.begin();
        
        return value;
    }
    
    void put(int key, int value) {
        if (cache_map.find(key) != cache_map.end()) {
            // Update existing key
            auto it = cache_map[key];
            cache_list.erase(it);
        } else if (cache_list.size() >= capacity) {
            // Remove LRU item
            auto last = cache_list.back();
            cache_map.erase(last.first);
            cache_list.pop_back();
        }
        
        // Add new item to front
        cache_list.push_front({key, value});
        cache_map[key] = cache_list.begin();
    }
};

int main() {
    // Test the custom implementation
    std::cout << "Testing Custom LRU Cache Implementation:" << std::endl;
    LRUCache lruCache(2);
    
    lruCache.put(1, 1);
    lruCache.printCache();
    
    lruCache.put(2, 2);
    lruCache.printCache();
    
    std::cout << "get(1): " << lruCache.get(1) << std::endl;  // returns 1
    lruCache.printCache();
    
    lruCache.put(3, 3);  // evicts key 2
    lruCache.printCache();
    
    std::cout << "get(2): " << lruCache.get(2) << std::endl;  // returns -1
    std::cout << "get(3): " << lruCache.get(3) << std::endl;  // returns 3
    lruCache.printCache();
    
    std::cout << "get(1): " << lruCache.get(1) << std::endl;  // returns 1
    lruCache.printCache();
    
    lruCache.put(4, 4);  // evicts key 3
    lruCache.printCache();
    
    std::cout << "get(1): " << lruCache.get(1) << std::endl;  // returns 1
    std::cout << "get(3): " << lruCache.get(3) << std::endl;  // returns -1
    std::cout << "get(4): " << lruCache.get(4) << std::endl;  // returns 4
    
    std::cout << "\nTesting STL-based LRU Cache Implementation:" << std::endl;
    LRUCacheSTL lruCacheSTL(2);
    
    lruCacheSTL.put(1, 1);
    lruCacheSTL.put(2, 2);
    std::cout << "get(1): " << lruCacheSTL.get(1) << std::endl;  // returns 1
    lruCacheSTL.put(3, 3);  // evicts key 2
    std::cout << "get(2): " << lruCacheSTL.get(2) << std::endl;  // returns -1
    std::cout << "get(3): " << lruCacheSTL.get(3) << std::endl;  // returns 3
    std::cout << "get(1): " << lruCacheSTL.get(1) << std::endl;  // returns 1
    lruCacheSTL.put(4, 4);  // evicts key 3
    std::cout << "get(1): " << lruCacheSTL.get(1) << std::endl;  // returns 1
    std::cout << "get(3): " << lruCacheSTL.get(3) << std::endl;  // returns -1
    std::cout << "get(4): " << lruCacheSTL.get(4) << std::endl;  // returns 4
    
    return 0;
}

/*
Algorithm Analysis:

1. Custom Doubly Linked List + HashMap:
   - Time Complexity: O(1) for both get() and put()
   - Space Complexity: O(capacity)
   - Advantages: True O(1) operations, memory efficient
   - Implementation: Manual memory management required

2. STL List + HashMap:
   - Time Complexity: O(1) for both get() and put()
   - Space Complexity: O(capacity)
   - Advantages: Simpler code, automatic memory management
   - Disadvantages: Slightly higher overhead due to STL

Key Design Decisions:

1. Data Structure Choice:
   - HashMap: O(1) access to nodes by key
   - Doubly Linked List: O(1) insertion, deletion, and movement

2. Node Organization:
   - Head: Most recently used end
   - Tail: Least recently used end
   - Dummy nodes: Simplify edge case handling

3. Operations:
   - get(): Move accessed node to head
   - put(): Add new node to head, remove LRU if capacity exceeded

Implementation Details:

1. Dummy Nodes:
   - Eliminates null checks for head/tail operations
   - Simplifies insertion and deletion logic

2. Memory Management:
   - Custom implementation requires careful cleanup
   - STL version handles memory automatically

3. Cache Consistency:
   - HashMap and linked list must stay synchronized
   - Update both structures atomically

Applications:
- Operating system page replacement
- Web browser cache
- Database buffer pools
- CPU cache management
- General caching systems

Common Interview Variations:
1. LFU (Least Frequently Used) Cache
2. Time-based expiration
3. Thread-safe implementation
4. Multi-level cache hierarchy
5. Cache with different eviction policies

Edge Cases:
- Capacity of 1
- Multiple operations on same key
- Cache at full capacity
- Empty cache operations
*/
