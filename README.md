# C++ Knowledge Base for High-Frequency Trading & Competitive Programming

A comprehensive C++ knowledge base designed for technical interviews, online assessments, and competitive programming in the financial technology sector. This repository contains detailed implementations, explanations, and solutions covering data structures, algorithms, and HFT-specific problems.

## 📚 Repository Structure

### 🔷 [CPP_Knowledge_Base.md](./CPP_Knowledge_Base.md)
The main knowledge base containing:
- C++ fundamentals and best practices
- STL containers and algorithms cheat sheet
- Memory management and smart pointers
- Object-oriented programming concepts
- Time & space complexity analysis
- Interview tips and coding patterns

### 📊 DataStructures/
Implementation of core data structures with competitive programming solutions:

- **Arrays/** - [Trapping Rain Water](./DataStructures/Arrays/trapping_rain_water.cpp) (Hard)
- **LinkedLists/** - [Merge k Sorted Lists](./DataStructures/LinkedLists/merge_k_sorted_lists.cpp) (Hard)  
- **Trees/** - [Serialize/Deserialize Binary Tree](./DataStructures/Trees/serialize_deserialize_binary_tree.cpp) (Hard)
- **Graphs/** - [Word Ladder II](./DataStructures/Graphs/word_ladder_ii.cpp) (Hard)
- **HashMaps/** - [LRU Cache](./DataStructures/HashMaps/lru_cache.cpp) (Medium/Hard)

### 🔢 Algorithms/
Advanced algorithmic solutions organized by category:

- **DynamicProgramming/** - [Regular Expression Matching](./Algorithms/DynamicProgramming/regex_matching.cpp) (Hard)
- **Sorting/** - [Merge Intervals](./Algorithms/Sorting/merge_intervals.cpp) (Medium)
- **BinarySearch/** - [Median of Two Sorted Arrays](./Algorithms/BinarySearch/median_two_sorted_arrays.cpp) (Hard)

### 🏦 HFT/ (High-Frequency Trading)
Company-specific coding questions and implementations:

#### [Citadel/](./HFT/Citadel/)
- **[Order Book Implementation](./HFT/Citadel/order_book.cpp)** - Low-latency order book with O(log n) operations, order matching, and market depth analysis

#### [Jane Street/](./HFT/Jane_Street/)  
- **[Market Making Strategy](./HFT/Jane_Street/market_making.cpp)** - Statistical arbitrage with inventory management, volatility estimation, and risk controls

#### [Two Sigma/](./HFT/Two_Sigma/)
- **[Statistical Arbitrage Engine](./HFT/Two_Sigma/statistical_arbitrage.cpp)** - Pairs trading with cointegration testing, Kalman filtering, and portfolio optimization

#### [Jump Trading/](./HFT/Jump_Trading/)
- **[Low-Latency Arbitrage](./HFT/Jump_Trading/latency_arbitrage.cpp)** - Sub-microsecond arbitrage detection with cache-aligned data structures and lock-free programming

## 🎯 Difficulty Levels

### Medium Level Problems
- LRU Cache implementation
- Merge Intervals with custom comparators
- Basic market making strategies

### Hard Level Problems  
- Trapping Rain Water with multiple approaches
- Merge k Sorted Lists using divide & conquer
- Regular Expression Matching with DP optimization
- Word Ladder II with bidirectional BFS

### Expert Level Problems
- Serialize/Deserialize Binary Tree with multiple approaches
- Median of Two Sorted Arrays in O(log(min(m,n)))
- Advanced HFT systems with microsecond precision
- Statistical arbitrage with real-time risk management

## 🛠️ Features

### Code Quality
- **Production-ready** implementations with error handling
- **Multiple approaches** for each problem (brute force → optimized)
- **Comprehensive test cases** including edge cases
- **Performance analysis** with Big O complexity
- **Memory optimization** techniques

### HFT-Specific Content
- **Low-latency programming** patterns
- **Cache-aligned data structures**
- **Lock-free programming** with atomics
- **Market microstructure** understanding
- **Risk management** implementations
- **Performance measurement** in nanoseconds

### Educational Value
- **Step-by-step explanations** of algorithms
- **Common interview patterns** and templates
- **Debugging tips** and best practices
- **Trade-off discussions** for different approaches
- **Real-world applications** and use cases

## 🚀 Getting Started

### Prerequisites
- C++17 or later
- GCC/Clang with optimization flags
- Basic understanding of data structures and algorithms

### Compilation
```bash
# Standard compilation
g++ -std=c++17 -O2 -Wall -Wextra filename.cpp -o output

# For HFT code (maximum optimization)
g++ -std=c++17 -O3 -march=native -DNDEBUG filename.cpp -o output

# With debugging symbols
g++ -std=c++17 -g -fsanitize=address filename.cpp -o output
```

### Running Examples
```bash
# Run any example
./output

# For performance testing
time ./output

# Memory profiling (if compiled with debug flags)
valgrind --tool=memcheck ./output
```

## 📈 Performance Benchmarks

### Data Structure Operations
| Operation | Array | Hash Map | Binary Tree | Order Book |
|-----------|-------|----------|-------------|------------|
| Insert    | O(n)  | O(1)*    | O(log n)    | O(log n)   |
| Search    | O(n)  | O(1)*    | O(log n)    | O(1)       |
| Delete    | O(n)  | O(1)*    | O(log n)    | O(log n)   |

### HFT Performance Targets
- **Latency**: < 1 microsecond for critical path
- **Throughput**: > 1M messages/second
- **Jitter**: < 100 nanoseconds P99
- **Memory**: Zero allocations in hot path

## 🎯 Interview Preparation

### Study Path
1. **Start with** fundamentals in `CPP_Knowledge_Base.md`
2. **Practice** medium-level data structure problems
3. **Master** hard algorithmic challenges
4. **Explore** HFT-specific implementations
5. **Simulate** interview conditions with time limits

### Key Concepts to Master
- **Time/Space Complexity Analysis**
- **C++ STL and Modern Features**
- **Memory Management and Optimization** 
- **Concurrent Programming Patterns**
- **Financial Markets Understanding**
- **System Design for Low Latency**

### Common Interview Topics
- Implement core data structures from scratch
- Optimize existing algorithms for performance
- Design systems for high-frequency trading
- Debug and explain complex C++ code
- Discuss trade-offs in algorithm selection

## 🔗 Additional Resources

### Online Judges
- [LeetCode](https://leetcode.com/) - Algorithm practice
- [Codeforces](https://codeforces.com/) - Competitive programming  
- [HackerRank](https://hackerrank.com/) - Technical interviews

### C++ References
- [cppreference.com](https://cppreference.com/) - Complete C++ reference
- [C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines) - Best practices
- [Awesome C++](https://github.com/fffaraz/awesome-cpp) - Curated resources

### Financial Technology
- [Quantitative Finance](https://quantstart.com/) - Algorithmic trading
- [Market Microstructure](https://github.com/stefan-jansen/machine-learning-for-trading) - Market structure analysis

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines
- Follow existing code style and formatting
- Include comprehensive test cases
- Add detailed explanations and comments
- Provide performance analysis
- Update documentation as needed

---

⭐ **Star this repository** if you find it helpful for your interview preparation!

💼 **Perfect for preparing for** technical interviews at top financial firms like Citadel, Jane Street, Two Sigma, Jump Trading, and other quantitative trading companies.