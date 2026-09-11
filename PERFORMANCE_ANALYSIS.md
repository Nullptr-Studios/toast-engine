# Toast Engine Performance Analysis

## Overview

This analysis identifies potential optimization opportunities and performance issues in the Toast engine C++ codebase. The engine uses complex systems for scene management, event handling, and multithreading that could be improved.

## Key Performance Considerations

### 1. Thread-Safe Data Structures
Several data structures in world.cpp use mutexes extensively for thread safety:

**Potential Issues:**
- Mutex granularity is quite fine-grained but may still cause contention:
  - \dependency_graph.connections\ and \inverse_connections\ 
  - \m.load_mutex\ used in load-related functions
  - \
odes_mutex\ accessed during dependency graph computation

**Optimization Opportunities:**
- Consider lock-free alternatives where possible for read-heavy operations
- Look at using shared mutexes for better read performance (std::shared_mutex)

### 2. Memory Management and Allocation
The engine heavily uses:
- \std::vector<Box<Node>>\ throughout the codebase for node collections (very frequent)
- Memory pools in event handling system (\Pool\ with monotonic_buffer_resource)
- Asynchronous futures (\std::future\) for loading operations

**Potential Bottlenecks:**
- Node allocation/deallocation frequently through Box<T> system
- Memory pool recycling might be inefficient during high-load periods
- Future creation and waiting in world's async loading pipeline

### 3. Algorithm Complexity 
Critical algorithms in world.cpp show O(N^2) complexity in some operations:

**Analysis Areas:**
1. \World::subgraphSeparation()\ using BFS with nested loops  
2. \World::tarjanAlgorithm()\ - Tarjan's algorithm with SCC detection
3. \World::assignWaves()\ - Complex wave assignment logic 
4. Node search/find algorithms that use recursive DFS with potential exponential behavior

**Optimization Opportunities:**
- Use more efficient graph traversal methods for the dependency system
- Cache lookup results during dependency graph updates
- Consider pre-computing expensive computations or using spatial data structures

### 4. Function Call Overhead
The reflection system (\eflect.hpp\) uses extensive type erasure:
- \FieldInfo::get()\ and \set()\ functions use \std::any\ for type erasure (significant overhead)
- Multiple virtual function calls in \Node::callTick()\ 
- Event system with generic callback mechanisms using \std::function\

**Optimization Potential:**
- Inline critical path functions like \Node::enabled()\, \Node::uid()\, etc. - already marked as noexcept
- Consider compile-time optimizations for field accessing (if reflection data can be made more efficient)
- Reduce number of virtual function calls in hot paths

### 5. Event System 
Critical areas:
1. \vent::send<T>(...)\ has a memory pool with queue operations 
2. Event dispatching occurs in \pollEvents()\ that processes all queued events
3. Callback registration with multimap (used for sorting by priority)

**Bottlenecks:**
- The event system's notification process is relatively complex
- Each event needs to iterate through all registered callbacks
- Type erasure in \EventSystem::EventInfo\ structure

### 6. Node Management Functions
Several key performance-sensitive functions:
- \Node::enabled(bool)\ - calls \callTick()\ and recursively updates children
- \Node::callTick()\ - core reflection system dispatch 
- \World::tick()\ - dispatches all waves using ThreadPool
- \World::findFrom()\, \searchFrom()\ - complex path-resolution with DFS

### 7. Asynchronous Loading System
The \World\ class uses async loading extensively:

**Potential Issues:**
1. Load futures are kept in a vector and not aggressively removed
2. \drainLoadQueue()\ processes all loaded nodes at once, could be batched for better memory usage
3. Load queue may cause memory pressure during heavy loading

### 8. STL Container Usage 
Many STL containers used with potential for optimization:
- \std::vector\ in numerous places (node collections, graph processing)
- \std::unordered_map\ used for dependency tracking 
- \std::queue\ and \std::stack\ used in graph algorithms
- Multiple uses of \std::future\ creating thread overhead

### 9. Caching Opportunities
The engine does have caches for loading and global nodes but:
- No explicit LRU or cache invalidation heuristics for loaded assets
- Dependency graph recomputation is done on every change (potentially expensive)

### 10. Virtual Function Usage
Several virtual functions exist in the node hierarchy, including:
- \Node::callTick()\ and related lifecycle methods 
- Various lifecycle events like on_enable, on_disable

**Opportunities:**
- Consider move virtual function calls to inline functions where possible
- Evaluate necessity of virtual dispatch vs. template approaches for performance-critical sections

## Optimization Recommendations

1. **Inlining Policy**: Functions already marked as noexcept, but could benefit from inlining (especially small accessors)
2. **Memory Pool Management**: Investigate if the event system can be optimized to reduce allocations
3. **Graph Traversal Optimizations**: Use more efficient algorithms for dependency graph processing when possible 
4. **Lazy Evaluation Patterns**: Introduce lazy evaluation where computations are deferred until needed
5. **Caching**: Add caching strategies for expensive operations like reflection lookups, node traversals, etc.
6. **Thread Safety**: Profile mutex usage to identify bottlenecks; consider lock-free containers where appropriate

## Identified No-Optimize Functions
From the codebase, many getters and setters are already marked as \
oexcept\:

- All Node member accessors (uid(), name(), enabled(), etc.)
- Most internal node state functions 
- Box<T> accessor functions
- Memory allocation functions with pools  

## Critical Hot Paths
1. \World::tick()\ - main game loop entry point, calls \un_phase()\
2. \Node::callTick()\ - core reflection system dispatch 
3. \pollEvents()\ - all event processing pipeline  
4. \Node::find()/search()\ - node tree traversal APIs

## Conclusion
The engine has a well-structured multithreaded architecture but potential bottlenecks exist in:
1. Event system and node state management
2. Complex dependency graph algorithms 
3. Asynchronous loading with futures
4. Memory allocation overhead during heavy node operations
