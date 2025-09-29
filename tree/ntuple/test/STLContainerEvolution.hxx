#ifndef ROOT_RNTuple_Test_STLContainerEvolution
#define ROOT_RNTuple_Test_STLContainerEvolution

#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

template <typename T>
struct CollectionProxy {
   using ValueType = T;
   std::vector<T> v; //!
};

#endif
