#ifndef ROOT_RNTuple_Test_STLContainerEvolution
#define ROOT_RNTuple_Test_STLContainerEvolution

#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

// The following std::hash specializations allow for creating unordered sets of pairs in the Collections test.
// It is a quick but bad hash but for this test though it should be ok.
template <>
struct std::hash<std::pair<int, int>> {
   std::size_t operator()(const std::pair<int, int> &p) const noexcept
   {
      std::size_t h1 = std::hash<int>{}(p.first);
      std::size_t h2 = std::hash<int>{}(p.second);
      return h1 ^ (h2 << 1);
   }
};

template <>
struct std::hash<std::pair<short int, short int>> {
   std::size_t operator()(const std::pair<short int, short int> &p) const noexcept
   {
      std::size_t h1 = std::hash<short int>{}(p.first);
      std::size_t h2 = std::hash<short int>{}(p.second);
      return h1 ^ (h2 << 1);
   }
};

template <typename T>
struct CollectionProxy {
   using ValueType = T;
   std::vector<T> v; //!
};

#endif
