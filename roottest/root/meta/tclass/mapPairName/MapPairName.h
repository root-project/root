#ifndef ROOTTEST_MAPPAIRNAME_H
#define ROOTTEST_MAPPAIRNAME_H

#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

struct MapPairKey {
   int fValue = 0;
};

struct MapPairLess {
   bool operator()(const MapPairKey &lhs, const MapPairKey &rhs) const { return lhs.fValue < rhs.fValue; }
};

struct MapPairHash {
   std::size_t operator()(const MapPairKey &key) const { return std::hash<int>{}(key.fValue); }
};

struct MapPairEqual {
   bool operator()(const MapPairKey &lhs, const MapPairKey &rhs) const { return lhs.fValue == rhs.fValue; }
};

struct MapPairHolder {
   std::map<MapPairKey, std::string, MapPairLess> fMap;
   std::multimap<MapPairKey, std::string, MapPairLess> fMultiMap;
   std::unordered_map<MapPairKey, std::vector<std::string>, MapPairHash, MapPairEqual> fUnorderedMap;
   std::unordered_multimap<MapPairKey, std::vector<std::string>, MapPairHash, MapPairEqual> fUnorderedMultiMap;
};

#endif
