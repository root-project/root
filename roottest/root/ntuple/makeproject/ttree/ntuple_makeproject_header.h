
#ifndef RNTUPLE_MAKEPROJECT_MYSTLEVENT
#define RNTUPLE_MAKEPROJECT_MYSTLEVENT

#include <bitset>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "TObject.h"

class MySTLEvent final : public TObject {

private:
   std::bitset<16> foo = 0xfa2;
   std::vector<int> bar = {1, 2};
   std::unordered_set<double> spam = {1, 3, 5, 6, 8};
   std::unordered_map<int, std::string> eggs = {{2, "two"}, {4, "four"}};
   std::unordered_multimap<std::string, std::vector<float>> strange = {
      {"one", {1, 2, 3}}, {"one", {4, 5, 6}}, {"two", {7, 8, 9}}};

public:
   MySTLEvent() = default;

   unsigned long get_bitset() const { return foo.to_ulong(); }
   bool correct_bar() const { return (bar.size() == 2) && (bar[0] == 1) && (bar[1] == 2); }

   ClassDefOverride(MySTLEvent, 1) // Event structure
};

#endif // RNTUPLE_MAKEPROJECT_MYSTLEVENT
