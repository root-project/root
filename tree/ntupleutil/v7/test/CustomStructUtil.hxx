#ifndef ROOT7_RNTupleUtil_Test_CustomStructUtil
#define ROOT7_RNTupleUtil_Test_CustomStructUtil

#include <string>
#include <vector>


struct Base {
   int base;
};

struct CustomStructUtil : Base {
   float a = 0.0;
   std::vector<float> v1;
   std::vector<std::vector<float>> nnlo;
   std::string s;
};

#endif
