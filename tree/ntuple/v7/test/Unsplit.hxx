#ifndef ROOT7_RNTuple_Test_Unsplit
#define ROOT7_RNTuple_Test_Unsplit

#include <vector>

struct UnsplitMember {
   float a = 0.0;
   std::vector<UnsplitMember> v; //|| Don't split
};

#endif // ROOT7_RNTuple_Test_Unsplit
