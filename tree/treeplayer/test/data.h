#ifndef TTREEREADER_LEAF_TEST_DATA
#define TTREEREADER_LEAF_TEST_DATA

#include <vector>
struct Data {
   unsigned int fUSize;
   int fSize;
   double* fArray; //[fSize]
   float* fUArray; //[fUSize]
   std::vector<double> fVec;
   Double32_t fDouble32;
   Float16_t fFloat16;
};

struct V {
   int a = -100;
};

struct W {
   int b;
   V v;
};

#endif
