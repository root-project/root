#ifndef ROOT_RNTuple_Test_StreamerBeyond
#define ROOT_RNTuple_Test_StreamerBeyond

#include <Rtypes.h>

#include <cstdint>
#include <vector>

struct StreamerBeyond {
   std::vector<std::int64_t> fOne;
   std::vector<std::int64_t> fTwo;
};

#endif
