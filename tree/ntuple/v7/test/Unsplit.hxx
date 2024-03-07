#ifndef ROOT7_RNTuple_Test_Unsplit
#define ROOT7_RNTuple_Test_Unsplit

#include <Rtypes.h>

#include <vector>

struct CyclicMember {
   float fB = 0.0;
   std::vector<CyclicMember> fV;
};

struct ClassWithUnsplitMember {
   float fA = 0.0;
   CyclicMember fUnsplit; // in the unit test, we set the "rntuple.unsplit" class attribute of CyclicMember
};

struct CustomStreamer {
   float a;
   ClassDefNV(CustomStreamer, 1);
};

#endif // ROOT7_RNTuple_Test_Unsplit
