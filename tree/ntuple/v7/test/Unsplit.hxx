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

struct CustomStreamerForceSplit {
   float a;
   ClassDefNV(CustomStreamerForceSplit, 1);
};

struct CustomStreamerForceUnsplit {
   float a;
};

// For the time being, RNTuple ignores the unsplit comment marker and does _not_ use an RUnsplitField for such members.
class IgnoreUnsplitComment {
   std::vector<float> v; //||
};

#endif // ROOT7_RNTuple_Test_Unsplit
