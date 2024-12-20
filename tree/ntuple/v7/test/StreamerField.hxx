#ifndef ROOT7_RNTuple_Test_StreamerField
#define ROOT7_RNTuple_Test_StreamerField

#include <Rtypes.h>

#include <memory>
#include <vector>

struct CyclicMember {
   float fB = 0.0;
   std::vector<CyclicMember> fV;
};

struct ClassWithStreamedMember {
   float fA = 0.0;
   CyclicMember fStreamed; // in the unit test, we set the "rntuple.streamerMode" class attribute of CyclicMember
};

struct CustomStreamer {
   float a;
   ClassDefNV(CustomStreamer, 1);
};

struct CustomStreamerForceNative {
   float a;
   ClassDefNV(CustomStreamerForceNative, 1);
};

struct CustomStreamerForceStreamed {
   float a;
};

// For the time being, RNTuple ignores the unsplit comment marker and does _not_ use an RStreamerField for such members.
class IgnoreUnsplitComment {
   std::vector<float> v; //||
};

// Test streamer field with polymorphic type

struct PolyBase {
   virtual ~PolyBase() {}
   int x;
};

struct PolyA : public PolyBase {
   int a;
};

struct PolyB : public PolyBase {
   int b;
};

struct PolyContainer {
   std::unique_ptr<PolyBase> fPoly;
};

#endif // ROOT7_RNTuple_Test_StreamerField
