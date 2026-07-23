#ifndef ROOT_TEST_STREAMERLOOP_MEMBERWISE
#define ROOT_TEST_STREAMERLOOP_MEMBERWISE

#include <Rtypes.h>

// Classes for the regression test of member-wise streaming of a variable-size
// array (`//[n]`, a TStreamerLoop element) that lives in a base class at a
// non-zero offset. See io/io/src/TStreamerInfoActions.cxx
// (TConfStreamerLoop::fCounterOffset) and TStreamerLoopMemberwise.cxx.

namespace ROOTTest {
namespace StreamerLoopMemberwise {

// Element stored in the variable-size array (a class, so the array is a
// TStreamerLoop and not a TStreamerBasicPointer).
class Hit {
public:
   Hit() = default;
   Hit(int a, int b) : fA(a), fB(b) {}
   bool operator==(const Hit &o) const { return fA == o.fA && fB == o.fB; }
   int fA = 0;
   int fB = 0;
   ClassDefNV(Hit, 1)
};

class Frame {
public:
   Frame() = default;
   ~Frame() { delete[] fHits; }
   Frame(const Frame &o) { Set(o.fN, o.fHits); }
   Frame &operator=(const Frame &o)
   {
      if (this != &o)
         Set(o.fN, o.fHits);
      return *this;
   }
   void Set(int n, const Hit *hits)
   {
      delete[] fHits;
      fHits = nullptr;
      fN = 0;
      if (n > 0) {
         fHits = new Hit[n];
         for (int i = 0; i < n; ++i)
            fHits[i] = hits[i];
         fN = n;
      }
   }
   int fN = 0;
   Hit *fHits = nullptr; //[fN]
   ClassDef(Frame, 1)
};

// A base preceding Frame, so that the Frame base does not sit at offset 0.
class Pad {
public:
   Pad() = default;
   int fX = 0;
   ClassDef(Pad, 1)
};

// The collection element: Frame is a base at a non-zero offset.
class Super : public Pad, public Frame {
public:
   Super() = default;
   ClassDef(Super, 1)
};

} // namespace StreamerLoopMemberwise
} // namespace ROOTTest

#endif
