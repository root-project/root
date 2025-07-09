#ifndef SHIPDATA_SHIPPARTICLE_H_
#define SHIPDATA_SHIPPARTICLE_H_

#include <Rtypes.h>
#include <vector>

class SplittableBase {
public:
   int fSplittableBaseInt{101};
   float fSplittableBaseFloat{102.f};

   SplittableBase() = default;
};

// Having a base which is not splittable (due to a custom streamer) was
// making the calculation of the base class offset fail for the case of a
// std::vector<Derived>
class UnsplittableBase {
public:
   float fUnsplittableBaseFloat{201.f};

   UnsplittableBase() = default;
   ClassDefNV(UnsplittableBase, 2);
};

class Derived : public SplittableBase, public UnsplittableBase {
public:
   int fDerivedInt{301};

   Derived() = default;
};

#endif // SHIPDATA_SHIPPARTICLE_H_
