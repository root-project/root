#ifndef B_h
#define B_h 1

#include "TObject.h"
#include "TClonesArray.h"
#include <vector>

class B : public TObject
{
public:
  B();
  ~B();

  TClonesArray *fA;//->
  std::vector<A> fVecA;
  A GetACopy() const { return *(A*)fA->At(0); };
  std::vector<A> GetVecACopy() const { return fVecA; }

  ClassDefOverride(B,1)
};

#endif
