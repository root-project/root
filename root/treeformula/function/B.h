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
  A GetA() { return *(A*)fA->At(0); };
  std::vector<A> GetVecA() { return fVecA; }

  ClassDef(B,1)
};

#endif
