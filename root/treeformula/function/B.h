#ifndef B_h
#define B_h 1

#include "TObject.h"
#include "TClonesArray.h"

class B : public TObject
{
public:
  B();
  ~B();

  TClonesArray *fA;//->

  ClassDef(B,1)
};

#endif
