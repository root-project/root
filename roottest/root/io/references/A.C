#ifndef MACRO_A_C
#define MACRO_A_C

#include "TRef.h"

class A : public TObject {
public:
  A(A* other = nullptr) { _o = other; }
  TRef _o;
  ClassDef(A, 1)
};

#ifdef __ROOTCLING__
#pragma link C++ class A+;
#endif

#endif
