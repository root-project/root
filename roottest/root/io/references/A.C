#include "TRef.h"

#ifndef A_C
#ifdef ClingWorkAroundMultipleInclude
#define A_C
#endif

class A : public TObject { 
public: 
A(A* other = 0) { _o = other; }

TRef _o; 

ClassDef(A, 1)
};

#endif
