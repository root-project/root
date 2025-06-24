#ifndef A_C
#define A_C

#include "TRef.h"

class A : public TObject {
public:
A(A* other = 0) { _o = other; }

TRef _o;

ClassDef(A, 1)
};

#endif
