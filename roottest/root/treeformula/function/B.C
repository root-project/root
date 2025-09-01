#include "B.h"

#if !defined(__CLING__)
#endif


B::B() {
  fA = new TClonesArray("A",1000);
}


B::~B() {
  delete fA;
}
