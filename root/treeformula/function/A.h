#ifndef A_h
#define A_h 1

#include "TObject.h"
#include "TVector3.h"

class A : public TObject
{
public:
  A();
  ~A();

  TVector3  tv;
  int       val;

  ClassDef(A,1)
};

#endif
