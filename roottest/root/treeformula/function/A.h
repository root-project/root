#ifndef A_h
#define A_h 1

#include "TObject.h"
#include "TVector3.h"

class A : public TObject
{
public:
  A();
  ~A() override;

  TVector3  GetV() const { return tv; }
  TVector3  tv;
  int       val;

  ClassDefOverride(A,1)
};

#endif
