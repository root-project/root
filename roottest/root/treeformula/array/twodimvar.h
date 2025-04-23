#ifndef A_h
#define A_h 1

#include "TObject.h"
#include <vector>
#include <iostream>
#include <iomanip>

class A : public TObject
{
public:
  A();
  ~A() override;
  void Fill(int n);
  void Dump2() const;
  
  int    n;
  int    *a;      //[n]
  int    *aa[3];  //[n]
  int    (*aaa)[3];  //![n]

  ClassDefOverride(A,1)
};

#endif
