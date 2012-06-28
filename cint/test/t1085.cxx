/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

typedef bool Bool_t;

class A {
public:
   A() {}
  virtual  ~A() {}
  virtual void QQ() {}
};

class B: public A {
public:
   B() {}
  virtual  ~B() {}
  virtual void QQ() { printf("B\n"); }
};

class C: public A {
private:
   B *fB;
   Bool_t fA;
public:
   C(Bool_t a):fA(a) { fB = new B();}
  virtual  ~C() { delete fB;}
  virtual void QQ() { printf("C\n"); }

  A* operator->()  { return (fA ? fB : (A*)this); }
};


int main() {
  C x(false);
  C y(true);
  for(int i=0;i<3;i++) {
    x->QQ();
    y->QQ();
  }
  return 0;
}

