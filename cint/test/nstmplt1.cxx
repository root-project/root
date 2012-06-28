/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

template<class T> class A {
 public:
  int x;
  A() ;
  void f();
};

template<class T> A<T>::A() { x=1; }
template<class T> void A<T>::f() { printf("A::f=%d\n",x); }

class B {
 public:
  double b;
  B();
  void g();
};

B::B() { b=3.14; }
void B::g() { printf("B::b=%g\n",b); }

class C {
 public:
  A<B> a;
};

int main() {
  A<int> a;
  a.f();
  B b;
  b.g();
  C c;
  c.a.f();
  printf("nstmplt1\n");
  return 0;
}
