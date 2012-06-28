/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>

namespace ns {
  class A {
  public:
  };

  class B {
  public:
    void Print(const ns::A* x) { printf("%p\n",x); }
  };
  
}

int main() {
  //ns::A a;
  void *x= 0;
  ns::B b;
  b.Print((const ns::A*)x);
#ifdef __CINT__
  G__calc("b.Print((const ns::A*)x)");
#else
  b.Print((const ns::A*)x);
#endif
  return 0;
}

