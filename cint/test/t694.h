/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>
class A {
 private:
  static const int x;
 public:
  int add(int a,int b=x) { return(a+b); }
};

const int A::x=5;

class B {
 public:
  int x;
  B(int a) : x(a) { }
  int get() const { return x; }
};

B b(789);
void f(const B& x=b) {
  printf("f(%d) ",x.get());
}
void g(const B& x=123) {
  printf("g(%d) ",x.get());
}
void h(int a,int b) {
#ifndef CINT_HIDE_FAILURE
  printf("%d %d ",a,b);
#endif
}
void endline() {
  printf("\n");
}



