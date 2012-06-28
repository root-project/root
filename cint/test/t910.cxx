/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

// This program can not be compiled with TEST defined 
// TEST is only for CINT.
//#define TEST

class A;

#ifdef TEST
void f(A& x) {
  printf("f() x.d=%d\n",x.d);
}
#else
void f(A& x) ; // function f 1
#endif

class A {
  int d;
public:
  A();  // constructor of A 1
  ~A();  // destructor of A 1
  friend void f(A& x); // function f 2
  friend void g(A& x); // function g 
};

A::A() { d=123; } // constructor of A 2
A::~A() {} 

#ifndef TEST
void f(A& x) {
  printf("f() x.d=%d\n",x.d);
}
#endif

void g(A& x) {
  printf("g() x.d=%d\n",x.d);
}

int main() {
  A x;
  g(x);
  f(x);
  return 0;
}
