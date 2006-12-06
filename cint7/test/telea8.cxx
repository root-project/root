/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>

class A {
 public:
  A() { printf("A()\n"); }
  ~A() { printf("~A()\n"); }
};

A f1() {
 A a;
 return a;
}

int main() {
  const A& b=f1();
  //A  c=f1();
  //G__pause();
  //printf("z\n");
  return 0;
}
