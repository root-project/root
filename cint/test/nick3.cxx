/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

class A { 
  float v[3];
 public:
  A() {
    v[0] = 11;
    v[1] = 22;
    v[2] = 33;
  }
  void f(float vec[]) {
    for(int i=0;i<3;i++) {
      vec[i] = v[i];
    }
  }
};

class B {
 public:
  void test() {
    A a;
    float v[3];
    for(int i=0;i<4;i++) {
      a.f(v);
      for(int j=0;j<3;j++) printf("%g\n",v[j]);
    }
  }
};


int main() {
  B b;
  b.test();
  return 0;
}
