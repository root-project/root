/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

class A {
  int a;
 public:
  A() { a=123; }
#ifdef DEST
  ~A() { printf("~A() %d\n",a); }
#endif
  int get() {return a;}
};

class B {
  A **ppa;
 public:
  B() { 
#ifndef DEST
    ppa = NULL; 
#endif
  }
  void test() {
    ppa = new A*[3]; 
    for(int i=0;i<3;i++) {
      ppa[i] = new A[i+1];
    }
  }
  void disp() {
    for(int i=0;i<3;i++) {
      for(int j=0;j<i+1;j++) 
	printf("%d\n",ppa[i][j].get());
    }
  }
  ~B() {
    for(int i=0;i<3;i++) {
      delete[] ppa[i];
    }
    delete[] ppa;
  }
};

int main() {
  B b;
  b.test();
  b.disp();
  return 0;
}

