/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef __hpux
#include <stdio.h>
#else
#include <cstdio>
using namespace std;
#endif

#define X 5
#define Y 4
#define Z 3

class A {
 public:
  void init(int x) {for(int i=0;i<X;i++) a[i] = i+x;}
  int a[X];
  int operator[](int i) { return a[i]; }
};

class B { 
 public:
  void init(int x) { for(int i=0;i<Y;i++) a[i].init(i+x); }
  A a[Y];
  A& operator[](int i) { return a[i]; }
};

class C { 
 public:
  C() {for(int i=0;i<Z;i++) a[i].init(i);}
  B a[Z];
  B& operator[](int i) { return a[i]; }
};

int main() {
  C a;
  a[1];
  a[1][2];
  a[1][2][3];

  C *p= new C;
  (*p)[1];
  (*p)[1][2];
  (*p)[1][2][3];

  for(int i=0;i<Z;i++) {
    for(int j=0;j<Y;j++) {
      for(int k=0;k<Z;k++) {
	printf("%d:%d ",a[i][j][k],(*p)[i][j][k]);
      }
      printf("\n");
    }
  }

  delete p;

  return 0;
}

