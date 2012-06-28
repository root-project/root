/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

int idx = 0 ;
class A {
 public:
  int a;
  int id;
  A() { a=0; id=idx++; }
  A(int x) { a = x; id=idx++; printf("A(int) %d  %d\n",a,id); }
  A(const A& x) { a = x.a; id=idx++; printf("A(A&) %d  %d\n",a,id); }
  //~A() { printf("~A() %d  %d\n",a,id); }
};


class B {
  A a[3];
 public:
  B() {
    for(int i=0;i<3;i++) a[i].a = i+1;
  }
  A& Get(int i) { return a[i]; }
};
