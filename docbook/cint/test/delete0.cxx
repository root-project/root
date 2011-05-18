/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

int a=0;

class A {
  int a;
 public:
  A() { printf("A()\n");  a=::a++; }
  ~A() { printf("~A() %d\n",a); }
};

int main() {
 A* p[4];
 p[0] = 0;
 p[1] = new A;
 p[2] = new A;
 p[3] = 0;
 int i;
 for(i=0;i<4;i++) {
   delete p[i];
 }
 return 0;
}

