/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif


class A {
};

void h(long* x) {
  cout << (long)x << endl;
}

void f(A* x) {
  cout << (long)x << endl;
}

void g(A**** x) {
  cout << (long)x << endl;
}

int main() {
  short *ps;
  A *pa;
  ps=0;
  ps=NULL;
  pa=0;
  pa=NULL;
  h(NULL);
  h(0);
  f(NULL);
  f(0);
  f((A*)NULL);
  //g(NULL);
  //g(0);
  g((A****)NULL);
  return 0;
}


