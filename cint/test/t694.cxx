/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#if defined(__CINT__) && !defined(INTERPRET)
#pragma include "test.dll"
#else
#include "t694.h"
#endif

int main() {
  A x;
  //printf("t694 causes problem due to 1558, default param evaluation scheme\n");
  for(int i=0;i<2;i++) {
    //printf("%d %d ",x.add(i),x.add(i,i+1));
     h(x.add(i),x.add(i,i+1));
#ifndef CINT_HIDE_FAILURE
    f(i);
#endif
    f();
    g(i);
    g();
    //printf("\n");
    endline();
  }
  return 0;
}
