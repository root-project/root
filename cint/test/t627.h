/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>

class TGC {
 public:
  float operator()(int x=1) {
    printf("TGC::operator()(%d)\n",x);
    return (float)1.23*x;
  }
};

TGC dx;

TGC& GetDefaultGC() {
  printf("GetDefaultGC()\n");
  return dx;
}

void f(float x) {
  printf("f(x=%g)\n",x);
}
