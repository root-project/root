/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t1048.h"
#endif

//////////////////////////////////////////////////////////////
// Plug in functions
//////////////////////////////////////////////////////////////
double f3(int* a,double b) {
  double result = b-(*a);
  return result;
}

double f4(int* a,double b) {
  double result = b*(*a)*2;
  return result;
}


int main() {
  test((void*)f1,1000);
  test((void*)f2,1000);
  test((void*)f3,100);
  test((void*)f4,100);
}
