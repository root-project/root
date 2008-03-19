/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

// File t02.C
#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t1027.h"
#endif

void t02() {
  myclass m;
  int a = m;
  printf("created an int of value %d\n",a);
}  

int main() {
  t02(); 
  return 0;
}
