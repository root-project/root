/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t1016.h"
#endif

int main() {

  aaa a;
  aaa1 *a1 = (aaa1*)&a;
  aaa2 *a2 = (aaa2*)&a;
  printf("aaa1 offset = %ld\n",(char*)a1 - (char*)&a);
  printf("aaa2 offset = %ld\n",(char*)a2 - (char*)&a);
  a.fun();
  a1->fun1();
  a2->fun2();

  return 0;
}
