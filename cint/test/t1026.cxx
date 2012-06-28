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
#include "t1026.h"
#endif

#include <stdio.h>

int main() {

  A a,b,c,d,e;

  a.a=3;
  b.a=4;

  c = a-b;
  d = -a;
  e = !a;
  printf("%d %d %d\n",c.a,d.a,e.a);
  
  return 0;
}

