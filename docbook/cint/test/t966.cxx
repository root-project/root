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
#include "t966.h"
#endif

#include <stdio.h>

void test() {
  f(f1,1.23,4.56);
  f(f2,1.23,4.56);
  f(f3,1.23,4.56);
  f(f4,1.23,4.56);
  g(f1,9.23,2.0);
  g(f2,9.23,2.0);
  g(f3,9.23,2.0);
  g(f4,9.23,2.0);
}

int main() {
  test(); 
  return 0;
}


