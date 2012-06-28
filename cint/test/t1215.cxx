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
#include "t1215.h"
#endif

int main() {
  C c;
  A a;
  a.doit(c);
  A a2(c);
  a2.doit(c);
  return 0;
}
