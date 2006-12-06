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
#include "t1178.h"
#endif

int main() {
  CTest a;
  PointVectorIterator iter;
  a.Execute(iter);
  a.f(iter);
  a.g(iter);
  a.h(iter);
  return 0;
}
