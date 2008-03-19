/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#if defined(__CINT__) && !defined(INTERPRET)
#pragma include "test1.dll"
#pragma include "test2.dll"
#pragma include "test3.dll"
#else
#include "t928.h"
#include "t928a.h"
#include "t928b.h"
#endif

int main() {
  A<int> x;
  x.f<int>();
  x.f<B>();
  x.f<C>();
  x.f<float>();
  x.x();

  f();

  A<short> y;
  y.f<long>();
  y.f<B>();
  y.f<double>();
  y.f<C>();
  y.x();

  return 0;
}

