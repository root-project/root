/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

// File 
#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t1115.h"
#endif


int main() {
  holder1 h;
  std::map<const helper*,int> x;
  h.f(x);
  std::map<const helper,int> y;
  h.g(y);
  return 0;
}

