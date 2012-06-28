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
#include "t1067.h"
#endif

int main() {
  TString a("abc");
  TTree t;
  t.Draw(a);

  TString b("def");
  t.Draw(a,b);

  return(0);
}

