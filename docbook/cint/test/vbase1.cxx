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
#include "vbase1.h"
#endif

int main(){
  btest();
  ctest();
  dtest();
  etest();
  ftest();
  gtest();
  cout << endl;
  return 0;
}
