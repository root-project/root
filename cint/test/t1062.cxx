/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>

#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t1062.h"
#endif

int main() {
  printf("success\n");
  return(0);
}
