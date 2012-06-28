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
#include "t995.h"
#endif

#include <stdio.h>

int main() {
  List<BaseStar*>::iterator first;
  printf("success\n");
  return 0;
}
