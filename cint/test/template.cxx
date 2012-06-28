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
#include "template.h"
#endif

void test(void);
int main()
{
   test();
   return 0;
}
