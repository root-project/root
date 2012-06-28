/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#ifdef __CINT__

#pragma include "test1.dll"
#pragma include "test2.dll"

#else

#include "t972a.h"
#include "t972b.h"

#endif


#include <stdio.h>

int main() {
  
  printf("%d\n",foo::a01());
  printf("%d\n",foo::a02());
  printf("%d\n",foo::a03());
  printf("%d\n",foo::a04());
  printf("%d\n",foo::a15());
  printf("%d\n",foo::b01());
  printf("%d\n",foo::b02());
  printf("%d\n",foo::b03());
  printf("%d\n",foo::b04());
  printf("%d\n",foo::b15());
  return 0;
}
