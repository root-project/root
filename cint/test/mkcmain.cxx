/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifdef __CINT__
#include DNAME
#else
#include HNAME
#endif

#include <stdio.h>

void test() {
  printf("success\n");
}

int main() {
  test(); 
  return 0;
}
