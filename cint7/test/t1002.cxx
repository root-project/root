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
#include "t1002.h"
#endif

//#include <stdio.h>
#include <iostream>
using namespace std;

int main() {
  //printf("success\n");
  cout <<"success"<<endl;
  Marshal<double> d;
  Marshal<int> i;
  return 0;
}

