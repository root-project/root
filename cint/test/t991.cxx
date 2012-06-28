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
#pragma include "test3.dll"
#else
#include "t991a.h"
#include "t991b.h"
#include "t991c.h"
#endif

//#include <stdio.h>
#include <iostream>
using namespace std;

int main() {
  //printf("success\n");
  cout << "success"<< endl;
  Master::Container x;
  Slave2::Object obj;
  x.func(&obj);
  return 0;
}
