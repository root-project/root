/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <cstdio>
#include <iostream>
#include <sstream>

#if defined(__CINT__) && !defined(INTERPRET)
#pragma include "test.dll"
#else
#include "t705.h"
#endif

using namespace std;

int main(int argc, char** argv)
{
   char a[10];
   int i;
   for (i = 0; i < 3; ++i) {
      if (!i) {
         continue;
      }
      istringstream is(" aa bb cc ");
      is >> a;
      cout << "a: " << a << endl;
   }
   for (i = 0; i < 3; ++i) {
      if (!i) {
         continue;
      }
      A b(i);
      printf("b:%d\n", b.get());
   }
   return 0;
}
