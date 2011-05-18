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
#include "t1025.h"
#endif

void arrayfunc1() {

   float f = arrayret(1)[1];

   fprintf(stdout,"the printed value should equal to 22: %f\n",f);
   std::cout << "should also be 22: " << arrayret(1)[1] << std::endl;

}

int main() {
  arrayfunc();
  arrayfunc1();
  return 0;
}

