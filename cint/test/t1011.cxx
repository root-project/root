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
#include "t1011.h"
#endif

#include <iostream>


int main() {
  std::cout << "success" << endl;
  MyClass a;
  a.something.no_op();
  return 0;
}
