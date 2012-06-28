/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif

int k(117);
double d(3.14);

int main() {
  cout << "k=" << (unsigned char)k << endl;
  cout << "k=" << k << endl;
  cout << "d=" << d << endl;
  return 0;
}
