/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifdef __hpux
#include <iostream.h>
#include <iomanip.h>
#else
#include <iostream>
#include <iomanip>
using namespace std;
#endif

int main() {
  int i;
  for(i=0;i<5;i++) {
    cout << setw(3) << i << "x" << endl;
  }
  for(i=0;i<20;i++) {
#if defined(_MSC_VER) || defined(G__VISUAL)
    cout.flags(ios::hex);
    cout << i << "x" << endl;
#else
    cout << setbase(16) << i << "x" << endl;
#endif
  }
  for(i=0;i<5;i++) {
    cout << setw(4) << setfill('?') << i << "x" << endl;
  }
  for(i=0;i<5;i++) {
    cout << setprecision(5) << i*1.23461815 << "x" << endl;
  }

  return 0;
}

