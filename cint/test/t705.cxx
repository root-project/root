/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifdef __hpux
#include <iostream.h>
#include <strstream.h>
#else
#include <iostream>
#include <strstream>
using namespace std;
#endif

#include <stdio.h>

#if defined(__CINT__) && !defined(INTERPRET)
#pragma include "test.dll"
#else
#include "t705.h"
#endif

int main() {
  char a[10];
  int i;
  for (i=0; i<3; i++){
    if(i == 0) continue;
    istrstream is(" aa bb cc ");
    is >> a;
    cout << "a: " << a << endl;
  }
  //abc:
  for(i=0;i<3;i++) {
    if(i==0) continue;
    A b(i);
    printf("b:%d\n",b.get());
    //if(0) goto abc;
  }
  return 0;
}
