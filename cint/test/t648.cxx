/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#include <iostream>
#ifndef __hpux
using namespace std;
#endif
#include <stdio.h>

int main() {
  long double a=3.141592815,b=2.000000,c;

  for(int i=0;i<3;i++) {
    c = a*b*i;
    printf("%d %d %d\n",(int)sizeof(long double),(int)sizeof(a),(int)sizeof(double));
    cout << a << " " << b << " " << c << endl;
  }
  return 0;
}
