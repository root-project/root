/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>

int main() {
  double a=1.2,b;
  int x=0;
  int y;
  y=4 + ++x;
  b=.2 + ++a;
  printf("x=%d y=%d a=%g b=%g\n",x,y,a,b);
  return 0;
}
