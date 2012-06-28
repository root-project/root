/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

double evaluate_me(double x){return   ( x - -1.0 );}
double evaluate_me2(double x){return   ( x + +1.0 );}

int main() {
  printf("%g\n",evaluate_me(5.0));
  printf("%g\n",evaluate_me2(5.0));
  return 0;
}
