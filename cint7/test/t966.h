/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// 021219p2ftypedefarg.txt
// complain that pointer to arg type is not indicated as original type

#include <stdio.h>

typedef double (*p2f)(double,double);

double f(p2f p,double a,double b) {
  double c;
  c = p(a,b);
  printf("%g\n",c);
  return(c);
}

double g(double (*p)(double,double),double a,double b) {
  double c;
  c = p(a,b);
  printf("%g\n",c);
  return(c);
}

double f1(double a,double b) {
  return a+b;
}

double f2(double a,double b) {
  return a-b;
}

double f3(double a,double b) {
  return a*b;
}

double f4(double a,double b) {
  return a/b;
}

