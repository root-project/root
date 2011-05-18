/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef __CINT__
#include <stdio.h>
#endif
class A {
 public:
  float f;
  union {
    char s;
    double d;
    int a;
  };
  short b;
};
int main()
{
  union {
    int a;
    char s;
    double d;
  };

  a=1234;
  printf("a=%d\n",a);
  s=123;
  printf("s=%d\n",s);
  d=3.14;
  printf("d=%g\n",d);

  A x;
  x.f=5.43;
  x.b=12;
  x.s=21;
  printf("f=%g s=%d b=%d\n",x.f,x.s,x.b);
  x.d=1.4142;
  printf("f=%g d=%g b=%d\n",x.f,x.d,x.b);
  x.a=3229;
  printf("f=%g a=%d b=%d\n",x.f,x.a,x.b);

  return 0;
}
