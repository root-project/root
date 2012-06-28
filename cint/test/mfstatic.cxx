/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
class A {
 public:
  int func() ;
};

int A::func()
{
  static int a=1;
  printf("a=%d\n",a);
  a ^= 1;
  return 0;
}


A a;

int func(void);

int func()
{
  static int b=2;
  printf("b=%d\n",b);
  b *=2;
  return 0;
}

int main()
{
  func();
  func();
  func();

  a.func();
  a.func();
  a.func();
  a.func();
  return 0;
}
