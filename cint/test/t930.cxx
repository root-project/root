/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

// test.C
//#include <iostream>
#include <stdio.h>
class A
{
 public:
  class B;
};

class A::B
{
public:
  int b;
};

int main()
{
  A::B bb;
  
  bb.b = 3;
  //cerr<<bb.b<<endl;
  printf("%d\n",bb.b);
  return 0;
}

