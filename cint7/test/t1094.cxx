/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//#include <iostream>
//using namespace std;
#include <stdio.h>

class A {
public:
  A &operator() ();
  A &operator() (double a);
  A &operator() (double a, double b);
};

A &A::operator() ()
{
  //cerr<< " no arg " <<endl;
  printf(" no arg\n");
  return *this;
}

A &A::operator() (double a)
{
  //cerr<<a<<endl;
  printf("%g\n",a);
  return *this;
}

A &A::operator() (double a, double b)
{
  //cerr<<a << " and " << b <<endl;
  printf("%g and %g\n",a,b);
  return *this;
}

int t01() {
  A a;
  a(3.2)(5.4);
  a()(3.2)(5.4,3.0);
  return 0;
}

int main() {
  t01();
  return 0;
}

