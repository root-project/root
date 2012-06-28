/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif

template<class T> class A {
 public:
  A(T in) { a = in; }
  T a;
  void disp() { cout << "a=" << a << "\n"; }
};

typedef A <unsigned int> Auint;
typedef A<unsigned long> Aulong;

void test1()
{
  Auint a = Auint(3229); 
  Auint b = A <unsigned int>(3228);
  Aulong c(3227);
  cout << "a.Auint::a=" << a.Auint::a << "\n";
  a.disp();
  cout << "b.Auint::a=" << b.Auint::a << "\n";
  b.disp();
  c.disp();
}

int main()
{
  test1();
  return 0;
}
