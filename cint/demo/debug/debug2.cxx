/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// debug2.cxx
//
// cint debug interface demo program.  There are 3 ways to run this demo.
//
// 1.
//    $ cint -X debug2.com
//    cint>  q
//
// 2.
//    $ cint
//    cint>  < debug2.com
//    cint>  q
//
// 3.
//    $ cint
//    cint>  L debug2.cxx
//    cint>  p test()
//    cint>  s test()
//    cint>  S
//    cint>  S
//    cint>  s
//    cint>  p x
//    cint>  S
//    cint>  p *this
//    cint>  S
//    cint>  S
//    cint>  S
//    cint>  U debug2.cxx
//    cint>  q
//
//

#include <iostream>

class A {
  int x;
 public:
  A(int xin=5) : x(xin) { cout << "A(" << x << ")" << endl; }
  ~A() { cout << "~A()" << endl; }
  int Get() const { cout << "A::Get()=" << x << endl; return x; }
  void Set(int xin) const { x=xin; cout << "A::Set(" << x << ")" << endl; }
};

void test() {
  A a;
  cout << a.Get() << endl;
  a.Set(3);
  cout << a.Get() << endl;
}


