/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

class A {
  int a;
public:
  A(int x) : a(x) {}
  ~A() {}
  int get() const { return a; }
};
