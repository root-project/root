/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

//#include <iostream>
#include <stdio.h>

#define __PRETTY_FUNCTION__ "abc"

struct C { };

struct A {
  A() { };
  template <class U> A(const U& u) { printf("A(const U&)\n"); }
  template <class U> void doit(const U& u) { printf("doit(const U&)\n"); } 
};


#ifdef __MAKECINT__
#pragma link C++ class A+;
#pragma link C++ class C+;
#pragma link C++ function A::A(const C&);
#pragma link C++ function A::doit(const C&);
#endif
