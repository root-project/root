/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

int x=0;
class A {
  int i;
 public:
  A() { i=x; printf("A() %d\n",x++); }
  ~A() { printf("~A() %d\n",i); }
};

class B : public A {
};

template <class T> class C : public A {
  int i;
 public:
  C() { i=x; printf("C() %d\n",x++); }
  ~C() { printf("~C() %d\n",i); }
};

A a;

int main() {
 printf("main()\n");
 A b;
 A *p=&b;
 a.~A();
 b.~A();
 p->~A();

 B c;
 c.A::~A();
 //c.~A::A();

 C<int> d;
#if !defined(CINT_HIDE_FAILURE)
 d.~C<int>(); // this must call both ~C<int>() and ~A(), need to fix this
#endif
 //d.~C<int>::A();
 d.A::~A();

 printf("end main()\n");

 return 0;
} 
