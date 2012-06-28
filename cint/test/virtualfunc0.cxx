/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//
// virtual function test
//
#include <stdio.h>
#include <stdlib.h>

class A {
 public:
  int a;
  A(int i) { a=i; }
  virtual void print(void) { printf("a=%d\n",a); }
};

class B {
 public:
  int b;
  B(int i) { b=i; }
  void print(void) { printf("b=%d\n",b); }
};

class C: public A, public B {
 public:
  int c;
  C(int i) : A(i+1) , B(i+2) { c=i; }
  void print(void) { printf("c=%d\n",c); }
};

class D: public C {
 public:
  int d;
  D(int i) : C(i+10) { d=i; }
  void print(void) { printf("d=%d\n",d); }
};

int main()
{
  A aobj=A(1);
  B bobj=B(101);
  C cobj=C(201);
  D dobj=D(301);

  A *pa, /* *pb, */ *pc,*pd;
  B *PB;

  pa = &aobj;
#if 0
  fprintf(stderr,"Intentional error, pb below\n");
  pb = &bobj;
#endif
  PB = &bobj;
  pc = &cobj;
  pd = &dobj;

  pa->print();
  PB->print();
  pc->print();
  pd->print();

  return 0;
}
