/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

class A {
  int a;
 public:
  A() { a = 1; }
  virtual void f(int x) {
    printf("A::f(%d) %d\n",x,a);
  }
};

class B : public A {
  int b;
 public:
  B() { b = 2; }
  virtual void f(int x) {
    printf("B::f(%d) %d\n",x,b);
  }
};


class C : public B {
  int c;
 public:
  C() { c = 3; }
};

class D { 
 int d;
public:
  D() { d = 4; }
};

//class E : public C, public D {
class E : public D, public C {
  int e;
 public:
  E() { e = 5; }
};

class F : public C, public D {
  int f;
 public:
  F() { f = 6; }
};

int main() {
  A obja;
  B objb;
  C objc;
  E obje;
  F objf;

  A *p[5] = { &obja , &objb , &objc , &obje };
  p[3] = &obje;
  p[4] = &objf;

  for(int i=0;i<5;i++) {
    p[i]->f(i);
  }

  return 0;
}
