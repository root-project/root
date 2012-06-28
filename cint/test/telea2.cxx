/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif
#include <stdio.h>

class A {
  double a1,a2;
 public:
  A() { a1=1; a2=3.14; }
  //virtual ~A() { cout << "~A()" << endl; };
  virtual ~A() { printf("~A()\n"); };
  virtual void disp() { printf("a1=%g a2=%g\n",a1,a2); }
};

class B : public virtual 
A  {
  double b1,b2;
 public:
  B() { b1=2; b2=6.28; }
  virtual ~B() { printf("~B()\n"); };
  virtual void disp() { printf("b1=%g b2=%g\n",b1,b2); }
};

class C : public virtual A  {
  double c1,c2;
 public:
  C() { c1=3; c2=16.28; }
  virtual ~C() { printf("~C()\n"); };
  virtual void disp() { printf("c1=%g c2=%g\n",c1,c2); }
};

class D : public B, public C {
  double d1,d2;
 public:
  D() { d1=4; d2=26.28; }
  virtual ~D() { printf("~D()\n"); };
  virtual void disp() { printf("d1=%g d2=%g\n",d1,d2); }
};


void test1() {
  B* pb = new B;
  A* pa = pb;
  pa->A::disp();
  // pa->B::disp();
  pa->disp();
  // pb = (B*)pa;
  pb->disp();
  delete pa;
}

void test2() {
  C* pc = new C;
  A* pa = pc;
  pa->A::disp();
  // pa->C::disp();
  pa->disp();
  // pc = (C*)pa;
  pc->disp();
  delete pa;
}

void test3() {
  D* pd = new D;
  A* pa = pd;
  pa->A::disp();
  //pa->B::disp();
  //pa->C::disp();
  //pa->D::disp();
  pa->disp();
  //pd = (D*)pa;
  pd->disp();
  delete pa;
}

void test4() {
  for(int i=0;i<5;i++) {
    B* pb = new B;
    A* pa = pb;
    pa->A::disp();
    // pa->B::disp();
    pa->disp();
    // pb = (B*)pa;
    pb->disp();
    delete pa;
  }
}

void test5() {
  for(int i=0;i<5;i++) {
    C* pc = new C;
    A* pa = pc;
    pa->A::disp();
    // pa->C::disp();
    pa->disp();
    // pc = (C*)pa;
    pc->disp();
    delete pa;
  }
}

void test6() {
  for(int i=0;i<5;i++) {
    D* pd = new D;
    A* pa = pd;
    pa->A::disp();
    //pa->B::disp();
    //pa->C::disp();
    //pa->D::disp();
    pa->disp();
    //pd = (D*)pa;
    pd->disp();
    delete pa;
  }
}

int main() {
  test1();
  test2();
  test3();

  test4();
  test5();
  test6();
  return 0;
}
