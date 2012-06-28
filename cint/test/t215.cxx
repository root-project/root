/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// class/struct without explicit dtor has class object with dtor as member
// or base class.

#include <stdio.h>

int x=0;
double y=100;

#define this 0

class A {
public:
  int a;
  A() { a=x++; printf("%x.A() %d\n",this,a); }
  ~A() { printf("%x.~A() %d\n",this,a); }
};

class B {
public:
  double a;
  B() { a=x++; printf("%x.B() %g\n",this,a); }
  ~B() { printf("%x.~B() %g\n",this,a); }
};

class A1 {
public:
  int a;
  A1() { a=x++; printf("%x.A1() %d\n",this,a); }
  ~A1() { printf("%x.~A1() %d\n",this,a); }
};

class B1 {
public:
  double a;
  B1() { a=x++; printf("%x.B1() %g\n",this,a); }
  ~B1() { printf("%x.~B1() %g\n",this,a); }
};

class A2 {
public:
  int a;
  A2() { a=x++; printf("%x.A2() %d\n",this,a); }
  ~A2() { printf("%x.~A2() %d\n",this,a); }
};

class B2 {
public:
  double a;
  B2() { a=x++; printf("%x.B2() %g\n",this,a); }
  ~B2() { printf("%x.~B2() %g\n",this,a); }
};

class C 
  : virtual public A1
   , public B1
   , public A2 
{
public:
  A a;
  B b;
  A a2[3];
  B b2[2];
  C* next;
#ifdef TESTC
  C() { printf("%x.C()\n",this); }
#endif
#ifdef TESTD
  ~C() { printf("%x.~C()\n",this); }
#endif
};


void test() {
  C* first=0;
  C* last=0;
  int i;
  for(i=0;i<3;i++) {
    printf("------------------------\n");
    if(0==first) {
      first = new C;
      first->next =0;
      last = first;
    }
    else {
      last->next = new C;
      last = last->next;
      last->next = 0;
    }
  }

  C* p=first;
  C* next;
  while(p) {
    printf("------------------------\n");
    next = p->next;
    delete p;
    p=next;
  }
}


class D {
public:
  D() { test(); }
};

void f() {
  D* p;
  p= new D;
  delete p;
}

int main() {
  test();
  //for(int i=0;i<2;i++) {f();}
  return 0;
}
