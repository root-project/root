/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// conversion operator to be supported

#include <stdio.h>

class A {
 public:
  double a1,a2;
  void disp() { printf("a1=%g a2=%g\n",a1,a2); }
  A() { a1=1;a2=3.14; } 
};

class B {
 public:
  double b1,b2;
  B() { b1=2.2;b2=6.28; }
  void disp() { printf("b1=%g b2=%g\n",b1,b2); }
  operator A() { A a; a.a1=b1; a.a2=b2; return a; }
  operator double() { return b1; }
  operator int() { return ((int)(b1+b2)); }
  operator void*() { return ((void*)((long)b1*0x1000+(long)b2)); }
};

void test1() {
  //A a; 
  B b;
  int i;
  int j;
  i = b;
  double d;
  d = b;
  void* pv;
  pv = b;
  printf("test1 i=%d d=%g pv=%p\n",i,d,pv);
  for(j=0;j<3;j++) {
    i=0; d =0; pv = NULL;
    i = b; d = b; pv = b;
    printf("test1 i=%d d=%g pv=%p\n",i,d,pv);
  }
}

void test2() {
  //A a; 
  B b;
  int i = b;
  int j;
  double d = b;
  void* pv = b;
  printf("test2 i=%d d=%g pv=%p\n",i,d,pv);
  for(j=0;j<3;j++) {
    int i = b; double d = b; void* pv=b;
    printf("test2 i=%d d=%g pv=%p\n",i,d,pv);
    i=0; d =0;
  }
}

typedef int Int_t;
typedef double Double_t;
typedef void* PVoid;

void test3() {
  //A a; 
  B b;
  Int_t i;
  i = b;
  Double_t d;
  d = b;
  PVoid pv;
  pv = b;
  printf("test3 i=%d d=%g pv=%p\n",i,d,pv);
  for(int j=0;j<3;j++) {
    i=0; d =0; pv = NULL;
    i = b; d = b; pv = b;
    printf("test3 i=%d d=%g pv=%p\n",i,d,pv);
  }
}

void test4() {
  A a; B b;
  int j;
  a.disp(); b.disp();
  a=b;
  a.disp(); b.disp();
  for(j=0;j<3;j++) {
    a.a1 =0; a.a2=0;
    a.disp(); b.disp();
    a=b;
    a.disp(); b.disp();
  }

  A *pa=&a; B *pb=&b;
  pa->a1 =0; pa->a2=0;
  pa->disp(); pb->disp();
  *pa = *pb;
  pa->disp(); pb->disp();
  *pa = *pb;
  for(j=0;j<3;j++) {
    pa->a1 =0; pa->a2=0;
    pa->disp(); pb->disp();
    *pa = *pb;
    pa->disp(); pb->disp();
  }
}

int main()
{
  test1();
  test2();
  test3();
  test4();
  return 0;
}
