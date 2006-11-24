/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

class B {
 public:
  int a;
};

class A {
  void f() const ;
 public:
  int a;
  A() { }
  A(const B& x) { printf("copy ctor\n"); }
  //void operator  = (const B& x) { printf("operator =\n"); }
};

class B1 {
 public:
  int a;
};

class A1 {
  void f() const ;
 public:
  int a;
  A1() { }
  A1(const B1& x) { printf("copy ctor\n"); }
  void operator = (const B1& x) { printf("operator =\n"); }
};

class B2 {
 public:
  int a;
};

class A2 {
  void f() const ;
 public:
  int a;
  A2() { }
  //A2(const B1& x) { printf("copy ctor\n"); }
  void operator = (const B2& x) { printf("operator =\n"); }
};

//void operator = (A2& obj,const B2& x) { printf("global operator =\n"); }


void test1() {
  A a; B b;
  a = b;
  A *pa=&a; B *pb=&b;
  *pa=*pb;

  A1 a1; B1 b1;
  a1 = b1;
  A1 *pa1=&a1; B1 *pb1=&b1;
  *pa1 = *pb1;

  A2 a2; B2 b2;
  a2 = b2;
  A2 *pa2=&a2; B2 *pb2=&b2;
  *pa2 = *pb2;
}

void test2() {
  A a[5]; B b[5];
  A *pa=a; B *pb=b;
  int i;
  for(i=0;i<5;i++) {
    a[i] = b[i];
    pa[i] = pb[i];
  }

  A1 a1[5]; B1 b1[5];
  A1 *pa1=a1; B1 *pb1=b1;
  for(i=0;i<5;i++) {
    a1[i] = b1[i];
    pa1[i] = pb1[i];
  }

  A2 a2[5]; B2 b2[5];
  A2 *pa2=a2; B2 *pb2=b2;
  for(i=0;i<5;i++) {
    a2[i] = b2[i];
    pa2[i] = pb2[i];
  }
}

int main()
{
  test1();
  test2();
  return 0;
}

