/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

class A {
public: 
  int a;
};

class B : public A {
public: 
  int a;
};

void f(A x) {
  printf("%d\n",x.a);
}

void test1() {
  A a;
  B b;
  a.a = -12;
  b.a = 1;
  b.A::a = 2;
  printf("%d %d %d\n", b.a , b.A::a , a.a);
  a=b;
  printf("%d %d %d\n", b.a , b.A::a , a.a);
  f(a);
  f(b);

  A *pa = &a;
  A *pb = &b;
  pa->a = 123;
  printf("%d %d %d\n", pb->a , pb->A::a , pa->a);
  *pa = *pb;
  printf("%d %d %d\n", pb->a , pb->A::a , pa->a);
}

void test2() {
  A a[5];
  B b[5];
  int i;
  for(i=0;i<5;i++) {
    a[i].a = -12;
    b[i].a = 1;
    b[i].A::a = 2;
    printf("%d %d %d\n", b[i].a , b[i].A::a , a[i].a);
    a[i]=b[i];
    printf("%d %d %d\n", b[i].a , b[i].A::a , a[i].a);
    f(a[i]);
    f(b[i]);
  }

  A *pa = a;
  A *pb = b;
  for(i=0;i<5;i++) {
    pa[i].a = 123;
    printf("%d %d %d\n", pb->a , pb[i].A::a , pa[i].a);
    pa[i] = pb[i];
    printf("%d %d %d\n", pb->a , pb[i].A::a , pa[i].a);
  }
}

int main()
{
  test1();
  test2();
  return 0;
}

