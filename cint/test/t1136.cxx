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
  double b;
  float f;
};

float& f1(float &x) { return x; }
A& f2(A& x) { return x; }

void test() {
  float g;
  float ary[10];
  A a;

  float* pf;

  for(int i=0;i<3;i++) {

    pf = &g;
    printf("%d ",&g==pf);
    
    pf = &(g);
    printf("%d ",&g==pf);
    
    pf = &((g));
    printf("%d ",&g==pf);
    
    pf = &f1(g);
    printf("%d ",&g==pf);
    
    pf = &ary[3];
    printf("%d ",&ary[3]==pf);
    
    pf = &(ary[3]);
    printf("%d ",&ary[3]==pf);
    
    pf = &((ary[3]));
    printf("%d ",&ary[3]==pf);
    
    pf = &a.f;
    printf("%d ",&a.f==pf);
    
    pf = &(a.f);
    printf("%d ",&a.f==pf);
    
    pf = &((a.f));
    printf("%d ",&a.f==pf);
    
    pf = &f2(a).f;
    printf("%d ",&a.f==pf);
    
    pf = &(f2(a).f);
    printf("%d ",&a.f==pf);
    printf("\n");
  }
}

A a;

void test2() {
  float g;
  float ary[10];

  float* pf;

  for(int i=0;i<3;i++) {

    pf = &g;
    printf("%d ",&g==pf);
    
    pf = &(g);
    printf("%d ",&g==pf);
    
    pf = &((g));
    printf("%d ",&g==pf);
    
    pf = &f1(g);
    printf("%d ",&g==pf);
    
    pf = &ary[3];
    printf("%d ",&ary[3]==pf);
    
    pf = &(ary[3]);
    printf("%d ",&ary[3]==pf);
    
    pf = &((ary[3]));
    printf("%d ",&ary[3]==pf);
    
    pf = &a.f;
    printf("%d ",&a.f==pf);
    
    pf = &(a.f);
    printf("%d ",&a.f==pf);
    
    pf = &((a.f));
    printf("%d ",&a.f==pf);
    
    pf = &f2(a).f;
    printf("%d ",&a.f==pf);
    
    pf = &(f2(a).f);
    printf("%d ",&a.f==pf);
    printf("\n");
  }
}

int main() {
  test();
  for(int i=0;i<2;i++) test2();
  return 0;
}
