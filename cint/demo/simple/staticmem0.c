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
  double b;
  B(double in) { b=in;}
  ~B() { printf("~B\n"); }
};

class A {
 public:
  static B objb;
  static int a;
  static int ary[5];
  static void f(int ain) ;
  A() { printf("a=%d objb.b=%g\n",a,objb.b); }
  ~A() { printf("~A\n"); }
};

void A::f(int ain)
{
  int i;
  printf("ain=%d a=%d objb.b=%g\n",ain,a,objb.b);

  for(i=0;i<5;i++) {
    ary[i]=i+1;
  }
  for(i=0;i<5;i++) {
    printf("A::ary[%d]=%d\n",i,A::ary[i]);
  }

}

int A::a=1234 ;
B A::objb=B(3.14);
int A::ary[5];


A obja;


main()
{
  int i;
 A::f(1);

  for(i=0;i<5;i++) {
    A::ary[i]=i+10;
  }
  for(i=0;i<5;i++) {
    printf("A::ary[%d]=%d\n",i,A::ary[i]);
  }
  
}
