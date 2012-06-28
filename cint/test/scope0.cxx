/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
char a='a';
int b=2351;
class complex {
 public:
  double re,im;
  complex(double rein=1,double imin=2) { re=rein; im=imin; }
  void disp() {
    int re=3,im=4;
    printf("re=%d im=%d complex::re=%g complex::im=%g\n"
	   ,re,im,complex::re,complex::im);
  }
};
complex c;

struct B;

struct A : public complex {
 public:
  int re,im;	
  int a;
  A(int ain=3229) : complex(2345,3456) { a=ain; re=9999; im=8888; }
  void disp() {
    printf("a=%d re=%d im=%d complex::re=%g complex::im=%g\n"
	   ,a,re,im,complex::re,complex::im);
    complex::disp();
  }
};

A obja;

struct B : public complex {
  int b;
  B(int bin=3230) : complex(456,789) { b=bin; }
  void disp() {
    printf("b=%d complex::re=%g complex::im=%g\n"
	   ,b,complex::re,complex::im);
    complex::disp();
  }
};

struct C: public B , public A {
  int c;
  C() { c=-1234; }
  void disp() {
    printf("c=%d\n",c);
    A::disp();
    B::disp();
    //A::complex::disp();
    //B::complex::disp();
  }
};

void test1(){
  long a=1234;
  double b=3.14;
  complex c(10,11);
  C objc;

  printf("test1\n");
  printf("a=%ld ::a='%c' b=%g ::b=%d\n",a,::a,b,::b);
  ::c.disp();
  c.disp();

  obja.disp();
  obja.complex::disp();

  objc.disp();
}

void test2()
{
  printf("test2\n");
  int i;
  C ary[5];	

  for(i=0;i<5;i++) {
    ary[i].A::a=i;
    ary[i].B::b=i*2;
    ary[i].A::re = i*3;
    ary[i].A::im = i*4;
  }
  for(i=0;i<5;i++) {
    printf("A::a=%d B::b=%d A::re=%d A::im=%d\n"
	   ,ary[i].A::a,ary[i].B::b,ary[i].A::re,ary[i].A::re);
  }
  for(i=0;i<5;i++) {
    ary[i].disp();
    ary[i].A::disp();
    ary[i].B::disp();
  }

}

int main() {
  test1();
  test2();
  return 0;
}
