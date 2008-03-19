/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

class B;
class C;

class A {
  int a;
 public:
  A(int in=3229) { a=in; }
  void disp() { printf("a=%d\n",a); }
  friend class B;
  friend class C;
  friend void f(A& a,B& b);
  friend void test(void);
};

class B {
  int b;
  B(int in=4000) { b = in; }
  void disp(A& a) { printf("a.a=%d b=%d\n",a.a,b); }
  friend void f(A& a,B& b);
  friend void test();
};

class C {
  static void disp(A& a) { printf("a.a=%d\n",a.a); }
  friend void test();
};


void f(A& a,B& b) {
  printf("a.a=%d b.b=%d\n",a.a,b.b);
}

void test()
{
  A a;
  B b;

  a.disp();
  b.disp(a);
  C::disp(a);
  f(a,b);
}

class complex {
 public:
  complex(double rein=0.0,double imin=0.0) { re=rein; im=imin; }
  complex& operator=(const complex& c);
  friend complex operator+(const complex& a,const complex& b);
  friend complex operator-(const complex& a,const complex& b);
  void disp() { printf("%g %g\n",re,im); }
 private:
  double re,im;
};

complex& complex::operator=(const complex& c)
{
  re=c.re;
  im=c.im;
  return(*this);
}

complex operator+(const complex& a,const complex& b)
{
  complex c;
  c.re = a.re+b.re;
  c.im = a.im+b.im;
  return(c);
}

complex operator-(const complex& a,const complex& b)
{
  complex c;
  c.re = a.re-b.re;
  c.im = a.im-b.im;
  return(c);
}

void test2() {
  complex a(1,2),b(5,6),c;
  a.disp(); b.disp(); c.disp();
  c=a+b;
  c.disp();
  c=a-b;
  c.disp();
}

int main()
{
  test();
  test2();
  return 0;
}

