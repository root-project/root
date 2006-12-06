/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

class complex {
   double re,im;
 public:
   complex(double rein=0.0,double imin=0) { re=rein;im=imin;}
   complex& operator+=(complex& a) { re+=a.re; im+=a.im; return(*this);}
   complex& operator-=(complex& a) { re-=a.re; im-=a.im; return(*this);}
   void disp() { printf("%g %g\n",re,im); }
 friend complex operator+(complex& a, complex& b);
 friend complex operator-(complex& a, complex& b);
   complex& operator++() { re++; return(*this);}
   complex& operator++(int i) { im++; return(*this);}
 friend complex& operator--(complex& a);
 friend complex& operator--(complex& a, int i);
};

complex& operator--(complex& a) { a.re--; return(a);}
complex& operator--(complex& a, int) { a.im--; return(a);}

complex operator+(complex& a, complex& b) {
 complex c;
 c.re = a.re+b.re;
 c.im = a.im+b.im;
 return(c);
}

complex operator-(complex& a, complex& b) {
 complex c;
 c.re = a.re-b.re;
 c.im = a.im-b.im;
 return(c);
}


int main() {
  int i;
  complex a(1,2),b(4,5),c;
  c.disp();

  c += a;
  c.disp();

  c -= b;
  c.disp();

  c++;
  c.disp();

  ++c;
  c.disp();

  c--;
  c.disp();

  --c;
  c.disp();

  for(i=0;i<5;i++) {
    c += a;
    c.disp();
  }

  for(i=0;i<5;i++) {
    c -= b;
    c.disp();
  }

  for(i=0;i<5;i++) {
    c++;
    c.disp();
  }
  for(i=0;i<5;i++) {
    ++c;
    c.disp();
  }
  for(i=0;i<5;i++) {
    c--;
    c.disp();
  }
  for(i=0;i<5;i++) {
    --c;
    c.disp();
  }

  return 0;
}
