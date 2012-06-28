/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

class A {
 public:
  unsigned int a : 1 ;
  unsigned int b : 2 ;
  unsigned int c : 3 ;
  unsigned int   : 0 ;

  unsigned int d : 4 ;
  unsigned int   : 2 ;
  unsigned int e : 1 ;

  void disp() {
    printf("%x %x %x %x %x\n",a,b,c,d,e);
  }
};

int main() {
 A a;

 a.a = 0;
 a.b = 0;
 a.c = 0;
 a.d = 0;
 a.e = 0;

 a.a = 0xffff;
 a.disp();
 a.b = 0xffff;
 a.disp();
 a.c = 0xffff;
 a.disp();
 a.d = 0xffff;
 a.disp();
 a.e = 0xffff;
 a.disp();

 a.a = 0;
 a.disp();
 a.b = 0;
 a.disp();
 a.c = 0;
 a.disp();
 a.d = 0;
 a.disp();
 a.e = 0;
 a.disp();

 a.a = 1;
 a.disp();
 a.b = 1;
 a.disp();
 a.c = 1;
 a.disp();
 a.d = 1;
 a.disp();
 a.e = 1;
 a.disp();

 return 0;
}
