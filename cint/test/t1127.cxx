/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t1127.h"
#endif

int testi() {
  struct vadat x;
  A a;
  a.a = 345;
  a.b = 6.28;
  a.c = 3229;
  a.d = 'x';
  a.e = 1.4142;
  x.buf[0] = 0;
  f("sdisrcu",2,"abcdefghijklmn",3.14,1234,"A",(short)12,'a',a);
  //g("sdis",2,x);
  return 0;
}

int main() {
  testc();
  testi();
  return 0;
}
