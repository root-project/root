/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <ertti.h>

void test(int a) {
 printf("test(%d)\n",a);
}

doit() {
  G__ClassInfo c;
  G__CallFunc c1;
  long l=0;
  c1.SetFunc(&c,"test","123",&l);
  c1.Exec(&l);
}
