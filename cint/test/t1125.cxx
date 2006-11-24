/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>
enum BIT {ZERO,ONE,X};

void f(BIT* x) { printf("f((BIT*))\n"); }

int main(){
  BIT *a;
  a=new BIT[3]; // memory leak, G__alloc_newarraylist

  void *p = a;
  BIT *b;
  b = (BIT*)p;

  f((BIT*)p);

  delete[] a;

  return(0);
}

