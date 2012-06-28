/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// Src.C , compiled function body

#include "Src.h"
#include "Stub.h"
#include <stdio.h>

double f2(double din,int iin) { 
  return(f1(iin,din));  // f1() is interpreted
}


int pain() {
  printf("pain result=%g\n",f2(3.14,2)); // f2() is compiled
  StubClass a;
  double y=a.set(2,3.14);
  int x=a.disp();
  printf("pain x,y: %d %g\n",x,y);

  StubBase *ary[5];
  ary[0] = new StubClass(3,1.2);
  ary[1] = new StubBase(34);
  ary[2] = new StubBase(56);
  ary[3] = new StubClass(-2,2.3);
  ary[4] = new StubBase(-12);
  for(int i=0;i<5;i++) ary[i]->disp();

  return(0);
}

