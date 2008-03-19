/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// Stub.C , interpreted source code

#include "Stub.h"
#include "Src.h"

main() {
  pain();
  printf("result=%g\n",f2(3.14,2)); // f2() is compiled
  StubClass a;
  double y=a.set(2,3.14);
  int x=a.disp();
  printf("%d %g\n",x,y);

  StubBase *ary[5];
  ary[0] = new StubClass(3,1.2);
  ary[1] = new StubBase(34);
  ary[2] = new StubBase(56);
  ary[3] = new StubClass(-2,2.3);
  ary[4] = new StubBase(-12);
  for(int i=0;i<5;i++) ary[i]->disp();
}

double f1(int a,double b) {
  return(a*b);	
}

#ifndef TEST
short StubBase::set(short bin) {
  b=bin;
  return(b);
}

int StubBase::disp() {
  printf("%d\n",b);
  return(b);
}

double StubClass::set(int ain,double din) {
  a=ain;
  d=din;
  return(a*d);
}

int StubClass::disp() {
  printf("%d %g\n",a,d);
  return(a);
}

#else

short StubBase::$set(short bin) {
  b=bin;
  return(b);
}

int StubBase::$disp() {
  printf("%d\n",b);
  return(b);
}

double StubClass::$set(int ain,double din) {
  a=ain;
  d=din;
  return(a*d);
}

int StubClass::$disp() {
  printf("%d %g\n",a,d);
  return(a);
}

#endif
