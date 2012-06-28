/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//
// Array precompiled class simple test program
//
#include <iostream.h>
#include <stdio.h>
#include "Fundament.h"
#include "Complex.h"
#include "Array.h"

#if defined(_WINDOWS) || defined(_WIN32)
#define G__NOIMPLICITCONV
#endif

void test1()  // Array<int> 
{
  printf("Array<int> test\n");
  iarray x(0,9,10),y(1,10,10);
  iarray z;

  z = x*y;

  for(int i=0;i<10;i++) {
    printf("(x[%d]=%d)*(y[%d]=%d)=(z[%d]=%d)\n",i,x[i],i,y[i],i,z[i]);
  }
}

void test2()  // Array<double> 
{
  printf("Array<double> test\n");
  darray x(0.0,9.0,10) , y(2.0,20.0,10);
  darray z;
  z = x/y;

  for(int i=0;i<10;i++) {
    printf("(x[%d]=%g)*(y[%d]=%g)=(z[%d]=%g)\n",i,x[i],i,y[i],i,z[i]);
  }
}

void test3()  // Array<Complex> 
{
  printf("Array<Complex> test\n");
#ifdef G__NOIMPLICITCONV
  Complex a(0,0),b(4,0),c(2,0),d(7,0);
  carray x=Array<Complex>(a,b,5);
  carray y=Array<Complex>(c,d,5);
#else
#if (G__CINTVERSION<5014035)
  carray x(0,4,5);
#else
  carray x(0.0,4,5);
#endif
  carray y(2,7,5);
#endif
  carray z;

  z = x+y;

#ifndef TEST
  x.disp(cout);
  y.disp(cout);
  z.disp(cout);
#else
  cout << x ;
  cout << y ;
  cout << z ;
#endif
}


main()
{
  test1();
  test2();
  test3();
}
