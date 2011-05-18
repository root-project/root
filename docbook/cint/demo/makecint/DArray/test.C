/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//
// Array precompiled class simple test program
//
#include <stdio.h>
#include "DArray.h"

void test1()  
{
  printf("Array<double> test\n");
  DArray x(0.0,9.0,10) , y[5];
  int n;
  for(n=0;n<5;n++) {
    if(0) break;
    if(n%2) {
	y[n] = x*x;
    }
    else {
	y[n] = x*x*x;
    }
  }

  int i;
  for(n=0;n<5;n++) {
    printf("%d : ",n);
    for(i=0;i<10;i++) {
      printf("%g ",y[n][i]);
    }
    printf("\n");
  }
}

void test2()  
{
  printf("Array<double> test\n");
  DArray x(0.0,9.0,10) , y[5];
  int n;
  for(n=0;n<5;n++) {
    if(n%2) {
      y[n] = x*x;
    }
    else {
      y[n] = x*x*x;
    }
  }

  int i;
  for(n=0;n<5;n++) {
    printf("%d : ",n);
    for(i=0;i<10;i++) {
      printf("%g ",y[n][i]);
    }
    printf("\n");
  }
}

void test3()  
{
  printf("Array<double> test\n");
  DArray x(0.0,9.0,10) , y[5];
  int n;
  for(n=0;n<5;n++) {
    if(0) goto xxx;
    if(n%2) {
      y[n] = x*n*2;
    }
    else {
      y[n] = x*n*(-1);
    }
  }
xxx:

  int i;
  for(n=0;n<5;n++) {
    printf("%d : ",n);
    for(i=0;i<10;i++) {
      printf("%g ",y[n][i]);
    }
    printf("\n");
  }
}

void test4()  
{
  printf("Array<double> test\n");
  DArray x(0.0,9.0,10) , y[5];
  int n;
  for(n=0;n<5;n++) {
    if(n%2) {
      y[n] = x*n*2;
      //y[n] = x;
    }
    else {
      y[n] = x*n*(-1);
      //y[n] = x*1;
    }
  }

  int i;
  for(n=0;n<5;n++) {
    printf("%d : ",n);
    for(i=0;i<10;i++) {
      printf("%g ",y[n][i]);
    }
    printf("\n");
  }
}


main()
{
  test1(); 
  test2();
  test3();
  test4();
}
