/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>
#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t1024.h"
#endif


void disp1(double a[][10]) {
  printf("-------------------------------\n");
  for(int i=0;i<20;i++) {
    for(int j=0;j<10;j++) {
      printf("%5g ",a[i][j]);
    }
    printf("\n");
  }
}

void test1() {
  double x[20][10];

  for(int i=0;i<20;i++) {
    for(int j=0;j<10;j++) {
      x[i][j] = i*100+j;
    }
  }
  disp1(x);

}


///////////////////////////////////////////////////////////
void disp2(double** a) {
  printf("-------------------------------\n");
  for(int i=0;i<20;i++) {
    for(int j=0;j<10;j++) {
      printf("%5g ",a[i][j]);
    }
    printf("\n");
  }
}


void test2() {
  int i,j;
  double* x[20];
  double y[20][10];

  for(i=0;i<20;i++) {
    x[i] = new double[10];
    for(j=0;j<10;j++) {
      x[i][j] = i*200+j;
    }
  }

  disp2(x);

  A a;
  a.d=x;
  for(i=0;i<20;i++) {
    for(j=0;j<10;j++) {
      //printf("%d %d\n",i,j);
      y[i][j] = a.ev_sc()[i][j];
    }
  }

  disp1(y);

  for(i=0;i<20;i++) delete[] x[i];
}


int main() {
  test1();
  test2();
  return 0;
}

