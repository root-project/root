/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

void testx(double d[][4][5]) {
  int i,j,k;
  for(i=0;i<3;i++) {
    for(j=0;j<4;j++) {
      for(k=0;k<5;k++) {
	printf("%g ",d[i][j][k]);
      }
      printf(" : ");
    }
    printf("\n");
  }
}

void test1() {
  double d[3][4][5] = 
    {
      { { 1, 2 } , { 3, 4 } , { 5, 6 , 7 } } ,
      { { 8, 9 } , { 10, 11 } } ,
      { { 12, 13 } , { 14, 15 } }
    };

  int i,j,k;
  for(i=0;i<3;i++) {
    for(j=0;j<4;j++) {
      for(k=0;k<5;k++) {
	printf("%g ",d[i][j][k]);
      }
      printf(" : ");
    }
    printf("\n");
  }
  testx(d);
}

void test2() {
  double d[3][4][5] = 
    { 1, 2 , 3, 4 , 5, 6 , 7 , 8, 9 
      , 10, 11, 12, 13,14, 15, 16, 17, 18,19, 20, 21,22,23,24,25,26 };

  int i,j,k;
  for(i=0;i<3;i++) {
    for(j=0;j<4;j++) {
      for(k=0;k<5;k++) {
	printf("%g ",d[i][j][k]);
      }
      printf(" : ");
    }
    printf("\n");
  }
}

void test3() {
  double d[][4][5] = 
    {
      { { 1, 2 } , { 3, 4 } , { 5, 6 , 7 } } ,
      { { 8, 9 } , { 10, 11 } } ,
      { { 12, 13 } , { 14, 15 } }
    };

  int i,j,k;
  for(i=0;i<3;i++) {
    for(j=0;j<4;j++) {
      for(k=0;k<5;k++) {
	printf("%g ",d[i][j][k]);
      }
      printf(" : ");
    }
    printf("\n");
  }
  testx(d);
}

int main() {
  test1();
  test2();
  test3();
  return 0;
}

