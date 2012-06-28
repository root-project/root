/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

void test1()
{
  printf("test1\n");
  for(int i=0;i<5;i++) {
    double w=i*2;
    printf("w=%g\n",w);
  }
}


void test2()
{
  printf("test2\n");
  for(int i=0;i<5;i++) {
    if(i) {
      double w=i*3;
      printf("w=%g\n",w);
    }
  }
}

void test3()
{
  printf("test3\n");
  for(int i=0;i<5;i++) {
    double w;
    w=i*4;
    printf("w=%g\n",w);
  }
}

void test4()
{
  printf("test4\n");
  for(int i=0;i<5;i++) {
    if(i) {
      double w;
      w=i*5;
      printf("w=%g\n",w);
    }
  }
}

void test5()
{
  printf("test5\n");
  for(int i=0;i<5;i++) {
    if(i%2) {
      double w=i*6;
      printf("w=%g\n",w);
    }
    else {
      double w=i*7;
      printf("w=%g\n",w);
    }
  }
}

void test6()
{
  printf("test6\n");
  for(int i=0;i<5;i++) {
    if(i) {
      double w;
      w=i*8;
      printf("w=%g\n",w);
    }
    else {
      double w;
      w=i*9;
      printf("w=%g\n",w);
    }
  }
}


int main()
{
  test1();
  test2();
  test3();
  test4();
  test5();
  test6();
  return 0;
}
