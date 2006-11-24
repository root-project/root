/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>

struct A {
  int ref;
};

struct A obja;


void test1()
{
  int i=0;

  while(1) {
    if(i) {
      i=obja.ref;
      printf("i=%d\n",i);
      return;
    }
    break;
  }

  printf("test1\n");
}

void test2()
{
  int i,j;
#if defined(__GNUC__) && (__GNUC__>=3)
  int ary[5][5] = {
    {0,0,0,0,0},
    {0,0,0,0,0},
    {0,0,0,0,0},
    {0,0,0,0,0},
    {0,0,0,0,0} 
  };
#else
  int ary[5][5] = {0};
#endif
  for(i=0;i<5;i++) {
    for(j=0;j<i;j++) {
      ary[i][j]= (i+1)*10 + (j+1);
    }
  }
  for(i=0;i<5;i++) {
    for(j=0;j<5;j++) {
      printf("ary[%d][%d]=%d\n",i,j,ary[i][j]);
    }
  }
  printf("test2\n");
}

int main()
{
  test1();
  test2();
  return 0;
}
