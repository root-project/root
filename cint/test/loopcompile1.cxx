/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*
* loop compilation if statement test
*/


#include <stdio.h>


void test1()
{
  int i;
  int a[10];
  for(i=0;i<10;i++) {
    a[i]=i*3;
  }
  for(i=0;i<10;i++) {
    printf("a[%d]=%d\n",i,a[i]);
  }
}

/* if else compile */
void test2()
{
  int i;
  int a[10];
  for(i=0;i<10;i++) {
    a[i]=i;
    if(i%2) {
      a[i]=a[i]*2;
    }
    else {
      a[i]= -1*a[i];
    }
  }
  for(i=0;i<10;i++) {
    printf("a[%d]=%d\n",i,a[i]);
  }
}

#define NUM 10

/* nested if compile */
void test3()
{ 
  int i;
  int a[NUM];
  for(i=0;i<NUM;i++) {
    a[i]=i;
    if(i%2) {
      if(i==5) a[i]=1234;
      else a[i]=a[i]*2;
    }
    else {
      if(i==4) a[i]=3229;
      else a[i]= -1*a[i];
    }
    if(i==100) {
      a[0]=122;
    }
  }
  for(i=0;i<10;i++) {
    printf("a[%d]=%d\n",i,a[i]);
  }
}

/* abort compile */
void test4()
{
  int i;
  int a[NUM];
  for(i=0;i<NUM;i++) { 
    a[i]=i;
    if(i%2) {
      if(i==5) a[i]=1234;
      else a[i]=a[i]*2;
    }
    else {
      switch(i) { /* abort compile */
      case 4:
	a[i]=3229;
	break;
      default:
	a[i]= -1*a[i];
	break;
      }
    }
  }
  for(i=0;i<10;i++) {
    printf("a[%d]=%d\n",i,a[i]);
  }
}

/* nested loop in if */
void test5()
{
  int i,j;
  int a[NUM];
  for(i=0;i<NUM;i++) {
    a[i]=i;
    if(i==8) {
      for(j=0;j<5;j++) {
	a[0]=j;
      }
      a[8]=16;
    }
    a[1]= -1;
  }
  for(i=0;i<10;i++) {
    printf("a[%d]=%d\n",i,a[i]);
  }
}

/* nested loop in if */
void test6()
{
  int i,j;
  int a[NUM];
  for(i=0;i<NUM;i++) {
    a[i]=i;
    if(i==8) {
      for(j=0;j<i;j++) {
	a[0]+=j;
      }
      a[8]=16;
    }
    a[1]= -1;
  }
  for(i=0;i<10;i++) {
    printf("a[%d]=%d\n",i,a[i]);
  }
}

void test7()
{
  int i,j;
  int a[NUM];
  for(i=0;i<NUM;i++) {
    a[i]=i;
    if(i==8) {
      j=0;
      do {
	a[0]+=j;
	j++;
      } while(j<i);
      a[8]=16;
    }
    a[1]= -1;
  }
  for(i=0;i<10;i++) {
    printf("a[%d]=%d\n",i,a[i]);
  }
}

int main() {
  test7();   // a[0] 28 vs 0
  test6();   // ok
  test5();   // ok
  test4();   // a[0],[2],[6],[8]  0,-2,-6,-8 vs 3229,3229,3229,3229
  test3();   // ok
  test2();   // ok
  test1();   // ok

  return 0;
}

