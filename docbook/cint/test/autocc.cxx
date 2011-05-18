/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*
 *   #pragma compile, endcompile test
 */
#include <stdio.h>
#include <string.h>


// Apply following if you do not want to compile
// #pragma disablecompile

#ifdef __CINT__
#pragma compile  /* following part is compiled */
int sum=0;
void loop(int num) {
  int i=0;
  for(i=num;num>0;num--) {
    sum += num;
  }
}
#pragma endcompile
#else
int sum=0;
void loop(int num) {
  int i=0;
  for(i=num;num>0;num--) {
    sum += num;
  }
}
#endif

void endfunc() {
  printf("end of execution sum=%d\n",sum);
}

int main(int argc,char **argv) // main can not be compiled
{
  loop(100000);
  endfunc();
  return 0;
}

