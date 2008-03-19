/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
int main()
{
  int a = 1<<3;
  int b = 2 << 5;
  int c = ( 3 << 2 );
  printf("a=%d b=%d c=%d\n",a,b,c);

  return 0;
}
