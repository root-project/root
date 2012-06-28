/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

#if defined(interp) && defined(makecint)
#include "test.dll"
#else
#include "t516.h"
#endif

int main() {
  arr b;
  
  b[0][1] = 3.14;

  arr *c;
  c = new arr[3];

  c[0][0][1];
  c[0][0][1] = 3.14;
  printf("%g\n",c[0][0][1]);
  int i,j,k;
  for(i=0;i<3;i++) 
    for(j=0;j<2;j++) 
      for(k=0;k<2;k++) c[i][j][k] = i*100+j*10+k;
  for(i=0;i<3;i++) 
    for(j=0;j<2;j++) 
      for(k=0;k<2;k++) printf("%g\n",c[i][j][k]);
  delete[] c;
  return 0;
}
