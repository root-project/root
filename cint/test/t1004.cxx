/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

int main() {
  bool b[3] = {true,false,true};
  for(int i=0;i<3;i++) printf("bool b[%d]=%d\n",i,b[i]);
  return 0;
}
