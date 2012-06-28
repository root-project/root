/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

int main() {
  int fab[4] = {4500,4900,5000,5606};
  int runnumber;
  int p=0;
  while(1) {
    runnumber = fab[p++];
    printf("%d\n",runnumber);
    if(runnumber == 4500) {break;}
  }
  return 0;
}


